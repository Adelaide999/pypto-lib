# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose existing Prefill computations into L2 regions between host communication boundaries."""

from hashlib import sha256
import keyword
import linecache

import pypto.language as pl
from pypto import ir
from pypto.ir.compiled_program import extract_param_infos


# Each binding names one external tensor or scalar, shared by all uses in a region.
REGIONS = {
    "input_l2": (
        ("reduce", ("gathered", "embedding", "active")),
        ("initial_hc", ("embedding", "streams", "premix", "tp_rank", "physical", "active")),
    ),
    "head_l2": (
        ("sp_unpack", ("gathered", "collapsed", "physical", "active")),
        ("norm", ("collapsed", "norm_weight", "normalized", "active")),
        ("projection", ("normalized", "weight", "local_logits", "active")),
    ),
    "qkv_l2": (
        ("wq_a", ("q_input", "wq_a", "wq_a_scale", "wqa", "query_rows")),
        ("q_norm", ("wqa", "q_norm", "latent", "query_rows")),
        ("wq_b", ("latent", "wq_b", "wq_b_scale", "qb", "query_rows")),
        ("q_rope", ("qb", "q_cos", "q_sin", "query", "query_rows")),
        ("wkv", ("full_norm", "wkv", "wkv_scale", "kv_raw", "total")),
        ("kv_norm", ("kv_raw", "kv_norm", "kv_normalized", "total")),
        ("kv_rope", ("kv_normalized", "cosine", "sine", "kv_rope", "total")),
    ),
    "attention_pack_l2": (
        ("inverse_rope", ("context", "q_cos", "q_sin", "inverse", "query_rows")),
        ("head_pack", ("inverse", "packet", "query_rows")),
    ),
    "output_projection_l2": (
        ("head_unpack", ("exchanged", "heads", "per_rank", "total")),
        ("grouped_wo_a", ("heads", "weight_a", "woa", "total")),
        ("wo_b", ("woa", "weight_b", "wob", "total")),
        ("reduce_pack", ("wob", "packed", "physical", "total")),
    ),
    "ffn_dispatch_l2": (
        ("hc_post", ("scattered", "hidden", "attn_post", "attn_comb", "changed", "physical")),
        ("hc_pre", ("changed", "hc_fn", "hc_scale", "hc_base", "attn_pre",
                    "ffn_pre", "ffn_post", "ffn_comb", "collapsed", "physical")),
        ("norm_cast", ("collapsed", "ffn_norm", "normed", "router_input", "physical")),
        ("router", ("router_input", "gate", "bias", "bias_vl", "token_ids",
                    "logits", "ids", "weights", "routing_rows")),
        ("shared", ("normed", "gate_up_weight", "gate_up_scale", "down_weight", "down_scale", "shared", "physical")),
        ("dispatch_pack", ("normed", "ids", "routes", "packet_hidden", "counts", "valid", "rank")),
    ),
    "moe_finish_l2": (
        ("combine", ("routes", "packet_hidden", "counts", "weights", "shared",
                     "routed", "combined", "returned_routes", "physical")),
        ("mask_moe", ("combined", "valid", "physical")),
        ("hc_post", ("combined", "changed", "post", "comb", "output", "physical")),
    ),
}

STAGE_REGIONS = {
    "embedding": ("input_l2",),
    "output": ("head_l2",),
    "layer": ("qkv_l2", "attention_pack_l2", "output_projection_l2", "ffn_dispatch_l2", "moe_finish_l2"),
}


def region_bindings(steps):
    """Return the stable external ABI, including intermediate observation buffers."""
    return tuple(dict.fromkeys(binding for _, bindings in steps for binding in bindings))


def _parameter_type(param, capacity):
    """Bind only the existing leading dynamic token dimension to the network capacity."""
    shape = None if param.shape is None else tuple(capacity if size == -1 and axis == 0 else size
                                                  for axis, size in enumerate(param.shape))
    return shape, str(param.dtype), param.layout


def _annotation(param, writable, capacity):
    """Serialize a specialized child parameter without changing its storage type."""
    names = {"bfloat16": "BF16", "fp16": "FP16", "fp32": "FP32", "int8": "INT8",
             "uint8": "UINT8", "int16": "INT16", "int32": "INT32", "int64": "INT64"}
    dtype = names.get(str(param.dtype))
    if dtype is None:
        raise ValueError("Unsupported A3 L2 parameter dtype: " + str(param.dtype))
    if param.shape is None:
        if writable:
            raise ValueError("An L2 scalar cannot be writable")
        return f"pl.Scalar[pl.{dtype}]"
    shape, _, _ = _parameter_type(param, capacity)
    if any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("L2 regions require the exact positive specialized child shape")
    if param.layout not in ("ND", "NZ"):
        raise ValueError("Unsupported A3 L2 parameter layout: " + str(param.layout))
    layout = ", pl.NZ" if param.layout == "NZ" else ""
    annotation = f"pl.Tensor[{list(shape)}, pl.{dtype}{layout}]"
    return "pl.InOut[" + annotation + "]" if writable else annotation


def compose_l2(name, definitions, steps, capacity):
    """Inline the same child bodies and thread every returned tensor alias in call order.

    The generated body contains only calls and return bindings. It introduces no
    arithmetic, casts, reductions, quantization or cache-layout transformations.
    """
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("L2 regions require an explicit positive token capacity")
    bindings = region_bindings(steps)
    if not steps or any(not value.isidentifier() or keyword.iskeyword(value) for value in (name, *bindings)):
        raise ValueError("L2 regions require safe Python identifiers and at least one computation")
    namespace = {"pl": pl}
    params, writable, calls = {}, set(), []
    for index, (child_name, arguments) in enumerate(steps):
        child = definitions[child_name]
        program = child.specialize()
        infos, _, returns = extract_param_infos(program)
        if len(infos) != len(arguments):
            raise ValueError("L2 child arity differs: " + child_name)
        functions = [item for item in program.functions.values() if item.func_type == ir.FunctionType.Orchestration]
        if len(functions) != 1:
            raise ValueError("L2 children require one unambiguous orchestration entry")
        aliases = ir.returned_param_indices(functions[0], program)
        if len(aliases) != len(returns) or any(type(slot) is not int or not 0 <= slot < len(infos) for slot in aliases):
            raise ValueError("L2 child returns must alias actual arguments: " + child_name)
        for slot, (binding, info) in enumerate(zip(arguments, infos)):
            signature = _parameter_type(info, capacity)
            if binding in params:
                prior = params[binding]
                if signature != _parameter_type(prior, capacity):
                    raise ValueError("L2 binding has incompatible child types: " + binding)
            params[binding] = info
            if info.direction.name != "In" or slot in aliases:
                writable.add(binding)
        symbol = f"child_{index}"
        namespace[symbol] = pl.jit.inline(child._func)
        invocation = f"{symbol}({', '.join(arguments)})"
        targets = [arguments[slot] for slot in aliases]
        calls.append("    " + (", ".join(targets) + " = " if targets else "") + invocation)
    signature = [f"    {binding}: {_annotation(params[binding], binding in writable, capacity)}," for binding in bindings]
    results = [binding for binding in bindings if binding in writable]
    body = ["@pl.jit", f"def {name}(", *signature, "):", *calls, "    return " + ", ".join(results)]
    source = "\n".join(body) + "\n"
    filename = f"<ds41-prefill-l2-{sha256(source.encode()).hexdigest()}>"
    # PyPTO reads the function source through inspect, including generated orchestration.
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]


def make_network_l2_entries(definitions, capacity, stage="layer"):
    """Compose each stage only between its existing host communication boundaries."""
    regions = {name: REGIONS[name] for name in STAGE_REGIONS[stage]}
    covered = {name for steps in regions.values() for name, _ in steps}
    entries = {name: function for name, function in definitions.items() if name not in covered}
    entries.update({name: compose_l2(name, definitions, steps, capacity) for name, steps in regions.items()})
    return entries


def execute_region(runtime, name, rank, case, operations):
    """Run a network L2 region or the same child sequence in independent-layer validation."""
    steps = REGIONS[name]
    if tuple(item[0] for item in operations) != tuple(item[0] for item in steps):
        raise ValueError("L2 call order differs from the shared layer: " + name)
    callback = runtime.get("execute_l2")
    if callback is None:
        for operation in operations:
            child, arguments, outputs, role = operation[:4]
            child_case = operation[4] if len(operation) == 5 else case
            runtime["execute"](child, rank, child_case, arguments, outputs, role)
        return
    arguments, outputs, roles = region_arguments(name, operations)
    callback(name, rank, case, arguments, outputs, roles)


def region_arguments(name, operations):
    """Bind the shared child operands and observations without copying or changing tensors."""
    steps = REGIONS[name]
    if tuple(item[0] for item in operations) != tuple(item[0] for item in steps):
        raise ValueError("L2 call order differs from the shared stage: " + name)
    values, outputs, roles = {}, {}, []
    for operation, (_, bindings) in zip(operations, steps):
        child, arguments, observed, role = operation[:4]
        if len(arguments) != len(bindings):
            raise ValueError("L2 argument count differs: " + child)
        for binding, value in zip(bindings, arguments):
            if binding in values:
                prior = values[binding]
                same = prior is value if hasattr(value, "shape") else type(prior) is type(value) and prior == value
                if not same:
                    raise ValueError("L2 binding refers to different runtime operands: " + binding)
            values[binding] = value
        outputs.update({child + "." + key: value for key, value in observed.items()})
        roles.extend([role] if isinstance(role, str) else role)
    return [values[binding] for binding in region_bindings(steps)], outputs, tuple(roles)
