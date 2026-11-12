# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash A3 Prefill and Decode product operators."""

from importlib import import_module


_EXPORTS = {
    "run_prefill": ("prefill_fwd", "run_prefill"),
    "A2A3OperatorConfig": ("operator_config", "A2A3OperatorConfig"),
    "A2A3_DTYPE_CONTRACT": ("operator_config", "A2A3_DTYPE_CONTRACT"),
    "make_candidate_block_pool": ("indexer", "make_candidate_block_pool"),
    "dequantize_int32": ("quant", "dequantize_int32"),
    "int8_matmul": ("quant", "int8_matmul"),
    "quantize_per_channel_int8": ("quant", "quantize_per_channel_int8"),
    "quantize_per_token_int8": ("quant", "quantize_per_token_int8"),
    "w8a8_linear": ("quant", "w8a8_linear"),
    "load_layer0_qkv": ("weights", "load_layer0_qkv"),
    "load_layer0_output": ("weights", "load_layer0_output"),
    "load_layer0_router": ("weights", "load_layer0_router"),
    "load_layer0_shared_expert": ("weights", "load_layer0_shared_expert"),
    "load_tensor_bundle": ("weights", "load_tensor_bundle"),
    "open_layer0_moe": ("weights", "open_layer0_moe"),
    "QKVShape": ("qkv", "QKVShape"),
    "make_qkv_w8a8": ("qkv", "make_qkv_w8a8"),
    "qkv_reference": ("qkv", "qkv_reference"),
    "qkv_shape": ("qkv", "qkv_shape"),
    "make_routed_expert_w8a8": ("moe", "make_routed_expert_w8a8"),
    "make_shared_expert_w8a8": ("moe", "make_shared_expert_w8a8"),
    "make_prefill_layer_entries": ("prefill_layer", "make_prefill_layer_entries"),
    "run_prefill_layer": ("prefill_layer", "run_prefill_layer"),
    "make_physical_compressed_entries": ("prefill_c2a", "make_physical_compressed_entries"),
    "make_prefill_engram_entries": ("engram", "make_prefill_engram_entries"),
    "engram_reference": ("engram", "engram_reference"),
    "make_gate": ("engram", "make_gate"),
    "make_lookup": ("engram", "make_lookup"),
    "make_bf16_linear": ("gemm", "make_bf16_linear"),
    "make_fp32_linear": ("gemm", "make_fp32_linear"),
    "make_grouped_wo_a": ("output_proj", "make_grouped_wo_a"),
    "make_index_quantize": ("indexer", "make_index_quantize"),
    "make_linear": ("gemm", "make_int8_linear_direct"),
    "make_norm": ("norm_rope", "make_rmsnorm_entry"),
    "make_rope_kernel": ("norm_rope", "make_rope_kernel"),
    "make_swa": ("swa_attention", "make_swa_entry"),
    "make_swiglu": ("activation", "make_swiglu"),
    "make_sparse_attention": ("sparse_attention", "make_sparse_attention"),
    "sparse_attention_reference": ("sparse_attention", "sparse_attention_reference"),
    "indexer_query_reference": ("indexer", "indexer_query_reference"),
    "make_indexer_query_quantize": ("indexer", "make_indexer_query_quantize"),
    "make_l3_indexer_query": ("indexer", "make_l3_indexer_query"),
    "quantize_indexer_query_reference": ("indexer", "quantize_indexer_query_reference"),
    "indexer_select_reference": ("indexer_select", "indexer_select_reference"),
    "make_indexer_select": ("indexer_select", "make_indexer_select"),
    "make_indexer_prepare": ("indexer", "make_indexer_prepare"),
    "prepare_indexer_indices_reference": ("indexer", "prepare_indexer_indices_reference"),
    "RawCheckpoint": ("weights", "RawCheckpoint"),
    "flatten_scale": ("weights", "flatten_scale"),
    "load_attention_weights": ("weights", "load_attention_weights"),
    "load_decoder_layer_weights": ("weights", "load_decoder_layer_weights"),
    "compile_prefill": ("prefill_fwd", "compile_prefill"),
    "make_decode_step": ("decode_fwd", "make_decode_step"),
    "validate_decode_contract": ("decode_fwd", "validate_decode_contract"),
    "decode_ep2": ("decode_moe", "decode_ep2"),
    "get_layer_entry": ("decode_layer", "get_layer_entry"),
}


__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load a public operator only when a caller requests it."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager package names and lazy public operator names."""

    return sorted(set(globals()) | set(__all__))
