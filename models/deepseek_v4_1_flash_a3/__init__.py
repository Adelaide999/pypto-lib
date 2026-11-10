# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash A3 Prefill product and retained shared operators."""

from importlib import import_module


_EXPORTS = {
    "run_prefill": ("prefill_fwd", "run_prefill"),
    "A2A3OperatorConfig": ("operator_config", "A2A3OperatorConfig"),
    "A2A3_DTYPE_CONTRACT": ("operator_config", "A2A3_DTYPE_CONTRACT"),
    "make_candidate_block_pool": ("candidate_pool", "make_candidate_block_pool"),
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
    "make_physical_compressed_entries": ("prefill_full2", "make_physical_compressed_entries"),
    "make_prefill_engram_entries": ("prefill_engram", "make_prefill_engram_entries"),
    "layer0_reference": ("prefill_layer0", "layer0_reference"),
    "route_reference": ("prefill_layer0", "route_reference"),
    "engram_reference": ("engram", "engram_reference"),
    "make_gate": ("engram", "make_gate"),
    "make_lookup": ("engram", "make_lookup"),
    "make_bf16_linear": ("gemm", "make_bf16_linear"),
    "make_fp32_linear": ("gemm", "make_fp32_linear"),
    "make_grouped_wo_a": ("output_proj", "make_grouped_wo_a"),
    "make_index_quantize": ("index_cache", "make_index_quantize"),
    "make_linear": ("gemm", "make_int8_linear_direct"),
    "make_norm": ("norm_rope", "make_rmsnorm_entry"),
    "make_rope_kernel": ("norm_rope", "make_rope_kernel"),
    "make_swa": ("swa_attention", "make_swa_entry"),
    "make_swiglu": ("activation", "make_swiglu"),
    "make_sparse_attention": ("sparse_attention", "make_sparse_attention"),
    "sparse_attention_reference": ("sparse_attention", "sparse_attention_reference"),
    "indexer_query_reference": ("indexer_query", "indexer_query_reference"),
    "make_indexer_query_quantize": ("indexer_query", "make_indexer_query_quantize"),
    "make_l3_indexer_query": ("indexer_query", "make_l3_indexer_query"),
    "quantize_indexer_query_reference": ("indexer_query", "quantize_indexer_query_reference"),
    "indexer_select_reference": ("indexer_select", "indexer_select_reference"),
    "make_indexer_select": ("indexer_select", "make_indexer_select"),
    "make_indexer_prepare": ("indexer_prepare", "make_indexer_prepare"),
    "prepare_indexer_indices_reference": ("indexer_prepare", "prepare_indexer_indices_reference"),
    "LAYER14": ("layer14_15", "LAYER14"),
    "LAYER15": ("layer14_15", "LAYER15"),
    "Layer14A2A3Contract": ("layer14_15", "Layer14A2A3Contract"),
    "Layer15A2A3Contract": ("layer14_15", "Layer15A2A3Contract"),
    "index_key_reference": ("layer14_15", "index_key_reference"),
    "dequantize_engram_rows_reference": ("layer14_15", "dequantize_engram_rows_reference"),
    "engram_a2a3_reference": ("layer14_15", "engram_a2a3_reference"),
    "layer14_source_reference": ("layer14_15", "layer14_source_reference"),
    "layer15_reuse_reference": ("layer14_15", "layer15_reuse_reference"),
    "load_a2a3_layer14_15_bundle": ("layer14_15", "load_a2a3_layer14_15_bundle"),
    "make_layer14_stages": ("layer14_15", "make_layer14_stages"),
    "make_layer15_stages": ("layer14_15", "make_layer15_stages"),
    "make_ratio2_projection": ("layer14_15", "make_ratio2_projection"),
    "quantize_engram_rows_reference": ("layer14_15", "quantize_engram_rows_reference"),
    "ratio2_pool_reference": ("layer14_15", "ratio2_pool_reference"),
    "update_ratio2_state_reference": ("layer14_15", "update_ratio2_state_reference"),
    "validate_layer14_15_contract": ("layer14_15", "validate_layer14_15_contract"),
    "RawCheckpoint": ("raw_checkpoint", "RawCheckpoint"),
    "flatten_scale": ("raw_checkpoint", "flatten_scale"),
    "load_attention_weights": ("raw_checkpoint", "load_attention_weights"),
    "load_decoder_layer_weights": ("raw_checkpoint", "load_decoder_layer_weights"),
    "LAYER19": ("layers19_25", "LAYER19"),
    "LAYER20": ("layers19_25", "LAYER20"),
    "LAYER21": ("layers19_25", "LAYER21"),
    "LAYER24": ("layers19_25", "LAYER24"),
    "LAYER25": ("layers19_25", "LAYER25"),
    "TARGET_LAYERS": ("layers19_25", "TARGET_LAYERS"),
    "c1a_full_reference": ("layers19_25", "c1a_full_reference"),
    "c1a_reindex_reference": ("layers19_25", "c1a_reindex_reference"),
    "c1a_reuse_reference": ("layers19_25", "c1a_reuse_reference"),
    "c2a_reuse_reference": ("layers19_25", "c2a_reuse_reference"),
    "layer_contract": ("layers19_25", "layer_contract"),
    "make_decoder_layer_stages": ("layers19_25", "make_decoder_layer_stages"),
    "select_candidate_block_pool_reference": ("layers19_25", "select_candidate_block_pool_reference"),
    "validate_layer19_20_21_24_25_contracts": (
        "layers19_25",
        "validate_layer19_20_21_24_25_contracts",
    ),
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
