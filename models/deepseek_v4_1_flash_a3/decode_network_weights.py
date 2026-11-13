# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stream physical-layer checkpoint operands and sparse Engram rows for Decode."""

from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from .config import FLASH
from .decode_weights import DecodeSafetensorsCheckpoint
from .decode_weight_abi import iter_routed_experts, load_router, load_shared_expert
from .engram_hash import build_compressed_token_map, hash_ids, make_layout
from .engram_host import load_rotation
from .weights import prepare_product_operands


class DecodeNetworkWeights:
    """Keep one layer of host operands and memory-map both original Engram tables."""

    def __init__(self, checkpoint, world):
        self.reader = DecodeSafetensorsCheckpoint(checkpoint)
        self.world = world
        self._stack = ExitStack()
        self._slices = {}
        self.token_map = None
        text = self.reader.config["text_config"]
        for name in ("hidden_size", "vocab_size", "num_hidden_layers", "n_routed_experts",
                     "num_experts_per_tok", "num_attention_heads", "head_dim", "hc_mult"):
            if text[name] != getattr(FLASH, name):
                raise ValueError("Checkpoint model geometry differs: " + name)
        if tuple(text["compress_ratios"][:40]) != FLASH.compress_ratios:
            raise ValueError("Checkpoint backbone compression schedule differs")
        self.image_ids = (self.reader.config["image_token_id"],
                          self.reader.config.get("image_pad_token_id", self.reader.config["image_token_id"] + 1))

    def close(self):
        self._stack.close()
        self._slices.clear()

    def layer(self, layer):
        """Load replicated operands and all 384 owner experts for one physical layer."""
        with self.reader.cached_files():
            prefix = f"layers.{layer}."
            names = [name for name in self.reader.layer_tensor_names(layer)
                     if ".ffn.experts." not in name and ".engram.embed." not in name]
            raw = self.reader.require(names)
            prepared = prepare_product_operands(raw, layer)
            rename = dict(attn_norm="attn_norm_weight", q_norm="q_norm_weight", kv_norm="kv_norm_weight",
                          hc_function="attn_hc_function", hc_scale="attn_hc_scale", hc_base="attn_hc_base",
                          index_weights="index_weights_proj", index_wqb="index_wq_b",
                          index_wqb_scale="index_wq_b_scale")
            result = {rename.get(name, name): value for name, value in prepared.items()
                      if name != "wo_a_grouped_prefill"}
            for name, suffix in dict(ffn_hc_function="hc_ffn_fn", ffn_hc_scale="hc_ffn_scale",
                                     ffn_hc_base="hc_ffn_base", ffn_norm_weight="ffn_norm.weight").items():
                result[name] = raw[prefix + suffix].contiguous()
            result.update(load_router(self.reader, layer))
            result.update(load_shared_expert(self.reader, layer))
            result = {name: value.unsqueeze(0).expand(self.world, *value.shape).contiguous()
                      for name, value in result.items()}
            for rank in range(self.world):
                for expert, packed in iter_routed_experts(self.reader, layer, rank, ep_size=self.world):
                    local = expert % (384 // self.world)
                    for name, value in packed.items():
                        target = "routed_" + name
                        if target not in result:
                            result[target] = torch.empty((self.world, 384 // self.world, *value.shape), dtype=value.dtype)
                        result[target][rank, local].copy_(value)
            if layer in FLASH.engram_layer_ids:
                # The repeated block32 rotation is shared with the Prefill checkpoint ABI.
                result["engram_rotation"] = load_rotation(self.reader.root, FLASH.hidden_size)[0]
                result["engram_wkv"] = raw[prefix + "engram.wkv.weight"].T.contiguous()
                q = raw[prefix + "engram.q_weight"].float()
                k = raw[prefix + "engram.k_weight"].float()
                result["engram_channel_weight"] = (q * k).reshape(4, 5120).contiguous()
            return result

    def embedding(self):
        return self.reader.tensor("embed.weight")

    def head(self):
        return self.reader.tensor("norm.weight"), self.reader.tensor("head.weight").T.contiguous()

    def _table_slice(self, layer, kind):
        key = layer, kind
        if key not in self._slices:
            name = f"layers.{layer}.engram.embed.{kind}"
            path = (self.reader.shard_for(name) if name in self.reader.weight_map else
                    Path(self.reader.root) / f"engram_embed_{kind}_l{layer}.safetensors")
            handle = self._stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            keys = list(handle.keys())
            if name not in keys:
                if len(keys) != 1:
                    raise ValueError("Engram table file has ambiguous tensor names")
                name = keys[0]
            self._slices[key] = handle.get_slice(name)
        return self._slices[key]

    def engram_rows(self, tokens, histories):
        """Hash each request with its last three tokens and gather only 24 rows per slot."""
        if self.token_map is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.reader.root, local_files_only=True)
            mapping, compressed = build_compressed_token_map(tokenizer)
            if compressed != FLASH.engram_compressed_vocab_size:
                raise ValueError("Checkpoint tokenizer compressed vocabulary differs")
            self.token_map = torch.tensor(mapping, dtype=torch.int64)
            self.layout = make_layout()
        rows = {layer: torch.zeros((self.world, 16, 6144), dtype=torch.bfloat16) for layer in (1, 14)}
        masks = torch.zeros((self.world, 16), dtype=torch.int32)
        for rank, (token, history) in enumerate(zip(tokens, histories, strict=True)):
            previous = torch.tensor([(list(reversed(history[-3:])) + [-1] * 3)[:3]], dtype=torch.int64)
            dead = token in self.image_ids
            hashes = hash_ids(torch.tensor([token]), self.token_map, torch.tensor([len(history)]),
                              torch.tensor([0, 1], dtype=torch.int32), layout=self.layout,
                              dead_mask=torch.tensor([dead]), lookback_token_ids=previous,
                              lookback_dead_mask=(previous == self.image_ids[0]) | (previous == self.image_ids[1]))
            masks[rank, 0] = int(not dead)
            for slot, layer in enumerate((1, 14)):
                codes = self._table_slice(layer, "weight")
                scales = self._table_slice(layer, "scale")
                for col, row in enumerate(hashes[0, slot].tolist()):
                    code = codes[row:row + 1].float().reshape(8, 32)
                    scale = scales[row:row + 1].float().reshape(8, 1)
                    rows[layer][rank, 0, col * 256:(col + 1) * 256] = (code * scale).flatten().to(torch.bfloat16)
        return rows, masks
