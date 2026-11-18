# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Grouped blocked-INT8 routed experts with token-first Gate-Up scaling."""

import pypto.language as pl

from .quant import make_cast_tile, make_quantize_row


def make_grouped_routed_experts(capacity, config, local_experts):
    """Build one synchronous entry for the complete local expert slab."""
    hidden = config["hidden_size"]
    intermediate = config["moe_intermediate_size"]
    width = intermediate * 2
    limit = config["swiglu_limit"]
    total_rows = local_experts * capacity
    gate_rows = hidden * (width // 64)
    down_rows = intermediate * (hidden // 64)
    row_tile = min(capacity, 64)
    gate_k_tile = hidden // 4
    max_tiles = local_experts * ((capacity + row_tile - 1) // row_tile)
    quantize_x = make_quantize_row(hidden)
    quantize_hidden = make_quantize_row(intermediate, fused_requantize=True)
    widen_x = make_cast_tile(1, hidden, pl.BF16, pl.FP32)

    @pl.jit
    def grouped_routed_experts(
        x: pl.Tensor[[total_rows, hidden], pl.BF16],
        gate_up_weight: pl.Tensor[[local_experts * gate_rows, 64], pl.INT8],
        gate_up_scale: pl.Tensor[[local_experts * width], pl.FP32],
        down_weight: pl.Tensor[[local_experts * down_rows, 64], pl.INT8],
        down_scale: pl.Tensor[[local_experts * hidden], pl.FP32],
        counts: pl.Tensor[[local_experts], pl.INT64],
        output: pl.InOut[pl.Tensor[[total_rows, hidden], pl.BF16]],
    ):
        row_map = pl.create_tensor([total_rows], dtype=pl.INT64)
        tile_map = pl.create_tensor([max_tiles, 8], dtype=pl.INT64)
        schedule_size = pl.create_tensor([8], dtype=pl.INT64)
        for _ in pl.spmd(1, name_hint="grouped_schedule"):
            row_cursor = 0
            tile_cursor = 0
            for local in pl.range(local_experts):
                local_count = pl.cast(pl.read(counts, [local]), pl.INDEX)
                compact_begin = row_cursor
                for local_row in pl.range(local_count):
                    pl.write(row_map, [row_cursor], pl.cast(local * capacity + local_row, pl.INT64))
                    row_cursor = row_cursor + 1
                for local_row in pl.range(0, local_count, row_tile):
                    pl.write(tile_map, [tile_cursor, 0], pl.cast(local, pl.INT64))
                    pl.write(tile_map, [tile_cursor, 1], pl.cast(compact_begin + local_row, pl.INT64))
                    pl.write(tile_map, [tile_cursor, 2], pl.cast(pl.min(row_tile, local_count - local_row), pl.INT64))
                    tile_cursor = tile_cursor + 1
            pl.write(schedule_size, [0], pl.cast(row_cursor, pl.INT64))
            pl.write(schedule_size, [1], pl.cast(tile_cursor, pl.INT64))
        active_rows = pl.cast(pl.read(schedule_size, [0]), pl.INDEX)
        active_tiles = pl.cast(pl.read(schedule_size, [1]), pl.INDEX)
        scratch_rows = pl.max(active_rows, 1)

        payload = pl.create_tensor([scratch_rows, hidden], dtype=pl.INT8)
        scales = pl.create_tensor([scratch_rows, 8], dtype=pl.FP32)
        for block in pl.spmd(active_rows, name_hint="grouped_input_quant"):
            token = pl.cast(pl.read(row_map, [block]), pl.INDEX)
            input_row = pl.slice(x, [1, hidden], [token, 0], valid_shape=[1, hidden])
            input_value = pl.create_tensor([1, hidden], dtype=pl.FP32)
            input_value[:, :] = widen_x(input_row)
            input_codes, input_scale = quantize_x(input_value)
            payload[block : block + 1, :] = input_codes
            scales[block : block + 1, :] = input_scale

        gate_acc = pl.create_tensor([scratch_rows, width], dtype=pl.INT32)
        for block in pl.spmd(active_tiles * (width // 192), name_hint="grouped_gate_up"):
            tile = block // (width // 192)
            expert = pl.cast(pl.read(tile_map, [tile, 0]), pl.INDEX)
            first_nb = block % (width // 192) * 3
            rows = pl.cast(pl.read(tile_map, [tile, 2]), pl.INDEX)
            gate_row = pl.cast(pl.read(tile_map, [tile, 1]), pl.INDEX)
            for nb in pl.range(first_nb, first_nb + 3):
                packed_begin = expert * gate_rows + nb * hidden
                storage = pl.tile.create([row_tile, 64], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc, compact=True)
                gate_product = pl.tile.set_validshape(storage, rows, 64)
                for k0 in pl.pipeline(0, hidden, gate_k_tile, stage=2):
                    gate_source = pl.load(
                        payload, [gate_row, k0], [row_tile, gate_k_tile], [rows, gate_k_tile],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    weight_begin = packed_begin + k0
                    gate_weight = pl.load(
                        gate_up_weight, [weight_begin, 0], [gate_k_tile, 64], target_memory=pl.MemorySpace.Mat,
                    )
                    gate_product = pl.tile.matmul_acc(gate_product, gate_source, gate_weight, init_cond=k0 == 0)
                pl.store(gate_product, [gate_row, nb * 64], gate_acc)

        activated = pl.create_tensor([scratch_rows, intermediate], dtype=pl.INT8)
        for block in pl.spmd(active_rows, name_hint="grouped_activation_quant"):
            token = pl.cast(pl.read(row_map, [block]), pl.INDEX)
            expert = token // capacity
            scale = pl.read(scales, [block, 0])
            gate = pl.cast(gate_acc[block : block + 1, :intermediate], pl.FP32)
            up = pl.cast(gate_acc[block : block + 1, intermediate:], pl.FP32)
            gate = pl.mul(gate, scale)
            up = pl.mul(up, scale)
            channel_begin = expert * width
            gate_channel = pl.reshape(gate_up_scale[channel_begin : channel_begin + intermediate], [1, intermediate])
            up_begin = channel_begin + intermediate
            up_channel = pl.reshape(gate_up_scale[up_begin : up_begin + intermediate], [1, intermediate])
            gate = pl.col_expand_mul(gate, gate_channel)
            up = pl.col_expand_mul(up, up_channel)
            gate = pl.minimum(gate, limit)
            up = pl.maximum(pl.minimum(up, limit), -limit)
            denominator = pl.add(pl.exp(pl.neg(gate)), 1.0)
            silu = pl.div(gate, denominator)
            activation_value = pl.create_tensor([1, intermediate], dtype=pl.FP32)
            activation_value[:, :] = pl.mul(silu, up)
            activation_codes, activation_scale = quantize_hidden(activation_value)
            activated[block : block + 1, :] = activation_codes
            scales[block : block + 1, :] = activation_scale

        down_acc = pl.create_tensor([scratch_rows, hidden], dtype=pl.INT32)
        for block in pl.spmd(active_tiles * (hidden // 256), name_hint="grouped_down"):
            tile = block // (hidden // 256)
            expert = pl.cast(pl.read(tile_map, [tile, 0]), pl.INDEX)
            first_nb = block % (hidden // 256) * 4
            rows = pl.cast(pl.read(tile_map, [tile, 2]), pl.INDEX)
            compact_row = pl.cast(pl.read(tile_map, [tile, 1]), pl.INDEX)
            down_source = pl.slice(activated, [row_tile, intermediate], [compact_row, 0], valid_shape=[rows, intermediate])
            for nb in pl.pipeline(first_nb, first_nb + 4, stage=2):
                packed_begin = expert * down_rows + nb * intermediate
                down_matrix = down_weight[packed_begin : packed_begin + intermediate, :]
                product = pl.matmul(down_source, down_matrix, out_dtype=pl.INT32)
                valid_product = pl.set_validshape(product, rows, 64)
                down_acc[compact_row : compact_row + row_tile, nb * 64 : (nb + 1) * 64] = valid_product

        for block in pl.spmd(active_rows, name_hint="grouped_down_epilogue"):
            token = pl.cast(pl.read(row_map, [block]), pl.INDEX)
            expert = token // capacity
            channel_begin = expert * hidden
            channels = pl.reshape(down_scale[channel_begin : channel_begin + hidden], [1, hidden])
            value = pl.cast(down_acc[block : block + 1, :], pl.FP32, mode="rint")
            value = pl.col_expand_mul(value, channels)
            # The activation quantizer repeats its per-token scale across all eight lanes.
            value = pl.mul(value, pl.read(scales, [block, 0]))
            output[token : token + 1, :] = pl.cast(value, pl.BF16, mode="rint")
        return output

    return grouped_routed_experts
