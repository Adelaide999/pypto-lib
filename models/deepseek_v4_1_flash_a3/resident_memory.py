# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AIV byte transfers for scratch storage inside the resident Decode graph."""
import pypto.language as pl

BYTE_COUNT = pl.dynamic("DECODE_MEMORY_BYTES")
BYTE_TILE = 1024


@pl.jit
def copy_bytes(
    source: pl.Tensor[[1, BYTE_COUNT], pl.UINT8],
    destination: pl.InOut[pl.Tensor[[1, BYTE_COUNT], pl.UINT8]],
    count: pl.Scalar[pl.INT32],
):
    aligned = count // 32 * 32
    tiles = (aligned + BYTE_TILE - 1) // BYTE_TILE
    workers = pl.min(16, pl.max(1, tiles))
    for worker in pl.spmd(workers, name_hint="decode_copy_bytes"):
        for i in pl.range(worker, tiles, workers):
            start = i * BYTE_TILE
            active = pl.min(BYTE_TILE, aligned - start)
            data = pl.load(source, [0, start], [1, BYTE_TILE], valid_shape=[1, active])
            pl.store(data, [0, start], destination)
    if count > aligned:
        for _ in pl.spmd(1, name_hint="decode_copy_tail"):
            for i in pl.range(aligned, count):
                byte = pl.read(source, [0, i])
                pl.write(destination, [0, i], byte)
    return destination


def make_fill_bytes(byte):
    @pl.jit
    def fill_bytes(
        destination: pl.InOut[pl.Tensor[[1, BYTE_COUNT], pl.UINT8]],
        count: pl.Scalar[pl.INT32],
    ):
        aligned = count // 32 * 32
        tiles = (aligned + BYTE_TILE - 1) // BYTE_TILE
        workers = pl.min(16, pl.max(1, tiles))
        for worker in pl.spmd(workers, name_hint="decode_fill_bytes"):
            constant = pl.tile.full([1, BYTE_TILE], dtype=pl.FP16, value=byte)
            full = pl.tile.cast(constant, pl.UINT8, mode="rint")
            for i in pl.range(worker, tiles, workers):
                start = i * BYTE_TILE
                active = pl.min(BYTE_TILE, aligned - start)
                data = pl.tile.set_validshape(full, 1, active)
                pl.store(data, [0, start], destination)
        if count > aligned:
            for _ in pl.spmd(1, name_hint="decode_fill_tail"):
                tail_byte = pl.cast(byte, pl.UINT8)
                for i in pl.range(aligned, count):
                    pl.write(destination, [0, i], tail_byte)
        return destination

    return fill_bytes
