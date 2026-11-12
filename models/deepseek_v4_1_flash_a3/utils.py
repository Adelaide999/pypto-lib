# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side weight layouts and per-row INT8 quantization utilities."""

import torch


INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4


NZ_FRACTAL_ROWS = 16


NZ_C0_BYTES = 32


BLOCKED_INT8_COLUMNS = 64


def pack_blocked_weight(weight_kn: torch.Tensor) -> torch.Tensor:
    """Pack CPU INT8 ``[K, N]`` into ``[N/64, K, 64]`` flattened to 2D.

    This is the A3 blocked GEMM operand, distinct from the NZ expert layout.
    Only byte order changes; checkpoint scales and offsets stay separate.
    """
    if weight_kn.device.type != "cpu" or weight_kn.dtype != torch.int8 or weight_kn.ndim != 2:
        raise ValueError("Blocked weights require a CPU INT8 [K, N] tensor")
    rows, columns = weight_kn.shape
    if rows <= 0 or columns <= 0 or rows % 256 or columns % BLOCKED_INT8_COLUMNS:
        raise ValueError("Blocked weights require positive K/256 and N/64")
    blocks = columns // BLOCKED_INT8_COLUMNS
    return (
        weight_kn.reshape(rows, blocks, BLOCKED_INT8_COLUMNS)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(rows * blocks, BLOCKED_INT8_COLUMNS)
    )


def unpack_blocked_weight(packed: torch.Tensor, input_width: int, output_width: int) -> torch.Tensor:
    """Restore the logical ``[K, N]`` weight for Torch golden computation."""
    if input_width <= 0 or output_width <= 0 or input_width % 256 or output_width % BLOCKED_INT8_COLUMNS:
        raise ValueError("Blocked weights require positive K/256 and N/64")
    blocks = output_width // BLOCKED_INT8_COLUMNS
    if packed.dtype != torch.int8 or tuple(packed.shape) != (input_width * blocks, BLOCKED_INT8_COLUMNS):
        raise ValueError("Packed weight shape does not match the logical K and N")
    return (
        packed.reshape(blocks, input_width, BLOCKED_INT8_COLUMNS)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(input_width, output_width)
    )


def pack_nz(logical: torch.Tensor) -> torch.Tensor:
    """Reorder the trailing ``[R, C]`` of a row-major tensor into NZ fractal order."""
    rows, cols = logical.shape[-2:]
    c0 = NZ_C0_BYTES // logical.element_size()
    if rows % NZ_FRACTAL_ROWS:
        raise ValueError(f"NZ needs {NZ_FRACTAL_ROWS}-row fractals, got {rows} rows")
    if cols % c0:
        raise ValueError(f"NZ needs whole C0 lines of {c0} elements, got {cols} cols")
    blocked = logical.reshape(-1, rows // NZ_FRACTAL_ROWS, NZ_FRACTAL_ROWS, cols // c0, c0)
    return blocked.permute(0, 3, 1, 2, 4).contiguous().reshape(logical.shape)


def unpack_nz(packed: torch.Tensor) -> torch.Tensor:
    """Restore NZ-ordered bytes to the logical row-major ``[..., R, C]`` tensor."""
    rows, cols = packed.shape[-2:]
    c0 = NZ_C0_BYTES // packed.element_size()
    blocked = packed.reshape(-1, cols // c0, rows // NZ_FRACTAL_ROWS, NZ_FRACTAL_ROWS, c0)
    return blocked.permute(0, 2, 3, 1, 4).contiguous().reshape(packed.shape)


def int8_quant_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path.

    Rounds to int8 through fp16 to match the device rounding; returns the
    per-row dequant scale (``1 / scale_quant``).
    """
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    # Explicit division preserves half-amax ties as in device pl.div.
    scale_quant = torch.div(torch.full_like(amax, INT8_SCALE_MAX), amax)
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)
