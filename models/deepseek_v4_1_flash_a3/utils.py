# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host weight layouts, integer window scheduling and file utilities."""

import hashlib
import json
from pathlib import Path
import torch


INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4


NZ_FRACTAL_ROWS = 16


NZ_C0_BYTES = 32


BLOCKED_INT8_COLUMNS = 64


WINDOW_HEADS = 64
WINDOW_HEAD_DIM = 512
WINDOW_PAGE_ROWS = WINDOW_SIZE = 128
SCHEDULE_INT32_MAX = (1 << 31) - 1


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


def file_sha(path):
    """Hash a source, runtime library, or staging artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    """Hash actual contiguous CPU tensor bytes, including BF16 payloads."""

    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _require_schedule(condition, message):
    """Reject controls that violate the original native operator contract."""
    if not condition:
        raise RuntimeError(message)


def _schedule_integer(value, *, minimum=0):
    _require_schedule(
        type(value) is int and minimum <= value <= SCHEDULE_INT32_MAX,
        "A bounded exact integer scheduler control is required",
    )
    return value


def build_window_schedule(query_lengths, prefix_lengths, block_table, *, cache_blocks, capacity):
    """Derive physical windows and publications for every request and active row."""
    _schedule_integer(cache_blocks, minimum=1)
    _require_schedule(
        cache_blocks * WINDOW_PAGE_ROWS <= SCHEDULE_INT32_MAX,
        "Physical window indices must fit the original INT32 ABI",
    )
    _schedule_integer(capacity, minimum=1)
    _require_schedule(
        type(query_lengths) is list
        and type(prefix_lengths) is list
        and type(block_table) is list
        and len(query_lengths) == len(prefix_lengths) == len(block_table),
        "Explicit own request lengths/prefixes/page rows are required",
    )
    requests = len(query_lengths)
    _require_schedule(
        all(type(row) is list and row for row in block_table)
        and (not requests or len({len(row) for row in block_table}) == 1),
        "A rectangular nonempty own block table is required",
    )
    starts, seq_lens, positions, flat_slots, slots, indices, logical = [0], [], [], [], [], [], []
    allocated = set()
    for query_len, prefix, pages in zip(query_lengths, prefix_lengths, block_table, strict=True):
        _schedule_integer(query_len, minimum=1)
        _schedule_integer(prefix)
        end = _schedule_integer(prefix + query_len, minimum=1)
        required_pages = (end + WINDOW_PAGE_ROWS - 1) // WINDOW_PAGE_ROWS
        _require_schedule(required_pages <= len(pages), "Own cache pages must cover the complete sequence")
        _require_schedule(
            all(type(page) is int and -1 <= page < cache_blocks for page in pages),
            "Own page IDs must be valid physical pages or unused -1 padding",
        )
        used = pages[:required_pages]
        _require_schedule(
            all(page >= 0 for page in used)
            and len(set(used)) == len(used)
            and not allocated.intersection(used),
            "Live own request pages must be distinct",
        )
        allocated.update(used)

        def physical(position, current_pages=pages):
            return (
                current_pages[position // WINDOW_PAGE_ROWS] * WINDOW_PAGE_ROWS + position % WINDOW_PAGE_ROWS
            )

        for position in range(prefix, end):
            address = physical(position)
            visible = list(range(max(0, position - WINDOW_SIZE + 1), position + 1))
            padding = [-1] * (WINDOW_SIZE - len(visible))
            positions.append(position)
            flat_slots.append(address)
            slots.append([address // WINDOW_PAGE_ROWS, address % WINDOW_PAGE_ROWS])
            logical.append(visible + padding)
            indices.append([physical(token) for token in visible] + padding)
        starts.append(starts[-1] + query_len)
        seq_lens.append(end)
    _require_schedule(
        starts[-1] <= capacity and len(set(flat_slots)) == len(flat_slots),
        "Own active rows must fit capacity with unique cache publications",
    )
    return dict(
        active=starts[-1],
        capacity=capacity,
        cache_blocks=cache_blocks,
        num_reqs=requests,
        query_lengths=list(query_lengths),
        prefix_lengths=list(prefix_lengths),
        query_start_loc=starts,
        seq_lens=seq_lens,
        block_table=[list(row) for row in block_table],
        positions=positions,
        flat_slot_mapping=flat_slots,
        slot_mapping=slots,
        physical_window_indices=indices,
        logical_window_indices=logical,
        max_query_len=max(query_lengths, default=0),
        max_seq_len=max(seq_lens, default=0),
        cache_shape=[cache_blocks, WINDOW_PAGE_ROWS, 1, WINDOW_HEAD_DIM],
        cache_dtype="BF16",
        heads=WINDOW_HEADS,
        head_dim=WINDOW_HEAD_DIM,
        sliding_window=WINDOW_SIZE,
        cmp_ratio=0,
        causal=True,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=127,
        ori_win_right=0,
        ori_topk=0,
        cmp_topk=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )


def window_schedule_from_controls(starts, seq_lens, block_table, *, cache_blocks, capacity):
    _require_schedule(
        type(starts) is list
        and len(starts) >= 1
        and starts[0] == 0
        and type(starts[0]) is int
        and type(seq_lens) is list
        and len(seq_lens) == len(starts) - 1,
        "Own cumulative query starts must partition requests",
    )
    for value in starts:
        _schedule_integer(value)
    for value in seq_lens:
        _schedule_integer(value, minimum=1)
    lengths = [right - left for left, right in zip(starts[:-1], starts[1:], strict=True)]
    prefixes = [end - count for end, count in zip(seq_lens, lengths, strict=True)]
    return build_window_schedule(lengths, prefixes, block_table, cache_blocks=cache_blocks, capacity=capacity)
