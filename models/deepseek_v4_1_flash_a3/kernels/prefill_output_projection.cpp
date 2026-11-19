// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <pto/pto-inst.hpp>
#include "intrinsic.h"
#include "tensor.h"

namespace
{
using namespace pto;
using BF16 = bfloat16_t;

template <int Rows, int Cols, int RowStride, int ColStride, Layout Order = Layout::ND>
using Panel = GlobalTensor<BF16, Shape<1, 1, 1, Rows, Cols>, Stride<1, 1, 1, RowStride, ColStride>, Order>;

__aicore__ inline __gm__ BF16 *data(__gm__ int64_t *args, int slot)
{
    auto tensor = reinterpret_cast<__gm__ Tensor *>(args[slot]);
    return reinterpret_cast<__gm__ BF16 *>(tensor->buffer.addr) + tensor->start_offset;
}

constexpr int M = 128, N = 256;
struct OutputTile {
    int row, col, rows;
};

// Adjacent workers share a weight panel across up to eight row tiles.
template <int Width> __aicore__ inline OutputTile outputTile(int block, int active, int rowTiles)
{
    constexpr int stripRows = 8;
    constexpr int stripSize = stripRows * (Width / N);
    const int first = block / stripSize * stripRows;
    const int height = rowTiles - first < stripRows ? rowTiles - first : stripRows;
    const int offset = block % stripSize;
    const int row = (first + offset % height) * M;
    return {row, offset / height * N, active - row < M ? active - row : M};
}

template <bool Grouped, int Groups> __aicore__ inline void project(__gm__ int64_t *args)
{
    constexpr int panelK = 256, cubeK = 64;
    constexpr int K = Grouped ? 4096 : Groups * 1024;
    constexpr int inWidth = Grouped ? Groups * K : K;
    constexpr int outWidth = Grouped ? Groups * 1024 : 5120;
    const int active = static_cast<int>(args[3]);
    const int rowTiles = (active + M - 1) / M;
    const int blocks = rowTiles * (outWidth / N);
    const int worker = get_block_idx(args), workers = get_block_num(args);
    if (workers == 0 || worker >= blocks) {
        return;
    }
    const int tileCount = (blocks - 1 - worker) / workers + 1;
    auto input = data(args, 0), weights = data(args, 1), output = data(args, 2);

    using Lhs = Tile<TileType::Mat, BF16, M, panelK, BLayout::ColMajor, DYNAMIC, panelK, SLayout::RowMajor>;
    using Rhs = Tile<TileType::Mat, BF16, panelK, N, Grouped ? BLayout::RowMajor : BLayout::ColMajor, panelK,
                     N, Grouped ? SLayout::ColMajor : SLayout::RowMajor>;
    Lhs lhs[2];
    Rhs rhs[2];
    TileLeft<BF16, M, cubeK> left[2];
    TileRight<BF16, cubeK, N> right[2];
    TileAcc<float, M, N, DYNAMIC, N> acc;
    TASSIGN(acc, 0);
    for (int i = 0; i < 2; ++i) {
        TASSIGN(lhs[i], i * Lhs::StorageBytes);
        TASSIGN(rhs[i], 2 * Lhs::StorageBytes + i * Rhs::StorageBytes);
        TASSIGN(left[i], i * M * cubeK * sizeof(BF16));
        TASSIGN(right[i], i * N * cubeK * sizeof(BF16));
        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(i));
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(i));
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    constexpr int panels = K / panelK;
    // Prime the next L1 panel while the Cube consumes the previous one, even across output tiles.
    for (int t = 0; t <= tileCount * panels; ++t) {
        if (t < tileCount * panels) {
            int block = worker + (t / panels) * workers;
            const auto tile = outputTile<outWidth>(block, active, rowTiles);
            int group = Grouped ? tile.col / 1024 : 0;
            int step = t % panels;
            const int slot = step % 2;
            // Stagger the first K panel across tiles to spread concurrent GM reads.
            const int k = ((step + 3 * block) % panels) * panelK;
            lhs[slot].SetValidRow(tile.rows);
            using InputPanel = Panel<DYNAMIC, panelK, inWidth, 1>;
            InputPanel a(input + tile.row * inWidth + group * K + k, typename InputPanel::Shape(tile.rows));
            wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            TLOAD(lhs[slot], a);
            if constexpr (Grouped) {
                Panel<panelK, N, 1, K, Layout::DN> b(weights + tile.col * K + k);
                TLOAD(rhs[slot], b);
            } else {
                Panel<panelK, N, outWidth, 1> b(weights + k * outWidth + tile.col);
                TLOAD(rhs[slot], b);
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
        }
        if (t == 0) {
            continue;
        }
        int step = (t - 1) % panels + 1;
        if (step == 1) {
            acc.SetValidRow(M);
        }
        int slot = (step - 1) % 2;
        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
#pragma unroll
        for (int sub = 0; sub < panelK / cubeK; ++sub) {
            int buf = sub % 2;
            auto event = static_cast<event_t>(buf);
            wait_flag(PIPE_M, PIPE_MTE1, event);
            TEXTRACT(left[buf], lhs[slot], 0, sub * cubeK);
            TEXTRACT(right[buf], rhs[slot], sub * cubeK, 0);
            if (sub == panelK / cubeK - 1) {
                set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            }
            set_flag(PIPE_MTE1, PIPE_M, event);
            wait_flag(PIPE_MTE1, PIPE_M, event);
            if (step == 1 && sub == 0) {
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
                TMATMUL(acc, left[buf], right[buf]);
            } else {
                TMATMUL_ACC(acc, acc, left[buf], right[buf]);
            }
            set_flag(PIPE_M, PIPE_MTE1, event);
        }

        if (step == panels) {
            int block = worker + (t - 1) / panels * workers;
            const auto tile = outputTile<outWidth>(block, active, rowTiles);
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            acc.SetValidRow(tile.rows);
            using OutputPanel = Panel<DYNAMIC, N, outWidth, 1>;
            OutputPanel c(output + tile.row * outWidth + tile.col, typename OutputPanel::Shape(tile.rows));
            TSTORE(c, acc);
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        }
    }
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int i = 0; i < 2; ++i) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(i));
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(i));
    }
}
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args)
{
    auto weight = reinterpret_cast<__gm__ Tensor *>(args[1]);
    if (weight->ndims == 3) {
        switch (weight->shapes[0]) {
        case 2:
            project<true, 2>(args);
            break;
        case 4:
            project<true, 4>(args);
            break;
        case 8:
            project<true, 8>(args);
            break;
        }
    } else {
        switch (weight->shapes[0]) {
        case 2048:
            project<false, 2>(args);
            break;
        case 4096:
            project<false, 4>(args);
            break;
        case 8192:
            project<false, 8>(args);
            break;
        }
    }
}
