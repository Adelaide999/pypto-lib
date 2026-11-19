// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <stdint.h>

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif

#include "acl/acl.h"
#include PYPTO_VECTOR_BODY

extern "C" __global__ AICORE __attribute__((aiv)) void prefill_vector_leaf(
    __gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c, __gm__ uint8_t *d,
    int32_t active, int32_t workers) {
#if defined(__DAV_VEC__)
    const int32_t block = get_block_idx();
    if (block < 0 || block >= workers || get_subblockid() != 0 || get_subblockdim() != 1) {
        return;
    }
    set_atomic_none();
#if PYPTO_VECTOR_QUANT_ROWS
    constexpr auto rows = PYPTO_VECTOR_QUANT_ROWS;
    const int64_t tiles = (static_cast<int64_t>(active) + rows - 1) / rows;
    prefill_token_quantize_whole_row((__gm__ int8_t *)b, (__gm__ float *)c, (__gm__ bfloat16_t *)a,
                                    tiles, static_cast<int64_t>(workers), active, block, workers);
#elif PYPTO_VECTOR_ROPE_TABLE_ROWS
    constexpr auto rows = PYPTO_VECTOR_ROPE_TABLE_ROWS;
    const int64_t tiles = (static_cast<int64_t>(active) + rows - 1) / rows;
    prefill_rope_tables((__gm__ int64_t *)a, (__gm__ float *)b, (__gm__ float *)c,
                        tiles, static_cast<int64_t>(workers), active, block, workers);
#else
    const int64_t tiles = (static_cast<int64_t>(active) + 7) / 8;
    prefill_norm_dual_output((__gm__ bfloat16_t *)c, (__gm__ float *)d, (__gm__ bfloat16_t *)a,
                            (__gm__ bfloat16_t *)b, tiles, static_cast<int64_t>(workers), active, block, workers);
#endif
#endif
}

extern "C" aclError launch_prefill_vector_leaf(
    void *a, void *b, void *c, void *d, int32_t active, int32_t workers, void *stream) {
    prefill_vector_leaf<<<workers, nullptr, stream>>>(
        (__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)c,
        (__gm__ uint8_t *)d, active, workers);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
