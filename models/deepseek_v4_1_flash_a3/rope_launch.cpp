// Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include ROPE_BODY

extern "C" __global__ AICORE void rope_direct_entry(
    __gm__ bfloat16_t *x, __gm__ float *cosine, __gm__ float *sine, __gm__ bfloat16_t *output,
    int32_t active, int32_t workers) {
#if defined(__DAV_VEC__)
    const int32_t block = get_block_idx();
    if (block < 0 || block >= workers || get_subblockid() != 0 || get_subblockdim() != 1) {
        return;
    }
    const int64_t blocks = (static_cast<int64_t>(active) + ROPE_TILE - 1) / ROPE_TILE;
#if ROPE_ABI == 1
    const int64_t quota = (blocks + 47) / 48;
    ROPE_FUNCTION(output, x, cosine, sine, quota, blocks, active, block, workers);
#elif ROPE_ABI == 2
    ROPE_FUNCTION(x, cosine, sine, output, blocks, workers, active, block, workers);
#elif ROPE_ABI == 3
    ROPE_FUNCTION(x, cosine, sine, output, blocks, workers, block, workers);
#else
#error Unsupported RoPE ABI
#endif
#endif
}

extern "C" aclError launch_rope_on_stream(
    uint16_t *x, float *cosine, float *sine, uint16_t *output, int32_t active, int32_t workers, void *stream) {
    rope_direct_entry<<<workers, nullptr, stream>>>(
        (__gm__ bfloat16_t *)x, (__gm__ float *)cosine, (__gm__ float *)sine,
        (__gm__ bfloat16_t *)output, active, workers);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
