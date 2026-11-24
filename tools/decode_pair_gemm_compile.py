# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile-only check for the canonical paired Decode INT8 GEMM."""

import pypto.language as pl
import torch

from golden import TensorSpec, run
from models.deepseek_v4_1_flash_a3.gemm import gemm_int8_pair_columns


CAPACITY = 64
WIDTH = 1280
OUTPUT_WIDTH = 8192


@pl.jit
def l2(
    payload: pl.Tensor[[CAPACITY, WIDTH], pl.INT8],
    activation_scale: pl.Tensor[[CAPACITY, 1], pl.FP32],
    weight: pl.Tensor[[WIDTH, OUTPUT_WIDTH], pl.INT8],
    weight_scale: pl.Tensor[[1, OUTPUT_WIDTH], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.Out[pl.Tensor[[CAPACITY, OUTPUT_WIDTH], pl.BF16]],
):
    count = pl.read(counts, [0])
    return gemm_int8_pair_columns(
        payload, activation_scale, weight, weight_scale, output, count,
        WIDTH, OUTPUT_WIDTH, "decode_pair_compile", 256, 256, False, 16,
    )


def main():
    gen = torch.Generator().manual_seed(1122)
    specs = [
        TensorSpec("payload", [CAPACITY, WIDTH], torch.int8, init_value=torch.randint(-127, 128, (CAPACITY, WIDTH), dtype=torch.int8, generator=gen)),
        TensorSpec("activation_scale", [CAPACITY, 1], torch.float32, init_value=torch.full((CAPACITY, 1), 0.02 / 127)),
        TensorSpec("weight", [WIDTH, OUTPUT_WIDTH], torch.int8, init_value=torch.randint(-127, 128, (WIDTH, OUTPUT_WIDTH), dtype=torch.int8, generator=gen)),
        TensorSpec("weight_scale", [1, OUTPUT_WIDTH], torch.float32, init_value=torch.full((1, OUTPUT_WIDTH), 0.02 / 127)),
        TensorSpec("counts", [1], torch.int32, init_value=torch.tensor([1], dtype=torch.int32)),
        TensorSpec("output", [CAPACITY, OUTPUT_WIDTH], torch.bfloat16),
    ]
    result = run(fn=l2, specs=specs, config={"backend_type": "a2a3", "execution_mode": False, "device_id": 0}, compile_only=True)
    print({"passed": result.passed, "error": str(result.error) if result.error else None, "build": str(result.work_dir)})
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
