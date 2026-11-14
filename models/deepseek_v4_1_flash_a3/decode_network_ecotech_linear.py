# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eco-Tech dynamic INT8 linear with the native BF16 parameter scale boundary."""

import math

import torch
import pypto.language as pl

from .decode_network_linear_primitives import make_dequantize, make_dot, make_pack_weight, make_quantize


def prepare_ecotech_dynamic_weight(weight_nk, source_scale, source_offset):
    """Preprocess CPU checkpoint operands into KN INT8 and effective FP32 scales."""
    values = (weight_nk, source_scale, source_offset)
    if any(value.device.type != "cpu" for value in values):
        raise ValueError("Checkpoint preprocessing requires CPU source tensors")
    if weight_nk.ndim != 2 or weight_nk.dtype != torch.int8:
        raise ValueError("The source weight must be a complete NK INT8 tensor")
    output_width, input_width = weight_nk.shape
    if input_width % 256 or output_width % 64:
        raise ValueError("The current A2 projection requires K/256 and N/64 alignment")
    for value in (source_scale, source_offset):
        if value.shape != (output_width, 1) or value.dtype != torch.float32:
            raise ValueError("Source per-channel scale and offset require FP32 [N,1]")
        if any(not math.isfinite(item) for item in value.flatten().tolist()):
            raise ValueError("Source scale and offset must be finite")
    if any(item != 0 for item in source_offset.flatten().tolist()):
        raise ValueError("The symmetric native INT8 path requires zero checkpoint offsets")
    native_scale = source_scale.to(torch.bfloat16)
    if any(not math.isfinite(item) or item <= 0 for item in native_scale.flatten().tolist()):
        raise ValueError("Effective BF16 channel scales must remain finite and positive")
    weight_kn = weight_nk.T.contiguous()
    channel_scale = native_scale.to(torch.float32).flatten()
    return weight_kn, channel_scale


def make_ecotech_dynamic_linear(capacity, input_width, output_width, combine_scales=False):
    """Compose existing INT8 quantization, full matmul and FP32 scale operators."""
    if capacity <= 0 or capacity % 16 or input_width % 256 or output_width % 64:
        raise ValueError("Use a positive capacity/16, K/256 and N/64")
    packed_rows = input_width * (output_width // 64)
    pack = make_pack_weight(input_width, output_width)
    quantize = make_quantize(capacity, input_width)
    dot = make_dot(capacity, input_width, output_width)
    dequantize = make_dequantize(capacity, output_width, combine_scales=combine_scales)

    @pl.jit
    def linear(
        x: pl.Tensor[[capacity, input_width], pl.BF16],
        weight: pl.Tensor[[input_width, output_width], pl.INT8],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        packed = pl.create_tensor([packed_rows, 64], dtype=pl.INT8)
        pack(weight, packed)
        payload = pl.create_tensor([capacity, input_width], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(x, payload, token_scale, active)
        accumulator = pl.create_tensor([capacity, output_width], dtype=pl.INT32)
        dot(payload, packed, accumulator, active)
        dequantize(accumulator, channel_scale, token_scale, output, active)
        return output

    return linear

# ---- Decode tail padding and prepared INT8 linears ----

def padded_dimensions(input_width, output_width):
    if type(input_width) is not int or type(output_width) is not int or input_width <= 0 or output_width <= 0:
        raise ValueError("Require positive logical dimensions")
    if input_width % 32 or output_width % 32:
        raise ValueError("This tail path requires logical K/32 and N/32")
    return ((input_width + 255) // 256) * 256, ((output_width + 63) // 64) * 64


def prepare_ecotech_tail_weight(weight_nk, source_scale, source_offset):
    values = (weight_nk, source_scale, source_offset)
    if any(value.device.type != "cpu" for value in values):
        raise ValueError("Checkpoint preprocessing requires CPU operands")
    if weight_nk.ndim != 2 or weight_nk.dtype != torch.int8:
        raise ValueError("Require source NK INT8 bytes")
    output_width, input_width = weight_nk.shape
    padded_k, padded_n = padded_dimensions(input_width, output_width)
    for value in (source_scale, source_offset):
        if value.shape != (output_width, 1) or value.dtype != torch.float32:
            raise ValueError("Require original F32 [N,1] scale/offset")
    padded_weight = torch.zeros((padded_n, padded_k), dtype=torch.int8)
    padded_weight[:output_width, :input_width].copy_(weight_nk)
    padded_scale = torch.ones((padded_n, 1), dtype=torch.float32)
    padded_scale[:output_width].copy_(source_scale)
    padded_offset = torch.zeros((padded_n, 1), dtype=torch.float32)
    padded_offset[:output_width].copy_(source_offset)
    return prepare_ecotech_dynamic_weight(padded_weight, padded_scale, padded_offset)


def _make_pad_input(capacity, input_width, padded_k, input_dtype):
    @pl.jit.inline
    def pad_input(
        x: pl.Tensor[[capacity, input_width], input_dtype],
        padded: pl.Tensor[[capacity, padded_k], input_dtype],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + 7) // 8, name_hint="ecotech_tail_pad_input"):
            t0 = block * 8
            rows = pl.min(8, active - t0)
            for k0 in pl.range(0, padded_k, 256):
                columns = pl.min(256, input_width - k0)
                source = pl.slice(x, [8, 256], [t0, k0], valid_shape=[rows, columns])
                filled = pl.fillpad(source, pad_value=pl.PadValue.zero)
                padded[t0:t0 + 8, k0:k0 + 256] = pl.set_validshape(filled, 8, 256)
        return padded

    return pad_input


def _make_crop_output(capacity, output_width, padded_n):
    @pl.jit.inline
    def crop_output(
        padded: pl.Tensor[[capacity, padded_n], pl.BF16],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 7) // 8) * (output_width // 32), name_hint="ecotech_tail_crop_output"):
            t0 = (block // (output_width // 32)) * 8
            n0 = (block % (output_width // 32)) * 32
            rows = pl.min(8, active - t0)
            value = padded[t0:t0 + 8, n0:n0 + 32]
            output[t0:t0 + 8, n0:n0 + 32] = pl.set_validshape(value, rows, 32)
        return output

    return crop_output


def make_ecotech_tail_linear(capacity, input_width, output_width):
    if capacity <= 0 or capacity % 16:
        raise ValueError("Require a positive capacity/16")
    padded_k, padded_n = padded_dimensions(input_width, output_width)
    standalone = make_ecotech_dynamic_linear(capacity, padded_k, padded_n)
    aligned = pl.jit.inline(standalone._func)

    pad_input = _make_pad_input(capacity, input_width, padded_k, pl.BF16)
    crop_output = _make_crop_output(capacity, output_width, padded_n)

    @pl.jit
    def linear(
        x: pl.Tensor[[capacity, input_width], pl.BF16],
        weight: pl.Tensor[[padded_k, padded_n], pl.INT8],
        channel_scale: pl.Tensor[[padded_n], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        padded_x = pl.create_tensor([capacity, padded_k], dtype=pl.BF16)
        pad_input(x, padded_x, active)
        padded_output = pl.create_tensor([capacity, padded_n], dtype=pl.BF16)
        aligned(padded_x, weight, channel_scale, padded_output, active)
        crop_output(padded_output, output, active)
        return output

    return linear


def prepare_ecotech_token_scales(source_scale):
    """Broadcast exact CPU F32 per-token scale bits to the eight-lane ABI."""
    if source_scale.device.type != "cpu" or source_scale.dtype != torch.float32 or source_scale.ndim != 1:
        raise ValueError("Require original CPU F32 per-token vector")
    if not bool(torch.isfinite(source_scale).all()) or not bool((source_scale > 0).all()):
        raise ValueError("Per-token scales must be finite and positive")
    return source_scale[:, None].expand(-1, 8).contiguous()


def make_ecotech_int8_tail_linear(capacity, input_width, output_width, *, combine_scales=False):
    if capacity <= 0 or capacity % 16:
        raise ValueError("Require a positive capacity/16")
    padded_k, padded_n = padded_dimensions(input_width, output_width)
    packed_rows = padded_k * (padded_n // 64)
    pack = make_pack_weight(padded_k, padded_n)
    dot = make_dot(capacity, padded_k, padded_n)
    dequantize = make_dequantize(capacity, padded_n, combine_scales=combine_scales)
    pad_input = _make_pad_input(capacity, input_width, padded_k, pl.INT8)
    crop_output = _make_crop_output(capacity, output_width, padded_n)

    @pl.jit
    def linear(
        payload: pl.Tensor[[capacity, input_width], pl.INT8],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        weight: pl.Tensor[[padded_k, padded_n], pl.INT8],
        channel_scale: pl.Tensor[[padded_n], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        padded_payload = pl.create_tensor([capacity, padded_k], dtype=pl.INT8)
        pad_input(payload, padded_payload, active)
        packed = pl.create_tensor([packed_rows, 64], dtype=pl.INT8)
        pack(weight, packed)
        accumulator = pl.create_tensor([capacity, padded_n], dtype=pl.INT32)
        dot(padded_payload, packed, accumulator, active)
        padded_output = pl.create_tensor([capacity, padded_n], dtype=pl.BF16)
        dequantize(accumulator, channel_scale, token_scale, padded_output, active)
        crop_output(padded_output, output, active)
        return output

    return linear
