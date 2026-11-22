# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed views into shared resident device allocations."""


def _device_view(allocation, value, base_offset=0):
    """Bind a contiguous typed view to the allocation's owning Buffer."""
    from pypto.runtime import DeviceTensor

    offset = value.storage_offset() * value.element_size() - base_offset
    if offset < 0:
        raise ValueError("Resident tensor view begins before its allocation")
    if offset + value.numel() * value.element_size() > allocation.nbytes:
        raise ValueError("Resident tensor view exceeds its allocation")
    return DeviceTensor(allocation.data_ptr + offset, value.shape, value.dtype,
                        buffer=allocation.buffer, byte_offset=allocation.byte_offset + offset)
