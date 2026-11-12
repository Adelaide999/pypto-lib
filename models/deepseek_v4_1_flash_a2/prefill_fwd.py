# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unified A2/A3 Prefill forward composition entry points."""

from .attention import make_attention, make_attention_inline
from .attention_block import make_attention_block
from .layer0 import layer0_reference, make_l0_attention, route_reference
from .layer2_c1c2 import make_layer2_c1c2
from .layer2_prefill import make_layer2_prefill_producer
from .prefill import make_prefill_swa


__all__ = [
    "layer0_reference",
    "make_attention",
    "make_attention_block",
    "make_attention_inline",
    "make_l0_attention",
    "make_layer2_c1c2",
    "make_layer2_prefill_producer",
    "make_prefill_swa",
    "route_reference",
]
