# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode parameter binding to the generation session's canonical weight bank."""


class DecodeNetworkWeights:
    """Bind the shared network ABI to the generation owner's existing device handles."""

    def __init__(self, owner):
        self.owner = owner
        self.world = owner.world

    def layer(self, layer):
        if not self.owner.ready:
            raise RuntimeError("Decode operands require the session's completed preload")
        return self.owner.for_layer(layer)

    def handles(self, rank):
        if not 0 <= rank < self.world:
            raise ValueError("Decode rank is outside the session topology")
        return self.owner.decode_handles[rank]

    def preload(self, runtime):
        raise RuntimeError("Decode ABI adapters cannot read or upload a second weight bank")
