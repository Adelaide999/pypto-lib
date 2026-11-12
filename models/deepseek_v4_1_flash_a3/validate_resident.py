# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check resident ownership, aliases and failure cleanup on CPU; never initialize an NPU."""

import ctypes
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .prefill_resident import ResidentNetworkRuntime


class HostWorker:
    """Use CPU allocations as a bounded substitute for the public device memory API."""

    def __init__(self, rank):
        self.rank, self.live, self.closed = rank, {}, False
        self.uploads = 0

    def alloc_tensor(self, shape, dtype, *, init):
        from pypto.runtime import DeviceTensor

        value = init.clone().reshape(shape)
        owner = SimpleNamespace(base=value.data_ptr(), nbytes=value.numel() * value.element_size(),
                                value=value, tensor=lambda *args: None)
        result = DeviceTensor(owner.base, shape, dtype, buffer=owner)
        self.live[id(result)] = result
        self.uploads += 1
        return result

    def free_tensor(self, tensor):
        del self.live[id(tensor)]

    def copy_to(self, destination, source, count):
        ctypes.memmove(destination, source, count)

    copy_from = copy_to

    @staticmethod
    def host(value):
        strides, step = [], 1
        for dimension in reversed(value.shape):
            strides.insert(0, step)
            step *= dimension
        result = torch.empty(0, dtype=value.dtype)
        return result.set_(value.buffer.value.untyped_storage(), value.byte_offset // result.element_size(),
                           value.shape, tuple(strides))

    def run(self, compiled, *arguments):
        compiled(*(self.host(value) if hasattr(value, 'buffer') else value for value in arguments))

    def close(self):
        self.closed = True
        self.live.clear()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def reject(call, message):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise RuntimeError(message)


def abi(*directions):
    return dict(params=[dict(shape=[1]) for _ in directions], effective_directions=list(directions))


def check_lifecycle():
    topology = dict(dp=1, tp=2, ep=2)
    workers = []

    def factory(rank):
        worker = HostWorker(rank)
        workers.append(worker)
        return worker

    with ResidentNetworkRuntime([4, 7], topology, worker_factory=factory) as runtime:
        full_weight = torch.arange(12, dtype=torch.float32)
        weight = full_weight[4:8]
        runtime.pin(weight, [0, 1], readonly=True, name='nonzero-offset shared weight')
        output = torch.zeros(4)
        for rank in range(2):
            for _ in range(3):
                runtime.execute(lambda source, out: out.copy_(source), rank, [weight, output], abi('In', 'Out'))
                require(torch.equal(output, weight), 'Rank-local weight slice changed')
        require(runtime.stats['weight_bytes'] == [16, 16], 'Weights were duplicated across repeated dispatches')
        require([worker.uploads for worker in workers] == [4, 4], 'Resident weight was uploaded per token')
        require(all(len(worker.live) == 1 for worker in workers), 'Transient arguments leaked')

        raw = torch.zeros(16, dtype=torch.uint8)
        alias = torch.empty(0, dtype=torch.int32).set_(raw.untyped_storage(), 0, (4,), (1,))
        runtime.pin(raw, [0], readonly=False, name='raw cache')
        runtime.pin(alias, [0], readonly=False, name='typed cache alias')
        runtime.execute(lambda state: state.fill_(5), 0, [alias], abi('InOut'))
        require(alias.tolist() == [5] * 4 and raw[::4].tolist() == [5] * 4, 'Typed cache readback lost byte aliasing')
        alias.fill_(13)
        runtime.execute(lambda state, out: out.copy_(state.float()), 0, [alias, output], abi('In', 'Out'))
        require(output.tolist() == [13.0] * 4, 'Host mutation through an independent alias was not uploaded')
        fresh = torch.empty(0, dtype=torch.int32).set_(raw.untyped_storage(), 0, (4,), (1,))
        fresh.fill_(17)
        runtime.execute(lambda state, out: out.copy_(state.float()), 0, [fresh, output], abi('In', 'Out'))
        require(output.tolist() == [17.0] * 4, 'New typed host alias used stale device state')

        unpinned = torch.zeros(8)
        runtime.execute(lambda first, second: (first.fill_(9), second.add_(1)), 0,
                        [unpinned[:6], unpinned[2:]], abi('InOut', 'InOut'))
        require(unpinned.tolist() == [9, 9, 10, 10, 10, 10, 1, 1], 'Transient overlapping arguments lost aliases')
        reject(lambda: runtime.execute(lambda *_: None, 0, [weight], abi('InOut')),
               'Writable immutable weights were accepted')
        weight.add_(1)
        reject(lambda: runtime.execute(lambda *_: None, 0, [weight], abi('In')),
               'Host mutation of immutable weights was accepted')
        expert = runtime.weight(torch.ones(2), 0)
        reject(lambda: runtime.execute(lambda *_: None, 1, [expert], abi('In')),
               'A resident device operand was sent to the wrong rank')
        reject(lambda: runtime.exchange('invalid', [{'x': {'a': torch.ones(2)}}, {'x': {'a': torch.ones(3)}}],
                                         'all_gather', 'ep'), 'Incompatible collective sizes were admitted')
        reject(lambda: runtime.exchange('invalid', [{'x': {'a': torch.ones(3)}}, {'x': {'a': torch.ones(3)}}],
                                         'reduce_scatter', 'ep'), 'Incompatible collective buckets were admitted')
        previous = len(workers[0].live)

        def fail(*args):
            raise RuntimeError('injected launch failure')

        reject(lambda: runtime.execute(fail, 0, [torch.ones(2)], abi('In')), 'Launch failure was swallowed')
        require(len(workers[0].live) == previous, 'Failed launch leaked transient allocations')
        summary = runtime.summary()
        require(summary['single_device_graph'] is False, 'Host scheduling was mislabeled as a device graph')
    require(all(worker.closed and not worker.live for worker in workers), 'Runtime teardown leaked workers')
    runtime.close()
    reject(lambda: runtime.execute(None, 0, [], abi()), 'Closed runtime admitted a dispatch')

    partial = []

    def failing_factory(rank):
        if rank:
            raise RuntimeError('injected initialization failure')
        partial.append(HostWorker(rank))
        return partial[-1]

    reject(lambda: ResidentNetworkRuntime([0, 1], topology, worker_factory=failing_factory),
           'Partial worker initialization was accepted')
    require(partial[0].closed, 'Partial initialization leaked an earlier worker')
    return dict(weight_reuse='checked across 6 dispatches', typed_cache_aliases='checked',
                host_mutations='checked', overlapping_arguments='checked', failure_cleanup='checked',
                wrong_rank_and_invalid_collectives='rejected')


def check_weight_bank():
    from . import prefill_resident_weights as module

    suffixes = ('hc_ffn_fn', 'hc_ffn_scale', 'hc_ffn_base', 'ffn_norm.weight',
                'ffn.gate.weight', 'ffn.gate.bias', 'ffn.gate.bias_vl',
                'engram.wkv.weight', 'engram.q_weight', 'engram.k_weight')

    class Reader:
        def __init__(self, path):
            self.index = {f'layers.{layer}.{name}': 'synthetic' for layer in range(40) for name in suffixes}
            self.boundaries, self.read_calls, self.read_seconds, self.expert_calls = {}, 0, 0.0, []

        def load(self, name):
            self.read_calls += 1
            if name not in self.boundaries:
                self.boundaries[name] = torch.ones((16, 2) if name in ('head.weight', 'embed.weight') else (2, 2))
            return self.boundaries[name]

        def expert(self, name):
            self.expert_calls.append(name)
            return (torch.ones((2, 2), dtype=torch.int8), torch.ones(2),
                    torch.ones((2, 2), dtype=torch.int8), torch.ones(2))

    class Runtime:
        def __init__(self):
            self.counts = [0] * 16
            self.stats = {}

        def pin(self, *args, **kwargs):
            return None

        def weight(self, value, rank):
            self.counts[rank] += 1
            return value

    def prepare(*args, **kwargs):
        return dict(wo_a=torch.ones(1), wo_a_grouped_prefill=torch.ones(8, 2), wo_b=torch.ones(8192, 1))

    runtime = Runtime()
    with patch.object(module, 'ResidentCheckpointOperands', Reader), patch.object(module, 'prepare_product_operands', prepare):
        bank = module.ResidentNetworkWeights('synthetic', dict(dp=4, tp=4, ep=16), {}, runtime)
        bank.preload()
    require(len(bank.layers) == 40 and len(bank.experts) == 40 * (384 + 16), 'Resident bank omitted layers or experts')
    require(runtime.counts == [40 * 25 * 4] * 16, 'Routed expert distribution or shared replication differs')
    require(len(bank.reader.expert_calls) == 40 * 385, 'Preload read routed or shared experts more than once')
    reads = bank.reader.read_calls
    for _ in range(2):
        for layer in range(40):
            values = bank.for_layer(layer)
            require(values['expert'](0, 0) is bank.experts[layer, 0, 0], 'Token changed expert allocation identity')
            bank.release_current()
    require(reads == bank.reader.read_calls, 'Layer/token traversal reloaded checkpoint weights')
    reject(lambda: bank.expert(0, 0, 24), 'Another rank expert was accepted')
    reject(lambda: bank.preload(), 'Repeated generation preload was accepted')
    bank.close()
    require(not bank.layers and not bank.experts, 'Weight bank teardown retained its owners')
    return dict(layers=40, routed_experts_per_layer=384, shared_replicas_per_layer=16,
                two_token_traversal='no checkpoint reads; allocation identities preserved',
                operands='small synthetic lifecycle fixtures, not numerical evidence')


def check_hccl_cleanup():
    from . import moe_transport as transport

    destroyed = []

    class Function:
        def __init__(self, call=lambda *args: 0):
            self.call = call

        def __call__(self, *args):
            return self.call(*args)

    def create(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 17
        return 0

    def destroy(pointer):
        destroyed.append(pointer.value)
        return 0

    class Library:
        def __init__(self):
            self.functions = {}

        def __getattr__(self, name):
            return self.functions.setdefault(name, Function())

    library = Library()
    memory = SimpleNamespace(select=lambda rank: None, library=SimpleNamespace(
        aclrtCreateStream=Function(create), aclrtDestroyStream=Function(destroy)))
    with patch.object(transport.ctypes, 'CDLL', return_value=library):
        library.functions['aclInit'] = Function(lambda *args: 100002)
        transport.CANNMemory(diagnostics=False)
        library.functions['aclInit'] = Function(lambda *args: 91)
        reject(lambda: transport.CANNMemory(diagnostics=False), 'An unknown ACL initialization failure was accepted')
        reject(lambda: transport.HCCLBytes(memory, 0, 2, None, root_info=b'x'),
               'Malformed root info was accepted')
        library.functions['HcclCommInitRootInfo'] = Function(lambda *args: 91)
        reject(lambda: transport.HCCLBytes(memory, 0, 2, None, root_info=bytes(4108)),
               'Failed HCCL initialization was accepted')
    require(destroyed == [17, 17], "Partial HCCL initialization leaked its stream")
    return dict(partial_communicator_initialization='stream cleanup checked twice')


def main():
    print(json.dumps(dict(lifecycle=check_lifecycle(), weight_bank=check_weight_bank(),
                          hccl_cleanup=check_hccl_cleanup(),
                          device_initialization='not_run', NPU_numerics='not_run', performance='not_measured')))


if __name__ == '__main__':
    main()
