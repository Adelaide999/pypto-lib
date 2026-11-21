// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// The generated header contains the unchanged model orchestration functions and their child IDs.
#include "runtime.h"
#include "orchestration_api.h"
#include "aicpu/cache_maintenance.h"
#include <algorithm>
#include <cstring>

static TaskOutputTensors finish_task(TaskOutputTensors result) {
    auto id = result.task_id();
    if (id.is_valid() && !rt_is_fatal()) {
        uint32_t shape[] = {1};
        TensorCreateInfo ready(shape, 1, DataType::INT32);
        CoreTaskArgs join;
        join.set_dependencies(&id, 1);
        join.add_output(ready);
        auto fence = rt_submit_dummy_task(join);
        uint32_t index[] = {0};
        if (!rt_is_fatal()) {
            get_tensor_data<int32_t>(fence.get_ref(0), 1, index);
        }
    }
    return result;
}

static TaskOutputTensors submit_mixed(int base, MixedKernels ids, const CoreTaskArgs &args) {
    if (ids.aic_kernel_id >= 0) ids.aic_kernel_id += base;
    if (ids.aiv0_kernel_id >= 0) ids.aiv0_kernel_id += base;
    if (ids.aiv1_kernel_id >= 0) ids.aiv1_kernel_id += base;
    return finish_task(rt_submit_task(ids, args));
}

#include "resident_decode_stages.inc"

extern "C" __attribute__((visibility("default")))
OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 3};
}

extern "C" __attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const Tensor &plan = args.tensor(0).ref();
    if (args.tensor(1).ref().numel() != 1) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Expected a single Decode status word");
        return;
    }
    int32_t *status = args.tensor(1).ref().data_as<int32_t>();
    const int64_t *cursor = plan.data_as<int64_t>();
    const int64_t *end = cursor + plan.numel();
    const Tensor &inputs = args.tensor(2).ref();
    const uint8_t *payload = inputs.data_as<uint8_t>();
    cache_invalidate_range(payload, inputs.nbytes());
    int32_t step = 0;
    cache_invalidate_range(cursor, plan.nbytes());
    cache_invalidate_range(status, args.tensor(1).ref().nbytes());
    *status = 0;
    while (cursor < end && !rt_is_fatal()) {
        ++step;
        if (end - cursor < 3) {
            rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Truncated Decode instruction");
            return;
        }
        int64_t op = *cursor++;
        int64_t nt = *cursor++;
        int64_t ns = *cursor++;
        if (nt < 0 || nt > 64 || ns < 0 || ns > 32) {
            rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Invalid Decode operand count");
            return;
        }
        ChipTaskArgs nested;
        nested.reset();
        Tensor values[64];
        for (int i = 0; i < nt; ++i) {
            if (end - cursor < 3) {
                rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Missing Decode tensor descriptor");
                return;
            }
            int64_t ref = *cursor++;
            if (ref < 0 && static_cast<uint64_t>(-(ref + 1)) >= inputs.nbytes()) {
                rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Invalid replay tensor offset");
                return;
            }
            auto address = ref < 0 ? reinterpret_cast<uintptr_t>(payload + (-(ref + 1))) : static_cast<uintptr_t>(ref);
            auto dtype = *cursor++;
            auto n = *cursor++;
            if (n < 1 || n > MAX_TENSOR_DIMS || end - cursor < n || dtype < 0 || dtype > 10 || !address) {
                rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Invalid Decode tensor descriptor");
                return;
            }
            uint32_t dims[MAX_TENSOR_DIMS];
            for (int d = 0; d < n; ++d) dims[d] = static_cast<uint32_t>(*cursor++);
            values[i] = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(address), dims, n,
                                                            static_cast<DataType>(dtype));
            if (ref < 0 && values[i].nbytes() > inputs.nbytes() - static_cast<uint64_t>(-(ref + 1))) {
                rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Decode tensor outside replay input storage");
                return;
            }
            nested.add_input(values[i]);
        }
        if (end - cursor < ns) {
            rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Missing Decode scalar");
            return;
        }
        constexpr int64_t input_ref = -(INT64_C(1) << 60);
        int64_t scalars[32];
        for (int i = 0; i < ns; ++i) {
            int64_t value = *cursor++;
            if (value >= input_ref && value < input_ref / 2) {
                uint64_t offset = static_cast<uint64_t>(value - input_ref);
                if (offset % 8 || offset + sizeof(int64_t) > inputs.nbytes()) {
                    rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Invalid replay scalar offset");
                    return;
                }
                value = *reinterpret_cast<const int64_t *>(payload + offset);
            }
            scalars[i] = value;
            nested.add_scalar(static_cast<uint64_t>(value));
        }
        if (op >= 0 && static_cast<size_t>(op) < sizeof(stages) / sizeof(*stages)) {
            SIMPLER_SCOPE() {
                stages[op](nested);
            }
        } else if (op == -1 && nt == 2 && ns == 0 && values[0].nbytes() == values[1].nbytes()) {
            cache_invalidate_range(values[1].data_as<void>(), values[1].nbytes());
            cache_invalidate_range(values[0].data_as<void>(), values[0].nbytes());
            std::memmove(values[0].data_as<void>(), values[1].data_as<void>(), values[0].nbytes());
            cache_flush_range(values[0].data_as<void>(), values[0].nbytes());
        } else if (op == -3 && nt == 1 && ns == 0) {
            const auto *finite = values[0].data_as<int32_t>();
            cache_invalidate_range(finite, values[0].nbytes());
            for (size_t i = 0; i < values[0].numel(); ++i) {
                if (!finite[i] && !*status) *status = step;
            }
        } else if (op == -5 && nt == 3 && ns == 2) {
            for (int i = 0; i < nt; ++i) cache_invalidate_range(values[i].data_as<void>(), values[i].nbytes());
            auto *indices = values[0].data_as<int32_t>();
            bool corrected = false;
            for (size_t i = 0; i < values[0].numel(); ++i) {
                int plane = i < 128 ? 0 : 1;
                int32_t id = indices[i];
                const auto *pages = values[plane + 1].data_as<int32_t>();
                const auto *last = pages + values[plane + 1].numel();
                if (id < -1 || (id >= 0 && (id % 128 >= scalars[plane] ||
                    std::find(pages, last, id / 128) == last))) {
                    if (!*status) *status = step;
                    indices[i] = -1;
                    corrected = true;
                }
            }
            if (corrected) cache_flush_range(indices, values[0].nbytes());
        } else if (op == -6 && nt == 1 && ns == 1) {
            const auto *ids = values[0].data_as<int32_t>();
            cache_invalidate_range(ids, values[0].nbytes());
            const auto *last = ids + values[0].numel();
            for (int64_t block = 0; block < scalars[0]; ++block) {
                if (std::find(ids, last, block) == last && !*status) *status = step;
            }
        } else {
            rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Unknown Decode instruction %lld", static_cast<long long>(op));
            return;
        }
    }
    cache_flush_range(status, args.tensor(1).ref().nbytes());
}
