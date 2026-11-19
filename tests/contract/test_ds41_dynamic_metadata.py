# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise request admission, stale completions and independent runtime limits."""

import unittest

from models.deepseek_v4_1_flash_a3.metadata import RequestLedger, RequestQuery
from models.deepseek_v4_1_flash_a3.runtime_config import ParallelPlan, RuntimeLimits


class DynamicMetadataContracts(unittest.TestCase):
    def ledger(self, **kwargs):
        limits = RuntimeLimits(max_num_seqs=4, max_seqs_per_dp=2, max_num_batched_tokens=16, **kwargs)
        return RequestLedger(limits, ParallelPlan(tuple(range(16))))

    def query(self, key, tokens=(1,), dp=0, prompt=1, generation=None):
        return RequestQuery(key, dp, tokens, prompt, generation)

    def test_omission_reordering_release_and_slot_reuse_preserve_identity(self):
        ledger = self.ledger()
        first = ledger.begin([self.query("a"), self.query("b", dp=1)], phase="prefill")
        a, b = [row.handle for row in first.requests]
        ledger.commit(first)
        second = ledger.begin([self.query("b", dp=1, generation=b.generation)], phase="decode")
        ledger.commit(second)
        self.assertEqual(ledger.get(a).committed_length, 1)
        ledger.pause(a)
        ledger.resume(a)
        ledger.finish(a)
        self.assertEqual(ledger.get(a).status, "finished")
        with self.assertRaises(ValueError):
            ledger.begin([self.query("a", generation=a.generation)], phase="decode")
        resets = []
        ledger.release(a, resets.append)
        with self.assertRaises(ValueError):
            ledger.begin([self.query("c"), self.query("b", dp=1, generation=b.generation)], phase="prefill")
        self.assertNotIn("c", ledger.owners)
        third = ledger.begin([self.query("c")], phase="prefill")
        self.assertEqual(third.requests[0].handle.state_slot, a.state_slot)
        ledger.commit(third)
        self.assertEqual(ledger.get(b).committed_length, 2)
        self.assertEqual(resets, [a])

    def test_reused_slot_rejects_stale_generation_and_completion(self):
        ledger = self.ledger()
        first = ledger.begin([self.query("a")], phase="prefill")
        a = first.requests[0].handle
        ledger.commit(first)
        ledger.release(a, lambda handle: None)
        second = ledger.begin([self.query("b")], phase="prefill")
        b = second.requests[0].handle
        self.assertEqual(a.state_slot, b.state_slot)
        self.assertGreater(b.generation, a.generation)
        with self.assertRaises(ValueError):
            ledger.commit(first)
        ledger.commit(second)
        with self.assertRaises(ValueError):
            ledger.get(a)
        with self.assertRaises(ValueError):
            ledger.begin([self.query("b", generation=a.generation)], phase="decode")

    def test_preparation_failure_rolls_back_without_advancing_committed_history(self):
        ledger = self.ledger()
        batch = ledger.begin([self.query("a", prompt=4)], phase="prefill")
        ledger.rollback(batch)
        self.assertEqual(ledger.owners, {})
        self.assertEqual(ledger.epoch, 0)
        batch = ledger.begin([self.query("a", (1, 2), prompt=4)], phase="prefill")
        ledger.commit(batch)
        handle = batch.requests[0].handle
        batch = ledger.begin([self.query("a", (3,), prompt=4, generation=handle.generation)], phase="prefill")
        ledger.rollback(batch)
        self.assertEqual(ledger.get(handle).committed_length, 2)
        self.assertEqual(ledger.get(handle).lookback, (1, 2))

    def test_device_failure_invalidates_all_rows_and_failed_reset_prevents_reuse(self):
        ledger = self.ledger()
        batch = ledger.begin([self.query("a"), self.query("b", dp=1)], phase="prefill")
        calls = []
        def reset(handle):
            calls.append(handle.request_id)
            if handle.request_id == "a":
                raise RuntimeError("device reset failed")
        with self.assertRaises(RuntimeError):
            ledger.abort(batch, reset)
        self.assertEqual(calls, ["a", "b"])
        self.assertTrue(ledger.poisoned)
        self.assertNotIn("b", ledger.owners)
        self.assertEqual(ledger.owners["a"].status, "failed")
        with self.assertRaises(RuntimeError):
            ledger.begin([self.query("c")], phase="prefill")

    def test_request_token_and_context_limits_are_independent(self):
        ledger = self.ledger(max_model_len=1048576)
        batch = ledger.begin([self.query("a", (1,) * 15, prompt=1048576),
                              self.query("b", dp=1)], phase="prefill")
        self.assertEqual(batch.num_tokens, 16)
        self.assertEqual(len(batch.requests), 2)
        self.assertEqual(batch.token_capacity, 16)
        ledger.rollback(batch)
        with self.assertRaises(ValueError):
            ledger.begin([self.query("a", (1,) * 17, prompt=100)], phase="prefill")
        self.assertFalse(ledger.owners)
        batch = ledger.begin([self.query("a")], phase="prefill")
        ledger.commit(batch)
        handle = batch.requests[0].handle
        ledger.get(handle).committed_length = 1048576
        with self.assertRaises(ValueError):
            ledger.begin([self.query("a", generation=handle.generation)], phase="decode")

    def test_packed_coordinates_keep_c2_and_output_rows_request_local(self):
        ledger = self.ledger()
        batch = ledger.begin([self.query("b", (4, 5, 6), dp=1, prompt=3),
                              self.query("a", (1, 2), dp=1, prompt=4)], phase="prefill")
        values = batch.coordinates(1)
        self.assertEqual(values["query_start_loc"], [0, 3, 5])
        self.assertEqual(values["positions"], [0, 1, 2, 0, 1])
        self.assertEqual(values["token_to_req"], [0, 0, 0, 1, 1])
        self.assertEqual(values["max_query_len"], 3)
        self.assertEqual(values["output_rows"], [2])
        self.assertEqual(batch.coordinates(0)["query_start_loc"], [0])
        self.assertEqual([row["rank"] for row in batch.output_map()], [4, 4])
        self.assertEqual([row["row"] for row in batch.output_map()], [2, 4])
        self.assertEqual([row["sample"] for row in batch.output_map()], [True, False])

    def test_parallel_groups_cover_one_world_for_each_supported_plan(self):
        for dp, tp in ((4, 4), (8, 2), (16, 1)):
            plan = ParallelPlan(tuple(reversed(range(16))), dp, tp)
            self.assertEqual(sorted(rank for group in plan.tp_groups for rank in group), list(range(16)))
            self.assertEqual(sorted(rank for group in plan.dp_groups for rank in group), list(range(16)))
            self.assertEqual([expert for rank in range(16) for expert, _ in plan.expert_ownership(rank)],
                             list(range(384)))
            self.assertEqual(plan.token_range(0, 0), (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            ParallelPlan(tuple(range(16)), dp=4, tp=2)


if __name__ == "__main__":
    unittest.main()
