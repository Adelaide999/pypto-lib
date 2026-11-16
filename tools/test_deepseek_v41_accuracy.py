# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU regressions for fixed-history full-vocabulary precision comparisons."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import deepseek_v41_accuracy as accuracy
from tools import deepseek_v41_vllm_reference as reference


@pytest.fixture
def prepared(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    accuracy.write_json(checkpoint / "config.json", dict(
        architectures=["DeepseekV41ForCausalLM"], text_config=dict(num_hidden_layers=40, vocab_size=8),
        quantization_config=dict(quant_method="ascend", model_quant_type="W8A8_DYNAMIC"),
    ))
    accuracy.write_json(checkpoint / "quant_model_weights.safetensors.index.json",
                        dict(weight_map={"model.weight": "weights.safetensors"}))
    (checkpoint / "weights.safetensors").write_bytes(b"fixture-only")
    for name in ("tokenizer.json", "quant_model_description.json"):
        accuracy.write_json(checkpoint / name, {})
    source = tmp_path / "source.json"
    accuracy.write_json(source, dict(cases=[dict(id="trace", prompt_token_ids=[0, 1], trace_token_ids=[5, 6, 7])]))
    path = tmp_path / "cases.json"
    spec = accuracy.prepare_cases(checkpoint, source, path)
    return spec, path, source


@pytest.fixture
def captures(tmp_path, prepared):
    spec, path, _ = prepared
    logits = torch.tensor([[0, 1, 7, 2, 3, 4, 5, 6],
                           [2, 3, 4, 5, 6, 7, 0, 1],
                           [0, 7, 6, 5, 4, 3, 2, 1]], dtype=torch.float32)
    directories = []
    for backend, name in (("pypto", "actual"), ("vllm-ascend", "reference")):
        directory = tmp_path / name
        directory.mkdir()
        record = accuracy.save_case(directory, spec["cases"][0], logits, spec["vocab_size"])
        manifest = dict(spec, backend=backend, status="complete", output_kind="raw_logits", active_dp_rank=0,
                        chunk_size=16, case_spec_sha256=accuracy.file_sha256(path),
                        expected_case_ids=[case["id"] for case in spec["cases"]], cases=[record])
        accuracy.write_json(directory / "manifest.json", manifest)
        directories.append(directory)
    return directories[0], directories[1], logits


def edit_manifest(directory, mutate):
    path = directory / "manifest.json"
    manifest = accuracy.read_json(path)
    mutate(manifest)
    accuracy.write_json(path, manifest)


def test_strict_match_and_trace_token_are_independent(captures):
    actual, expected, _ = captures
    report = accuracy.compare_captures(actual, expected, topk=3, rtol=0, atol=0, require_top1=True)
    assert report["status"] == "pass"
    assert report["summary"]["top1_agreement"] == 1
    assert report["summary"]["max_abs_error"] == 0
    assert report["by_phase"]["prefill"]["steps"] == 1
    assert report["by_phase"]["decode"]["steps"] == 2
    assert [row["context_length"] for row in report["steps"]] == [2, 3, 4]
    assert report["steps"][0]["actual_top1"] == 2
    assert report["steps"][0]["trace_token_id"] == 5
    assert report["steps"][0]["topk_overlap"] == 1


def test_cli_measures_without_implicit_precision_threshold(captures, tmp_path):
    actual, expected, _ = captures
    output = tmp_path / "metrics.json"
    assert accuracy.main(["compare", "--actual", str(actual), "--reference", str(expected),
                          "--output", str(output), "--topk", "3"]) == 0
    assert accuracy.read_json(output)["status"] == "measured"


def test_decode_regression_fails_cli_and_preserves_prefill(captures, tmp_path):
    actual, expected, logits = captures
    logits[1, 5], logits[1, 6] = -2, 20
    manifest = accuracy.read_json(actual / "manifest.json")
    case = {key: manifest["cases"][0][key] for key in ("id", "prompt_token_ids", "trace_token_ids")}
    manifest["cases"][0] = accuracy.save_case(actual, case, logits, manifest["vocab_size"])
    accuracy.write_json(actual / "manifest.json", manifest)
    output = tmp_path / "failed.json"
    assert accuracy.main(["compare", "--actual", str(actual), "--reference", str(expected),
                          "--output", str(output), "--topk", "3", "--rtol", "0", "--atol", "0",
                          "--require-top1"]) == 1
    report = accuracy.read_json(output)
    assert report["status"] == "fail"
    assert report["by_phase"]["prefill"]["max_abs_error"] == 0
    assert report["by_phase"]["decode"]["max_abs_error"] == 20
    assert not report["steps"][1]["top1_match"]
    assert not report["steps"][1]["logits_pass"]
    assert report["steps"][2]["logits_pass"]


@pytest.mark.parametrize("field", ["checkpoint_identity", "case_spec_sha256", "checkpoint", "chunk_size"])
def test_rejects_different_checkpoint_cases_or_chunking(captures, field):
    actual, expected, _ = captures
    edit_manifest(actual, lambda manifest: manifest.update({field: "different"}))
    with pytest.raises(ValueError, match=field):
        accuracy.compare_captures(actual, expected, topk=3)


def test_rejects_shifted_decode_history(captures):
    actual, expected, _ = captures
    edit_manifest(actual, lambda manifest: manifest["cases"][0]["contexts"][1].append(6))
    with pytest.raises(ValueError, match="Decode history"):
        accuracy.compare_captures(actual, expected, topk=3)


def test_rejects_partial_capture(captures):
    actual, expected, _ = captures
    edit_manifest(actual, lambda manifest: manifest.update(status="running"))
    with pytest.raises(ValueError, match="incomplete"):
        accuracy.compare_captures(actual, expected, topk=3)


def test_rejects_complete_status_with_missing_prepared_cases(captures):
    actual, expected, _ = captures
    edit_manifest(actual, lambda manifest: manifest["expected_case_ids"].append("missing-case"))
    with pytest.raises(ValueError, match="all prepared cases"):
        accuracy.compare_captures(actual, expected, topk=3)


def test_rejects_modified_tensor(captures):
    actual, expected, _ = captures
    with (actual / "trace.pt").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="changed"):
        accuracy.compare_captures(actual, expected, topk=3)


@pytest.mark.parametrize("corruption", ["shape", "nan", "dtype"])
def test_rejects_invalid_full_vocabulary_tensor(captures, corruption):
    actual, expected, logits = captures
    if corruption == "shape":
        logits = logits[:, :-1]
    elif corruption == "nan":
        logits[1, 2] = float("nan")
    else:
        logits = logits.to(torch.float16)
    path = actual / "trace.pt"
    torch.save(logits, path)
    edit_manifest(actual, lambda manifest: manifest["cases"][0].update(logits_sha256=accuracy.file_sha256(path)))
    with pytest.raises(ValueError, match="shape|dtype|nonfinite"):
        accuracy.compare_captures(actual, expected, topk=3)


@pytest.mark.parametrize("corruption", ["legacy_topology", "invalid_id", "overflow", "duplicate_id"])
def test_prepare_rejects_legacy_topology_and_invalid_cases(prepared, tmp_path, corruption):
    spec, _, source = prepared
    raw = accuracy.read_json(source)
    if corruption == "legacy_topology":
        raw["topology"] = dict(accuracy.TOPOLOGY, tp=1, dp=16)
    elif corruption == "invalid_id":
        raw["cases"][0]["trace_token_ids"] = [8]
    elif corruption == "overflow":
        raw["cases"][0]["prompt_token_ids"] = [0] * accuracy.MAX_CONTEXT
    else:
        raw["cases"].append(deepcopy(raw["cases"][0]))
    accuracy.write_json(source, raw)
    with pytest.raises(ValueError):
        accuracy.prepare_cases(spec["checkpoint"], source, tmp_path / "invalid.json")


def test_load_cases_detects_changed_weight_shard(prepared):
    spec, path, _ = prepared
    (Path(spec["checkpoint"]) / "weights.safetensors").write_bytes(b"changed-shard")
    with pytest.raises(ValueError, match="metadata differs"):
        accuracy.load_cases(path)


def native_request(case, logits):
    rows = [{token: SimpleNamespace(logprob=float(row[token])) for token in reversed(range(len(row)))}
            for row in logits]
    completion = SimpleNamespace(token_ids=case["trace_token_ids"], logprobs=rows)
    return SimpleNamespace(prompt_token_ids=case["prompt_token_ids"], outputs=[completion])


def test_native_extraction_preserves_raw_scores_and_token_index(prepared):
    spec, _, _ = prepared
    case = spec["cases"][0]
    logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    request = native_request(case, logits)
    assert torch.equal(reference.extract_logits(request, case, 8), logits)


@pytest.mark.parametrize("corruption", ["partial_vocabulary", "wrong_history", "wrong_prompt", "missing_step"])
def test_native_extraction_rejects_incomplete_or_unaligned_results(prepared, corruption):
    spec, _, _ = prepared
    case = spec["cases"][0]
    request = native_request(case, torch.arange(24).reshape(3, 8))
    if corruption == "partial_vocabulary":
        request.outputs[0].logprobs[0].pop(2)
    elif corruption == "wrong_history":
        request.outputs[0].token_ids = [5, 7, 6]
    elif corruption == "wrong_prompt":
        request.prompt_token_ids = [1, 0]
    else:
        request.outputs[0].logprobs.pop()
    with pytest.raises(ValueError):
        reference.extract_logits(request, case, 8)


def test_native_entry_rejects_legacy_topology(prepared, tmp_path):
    _, path, _ = prepared
    with pytest.raises(SystemExit) as result:
        reference.main(["--cases", str(path), "--output", str(tmp_path / "reference"), "--tensor-parallel", "1"])
    assert result.value.code == 2


@pytest.mark.parametrize("arguments", [dict(rtol=0), dict(atol=0), dict(rtol=-1, atol=0),
                                      dict(max_error_ratio=0.1), dict(rtol=0, atol=0, max_error_ratio=1.1)])
def test_rejects_undefined_or_invalid_acceptance_policy(captures, arguments):
    actual, expected, _ = captures
    with pytest.raises(ValueError):
        accuracy.compare_captures(actual, expected, topk=3, **arguments)
