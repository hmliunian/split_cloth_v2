import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from split_cloth_v2.io_utils import read_json, sha256_file, write_json
from split_cloth_v2.search import RunConflictError, run_search


def prepare(tmp_path, frames=(1, 3, 7), frame_count=10):
    episode = tmp_path / "episode"
    episode.mkdir()
    nodes, evidence = [], {}
    for index, frame in enumerate(frames):
        node_id = f"n{index:04d}"
        path = episode / "evidence" / f"{node_id}.bin"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"frame-{frame}".encode())
        nodes.append({"node_id": node_id, "source_frame_id": frame, "timestamp_s": frame / 20, "opening_arms": ["right"]})
        evidence[node_id] = {"images": [{"path": str(path.relative_to(episode)), "source_frame_id": frame, "timestamp_s": frame / 20,
                                        "relative_time_s": 0, "role": "CANDIDATE", "sha256": sha256_file(path)}]}
    write_json(episode / "input.json", {"key": "synthetic", "frame_count": frame_count, "source_video_sha256": "video", "source_parquet_sha256": "state"})
    write_json(episode / "candidates.json", {"candidates": nodes})
    write_json(episode / "evidence" / "index.json", {"nodes": evidence})
    config = {"spec_version": "reverse_node_v1", "provider": {"model_id": "offline"}, "search_order": [3, 2, 1], "detector": {"settled_samples": 3}}
    return episode, config, nodes


class RecordingProvider:
    def __init__(self, decisions=None, crash_node=None, raw_attempt=False):
        self.decisions = decisions or {}
        self.calls = []
        self.crash_node = crash_node
        self.raw_attempt = raw_attempt

    def judge(self, request, call_dir):
        node_id, stage = request["node_id"], request["target_stage"]
        self.calls.append((request["images"][0]["source_frame_id"], stage))
        override = self.decisions.get(node_id, "match")
        response = {"node_id": node_id, "target_stage": stage, "decision": "match", "evidence": "visible shape"}
        response.update(override if isinstance(override, dict) else {"decision": override})
        raw = {"model": "offline-returned", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response)}}]}
        write_json(call_dir / ("response_attempt_01.json" if self.raw_attempt else "response.json"), raw)
        write_json(call_dir / "attempts.json", [{"usage": {"prompt_tokens": 3, "completion_tokens": 2}, "elapsed_s": 0.01}])
        if node_id == self.crash_node:
            raise KeyboardInterrupt("simulated power loss")
        return response


def boundaries(result):
    return [result["slots"][f"stage{i}"]["end_frame"] if result["slots"][f"stage{i}"] else None for i in (1, 2, 3)]


def test_000543_simulation_follows_reverse_targets_then_stops(tmp_path):
    episode, config, nodes = prepare(tmp_path, frames=(22, 76, 167, 169, 273, 274, 399, 400, 485, 562, 626), frame_count=681)
    provider = RecordingProvider({nodes[-1]["node_id"]: "skip"})
    result = run_search(episode, provider, config, "run")
    assert provider.calls == [(626, 3), (562, 3), (485, 2), (400, 1)]
    assert boundaries(result) == [400, 485, 562]
    assert result["status"] == "complete"
    assert result["logical_calls"] == 4
    assert result["http_attempts"] == 4
    assert result["input_tokens"] == 12
    assert result["output_tokens"] == 8
    assert result["returned_models"] == ["offline-returned"]
    labels = np.load(episode / "stage_candidates.npy", allow_pickle=False)
    assert labels[[400, 401, 485, 486, 562, 563]].tolist() == [1, 2, 2, 3, 3, 4]


def test_uncertain_keeps_target_and_flags_complete_chain(tmp_path):
    episode, config, nodes = prepare(tmp_path, frames=(1, 3, 6, 7))
    provider = RecordingProvider({nodes[-1]["node_id"]: "uncertain"})
    result = run_search(episode, provider, config, "run")
    assert provider.calls == [(7, 3), (6, 3), (3, 2), (1, 1)]
    assert boundaries(result) == [1, 3, 6]
    assert result["status"] == "needs_review"
    assert result["has_uncertainty"] is True
    assert (np.load(episode / "stage_candidates.npy") >= 1).all()


@pytest.mark.parametrize("last_decision,expected", [("skip", [None, None, None]), ("match", [None, None, 7])])
def test_exhaustion_preserves_fixed_slots_without_second_search(tmp_path, last_decision, expected):
    episode, config, nodes = prepare(tmp_path)
    decisions = {node["node_id"]: "skip" for node in nodes}
    decisions[nodes[-1]["node_id"]] = last_decision
    provider = RecordingProvider(decisions)
    result = run_search(episode, provider, config, "run")
    assert boundaries(result) == expected
    assert result["status"] == "needs_review"
    assert len(provider.calls) == len(nodes)
    assert np.load(episode / "stage_candidates.npy").tolist() == [-1] * 10


def test_last_source_frame_is_skipped_without_provider_call(tmp_path):
    episode, config, _ = prepare(tmp_path, frames=(1, 3, 7, 9))
    provider = RecordingProvider()
    result = run_search(episode, provider, config, "run")
    assert provider.calls == [(7, 3), (3, 2), (1, 1)]
    trace = [json.loads(line) for line in (episode / "trace.jsonl").read_text().splitlines()]
    assert trace[0]["reason"] == "no_stage4_frames"
    assert result["logical_calls"] == 3


@pytest.mark.parametrize("override", [{"node_id": "other"}, {"target_stage": 2}, {"decision": "accept"}, {"end_frame": 5}, {"evidence": ""}])
def test_invalid_response_is_execution_error_and_not_skip(tmp_path, override):
    episode, config, nodes = prepare(tmp_path)
    provider = RecordingProvider({nodes[-1]["node_id"]: override})
    result = run_search(episode, provider, config, "run")
    assert result["status"] == "execution_error"
    assert boundaries(result) == [None, None, None]
    assert provider.calls == [(7, 3)]
    assert not (episode / "calls" / nodes[-1]["node_id"] / "decision.json").exists()


def test_completed_search_replays_without_new_provider_calls_and_keeps_review(tmp_path):
    episode, config, _ = prepare(tmp_path)
    review_path = episode / "review.json"
    review_path.write_text('{"human":"do not overwrite"}')
    first = run_search(episode, RecordingProvider(), config, "run")
    provider = RecordingProvider()
    second = run_search(episode, provider, config, "run")
    assert provider.calls == []
    assert boundaries(first) == boundaries(second)
    assert second["cache_hits"] == 3
    assert second["http_attempts"] == 3
    assert review_path.read_text() == '{"human":"do not overwrite"}'


@pytest.mark.parametrize("raw_attempt", [False, True])
def test_crash_after_raw_response_recovers_without_repeating_call(tmp_path, raw_attempt):
    episode, config, nodes = prepare(tmp_path)
    provider = RecordingProvider(crash_node=nodes[-1]["node_id"], raw_attempt=raw_attempt)
    with pytest.raises(KeyboardInterrupt):
        run_search(episode, provider, config, "run")
    assert provider.calls == [(7, 3)]
    resumed = RecordingProvider()
    result = run_search(episode, resumed, config, "run")
    assert resumed.calls == [(3, 2), (1, 1)]
    assert boundaries(result) == [1, 3, 7]
    assert result["cache_hits"] == 1


def test_illegal_saved_semantic_answer_is_not_queried_again(tmp_path):
    episode, config, nodes = prepare(tmp_path)
    first = RecordingProvider({nodes[-1]["node_id"]: {"node_id": "other"}})
    assert run_search(episode, first, config, "run")["status"] == "execution_error"
    resumed = RecordingProvider()
    assert run_search(episode, resumed, config, "run")["status"] == "execution_error"
    assert resumed.calls == []


@pytest.mark.parametrize("change", ["model", "prompt", "detector", "sampling", "run", "input", "evidence"])
def test_changed_fingerprints_reject_resume_without_overwriting_result(tmp_path, change):
    episode, config, nodes = prepare(tmp_path)
    run_search(episode, RecordingProvider(), config, "run")
    prior = (episode / "result.json").read_bytes()
    changed = deepcopy(config)
    run_id = "run"
    if change == "model":
        changed["provider"]["model_id"] = "different"
    elif change == "prompt":
        changed["prompt_text"] = "changed prompt"
    elif change == "detector":
        changed["detector"]["settled_samples"] = 4
    elif change == "sampling":
        changed["context_offsets_s"] = [-2, 0, 2]
    elif change == "run":
        run_id = "different"
    elif change == "input":
        data = read_json(episode / "input.json")
        data["source_video_sha256"] = "changed"
        write_json(episode / "input.json", data)
    else:
        (episode / "evidence" / f"{nodes[0]['node_id']}.bin").write_bytes(b"changed image")
    with pytest.raises(RunConflictError):
        run_search(episode, RecordingProvider(), changed, run_id)
    assert (episode / "result.json").read_bytes() == prior


def test_replay_refuses_tampered_request_fingerprint(tmp_path):
    episode, config, nodes = prepare(tmp_path)
    run_search(episode, RecordingProvider(), config, "run")
    request_path = episode / "calls" / nodes[-1]["node_id"] / "request.json"
    request = read_json(request_path)
    request["request_fingerprint"] = "different"
    write_json(request_path, request)
    with pytest.raises(RunConflictError):
        run_search(episode, RecordingProvider(), config, "run")


@pytest.mark.parametrize("change", ["duplicate_frame", "unsafe_id", "empty_arms", "out_of_range"])
def test_malformed_candidate_rejected_before_any_request(tmp_path, change):
    episode, config, nodes = prepare(tmp_path)
    if change == "duplicate_frame":
        nodes[0]["source_frame_id"] = nodes[1]["source_frame_id"]
    elif change == "unsafe_id":
        nodes[0]["node_id"] = "../outside"
    elif change == "empty_arms":
        nodes[0]["opening_arms"] = []
    else:
        nodes[0]["source_frame_id"] = 10
    write_json(episode / "candidates.json", {"candidates": nodes})
    provider = RecordingProvider()
    with pytest.raises(ValueError):
        run_search(episode, provider, config, "run")
    assert provider.calls == []
    assert not (episode / "calls").exists()
