"""Exercise the v2 state machine using real frozen examples and offline replies."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from split_cloth_v2.config import load_config
from split_cloth_v2.expert_examples import freeze_expert_examples
from split_cloth_v2.io_utils import read_json, sha256_file, write_json
from split_cloth_v2.search import RunConflictError, run_search


def prepare_v2(tmp_path, frames, frame_count=12):
    inputs = tmp_path / "configuration"
    inputs.mkdir()
    default = Path(__file__).resolve().parents[1] / "configs/default.yaml"
    raw = yaml.safe_load(default.read_text())
    raw.update(spec_version="reverse_node_v2", prompt_file="prompt.txt",
               prompt_version="node_completion_prompt_v1",
               expert_examples={f"stage{i}": f"s{i}.png" for i in (1, 2, 3)})
    raw["provider"]["max_images"] = 6
    (inputs / "prompt.txt").write_text("Offline completion instructions.\n", encoding="utf-8")
    for number in (1, 2, 3):
        Image.new("RGB", (8, 8), (number * 50, 0, 0)).save(inputs / f"s{number}.png")
    config_path = inputs / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    run = tmp_path / "run"
    run.mkdir()
    freeze_expert_examples(run, config)
    episode = run / "episodes" / "synthetic"
    episode.mkdir(parents=True)
    nodes, evidence = [], {}
    for index, frame in enumerate(frames):
        node_id = f"n{index:04d}"
        path = episode / "evidence" / f"{node_id}.jpg"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (8, 8), (frame, 0, 0)).save(path)
        nodes.append({"node_id": node_id, "source_frame_id": frame,
                      "timestamp_s": frame / 20, "opening_arms": ["right"]})
        evidence[node_id] = {"source_frame_id": frame, "images": [{
            "path": str(path.relative_to(episode)), "sha256": sha256_file(path),
            "source_frame_id": frame, "timestamp_s": frame / 20,
            "relative_time_s": 0, "role": "CANDIDATE", "width": 8, "height": 8,
        }]}
    write_json(episode / "input.json", {"key": "synthetic", "frame_count": frame_count,
                                         "source_video_sha256": "video", "source_parquet_sha256": "state"})
    write_json(episode / "candidates.json", {"candidates": nodes})
    write_json(episode / "evidence/index.json", {
        "context_policy": "fixed_local_v1", "context_offsets_s": config["context_offsets_s"], "nodes": evidence,
    })
    return episode, config, nodes


class V2Provider:
    def __init__(self, decisions=None, node_override=None):
        self.decisions = decisions or {}
        self.node_override = node_override
        self.calls = []

    def judge(self, request, call_dir):
        candidate, = [item for item in request["images"] if item["role"] == "CANDIDATE"]
        example, = [item for item in request["images"] if item["role"] == "EXPERT_EXAMPLE"]
        self.calls.append((candidate["source_frame_id"], request["target_stage"], example["stage"]))
        response = {"node_id": self.node_override or request["node_id"],
                    "target_stage": request["target_stage"],
                    "decision": self.decisions.get(candidate["source_frame_id"], "match"),
                    "evidence": "Offline visible evidence."}
        write_json(call_dir / "response.json", response)
        return response


@pytest.mark.parametrize("frames,expected,states", [
    ((), [None, None, None], ["not_searched", "not_searched", "searched_not_found"]),
    ((7,), [None, None, 7], ["not_searched", "searched_not_found", "selected"]),
    ((3, 7), [None, 3, 7], ["searched_not_found", "selected", "selected"]),
])
def test_small_candidate_sets_preserve_partial_v2_slots(tmp_path, frames, expected, states):
    episode, config, _ = prepare_v2(tmp_path, frames)
    provider = V2Provider()
    result = run_search(episode, provider, config, "run")
    assert result["spec_version"] == "reverse_node_v2"
    assert len(provider.calls) == len(frames)
    assert [(result["slots"][f"stage{i}"] or {}).get("end_frame") for i in (1, 2, 3)] == expected
    assert [result["search_status"][f"stage{i}"]["status"] for i in (1, 2, 3)] == states
    assert result["status"] == "needs_review"
    assert np.load(episode / "stage_candidates.npy").tolist() == [-1] * 12


def test_skips_uncertainty_and_stage_examples_replay_without_new_calls(tmp_path):
    episode, config, _ = prepare_v2(tmp_path, (1, 3, 5, 7, 9, 11))
    provider = V2Provider({9: "skip", 7: "uncertain"})
    first = run_search(episode, provider, config, "run")
    assert provider.calls == [(9, 3, "stage3"), (7, 3, "stage3"), (5, 3, "stage3"),
                              (3, 2, "stage2"), (1, 1, "stage1")]
    assert first["status"] == "needs_review"
    assert first["search_status"]["stage3"]["uncertain_count"] == 1
    trace = [json.loads(line) for line in (episode / "trace.jsonl").read_text().splitlines()]
    assert trace[0]["reason"] == "no_stage4_frames"
    resumed = V2Provider()
    second = run_search(episode, resumed, config, "run")
    assert resumed.calls == []
    assert second["cache_hits"] == 5
    assert second["slots"] == first["slots"]


def test_unmatched_stage3_leaves_lower_stages_unsearched(tmp_path):
    episode, config, _ = prepare_v2(tmp_path, (1, 7))
    result = run_search(episode, V2Provider({1: "skip", 7: "uncertain"}), config, "run")
    assert result["search_status"]["stage3"] == {
        "status": "searched_not_found", "reason": "candidates_exhausted", "logical_calls": 2, "uncertain_count": 1,
    }
    assert result["search_status"]["stage2"]["status"] == "not_searched"
    assert result["search_status"]["stage1"]["status"] == "not_searched"


@pytest.mark.parametrize("node_override", ["EXPERT_EXAMPLE", "other_candidate"])
def test_v2_rejects_noncurrent_ids_and_preserves_execution_error_on_resume(tmp_path, node_override):
    episode, config, _ = prepare_v2(tmp_path, (1, 3, 7))
    result = run_search(episode, V2Provider(node_override=node_override), config, "run")
    assert result["status"] == "execution_error"
    assert result["search_status"]["stage3"]["status"] == "execution_error"
    assert result["search_status"]["stage2"]["status"] == "not_searched"
    provider = V2Provider()
    assert run_search(episode, provider, config, "run")["status"] == "execution_error"
    assert provider.calls == []


@pytest.mark.parametrize("tamper", ["unsearched_example", "stage_mapping"])
def test_all_frozen_examples_verified_before_replay_or_overwrite(tmp_path, tamper):
    episode, config, _ = prepare_v2(tmp_path, (7,))
    run_search(episode, V2Provider(), config, "run")
    original_result = (episode / "result.json").read_bytes()
    run = episode.parent.parent
    if tamper == "unsearched_example":
        (run / "expert_examples/s1.png").write_bytes(b"tampered unused example")
    else:
        index_path = run / "expert_examples/index.json"
        index = read_json(index_path)
        index["stages"]["stage1"], index["stages"]["stage2"] = index["stages"]["stage2"], index["stages"]["stage1"]
        write_json(index_path, index)
    provider = V2Provider()
    with pytest.raises(RunConflictError, match="frozen expert examples changed"):
        run_search(episode, provider, config, "run")
    assert provider.calls == []
    assert (episode / "result.json").read_bytes() == original_result
