"""V2 request and transport checks use local images and an in-process HTTP API."""

import base64
from copy import deepcopy
import json
from pathlib import Path

import httpx
from PIL import Image
import pytest
import yaml

from split_cloth_v2.config import load_config
from split_cloth_v2.expert_examples import freeze_expert_examples
from split_cloth_v2.io_utils import fingerprint, read_json, sha256_file, write_json
from split_cloth_v2.prompt import build_request
from split_cloth_v2.provider import DashScopeProvider, ProviderError
from split_cloth_v2.search import run_search


@pytest.fixture
def v2_episode(tmp_path):
    source = tmp_path / "configuration"
    (source / "images").mkdir(parents=True)
    for stage, color in enumerate(("red", "green", "blue"), 1):
        Image.new("RGB", (24, 20), color).save(source / f"images/s{stage}.png")
    fixed_text = "外置固定指令；只返回 JSON。\n保留此换行与空白。\n\n"
    (source / "prompt.txt").write_text(fixed_text, encoding="utf-8")
    repository_config = Path(__file__).resolve().parents[1] / "configs/reverse_node_v2.yaml"
    settings = yaml.safe_load(repository_config.read_text(encoding="utf-8"))
    settings["provider"].update(model_id="offline-model", base_url="https://offline.invalid/v1",
                                api_key_env="SPLIT_CLOTH_V2_OFFLINE_KEY")
    config_path = source / "v2.yaml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    config = load_config(config_path)
    run = tmp_path / "run"
    freeze_expert_examples(run, config)
    episode = run / "episodes/test"
    image_dir = episode / "evidence/frames"
    image_dir.mkdir(parents=True)
    node = {"node_id": "n0000", "source_frame_id": 20, "timestamp_s": 1., "opening_arms": ["right"]}
    images = []
    for frame in (0, 10, 20, 30, 40):
        path = image_dir / f"frame_{frame:06d}.jpg"
        Image.new("RGB", (32, 24), (frame, 10, 20)).save(path)
        images.append({"source_frame_id": frame, "timestamp_s": frame / 20,
                       "relative_time_s": frame / 20 - 1,
                       "role": "BEFORE" if frame < 20 else "AFTER" if frame > 20 else "CANDIDATE",
                       "path": str(path.relative_to(episode)), "sha256": sha256_file(path)})
    write_json(episode / "evidence/index.json", {
        "context_policy": config["context_policy"], "context_offsets_s": config["context_offsets_s"],
        "image_config": {"max_edge": 768, "jpeg_quality": 90},
        "nodes": {node["node_id"]: {"source_frame_id": 20, "images": images}},
    })
    write_json(episode / "input.json", {"key": "test", "frame_count": 45})
    write_json(episode / "candidates.json", {"candidates": [node]})
    return run, episode, node, config, fixed_text


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_v2_request_uses_exact_external_prompt_and_one_target_example(v2_episode, monkeypatch, stage):
    run, episode, node, config, fixed_text = v2_episode
    monkeypatch.setattr("split_cloth_v2.prompt.system_text", lambda: pytest.fail("v2 cannot fall back to v1 text"))
    request = build_request(node, stage, episode, config)
    assert request["system_text"] == fixed_text
    assert request["prompt_version"] == "node_completion_prompt_v1"
    assert request["context_offsets_s"] == [-1., -.5, 0., .5, 1.]
    assert request["expert_examples"] == read_json(run / "expert_examples/index.json")
    example, *local = request["images"]
    assert len(request["images"]) == 6
    assert example["role"] == "EXPERT_EXAMPLE"
    assert example["stage"] == f"stage{stage}"
    assert Path(example["path"]) == run / f"expert_examples/s{stage}.png"
    assert not {"source_frame_id", "timestamp_s", "relative_time_s", "node_id"} & example.keys()
    assert [item["source_frame_id"] for item in local] == [0, 10, 20, 30, 40]
    assert [item["role"] for item in local] == ["BEFORE", "BEFORE", "CANDIDATE", "AFTER", "AFTER"]
    assert local[2]["relative_time_s"] == 0
    assert all(word not in request["user_text"] for word in ("opening_arms", "participant_arms", "fold_cloth_v3"))


def test_source_edits_do_not_change_a_prepared_request(v2_episode):
    _, episode, node, config, _ = v2_episode
    before = build_request(node, 2, episode, config)
    Path(config["prompt_file"]).write_text("Changed source text", encoding="utf-8")
    for path in config["expert_examples"].values():
        Path(path).write_bytes(b"changed source bytes")
    assert build_request(node, 2, episode, config) == before


@pytest.mark.parametrize("change", ["target_image", "other_stage_image", "mapping", "prompt"])
def test_frozen_changes_are_rejected_before_request(v2_episode, change):
    run, episode, node, config, _ = v2_episode
    if change == "target_image":
        (run / "expert_examples/s3.png").write_bytes(b"changed frozen image")
    elif change == "other_stage_image":
        (run / "expert_examples/s1.png").write_bytes(b"changed frozen image")
    elif change == "mapping":
        index_path = run / "expert_examples/index.json"
        index = read_json(index_path)
        index["stages"]["stage3"]["path"] = index["stages"]["stage1"]["path"]
        write_json(index_path, index)
    else:
        config["prompt_text"] += "changed"
    with pytest.raises(ValueError, match="frozen_"):
        build_request(node, 3, episode, config)


@pytest.mark.parametrize("change", ["role", "relative_time", "candidate_time", "duplicate_frame", "hash", "path", "sampling"])
def test_invalid_local_evidence_is_rejected(v2_episode, change):
    _, episode, node, config, _ = v2_episode
    path = episode / "evidence/index.json"
    index = read_json(path)
    images = index["nodes"][node["node_id"]]["images"]
    if change == "role":
        images[0]["role"] = "AFTER"
    elif change == "relative_time":
        images[0]["relative_time_s"] = -.2
    elif change == "candidate_time":
        images[2]["timestamp_s"] += .01
        images[2]["relative_time_s"] = .01
    elif change == "duplicate_frame":
        images[0]["source_frame_id"] = images[1]["source_frame_id"]
    elif change == "hash":
        (episode / images[0]["path"]).write_bytes(b"changed local image")
    elif change == "path":
        images[0]["path"] = "../../outside.jpg"
    else:
        index["context_offsets_s"] = [-2., -1., 0., 1., 2.]
    write_json(path, index)
    with pytest.raises(ValueError):
        build_request(node, 3, episode, config)


def test_request_fingerprint_covers_sampling_and_complete_example_mapping(v2_episode):
    _, episode, node, config, _ = v2_episode
    request = build_request(node, 3, episode, config)
    changed = deepcopy(request)
    changed["expert_examples"]["stages"]["stage1"]["source_sha256"] = "different"
    assert fingerprint(changed) != fingerprint(request)
    changed = deepcopy(request)
    changed["context_offsets_s"] = [-2., 0., 2.]
    assert fingerprint(changed) != fingerprint(request)


@pytest.mark.parametrize("decision", ["skip", "uncertain"])
def test_http_payload_separates_example_and_context_without_semantic_retries(v2_episode, monkeypatch, decision):
    _, episode, node, config, fixed_text = v2_episode
    monkeypatch.setenv("SPLIT_CLOTH_V2_OFFLINE_KEY", "offline-key")
    calls, sleeps = [], []
    expected_image_bytes = [Path(item["path"]).read_bytes() for item in build_request(node, 3, episode, config)["images"]]

    def handler(http_request):
        payload = json.loads(http_request.content)
        calls.append(payload)
        assert payload["messages"][0] == {"role": "system", "content": fixed_text}
        user = payload["messages"][1]["content"]
        labels = [part["text"] for part in user[1:] if part["type"] == "text"]
        assert labels == ["EXPERT_EXAMPLE stage3", "BEFORE (-1.000s)", "BEFORE (-0.500s)",
                          "CANDIDATE (+0.000s)", "AFTER (+0.500s)", "AFTER (+1.000s)"]
        urls = [part["image_url"]["url"] for part in user if part["type"] == "image_url"]
        assert urls[0].startswith("data:image/png;base64,")
        assert [base64.b64decode(url.split(",", 1)[1]) for url in urls] == expected_image_bytes
        response = {"node_id": node["node_id"], "target_stage": 3, "decision": decision, "evidence": "visible evidence"}
        return httpx.Response(200, json={"model": "offline-returned", "choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(response)},
        }], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    provider = DashScopeProvider(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    result = run_search(episode, provider, config, "v2-run")
    assert result["logical_calls"] == result["http_attempts"] == len(calls) == 1
    assert result["status"] == "needs_review"
    assert result["has_uncertainty"] is (decision == "uncertain")
    assert sleeps == []
    assert run_search(episode, provider, config, "v2-run")["cache_hits"] == 1
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["image_bytes", "image_budget", "example_time"])
def test_payload_preflight_rejects_changes_before_http(v2_episode, monkeypatch, change):
    _, episode, node, config, _ = v2_episode
    monkeypatch.setenv("SPLIT_CLOTH_V2_OFFLINE_KEY", "offline-key")
    request = build_request(node, 3, episode, config)
    if change == "image_bytes":
        Path(request["images"][0]["path"]).write_bytes(b"changed after request assembly")
    elif change == "image_budget":
        request["images"].append(request["images"][-1])
    else:
        request["images"][0]["relative_time_s"] = 0
    calls = []

    def handler(http_request):
        calls.append(http_request)
        pytest.fail("invalid request must not reach HTTP")

    provider = DashScopeProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError):
        provider.judge(request, episode / "call")
    assert calls == []
    assert not (episode / "call/attempts.json").exists()
