"""Transport/search integration with an offline HTTP transport."""

import json
import re

import httpx
import pytest

import split_cloth_v2.search as search
from split_cloth_v2.io_utils import sha256_file, write_json
from split_cloth_v2.provider import DashScopeProvider


def episode_input(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    nodes, evidence = [], {}
    for index, frame in enumerate((1, 3, 7)):
        node_id = f"n{index}"
        image_path = episode / "evidence" / f"{node_id}.jpg"
        image_path.parent.mkdir(exist_ok=True)
        image_path.write_bytes(f"synthetic-frame-{frame}".encode())
        nodes.append({"node_id": node_id, "source_frame_id": frame, "timestamp_s": frame / 20, "opening_arms": ["right"]})
        evidence[node_id] = {"images": [{"path": str(image_path.relative_to(episode)), "source_frame_id": frame,
                                        "timestamp_s": frame / 20, "relative_time_s": 0, "role": "CANDIDATE", "sha256": sha256_file(image_path)}]}
    write_json(episode / "input.json", {"key": "synthetic", "frame_count": 10})
    write_json(episode / "candidates.json", {"candidates": nodes})
    write_json(episode / "evidence/index.json", {"nodes": evidence})
    config = {"spec_version": "reverse_node_v1", "provider": {"model_id": "offline-model", "base_url": "https://offline.invalid/v1", "max_attempts": 3}}
    return episode, config


def api_reply(node_id, stage, finish_reason="stop"):
    return {"model": "returned-offline-model", "choices": [{"finish_reason": finish_reason,
            "message": {"content": json.dumps({"node_id": node_id, "target_stage": stage, "decision": "match", "evidence": "visible folded shape"})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def transport_with_calls(calls, rate_limit_first=False):
    def handler(request):
        body = json.loads(request.content)
        user_text = body["messages"][1]["content"][0]["text"]
        node_id = re.search(r"当前候选：(\w+)", user_text).group(1)
        stage = int(re.search(r"当前目标：stage(\d)", user_text).group(1))
        calls.append((node_id, stage))
        if rate_limit_first and len(calls) == 1:
            return httpx.Response(429, json={"error": "offline rate limit"})
        return httpx.Response(200, json=api_reply(node_id, stage))
    return httpx.MockTransport(handler)


def test_completed_http_response_is_replayed_after_decision_save_crash(tmp_path, monkeypatch):
    episode, config = episode_input(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key")
    calls = []
    provider = DashScopeProvider(transport=transport_with_calls(calls), sleep=lambda _: None)
    normal_write = search.write_json
    def crash_decision(path, value):
        if path.name == "decision.json":
            raise KeyboardInterrupt("crash after raw response")
        return normal_write(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(search, "write_json", crash_decision)
        with pytest.raises(KeyboardInterrupt):
            search.run_search(episode, provider, config, "run")
    assert calls == [("n2", 3)]
    result = search.run_search(episode, provider, config, "run")
    assert calls == [("n2", 3), ("n1", 2), ("n0", 1)]
    assert result["status"] == "complete"
    assert result["http_attempts"] == 3
    assert result["cache_hits"] == 1
    assert result["input_tokens"] == 30
    assert result["output_tokens"] == 15
    assert result["returned_models"] == ["returned-offline-model"]


def test_transport_retry_stays_one_logical_node_request(tmp_path, monkeypatch):
    episode, config = episode_input(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key")
    calls = []
    provider = DashScopeProvider(transport=transport_with_calls(calls, rate_limit_first=True), sleep=lambda _: None)
    result = search.run_search(episode, provider, config, "run")
    assert calls == [("n2", 3), ("n2", 3), ("n1", 2), ("n0", 1)]
    assert result["logical_calls"] == 3
    assert result["http_attempts"] == 4
    assert result["status"] == "complete"


@pytest.mark.parametrize("response_filename", ["response.json", "response_attempt_01.json"])
def test_inflight_attempt_usage_is_recovered_from_saved_raw_response(tmp_path, monkeypatch, response_filename):
    episode, config = episode_input(tmp_path)
    class InterruptedProvider:
        def judge(self, request, call_dir):
            write_json(call_dir / "attempts.json", [{"attempt": 1, "status": "inflight"}])
            write_json(call_dir / response_filename, api_reply(request["node_id"], request["target_stage"]))
            raise KeyboardInterrupt("crash before attempt usage update")
    with pytest.raises(KeyboardInterrupt):
        search.run_search(episode, InterruptedProvider(), config, "run")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key")
    calls = []
    provider = DashScopeProvider(transport=transport_with_calls(calls), sleep=lambda _: None)
    result = search.run_search(episode, provider, config, "run")
    assert calls == [("n1", 2), ("n0", 1)]
    assert result["http_attempts"] == 3
    assert result["input_tokens"] == 30
    assert result["output_tokens"] == 15
    assert result["returned_models"] == ["returned-offline-model"]


@pytest.mark.parametrize("finish_reason", [None, "length"])
def test_unfinished_raw_attempt_does_not_become_a_cached_semantic_answer(tmp_path, monkeypatch, finish_reason):
    episode, config = episode_input(tmp_path)
    class InterruptedProvider:
        def judge(self, request, call_dir):
            write_json(call_dir / "response_attempt_01.json", api_reply(request["node_id"], request["target_stage"], finish_reason))
            write_json(call_dir / "attempts.json", [{"attempt": 1, "status": "inflight"}])
            raise KeyboardInterrupt("interrupted after an unfinished response")
    with pytest.raises(KeyboardInterrupt):
        search.run_search(episode, InterruptedProvider(), config, "run")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key")
    calls = []
    provider = DashScopeProvider(transport=transport_with_calls(calls), sleep=lambda _: None)
    result = search.run_search(episode, provider, config, "run")
    assert calls == [("n2", 3), ("n1", 2), ("n0", 1)]
    assert result["cache_hits"] == 0
    assert result["http_attempts"] == 4
    assert result["status"] == "complete"
