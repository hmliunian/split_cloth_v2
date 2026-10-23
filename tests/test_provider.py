"""Provider boundary checks use an in-process HTTP transport only."""

import json

import httpx
import pytest

from split_cloth_v2.io_utils import read_json, sha256_file, write_json
from split_cloth_v2.provider import DashScopeProvider, ProviderError
from split_cloth_v2.search import run_search

TEST_KEY_ENV = "SPLIT_CLOTH_OFFLINE_TEST_KEY"
TEST_KEY = "offline-test-secret-should-never-be-logged"


def request_for_test(max_attempts=3):
    return {"node_id": "n0", "target_stage": 3, "system_text": "Judge one fixed candidate.",
            "user_text": "Current candidate n0; target stage3.", "images": [],
            "provider": {"model_id": "offline-model", "base_url": "https://offline.invalid/v1",
                         "api_key_env": TEST_KEY_ENV, "max_attempts": max_attempts}}


def decision(**changes):
    return {"node_id": "n0", "target_stage": 3, "decision": "match", "evidence": "visible shape", **changes}


def raw_response(value=None, finish_reason="stop"):
    return {"model": "offline-returned", "choices": [{"finish_reason": finish_reason,
            "message": {"content": json.dumps(decision() if value is None else value)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


@pytest.fixture
def configured_key(monkeypatch):
    monkeypatch.setenv(TEST_KEY_ENV, TEST_KEY)


def invalid_reply(kind):
    if kind == "outer_json":
        return httpx.Response(200, content=b"not-json")
    if kind == "truncated":
        return httpx.Response(200, json=raw_response(finish_reason="length"))
    raw = raw_response()
    if kind == "inner_json":
        raw["choices"][0]["message"]["content"] = "{broken"
    elif kind == "non_json_constant":
        raw["choices"][0]["message"]["content"] = '{"decision": NaN}'
    elif kind == "nontext":
        raw["choices"][0]["message"]["content"] = [{"type": "text", "text": "{}"}]
    elif kind == "missing_choice":
        raw["choices"] = []
    else:
        raise AssertionError(kind)
    return httpx.Response(200, json=raw)


@pytest.mark.parametrize("kind", ["outer_json", "inner_json", "truncated", "non_json_constant", "nontext", "missing_choice"])
def test_json_or_completion_errors_retry_within_fixed_budget(tmp_path, configured_key, kind):
    calls, sleeps = [], []
    def handler(request):
        calls.append(request)
        return invalid_reply(kind) if len(calls) == 1 else httpx.Response(200, json=raw_response())
    provider = DashScopeProvider(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    result = provider.judge(request_for_test(), tmp_path / "call")
    assert result == decision()
    assert len(calls) == 2
    assert len(sleeps) == 1
    attempts = read_json(tmp_path / "call/attempts.json")
    assert [row["status"] for row in attempts] == ["failed", "responded"]
    assert [row["attempt"] for row in attempts] == [1, 2]


@pytest.mark.parametrize("kind", ["outer_json", "truncated"])
def test_invalid_responses_stop_at_attempt_limit(tmp_path, configured_key, kind):
    calls = []
    def handler(request):
        calls.append(request)
        return invalid_reply(kind)
    provider = DashScopeProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError, match="retry_exhausted"):
        provider.judge(request_for_test(max_attempts=2), tmp_path / "call")
    assert len(calls) == 2
    assert len(read_json(tmp_path / "call/attempts.json")) == 2
    assert not (tmp_path / "call/response.json").exists()


@pytest.mark.parametrize("semantic_decision", ["skip", "uncertain"])
def test_legal_semantic_answer_returns_once_without_retry(tmp_path, configured_key, semantic_decision):
    calls, sleeps = [], []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=raw_response(decision(decision=semantic_decision)))
    provider = DashScopeProvider(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    assert provider.judge(request_for_test(), tmp_path / "call")["decision"] == semantic_decision
    assert len(calls) == 1
    assert sleeps == []


def single_node_episode(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    image = episode / "evidence/node.jpg"
    image.parent.mkdir()
    image.write_bytes(b"synthetic image for offline transport")
    node = {"node_id": "n0", "source_frame_id": 3, "timestamp_s": 0.15, "opening_arms": ["right"]}
    write_json(episode / "input.json", {"key": "synthetic", "frame_count": 10})
    write_json(episode / "candidates.json", {"candidates": [node]})
    write_json(episode / "evidence/index.json", {"nodes": {"n0": {"images": [{"path": "evidence/node.jpg",
        "source_frame_id": 3, "timestamp_s": 0.15, "relative_time_s": 0, "role": "CANDIDATE", "sha256": sha256_file(image)}]}}})
    return episode


@pytest.mark.parametrize("reply", [
    {"node_id": "another"}, {"target_stage": 2}, {"decision": "accept"},
    {"end_frame": 2}, {"evidence": ""}, {"target_stage": True},
])
def test_valid_json_with_invalid_schema_is_not_resampled(tmp_path, configured_key, reply):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=raw_response(decision(**reply)))
    episode = single_node_episode(tmp_path)
    provider = DashScopeProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    config = {"spec_version": "reverse_node_v1", "provider": request_for_test()["provider"]}
    result = run_search(episode, provider, config, "frozen-run")
    assert result["status"] == "execution_error"
    assert result["logical_calls"] == 1
    assert result["http_attempts"] == 1
    assert result["slots"] == {"stage1": None, "stage2": None, "stage3": None}
    assert len(calls) == 1
    assert not (episode / "calls/n0/decision.json").exists()
    # Resume preserves the same invalid semantic answer without asking again.
    assert run_search(episode, provider, config, "frozen-run")["status"] == "execution_error"
    assert len(calls) == 1


def test_401_is_not_retried_and_error_body_or_key_is_not_logged(tmp_path, configured_key, capsys):
    calls = []
    upstream_detail = "sensitive-upstream-error-body"
    def handler(request):
        assert request.headers["Authorization"] == "Bearer " + TEST_KEY
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": TEST_KEY + upstream_detail}})
    provider = DashScopeProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError, match="provider_http_401") as error:
        provider.judge(request_for_test(), tmp_path / "call")
    assert len(calls) == 1
    recorded = "\n".join(path.read_text() for path in tmp_path.rglob("*.json"))
    captured = capsys.readouterr()
    recorded += str(error.value) + captured.out + captured.err
    assert TEST_KEY not in recorded
    assert upstream_detail not in recorded
    assert read_json(tmp_path / "call/attempts.json")[0]["error"] == "provider_http_401"


def test_rate_limit_exhaustion_and_reentry_do_not_reset_budget(tmp_path, configured_key):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": "rate limited"})
    provider = DashScopeProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    request, call_dir = request_for_test(max_attempts=3), tmp_path / "call"
    with pytest.raises(ProviderError, match="retry_exhausted"):
        provider.judge(request, call_dir)
    assert len(calls) == 3
    prior_attempts = (call_dir / "attempts.json").read_bytes()
    with pytest.raises(ProviderError, match="retry_exhausted"):
        provider.judge(request, call_dir)
    assert len(calls) == 3
    assert (call_dir / "attempts.json").read_bytes() == prior_attempts


def test_existing_attempts_leave_only_remaining_budget(tmp_path, configured_key):
    calls = []
    call_dir = tmp_path / "call"
    write_json(call_dir / "attempts.json", [{"attempt": 1, "status": "transport_error"}, {"attempt": 2, "status": "inflight"}])
    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": "rate limited"})
    provider = DashScopeProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError, match="retry_exhausted"):
        provider.judge(request_for_test(max_attempts=3), call_dir)
    assert len(calls) == 1
    assert [row["attempt"] for row in read_json(call_dir / "attempts.json")] == [1, 2, 3]


@pytest.mark.parametrize("key", [None, "contains whitespace"])
def test_missing_or_invalid_key_fails_before_network(tmp_path, monkeypatch, key):
    if key is None:
        monkeypatch.delenv(TEST_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(TEST_KEY_ENV, key)
    calls = []
    def handler(request):
        calls.append(request)
        raise AssertionError("network must not be contacted")
    provider = DashScopeProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="missing_or_invalid_credential"):
        provider.judge(request_for_test(), tmp_path / "call")
    assert calls == []
    assert not (tmp_path / "call").exists()
