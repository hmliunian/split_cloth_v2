"""DashScope compatible transport and deterministic offline simulation."""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import time
from pathlib import Path

import httpx

from .io_utils import now, read_json, write_json


class ProviderError(RuntimeError):
    pass


def parse_response(raw: dict) -> dict:
    choice = raw["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ProviderError("response_truncated")
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise ProviderError("response_content_not_text")
    def reject_constant(value):
        raise ValueError("non_json_constant")
    return json.loads(content, parse_constant=reject_constant)


def _messages(request: dict) -> list[dict]:
    content = [{"type": "text", "text": request["user_text"]}]
    for item in request["images"]:
        if item["role"] == "EXPERT_EXAMPLE":
            if item.get("stage") != f"stage{request['target_stage']}" or any(name in item for name in ("source_frame_id", "timestamp_s", "relative_time_s", "node_id")):
                raise ProviderError("invalid_expert_example_label")
            label = f"EXPERT_EXAMPLE {item['stage']}"
        else:
            label = f"{item['role']} ({item['relative_time_s']:+.3f}s)"
        content.append({"type": "text", "text": label})
        path = Path(item["path"])
        image_bytes = path.read_bytes()
        if item.get("sha256") and hashlib.sha256(image_bytes).hexdigest() != item["sha256"]:
            raise ProviderError("request_image_hash_mismatch")
        encoded = base64.b64encode(image_bytes).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    return [{"role": "system", "content": request["system_text"]}, {"role": "user", "content": content}]


class DashScopeProvider:
    def __init__(self, *, credential_file: str | Path | None = None, transport=None, sleep=time.sleep):
        self.credential_file = Path(credential_file) if credential_file else None
        self.transport = transport
        self.sleep = sleep

    def _credential(self, config):
        value = os.environ.get(config.get("api_key_env", "DASHSCOPE_API_KEY"))
        if not value and self.credential_file is not None:
            value = self.credential_file.read_text(encoding="utf-8").strip()
        if not value or any(char.isspace() for char in value):
            raise ProviderError("missing_or_invalid_credential")
        return value

    def judge(self, request: dict, call_dir: Path) -> dict:
        call_dir = Path(call_dir)
        config = request["provider"]
        if len(request["images"]) > config.get("max_images", 5):
            raise ProviderError("image_budget_exceeded")
        key = self._credential(config)
        payload = {
            "model": config["model_id"], "messages": _messages(request),
            "temperature": config.get("temperature", 0),
            "max_tokens": config.get("max_output_tokens", 512), "stream": False,
            "response_format": {"type": "json_object"}, "enable_thinking": False,
        }
        attempts_path = call_dir / "attempts.json"
        attempts = read_json(attempts_path, [])
        max_attempts = config.get("max_attempts", 3)
        with httpx.Client(timeout=config.get("timeout_s", 180), transport=self.transport) as client:
            while len(attempts) < max_attempts:
                event = {"attempt": len(attempts) + 1, "started_at": now(), "status": "inflight"}
                attempts.append(event)
                write_json(attempts_path, attempts)
                started = time.monotonic()
                retryable = False
                try:
                    response = client.post(config["base_url"].rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer " + key}, json=payload)
                    event["http_status"] = response.status_code
                    if response.status_code != 200:
                        retryable = response.status_code in (408, 429, 500, 502, 503, 504)
                        raise ProviderError(f"provider_http_{response.status_code}")
                    try:
                        raw = response.json()
                    except ValueError as exc:
                        retryable = True
                        raise ProviderError("response_invalid_json") from exc
                    # Retain every response, including malformed/truncated replies.
                    write_json(call_dir / f"response_attempt_{event['attempt']:02d}.json", raw)
                    event["usage"] = raw.get("usage", {}) if isinstance(raw, dict) else {}
                    event["model"] = raw.get("model") if isinstance(raw, dict) else None
                    try:
                        decision = parse_response(raw)
                    except (ValueError, KeyError, IndexError, TypeError, ProviderError) as exc:
                        retryable = True
                        raise ProviderError("response_truncated_or_invalid_json") from exc
                    write_json(call_dir / "response.json", raw)
                    event["status"] = "responded"
                    event["elapsed_s"] = round(time.monotonic() - started, 6)
                    write_json(attempts_path, attempts)
                    return decision
                except (httpx.TimeoutException, httpx.TransportError):
                    retryable = True
                    event.update(status="transport_error", error="network_or_timeout")
                except ProviderError as exc:
                    event.update(status="failed", error=str(exc))
                    if not retryable:
                        event["elapsed_s"] = round(time.monotonic() - started, 6)
                        write_json(attempts_path, attempts)
                        raise
                event["elapsed_s"] = round(time.monotonic() - started, 6)
                write_json(attempts_path, attempts)
                if retryable and len(attempts) < max_attempts:
                    self.sleep(min(2 ** len(attempts), 8))
        raise ProviderError("retry_exhausted")


class MockProvider:
    """Explicit simulation; decisions are never inferred from reference frames."""
    def __init__(self, decisions: dict | None = None):
        self.decisions = decisions or {}

    def judge(self, request: dict, call_dir: Path) -> dict:
        item = self.decisions.get(request["node_id"], "uncertain")
        response = {"node_id": request["node_id"], "target_stage": request["target_stage"],
                    "decision": item if isinstance(item, str) else item["decision"],
                    "evidence": "离线模拟结果，不代表模型视觉判断。"}
        if isinstance(item, dict):
            response.update(item)
        write_json(Path(call_dir) / "response.json", response)
        write_json(Path(call_dir) / "attempts.json", [])
        return response
