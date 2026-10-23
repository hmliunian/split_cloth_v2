"""Serial reverse-node search with persisted decisions and deterministic replay."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import fingerprint, read_json, sha256_file, write_json
from .prompt import build_request
from .results import STAGES, finalize_result, stage_labels, validate_decision


class RunConflictError(ValueError):
    """An existing run belongs to different inputs or inference settings."""


def _validate_candidates(candidates: Any, frame_count: int) -> list[dict]:
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    seen_ids, seen_frames = set(), set()
    for node in candidates:
        if not isinstance(node, dict):
            raise ValueError("candidate must be an object")
        node_id, frame = node.get("node_id"), node.get("source_frame_id")
        if not isinstance(node_id, str) or not node_id or Path(node_id).name != node_id or node_id in (".", ".."):
            raise ValueError("candidate node_id must be a safe nonempty filename")
        if type(frame) is not int or not 0 <= frame < frame_count:
            raise ValueError("candidate source_frame_id is outside the source video")
        timestamp = node.get("timestamp_s")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp):
            raise ValueError("candidate timestamp_s must be finite")
        arms = node.get("opening_arms")
        if not isinstance(arms, list) or not arms or any(arm not in ("left", "right") for arm in arms):
            raise ValueError("candidate must include at least one opening arm")
        if node_id in seen_ids or frame in seen_frames:
            raise ValueError("candidate ids and source frames must be unique")
        seen_ids.add(node_id)
        seen_frames.add(frame)
    return sorted(candidates, key=lambda node: node["source_frame_id"], reverse=True)


def _atomic_text(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _write_labels(path: Path, labels: np.ndarray) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, labels, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _recover_response(raw: Any) -> dict:
    """Recover a completed API response after a crash before decision persistence."""
    if isinstance(raw, dict) and {"node_id", "target_stage", "decision", "evidence"}.issubset(raw):
        return raw
    if isinstance(raw, dict) and "response" in raw:
        return _recover_response(raw["response"])
    if isinstance(raw, dict) and "decision" in raw and isinstance(raw["decision"], dict):
        return raw["decision"]
    try:
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("saved API response was not completed normally")
        content = choice["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if part.get("type") == "text")
        return json.loads(content)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("saved response does not contain a recoverable node decision") from exc


def _attempt_totals(episode_dir: Path, visited_ids: list[str]) -> dict:
    attempts_count = input_tokens = output_tokens = 0
    elapsed_s = 0.0
    returned_models = set()
    for node_id in visited_ids:
        call_dir = episode_dir / "calls" / node_id
        path = call_dir / "attempts.json"
        if not path.exists():
            continue
        data = read_json(path)
        if isinstance(data, list):
            attempts = data
        else:
            attempts = data.get("attempts", [])
            if isinstance(attempts, int):
                attempts_count += attempts
                attempts = []
            usage = data.get("usage", {})
            input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            elapsed_s += float(data.get("elapsed_s", 0) or 0)
        attempts_count += len(attempts)
        for index, attempt in enumerate(attempts, 1):
            attempt_number = attempt.get("attempt", index)
            raw_path = call_dir / f"response_attempt_{attempt_number:02d}.json"
            if not raw_path.exists() and index == len(attempts):
                raw_path = call_dir / "response.json"
            raw = read_json(raw_path) if raw_path.exists() else {}
            if not isinstance(raw, dict):
                raw = {}
            # A crash can happen after saving the raw response but before the
            # in-flight attempt record receives the returned usage and model.
            usage = attempt.get("usage") or raw.get("usage", {})
            input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            elapsed_s += float(attempt.get("elapsed_s", 0) or 0)
            if attempt.get("model"):
                returned_models.add(attempt["model"])
            elif raw.get("model"):
                returned_models.add(raw["model"])
        response_path = call_dir / "response.json"
        if response_path.exists():
            raw = read_json(response_path)
            if isinstance(raw, dict) and raw.get("model"):
                returned_models.add(raw["model"])
    return {
        "http_attempts": attempts_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider_elapsed_s": elapsed_s,
        "returned_models": sorted(returned_models),
    }


def run_search(episode_dir: Path, provider: Any, config: dict, run_fingerprint: str) -> dict:
    """Search stage3, then stage2, then stage1; never ask one node twice semantically.

    Existing decisions are replayed in exactly the same order. A changed run or
    request fingerprint raises before overwriting historical search products.
    """
    episode_dir = Path(episode_dir)
    input_info = read_json(episode_dir / "input.json")
    frame_count = input_info.get("frame_count")
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("input frame_count must be a positive integer")
    candidate_document = read_json(episode_dir / "candidates.json")
    candidates = _validate_candidates(candidate_document.get("candidates"), frame_count)
    if config.get("search_order", [3, 2, 1]) != [3, 2, 1]:
        raise ValueError("reverse_node_v1 requires search_order [3, 2, 1]")
    evidence_document = read_json(episode_dir / "evidence" / "index.json")
    # Validate frozen image bytes before replaying or replacing any search files.
    resolved_episode = episode_dir.resolve()
    for evidence_node in evidence_document["nodes"].values():
        for image in evidence_node["images"]:
            image_path = (episode_dir / image["path"]).resolve()
            if not image_path.is_relative_to(resolved_episode):
                raise RunConflictError("evidence path leaves episode directory")
            if sha256_file(image_path) != image["sha256"]:
                raise RunConflictError("prepared evidence changed; use a new run")
    context = {
        "run_fingerprint": run_fingerprint,
        "config": config,
        "input": input_info,
        "candidates": candidate_document,
        "evidence": evidence_document,
    }
    context_fingerprint = fingerprint(context)
    context_path = episode_dir / "search_context.json"
    if context_path.exists():
        if read_json(context_path).get("context_fingerprint") != context_fingerprint:
            raise RunConflictError("search inputs or configuration changed; use a new run")
    else:
        if (episode_dir / "result.json").exists() or (episode_dir / "calls").exists():
            raise RunConflictError("existing search products have no verifiable context; use a new run")
        write_json(context_path, {"context_fingerprint": context_fingerprint, "run_fingerprint": run_fingerprint})

    slots = dict.fromkeys(STAGES)
    trace: list[dict] = []
    visited_ids: list[str] = []
    target_stage = 3
    has_uncertainty = False
    execution_error = None
    started = time.monotonic()

    for node in candidates:
        row = {
            "node_id": node["node_id"],
            "source_frame_id": node["source_frame_id"],
            "target_stage": target_stage,
        }
        if node["source_frame_id"] == frame_count - 1:
            trace.append({**row, "event": "skip", "reason": "no_stage4_frames"})
            continue
        call_dir = episode_dir / "calls" / node["node_id"]
        try:
            request = build_request(node, target_stage, episode_dir, config)
            request_fingerprint = fingerprint({"context_fingerprint": context_fingerprint, "request": request})
            saved_request = {**request, "request_fingerprint": request_fingerprint}
            request_path = call_dir / "request.json"
            if request_path.exists():
                if read_json(request_path) != saved_request:
                    raise RunConflictError("request fingerprint changed; use a new run")
            else:
                call_dir.mkdir(parents=True, exist_ok=True)
                if any(call_dir.iterdir()):
                    raise RunConflictError("existing call artifacts have no request fingerprint")
                write_json(request_path, saved_request)
            visited_ids.append(node["node_id"])
            cached = False
            decision_path = call_dir / "decision.json"
            response_path = call_dir / "response.json"
            if decision_path.exists():
                saved = read_json(decision_path)
                if saved.get("request_fingerprint") != request_fingerprint:
                    raise RunConflictError("saved decision fingerprint changed; use a new run")
                response = saved["response"]
                cached = True
            elif response_path.exists():
                response = _recover_response(read_json(response_path))
                cached = True
            else:
                # Transport writes its raw reply before the final response file.
                # Recover that narrow crash window before making another request.
                previous_attempts = sorted(call_dir.glob("response_attempt_*.json"), reverse=True)
                recovered = False
                for attempt_path in previous_attempts:
                    try:
                        response = _recover_response(read_json(attempt_path))
                    except ValueError:
                        continue
                    cached = True
                    recovered = True
                    break
                if not recovered:
                    response = provider.judge(request, call_dir)
            decision = validate_decision(response, node["node_id"], target_stage)
            if not decision_path.exists():
                write_json(decision_path, {"request_fingerprint": request_fingerprint, "response": decision})
            row.update({"event": "decision", "decision": decision["decision"], "evidence": decision["evidence"], "cached": cached})
            if decision["decision"] == "uncertain":
                has_uncertainty = True
            elif decision["decision"] == "match":
                slots[f"stage{target_stage}"] = {
                    "node_id": node["node_id"],
                    "end_frame": node["source_frame_id"],
                    "timestamp_s": node["timestamp_s"],
                    "call_path": str(call_dir.relative_to(episode_dir)),
                    "request_fingerprint": request_fingerprint,
                }
                target_stage -= 1
            row["next_target_stage"] = target_stage if target_stage else None
            trace.append(row)
        except RunConflictError:
            raise
        except Exception as exc:
            execution_error = f"{type(exc).__name__}: {exc}"
            trace.append({**row, "event": "execution_error", "error": execution_error})
            break
        if target_stage == 0:
            break

    result = finalize_result(input_info, candidates, slots, has_uncertainty, run_fingerprint, trace, execution_error)
    result["logical_calls"] = len(visited_ids)
    result["context_fingerprint"] = context_fingerprint
    result["candidate_fingerprint"] = fingerprint(candidate_document)
    result["elapsed_s"] = time.monotonic() - started
    result.update(_attempt_totals(episode_dir, visited_ids))
    _atomic_text(episode_dir / "trace.jsonl", "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in trace))
    _write_labels(episode_dir / "stage_candidates.npy", stage_labels(frame_count, slots))
    write_json(episode_dir / "result.json", result)
    return result
