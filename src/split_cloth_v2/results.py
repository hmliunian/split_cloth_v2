"""Strict node decisions and original-frame stage labels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

STAGES = ("stage1", "stage2", "stage3")
DECISION_FIELDS = {"node_id", "target_stage", "decision", "evidence"}


def validate_decision(response: Any, node_id: str, target_stage: int) -> dict:
    """Reject interface errors rather than interpreting them as a semantic skip."""
    if not isinstance(response, dict) or set(response) != DECISION_FIELDS:
        raise ValueError("decision must contain exactly node_id, target_stage, decision, evidence")
    if response["node_id"] != node_id:
        raise ValueError("decision node_id does not match current candidate")
    stage = response["target_stage"]
    if type(stage) is not int or stage not in (1, 2, 3) or stage != target_stage:
        raise ValueError("decision target_stage does not match current target")
    if response["decision"] not in ("match", "skip", "uncertain"):
        raise ValueError("decision must be match, skip, or uncertain")
    if not isinstance(response["evidence"], str) or not response["evidence"].strip():
        raise ValueError("decision evidence must be a nonempty string")
    return dict(response)


def _boundary_frames(frame_count: int, slots: Mapping[str, Any]) -> list[int] | None:
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    if not isinstance(slots, Mapping) or set(slots) != set(STAGES):
        return None
    frames = []
    for stage in STAGES:
        slot = slots[stage]
        if not isinstance(slot, Mapping) or type(slot.get("end_frame")) is not int:
            return None
        frames.append(slot["end_frame"])
    b1, b2, b3 = frames
    return frames if 0 <= b1 < b2 < b3 < frame_count - 1 else None


def stage_intervals(frame_count: int, slots: Mapping[str, Any]) -> dict | None:
    """Return four inclusive intervals only when every boundary is legal."""
    frames = _boundary_frames(frame_count, slots)
    if frames is None:
        return None
    b1, b2, b3 = frames
    return {
        "stage1": [0, b1],
        "stage2": [b1 + 1, b2],
        "stage3": [b2 + 1, b3],
        "stage4": [b3 + 1, frame_count - 1],
    }


def stage_labels(frame_count: int, slots: Mapping[str, Any]) -> np.ndarray:
    """Boundary frames belong to the preceding stage; partial chains stay unknown."""
    intervals = stage_intervals(frame_count, slots)
    labels = np.full(frame_count, -1, dtype=np.int8)
    if intervals is not None:
        for stage_number, (start, end) in enumerate(intervals.values(), 1):
            labels[start : end + 1] = stage_number
    return labels


def finalize_result(
    input_info: dict,
    candidates: list[dict],
    slots: dict,
    has_uncertainty: bool,
    run_fingerprint: str,
    trace: list[dict],
    execution_error: str | None = None,
    *,
    spec_version: str = "reverse_node_v1",
) -> dict:
    """Build an unreviewed automatic result without claiming visual correctness."""
    frame_count = input_info["frame_count"]
    intervals = stage_intervals(frame_count, slots)
    by_id = {candidate["node_id"]: candidate for candidate in candidates}
    reasons = []
    for stage in STAGES:
        slot = slots[stage]
        if slot is None:
            reasons.append(f"missing_{stage}")
            continue
        original = by_id.get(slot.get("node_id"))
        if original is None or original["source_frame_id"] != slot.get("end_frame"):
            execution_error = execution_error or "selected node is not an unchanged input candidate"
    if intervals is None and all(slots[stage] is not None for stage in STAGES):
        reasons.append("invalid_boundary_order")
    if has_uncertainty:
        reasons.append("uncertain_decision")
    if execution_error:
        reasons.append(execution_error)
    # The target enters stage3 even when there are no candidates. A match
    # enters the preceding target before the next iteration, which can then
    # exhaust without making a request. Lower targets are not independently
    # searched when a later stage never matched.
    entered = {3}
    for row in trace:
        if row.get("target_stage") in (1, 2, 3):
            entered.add(row["target_stage"])
        if row.get("next_target_stage") in (1, 2, 3):
            entered.add(row["next_target_stage"])
    search_status = {}
    for number in (1, 2, 3):
        stage = f"stage{number}"
        rows = [row for row in trace if row.get("target_stage") == number]
        if slots[stage] is not None:
            stage_status, reason = "selected", None
        elif number not in entered:
            stage_status, reason = "not_searched", "later_stage_not_selected"
        elif any(row.get("event") == "execution_error" for row in rows):
            stage_status, reason = "execution_error", "search_interrupted"
        else:
            stage_status, reason = "searched_not_found", "candidates_exhausted"
        search_status[stage] = {
            "status": stage_status,
            "reason": reason,
            "logical_calls": sum(row.get("event") in ("decision", "execution_error") for row in rows),
            "uncertain_count": sum(row.get("decision") == "uncertain" for row in rows),
        }
    status = "execution_error" if execution_error else (
        "complete" if intervals is not None and not has_uncertainty else "needs_review"
    )
    return {
        "spec_version": spec_version,
        "key": input_info.get("key"),
        "frame_count": frame_count,
        "run_fingerprint": run_fingerprint,
        "slots": slots,
        "search_status": search_status,
        "intervals": intervals,
        "status": status,
        "has_uncertainty": bool(has_uncertainty),
        "reasons": reasons,
        "review_status": "unreviewed",
        "training_ready": False,
        "stage4_terminal_status": "not_evaluated",
        "trace_path": "trace.jsonl",
        "candidate_count": len(candidates),
        "logical_calls": sum(row.get("event") in ("decision", "execution_error") for row in trace),
        "cache_hits": sum(bool(row.get("cached")) for row in trace),
    }
