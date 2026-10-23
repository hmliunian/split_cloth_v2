import numpy as np
import pytest

from split_cloth_v2.results import finalize_result, stage_intervals, stage_labels, validate_decision


def slots_at(*frames):
    return {f"stage{i}": None if frame is None else {"node_id": f"n{i}", "end_frame": frame} for i, frame in enumerate(frames, 1)}


def test_closed_intervals_cover_every_frame_and_switch_after_boundary():
    slots = slots_at(400, 485, 562)
    intervals = stage_intervals(681, slots)
    assert intervals == {"stage1": [0, 400], "stage2": [401, 485], "stage3": [486, 562], "stage4": [563, 680]}
    labels = stage_labels(681, slots)
    assert labels.shape == (681,)
    assert labels[[0, 400, 401, 485, 486, 562, 563, 680]].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert np.unique(labels).tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("frames", [(None, None, None), (None, None, 7), (1, None, 7), (3, 2, 7), (2, 2, 7), (-1, 3, 7), (1, 3, 9), (1, 3, True)])
def test_missing_or_illegal_boundaries_leave_entire_video_unknown(frames):
    slots = slots_at(*frames)
    assert stage_intervals(10, slots) is None
    assert stage_labels(10, slots).tolist() == [-1] * 10


@pytest.mark.parametrize("count", [0, -1, True, 4.5])
def test_invalid_frame_count_rejected(count):
    with pytest.raises(ValueError):
        stage_labels(count, slots_at(1, 2, 3))


@pytest.mark.parametrize("change", [
    {"node_id": "other"}, {"target_stage": 2}, {"target_stage": True},
    {"target_stage": 3.0}, {"decision": "yes"}, {"evidence": "  "},
    {"evidence": 42}, {"end_frame": 12}, {"participant_arms": ["left"]},
])
def test_decision_rejects_wrong_target_and_extra_outputs(change):
    response = {"node_id": "n0", "target_stage": 3, "decision": "match", "evidence": "visible folded bundle"}
    response.update(change)
    with pytest.raises(ValueError):
        validate_decision(response, "n0", 3)


def test_valid_decision_is_copied_without_semantic_reinterpretation():
    response = {"node_id": "n0", "target_stage": 3, "decision": "uncertain", "evidence": "occluded"}
    assert validate_decision(response, "n0", 3) == response
    assert validate_decision(response, "n0", 3) is not response


def test_complete_chain_with_uncertainty_remains_unreviewed():
    slots = slots_at(1, 3, 7)
    candidates = [{"node_id": f"n{i}", "source_frame_id": frame} for i, frame in enumerate((1, 3, 7), 1)]
    result = finalize_result({"key": "sample", "frame_count": 10}, candidates, slots, True, "frozen", [])
    assert result["status"] == "needs_review"
    assert result["review_status"] == "unreviewed"
    assert result["training_ready"] is False
    assert result["stage4_terminal_status"] == "not_evaluated"
    assert "uncertain_decision" in result["reasons"]
    assert result["intervals"] is not None


def test_slot_moved_off_candidate_is_an_execution_error():
    result = finalize_result({"frame_count": 10}, [], slots_at(1, 3, 7), False, "frozen", [])
    assert result["status"] == "execution_error"
