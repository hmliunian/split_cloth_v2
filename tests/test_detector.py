import math

import pytest

from split_cloth_v2.detector import detect_candidates


def _run(left, right=None, **config):
    right = right if right is not None else [0.] * len(left)
    states = [[0.] * 14 for _ in left]
    for state, lvalue, rvalue in zip(states, left, right):
        state[6], state[13] = lvalue, rvalue
    times = [index / 20 for index in range(len(left))]
    return detect_candidates(states, times, list(range(len(left))), source_timestamps=times, config=config)


def test_either_arm_opens_and_plateau_is_backdated():
    result = _run([0.] * 20 + [1.] * 20)
    node, = result["candidates"]
    assert node["source_frame_id"] == 20
    assert node["opening_arms"] == ["left"]
    assert node["changes"][0]["confirmed_at_source_frame_id"] == 22
    assert node["changes"][0]["plateau_source_frame_ids"] == [20, 21, 22]
    right = _run([0.] * 40, [0.] * 20 + [1.] * 20)["candidates"]
    assert [(node["source_frame_id"], node["opening_arms"]) for node in right] == [(20, ["right"])]


def test_same_frame_merges_but_adjacent_frames_stay_separate():
    left = [0.] * 20 + [1.] * 20
    right = [0.] * 21 + [1.] * 19
    separate = _run(left, right)["candidates"]
    assert [(node["source_frame_id"], node["opening_arms"]) for node in separate] == [(20, ["left"]), (21, ["right"])]
    together = _run(left, left)["candidates"]
    assert len(together) == 1 and together[0]["opening_arms"] == ["left", "right"]


def test_initial_and_sustained_open_do_not_create_extra_events():
    assert _run([1.] * 40)["candidates"] == []
    nodes = _run([1.] * 15 + [0.] * 15 + [1.] * 20)["candidates"]
    assert [node["source_frame_id"] for node in nodes] == [30]


def test_spike_is_rejected_and_bounded_plateau_noise_is_one_node():
    values = [0.] * 10 + [1.] + [0.] * 9 + [1. + (.004 if index % 2 else 0.) for index in range(20)]
    assert [node["source_frame_id"] for node in _run(values)["candidates"]] == [20]


@pytest.mark.parametrize("error", ["missing_row", "shifted_frame", "shifted_time", "reverse_time", "nan", "dimension"])
def test_invalid_feedback_or_mapping_fails(error):
    states = [[0.] * 14 for _ in range(20)]
    times = [index / 20 for index in range(20)]
    source = times.copy()
    indices = list(range(20))
    if error == "missing_row":
        states.pop()
    elif error == "shifted_frame":
        indices = list(range(1, 21))
    elif error == "shifted_time":
        times = [value + .05 for value in times]
    elif error == "reverse_time":
        times.reverse()
    elif error == "nan":
        states[3][6] = math.nan
    else:
        states[3] = [0.] * 16
    with pytest.raises(ValueError):
        detect_candidates(states, times, indices, source_timestamps=source)


def test_closing_only_does_not_make_candidates():
    assert _run([1.] * 20 + [0.] * 20)["candidates"] == []


def test_only_registered_any_open_policy_and_indices_are_accepted():
    with pytest.raises(ValueError, match="any_open"):
        _run([0.] * 20, node_policy="all_transitions")
    with pytest.raises(ValueError, match="indices"):
        _run([0.] * 20, state_indices={"left": 5, "right": 12})


def test_candidate_timestamp_uses_exact_video_pts_after_float32_state_alignment():
    states = [[0.] * 14 for _ in range(40)]
    for state in states[21:]:
        state[6] = 1.
    source = [index / 20 for index in range(40)]
    measured_times = [round(value + 1e-7, 8) for value in source]
    node, = detect_candidates(states, measured_times, list(range(40)), source_timestamps=source)["candidates"]
    assert node["source_frame_id"] == 21
    assert node["timestamp_s"] == source[21]
    assert node["state_timestamp_s"] == measured_times[21]
