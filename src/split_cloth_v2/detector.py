"""Measured-state opening candidates, migrated from the frozen legacy detector.

Only the threshold crossing and confirmed plateau algorithms are retained.
An opening in either arm qualifies; no stage semantics are inferred here.
"""
from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping, Sequence

ARM_STATE_INDICES = {"left": 6, "right": 13}
DEFAULT_CANDIDATE_LEVELS = ((.5, .25), (.75, .45), (.9, .65))
DEFAULT_NODE_LEVELS = ((.25, .1),) + DEFAULT_CANDIDATE_LEVELS + ((.98, .9),)


def _quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite numbers")
    return result

def _validate_axis(timestamps, frame_indices, source_timestamps, tolerance):
    times = [_finite_float(v, "timestamps") for v in timestamps]
    indices = []
    for value in frame_indices:
        number = _finite_float(value, "frame_indices")
        if number < 0 or not number.is_integer():
            raise ValueError("frame_indices must contain nonnegative integers")
        indices.append(int(number))
    if len(times) != len(indices):
        raise ValueError("timestamps and frame_indices must have the same length")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing within one episode")
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("frame_indices must be strictly increasing within one episode")
    alignment = {"alignment_verified": False, "row_count": len(times),
                 "mapping": "explicit_frame_index", "timestamp_units": "seconds"}
    if source_timestamps is None:
        return times, indices, alignment
    source = [_finite_float(v, "source_timestamps") for v in source_timestamps]
    if not source or any(b <= a for a, b in zip(source, source[1:])):
        raise ValueError("source_timestamps must be a nonempty, strictly increasing video PTS axis")
    if indices and indices[-1] >= len(source):
        raise ValueError("parquet frame_index is outside the decoded video PTS axis")
    # A small fraction of a frame, not half a frame: a one-frame shift must fail.
    if tolerance is None:
        intervals = [b - a for a, b in zip(source, source[1:])]
        tolerance = max(1e-4, min(intervals) * .05) if intervals else 1e-4
    tolerance = _finite_float(tolerance, "timestamp_tolerance_s")
    if tolerance < 0:
        raise ValueError("timestamp_tolerance_s must be nonnegative")
    errors = [times[row] - source[index] for row, index in enumerate(indices)]
    mismatches = [{"row": row, "source_frame_id": indices[row],
                   "parquet_timestamp_s": times[row],
                   "video_timestamp_s": source[indices[row]], "error_s": error}
                  for row, error in enumerate(errors) if abs(error) > tolerance]
    if mismatches:
        raise ValueError(f"Parquet/video PTS alignment failed ({len(mismatches)} rows; "
                         f"tolerance={tolerance:g}s): {mismatches[:3]}")
    alignment.update({"alignment_verified": True, "source_frame_count": len(source),
                      "timestamp_tolerance_s": tolerance,
                      "max_abs_timestamp_error_s": max(map(abs, errors), default=0.0),
                      "median_signed_timestamp_error_s": _quantile(errors, .5) if errors else None})
    return times, indices, alignment

def detect_gripper_events(
    states: Sequence,
    timestamps: Sequence,
    frame_indices: Sequence,
    *,
    source_timestamps: Sequence | None = None,
    timestamp_tolerance_s: float | None = None,
    stable_samples: int = 2,
    stable_duration_s: float = .05,
    open_fraction: float = .9,
    rearm_fraction: float = .65,
    quantile_low: float = .05,
    quantile_high: float = .95,
    min_dynamic_range: float = 1e-6,
    min_dynamic_fraction: float = .01,
    opening_directions: Mapping[str, str] | None = None,
):
    """Find every supported opening candidate in a single episode.

    ``frame_indices`` are mandatory original frame IDs, not row numbers.
    ``source_timestamps`` must contain every decoded video PTS in presentation
    order, starting at frame 0. Supplying it verifies each row's explicit ID and
    timestamp, and rejects shifted or out-of-range axes. Without it output is
    explicitly unverified and must not be used for precise frame annotation.

    Each arm uses robust episode quantiles, hysteresis, and consecutive stable
    open feedback. The candidate is backdated to the *first* high-threshold
    sample; the later confirmation frame is kept separately. Initial open
    states, nonfinite/missing rows, and gaps cannot create an opening event.
    The default higher-is-open direction is an explicit configurable assumption
    requiring visual validation, not a fact inferred from endpoint values.
    """
    if isinstance(stable_samples, bool) or int(stable_samples) != stable_samples or stable_samples < 1:
        raise ValueError("stable_samples must be a positive integer")
    stable_samples = int(stable_samples)
    stable_duration_s = _finite_float(stable_duration_s, "stable_duration_s")
    if stable_duration_s < 0:
        raise ValueError("stable_duration_s must be nonnegative")
    if not 0 <= rearm_fraction < open_fraction <= 1:
        raise ValueError("require 0 <= rearm_fraction < open_fraction <= 1")
    if not 0 <= quantile_low < quantile_high <= 1:
        raise ValueError("require 0 <= quantile_low < quantile_high <= 1")
    min_dynamic_range = _finite_float(min_dynamic_range, "min_dynamic_range")
    min_dynamic_fraction = _finite_float(min_dynamic_fraction, "min_dynamic_fraction")
    if min_dynamic_range < 0 or min_dynamic_fraction < 0:
        raise ValueError("dynamic range minima must be nonnegative")
    directions = {"left": "higher", "right": "higher"}
    if opening_directions:
        if set(opening_directions) - set(ARM_STATE_INDICES):
            raise ValueError("opening_directions keys must be left and/or right")
        directions.update(opening_directions)
    if any(direction not in {"higher", "lower"} for direction in directions.values()):
        raise ValueError("opening direction must be higher or lower")
    times, indices, alignment = _validate_axis(
        timestamps, frame_indices, source_timestamps, timestamp_tolerance_s)
    if len(states) != len(times):
        raise ValueError("states, timestamps and frame_indices must have equal lengths")

    diagnostics = []
    if not alignment["alignment_verified"]:
        diagnostics.append({"code": "unverified_video_alignment",
                            "message": "No decoded source PTS supplied; row-to-video mapping is unverified."})
    valid_states = []
    invalid_rows = []
    for row_index, row in enumerate(states):
        try:
            if row is None or len(row) != 14:
                raise ValueError("state must have 14 components")
            values = [_finite_float(v, "state") for v in row]
        except (TypeError, ValueError, OverflowError):
            valid_states.append(None)
            invalid_rows.append(row_index)
        else:
            valid_states.append(values)
    if invalid_rows:
        diagnostics.append({"code": "missing_or_nonfinite_state", "count": len(invalid_rows),
                            "first_rows": invalid_rows[:20],
                            "message": "Invalid 14D feedback rows break event continuity; no action fallback."})
    gap_rows = [row for row in range(1, len(indices)) if indices[row] != indices[row - 1] + 1]
    if gap_rows:
        diagnostics.append({"code": "nonconsecutive_source_frames", "count": len(gap_rows),
                            "first_rows": gap_rows[:20],
                            "message": "Frame gaps break opening and confirmation continuity."})
    calibration = {**alignment, "state_dimensions": 14,
                   "signal_source": "measured_state_only", "raw_units": "unspecified",
                   "quantiles": [quantile_low, quantile_high],
                   "stable_samples": stable_samples, "stable_duration_s": stable_duration_s,
                   "open_fraction": open_fraction, "rearm_fraction": rearm_fraction,
                   "min_dynamic_range_raw": min_dynamic_range,
                   "min_dynamic_fraction": min_dynamic_fraction,
                   "valid_state_rows": len(states) - len(invalid_rows),
                   "invalid_state_rows": len(invalid_rows), "arms": {},
                   "interpretation": "feedback_opening_candidates_require_visual_release_and_stage_confirmation"}
    events = []
    for arm, state_index in ARM_STATE_INDICES.items():
        values = [row[state_index] if row is not None else None for row in valid_states]
        finite_values = [v for v in values if v is not None]
        arm_calibration = {"state_index": state_index, "opening_direction": directions[arm],
                           "direction_is_assumed": True, "finite_count": len(finite_values),
                           "status": "missing_feedback"}
        calibration["arms"][arm] = arm_calibration
        if not finite_values:
            diagnostics.append({"code": "missing_feedback", "arm": arm})
            continue
        low = _quantile(finite_values, quantile_low)
        high = _quantile(finite_values, quantile_high)
        dynamic_range = high - low
        required_range = max(min_dynamic_range, min_dynamic_fraction * max(abs(low), abs(high)))
        arm_calibration.update({"observed_min_raw": min(finite_values),
                                "observed_max_raw": max(finite_values),
                                "quantile_low_raw": low, "quantile_high_raw": high,
                                "dynamic_range_raw": dynamic_range,
                                "required_dynamic_range_raw": required_range})
        if len(finite_values) < max(3, stable_samples) or dynamic_range <= required_range:
            arm_calibration["status"] = "insufficient_dynamic_range"
            diagnostics.append({"code": "insufficient_dynamic_range", "arm": arm,
                                "message": "Cannot calibrate open/closed endpoints from this episode."})
            continue
        higher = directions[arm] == "higher"
        raw_threshold = lambda fraction: low + dynamic_range * (fraction if higher else 1 - fraction)
        arm_calibration.update({"status": "calibrated", "open_threshold_raw": raw_threshold(open_fraction),
                                "rearm_threshold_raw": raw_threshold(rearm_fraction)})
        armed = False
        candidate_rows = []
        for row, value in enumerate(values):
            if value is None or (row and indices[row] != indices[row - 1] + 1):
                armed = False
                candidate_rows = []
            if value is None:
                continue
            score = (value - low) / dynamic_range
            if not higher:
                score = 1 - score
            if score <= rearm_fraction:
                armed = True
                candidate_rows = []
            elif armed and score >= open_fraction:
                candidate_rows.append(row)
                onset = candidate_rows[0]
                if (len(candidate_rows) >= stable_samples and
                        times[row] - times[onset] + 1e-9 >= stable_duration_s):
                    previous = onset - 1
                    evidence = {"state_index": state_index, "raw_onset_value": values[onset],
                                "raw_confirmation_values": [values[i] for i in candidate_rows],
                                "confirmation_source_frame_ids": [indices[i] for i in candidate_rows],
                                "open_threshold_raw": arm_calibration["open_threshold_raw"],
                                "rearm_threshold_raw": arm_calibration["rearm_threshold_raw"],
                                "previous_source_frame_id": indices[previous] if previous >= 0 else None,
                                "previous_raw_value": values[previous] if previous >= 0 else None,
                                "stability_duration_s": times[row] - times[onset]}
                    events.append({"event_id": f"gripper_{arm}_{indices[onset]:06d}",
                                   "arm": arm, "source_frame_id": indices[onset],
                                   "timestamp_s": times[onset],
                                   "confirmed_at_source_frame_id": indices[row],
                                   "confirmed_at_timestamp_s": times[row],
                                   "kind": "feedback_open_threshold_candidate", "evidence": evidence})
                    armed = False
                    candidate_rows = []
            else:
                candidate_rows = []
    events.sort(key=lambda event: (event["source_frame_id"], event["arm"]))
    return {"alignment_verified": alignment["alignment_verified"], "calibration": calibration,
            "events": events, "diagnostics": diagnostics}

def detect_gripper_event_candidates(
    states, timestamps, frame_indices, *, source_timestamps=None,
    levels=DEFAULT_CANDIDATE_LEVELS, grouping_window_s=.25, **base_options,
):
    """Increase recall with several raw feedback thresholds, never physical truth.

    Each ``(open_fraction, rearm_fraction)`` level runs the existing strict
    feedback detector. Same-arm crossings within ``grouping_window_s`` of the
    group's *first* crossing form one proposal, anchored at the highest level
    actually observed in that group. Grouping is not transitive. A repeated
    crossing of the same level begins an independent group, preserving rapid
    regrip cycles rather than silently swallowing their separate events.

    Lower plateaus can be real openings even when a later opening sets a much
    higher episode quantile. Therefore all independent groups are retained,
    including groups with only a low threshold. Vision must confirm whether
    the participating gripper released and whether the operation completed.
    Base ``open_fraction``/``rearm_fraction`` options are superseded by levels;
    every other calibration, stability and axis option is passed unchanged.
    """
    window = _finite_float(grouping_window_s, "grouping_window_s")
    if window < 0:
        raise ValueError("grouping_window_s must be nonnegative")
    parsed_levels = []
    try:
        for level in levels:
            if len(level) != 2:
                raise ValueError("Each candidate level must be (open_fraction, rearm_fraction)")
            opening = _finite_float(level[0], "candidate open fraction")
            rearm = _finite_float(level[1], "candidate rearm fraction")
            if not 0 <= rearm < opening <= 1:
                raise ValueError("Each candidate level requires 0 <= rearm < open <= 1")
            parsed_levels.append((opening, rearm))
    except TypeError as exc:
        raise ValueError("levels must be a nonempty sequence of threshold pairs") from exc
    if not parsed_levels or len(set(parsed_levels)) != len(parsed_levels):
        raise ValueError("Candidate levels must be nonempty and unique")
    if len({opening for opening, _ in parsed_levels}) != len(parsed_levels):
        raise ValueError("Candidate open fractions must be unique")
    parsed_levels.sort()
    options = dict(base_options)
    options.pop("open_fraction", None)
    options.pop("rearm_fraction", None)
    results = []
    components = []
    for level_index, (opening, rearm) in enumerate(parsed_levels):
        result = detect_gripper_events(
            states, timestamps, frame_indices, source_timestamps=source_timestamps,
            open_fraction=opening, rearm_fraction=rearm, **options)
        results.append(result)
        for event in result["events"]:
            components.append({**event, "level_index": level_index,
                               "open_fraction": opening, "rearm_fraction": rearm})
    primary = results[-1]
    calibration = deepcopy(primary["calibration"])
    calibration["ensemble"] = {
        "method": "multiple_raw_feedback_thresholds",
        "levels": [{"open_fraction": opening, "rearm_fraction": rearm,
                    "event_count": len(result["events"])}
                   for (opening, rearm), result in zip(parsed_levels, results)],
        "primary_open_fraction": parsed_levels[-1][0],
        "grouping_window_s": window,
        "grouping_reference": "first_crossing_not_transitive",
        "repeated_same_level_starts_independent_group": True,
        "anchor": "highest_observed_open_fraction_within_group",
        "level_calibrations": [{"open_fraction": opening,
                                 "arms": result["calibration"]["arms"]}
                                for (opening, _), result in zip(parsed_levels, results)],
    }
    calibration["interpretation"] = "multithreshold_candidates_not_physical_release_or_stage_completion"
    groups = []
    for arm in ARM_STATE_INDICES:
        arm_components = sorted((event for event in components if event["arm"] == arm),
                                key=lambda event: (event["timestamp_s"], event["level_index"]))
        group = []
        for event in arm_components:
            if group and (event["timestamp_s"] - group[0]["timestamp_s"] > window + 1e-9
                          or event["level_index"] in {item["level_index"] for item in group}):
                groups.append(group)
                group = []
            group.append(event)
        if group:
            groups.append(group)
    events = []
    for group in groups:
        anchor = max(group, key=lambda event: event["level_index"])
        first = group[0]
        event = {key: deepcopy(value) for key, value in anchor.items()
                 if key not in {"level_index", "open_fraction", "rearm_fraction"}}
        event.update({
            "event_id": f"gripper_{first['arm']}_{first['source_frame_id']:06d}",
            "kind": "multithreshold_feedback_opening_candidate",
            "anchor_level": anchor["open_fraction"],
            "levels_detected": sorted(item["open_fraction"] for item in group),
            "candidate_group_start_source_frame_id": first["source_frame_id"],
            "candidate_group_start_timestamp_s": first["timestamp_s"],
            "level_evidence": [deepcopy(item) for item in group],
        })
        events.append(event)
    events.sort(key=lambda event: (event["source_frame_id"], event["arm"], event["event_id"]))
    diagnostics = deepcopy(primary["diagnostics"])
    diagnostics.append({"code": "multithreshold_feedback_is_not_physical_truth",
                        "message": "All raw opening levels are candidate evidence; neither 90% nor any lower threshold proves complete release or stage completion."})
    return {"alignment_verified": primary["alignment_verified"], "calibration": calibration,
            "events": events, "diagnostics": diagnostics}

def _positive_count(value, name):
    if isinstance(value, bool) or int(value) != value or value < 2:
        raise ValueError(f"{name} must be an integer >= 2")
    return int(value)

def _initial_status(values, times, indices, calibration, samples, duration, tolerance):
    """Only stable, calibrated episode endpoints establish an initial state."""
    if calibration["status"] != "calibrated" or len(values) < samples:
        return None
    stop = samples - 1
    while stop < len(values) and times[stop] - times[0] + 1e-9 < duration:
        stop += 1
    if stop >= len(values):
        return None
    window = values[:stop + 1]
    if max(window) - min(window) > tolerance + 1e-12:
        return None
    raw = sum(window) / len(window)
    score = (raw - calibration["quantile_low_raw"]) / calibration["dynamic_range_raw"]
    if calibration["opening_direction"] == "lower":
        score = 1 - score
    kind = "open" if score >= .9 else "close" if score <= .1 else None
    return {"kind": kind, "raw_plateau_value": raw,
            "confirmed_at_source_frame_id": indices[stop]} if kind else None

def _settle_candidate(candidate, values, times, indices, *, direction, tolerance,
                      reversal_tolerance, minimum_excursion,
                      samples, duration, search_window, cutoff):
    """Confirm a plateau while allowing bounded actuator overshoot/settling.

    Plateau noise and motion reversal use separate tolerances. A small retreat
    after an overshoot is normal settling, but the final plateau must retain a
    directional excursion from the earliest crossing's preceding measurement.
    An opposite crossing supplies an additional hard limit on the search.
    """
    index_to_row = {frame: row for row, frame in enumerate(indices)}
    onset = index_to_row[candidate["source_frame_id"]]
    first_crossing = index_to_row[candidate["candidate_group_start_source_frame_id"]]
    reference_row = max(0, first_crossing - 1)
    reference_value = values[reference_row]
    extreme = values[onset]
    limit_time = times[onset] + search_window
    for start in range(onset, len(values)):
        if times[start] > limit_time + 1e-9 or (cutoff is not None and indices[start] >= cutoff):
            break
        extreme = max(extreme, values[start]) if direction == "higher" else min(extreme, values[start])
        retreat = extreme - values[start] if direction == "higher" else values[start] - extreme
        if retreat > reversal_tolerance + 1e-12:
            break
        stop = start + samples - 1
        while stop < len(values) and times[stop] - times[start] + 1e-9 < duration:
            stop += 1
        if stop >= len(values) or times[stop] > limit_time + 1e-9:
            break
        if cutoff is not None and indices[stop] >= cutoff:
            break
        window = values[start:stop + 1]
        if max(window) - min(window) > tolerance + 1e-12:
            continue
        retained_excursion = (min(window) - reference_value if direction == "higher"
                              else reference_value - max(window))
        if retained_excursion + 1e-12 < minimum_excursion:
            continue
        return {"source_frame_id": indices[start], "timestamp_s": times[start],
                "confirmed_at_source_frame_id": indices[stop],
                "confirmed_at_timestamp_s": times[stop],
                "raw_plateau_value": sum(window) / len(window),
                "raw_plateau_values": window,
                "plateau_source_frame_ids": indices[start:stop + 1],
                "excursion_reference_source_frame_id": indices[reference_row],
                "excursion_reference_raw_value": reference_value,
                "retained_directional_excursion_raw": retained_excursion,
                "retreat_from_extreme_raw": retreat,
                "stability_duration_s": times[stop] - times[start]}
    return None


def detect_candidates(states, timestamps, frame_indices, *, source_timestamps, config=None):
    """Return fixed original-frame nodes for independently confirmed openings.

    Closing crossings only bound an opening's settling window. The other arm's
    state never gates an opening candidate. Full measured/video alignment is
    mandatory; caller-provided rows are never interpolated or renumbered.
    """
    options = dict(config or {})
    if options.pop("node_policy", "any_open") != "any_open":
        raise ValueError("Only node_policy=any_open is supported")
    indices_config = options.pop("state_indices", ARM_STATE_INDICES)
    if indices_config != ARM_STATE_INDICES:
        raise ValueError("Only verified 14D EEF indices left=6, right=13 are supported")
    if source_timestamps is None:
        raise ValueError("Decoded source_timestamps are required")
    times, indices, alignment = _validate_axis(
        timestamps, frame_indices, source_timestamps, options.get("timestamp_tolerance_s"))
    if not times or len(states) != len(times):
        raise ValueError("states and timestamps must have the same nonzero length")
    if indices != list(range(len(source_timestamps))):
        raise ValueError("Measured state requires every consecutive source frame starting at 0")
    measured = []
    for row, state in enumerate(states):
        try:
            if state is None or len(state) != 14:
                raise ValueError("expected 14 components")
            measured.append([_finite_float(value, "state") for value in state])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Expected finite 14D measured state at row {row}") from exc
    samples = _positive_count(options.pop("settled_samples", 3), "settled_samples")
    duration = _finite_float(options.pop("settled_duration_s", .10), "settled_duration_s")
    tolerance_fraction = _finite_float(options.pop("settled_tolerance_fraction", .02), "settled_tolerance_fraction")
    reversal_fraction = _finite_float(options.pop("settled_reversal_fraction", .10), "settled_reversal_fraction")
    excursion_fraction = _finite_float(options.pop("settled_min_excursion_fraction", .10), "settled_min_excursion_fraction")
    search_window = _finite_float(options.pop("settled_search_window_s", 2.), "settled_search_window_s")
    if duration < 0 or not 0 <= tolerance_fraction < .25 or search_window <= 0 or search_window < duration:
        raise ValueError("Invalid settled duration, tolerance or search window")
    if not 0 <= reversal_fraction <= 1 or not 0 < excursion_fraction <= 1:
        raise ValueError("Invalid reversal or minimum excursion fraction")
    levels = options.pop("candidate_levels", DEFAULT_NODE_LEVELS)
    directions = {"left": "higher", "right": "higher"}
    directions.update(options.pop("opening_directions", {}))
    opening = detect_gripper_event_candidates(
        measured, times, indices, source_timestamps=source_timestamps,
        opening_directions=directions, levels=levels, **options)
    reverse = {arm: "lower" if direction == "higher" else "higher" for arm, direction in directions.items()}
    closing = detect_gripper_event_candidates(
        measured, times, indices, source_timestamps=source_timestamps,
        opening_directions=reverse, levels=levels, **options)
    calibration = deepcopy(opening["calibration"])
    calibration.update({"method": "multithreshold_crossings_then_local_settled_plateaus",
                        "node_policy": "any_open", "physical_full_open_calibrated": False,
                        "boundary_frame": "first_frame_of_confirmed_plateau_not_confirmation_frame",
                        "same_frame_arm_merge_only": True,
                        "settled_samples": samples, "settled_duration_s": duration,
                        "settled_tolerance_fraction": tolerance_fraction,
                        "settled_reversal_fraction": reversal_fraction,
                        "settled_min_excursion_fraction": excursion_fraction,
                        "settled_search_window_s": search_window})
    crossings = [{**deepcopy(item), "transition_kind": kind}
                 for kind, result in (("open", opening), ("close", closing))
                 for item in result["events"]]
    diagnostics = deepcopy(opening["diagnostics"])
    openings = []
    for arm, state_index in ARM_STATE_INDICES.items():
        values = [state[state_index] for state in measured]
        dynamic_range = calibration["arms"][arm].get("dynamic_range_raw", 0)
        tolerance = dynamic_range * tolerance_fraction
        arm_crossings = sorted((item for item in crossings if item["arm"] == arm),
                               key=lambda item: (item["source_frame_id"], item["transition_kind"]))
        for candidate in arm_crossings:
            if candidate["transition_kind"] != "open":
                continue
            opposites = [item["candidate_group_start_source_frame_id"] for item in arm_crossings
                         if item["transition_kind"] == "close"
                         and item["candidate_group_start_source_frame_id"] > candidate["source_frame_id"]]
            plateau = _settle_candidate(
                candidate, values, times, indices, direction=directions[arm], tolerance=tolerance,
                reversal_tolerance=max(dynamic_range * reversal_fraction, 3 * tolerance),
                minimum_excursion=dynamic_range * excursion_fraction,
                samples=samples, duration=duration, search_window=search_window,
                cutoff=min(opposites, default=None))
            if plateau is None:
                diagnostics.append({"code": "crossing_without_settled_plateau", "arm": arm,
                                    "crossing_source_frame_id": candidate["source_frame_id"]})
                continue
            duplicate = next((item for item in openings
                              if item["arm"] == arm and item["source_frame_id"] == plateau["source_frame_id"]), None)
            if duplicate:
                duplicate["crossing_candidates"].append(candidate)
            else:
                openings.append({"arm": arm, **plateau, "state_index": state_index,
                                 "opening_direction": directions[arm],
                                 "plateau_tolerance_raw": tolerance,
                                 "crossing_candidates": [candidate]})
    grouped = {}
    for item in openings:
        grouped.setdefault(item["source_frame_id"], []).append(item)
    nodes = [{"node_id": f"n{number:04d}", "source_frame_id": frame,
              "timestamp_s": float(source_timestamps[frame]), "state_timestamp_s": times[frame],
              "opening_arms": sorted(item["arm"] for item in changes),
              "changes": changes, "alignment_verified": True,
              "physical_full_open_calibrated": False}
             for number, (frame, changes) in enumerate(sorted(grouped.items()))]
    return {"candidates": nodes, "calibration": calibration, "diagnostics": diagnostics,
            "alignment_verified": True, "alignment": alignment}
