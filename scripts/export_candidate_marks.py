"""Export all state candidates and saved decisions without changing inference.

Run with PYTHONPATH=src and --run runs/<id>. The named export preserves older
videos, prediction files, source frame counts, and presentation timestamps.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
import shutil

from PIL import ImageDraw

from split_cloth_v2 import cli, video_export as video
from split_cloth_v2.io_utils import read_json, sha256_file, write_json

COLORS = {"selected": "#64c396", "skip": "#f08080", "uncertain": "#efc45b",
          "not_queried": "#9cabb9", "ineligible": "#9cabb9"}
HOLD_S = 0.60


def candidate_points(episode: Path) -> list[dict]:
    metadata = read_json(episode / "input.json")
    result = read_json(episode / "result.json")
    axis = read_json(episode / "video_axis.json")["timestamps_s"]
    nodes = read_json(episode / "candidates.json")["candidates"]
    selected = {v["node_id"]: int(stage[-1]) for stage, v in result["slots"].items() if v}
    decisions = {}
    for line in (episode / "trace.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["node_id"] in decisions:
            raise ValueError("duplicate_candidate_trace")
        decisions[event["node_id"]] = event
    points, ids, frames = [], set(), set()
    for node in sorted(nodes, key=lambda n: n["source_frame_id"]):
        frame, node_id = node["source_frame_id"], node["node_id"]
        if node_id in ids or frame in frames or not 0 <= frame < metadata["frame_count"]:
            raise ValueError("invalid_candidate_marker")
        if not node["opening_arms"]:
            raise ValueError("candidate_without_opening_event")
        ids.add(node_id); frames.add(frame)
        event = decisions.get(node_id)
        state = "not_queried"
        if event:
            if event["source_frame_id"] != frame:
                raise ValueError("candidate_trace_frame_mismatch")
            if event["event"] == "skip":
                if event.get("reason") != "no_stage4_frames":
                    raise ValueError("unknown_program_skip")
                state = "ineligible"
            elif event["event"] == "decision":
                state = "selected" if event["decision"] == "match" else event["decision"]
            else:
                raise ValueError("unexpected_candidate_event")
        if state not in COLORS or (state == "selected") != (node_id in selected):
            raise ValueError("candidate_selection_disagrees_with_result")
        if node_id in selected and (event["target_stage"] != selected[node_id]
                                    or result["slots"][f"stage{selected[node_id]}"]["end_frame"] != frame):
            raise ValueError("selected_candidate_stage_mismatch")
        points.append({"node_id": node_id, "frame": frame, "timestamp_s": axis[frame],
                       "opening_arms": node["opening_arms"], "state": state,
                       "target_stage": event.get("target_stage") if event else None,
                       "selected_stage": selected.get(node_id)})
    if set(decisions) - ids or set(selected) - ids:
        raise ValueError("trace_or_result_has_unknown_candidate")
    return points


def focus_point(points: list[dict], frame_id: int, axis: list) -> tuple[dict | None, str]:
    if not points:
        return None, "NONE"
    times = [p["timestamp_s"] for p in points]
    index = bisect_left(times, axis[frame_id])
    neighbors = points[max(0, index - 1):min(len(points), index + 1)]
    point = min(neighbors, key=lambda p: (abs(p["timestamp_s"] - axis[frame_id]), p["frame"]))
    if point["frame"] == frame_id:
        return point, "NOW"
    if abs(point["timestamp_s"] - axis[frame_id]) <= HOLD_S:
        return point, "NEAR"
    if index < len(points):
        return points[index], "NEXT"
    return points[-1], "LAST"


def export_marked(run: Path, export_name: str) -> dict:
    run, record = cli._load_run(run, check_source=True)
    output = cli._export_root(run, export_name)
    protected_paths = [p for entry in record["manifest"]
                       for p in (run / "episodes" / entry["key"]).glob("*") if p.is_file()]
    protected_paths += [p for p in (run / "videos").rglob("*") if p.is_file()]
    protected = {str(p): sha256_file(p) for p in protected_paths}
    original_template, original_render = video._template, video.render_frame
    original_export, original_version = video.export_episode, video.EXPORTER_VERSION
    fonts = {"body": video._font(17), "bold": video._font(17, True),
             "small": video._font(15), "legend": video._font(13)}
    current = {}

    def marked_template(metadata, result, slots, segments, axis):
        canvas = original_template(metadata, result, slots, segments, axis)
        points = current["points"]
        canvas.info["candidate_points"] = points
        draw = ImageDraw.Draw(canvas)
        # A separate row above the stage bar marks every state candidate.
        for point in points:
            x = round(24 + (point["frame"] + .5) * 1232 / metadata["frame_count"])
            color = COLORS[point["state"]]
            draw.line((x, 651, x, 662), fill=color, width=2)
            draw.ellipse((x-3, 651, x+3, 657), fill=color)
        draw.rectangle((0, 693, 1279, 719), fill=video.BACKGROUND)
        x = 24
        for label, state in (("Selected", "selected"), ("Skip", "skip"),
                             ("Uncertain", "uncertain"), ("Not queried / excluded", "not_queried")):
            draw.rectangle((x, 701, x+9, 710), fill=COLORS[state])
            draw.text((x+15, 696), label, font=fonts["legend"], fill=video.MUTED)
            x += round(fonts["legend"].getlength(label)) + 35
        draw.text((x+8, 696), "All state candidates | S4 outcome not evaluated", font=fonts["legend"], fill=video.MUTED)
        return canvas

    def marked_frame(template, picture, frame_id, labels, axis):
        canvas = original_render(template, picture, frame_id, labels, axis)
        points = template.info["candidate_points"]
        point, relation = focus_point(points, frame_id, axis)
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((608, 563, 1263, 647), 7, fill="#1b2631")
        if point is None:
            draw.text((618, 575), "No state opening candidates", font=fonts["bold"], fill=video.MUTED)
            return canvas
        color = COLORS[point["state"]]
        label = f"CANDIDATE {relation}: {point['node_id']} | f{point['frame']} | {point['timestamp_s']:.2f}s"
        draw.text((618, 568), video._short(label, 630, fonts["bold"]), font=fonts["bold"], fill=color)
        target = f"S{point['target_stage']}" if point["target_stage"] else "--"
        outcome = {"selected": f"MATCH - selected as S{point['selected_stage']}",
                   "skip": "SKIP - not selected", "uncertain": "UNCERTAIN - not selected",
                   "not_queried": "NOT QUERIED", "ineligible": "EXCLUDED: last source frame"}[point["state"]]
        draw.text((618, 594), f"Target {target} | {outcome}", font=fonts["body"], fill=color)
        arms = "+".join("L" if arm == "left" else "R" for arm in point["opening_arms"])
        offset = axis[frame_id] - point["timestamp_s"]
        draw.text((618, 621), f"Open event: {arms} | Offset {offset:+.2f}s | Model output, unreviewed",
                  font=fonts["small"], fill=video.MUTED)
        if relation == "NOW":
            # Outside the original source-picture area; no source pixels covered.
            draw.rectangle((21, 85, 586, 650), outline=color, width=3)
        if relation in ("NOW", "NEAR"):
            x = round(24 + (point["frame"] + .5) * 1232 / len(labels))
            draw.ellipse((x-5, 650, x+5, 660), fill=color, outline="white", width=1)
        return canvas

    def marked_episode(episode_dir, video_dir, config):
        episode_dir = Path(episode_dir)
        current["points"] = candidate_points(episode_dir)
        annotation_hashes = {name: sha256_file(episode_dir/name)
                             for name in ("candidates.json", "trace.jsonl", "result.json")}
        exported = original_export(episode_dir, video_dir, config)
        if any(sha256_file(episode_dir/name) != value for name, value in annotation_hashes.items()):
            raise ValueError("candidate_annotations_changed_during_export")
        exported["candidate_markers"] = {
            "count": len(current["points"]), "points": current["points"],
            "state_counts": dict(Counter(p["state"] for p in current["points"])),
            "annotation_sha256": annotation_hashes, "exact_candidate_frame_border": True,
            "nearby_info_window_s": HOLD_S, "nearby_is_labelled_near": True,
            "renderer_sha256": sha256_file(Path(__file__)), "sampling": "state_open_only",
        }
        write_json(Path(exported["video_path"]).with_suffix(".json"), exported)
        print(f"{exported['key']}: marked {len(current['points'])} candidates; verified {exported['frame_count']} frames", flush=True)
        return exported

    try:
        video._template, video.render_frame = marked_template, marked_frame
        video.export_episode = marked_episode
        video.EXPORTER_VERSION = "review_video_candidate_marks_v1"
        summary = video.export_run(run, record["config"], export_dir=output)
    finally:
        video._template, video.render_frame = original_template, original_render
        video.export_episode, video.EXPORTER_VERSION = original_export, original_version
    if any(sha256_file(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError("original_run_artifacts_changed")
    shutil.copyfile(Path(__file__), output / "renderer.py")
    marked_report = cli.report(argparse.Namespace(run=str(run), export_name=export_name))
    # main's report command writes its summary under the run; also retain the
    # named export's own report so its verification is self-contained.
    write_json(output / "reports/summary.json", marked_report)
    for name in ("phase1_report.md", "request_images.json"):
        shutil.copyfile(run / "reports" / name, output / "reports" / name)
    write_json(output / "reports/candidate_marker_verification.json", {
        "all_passed": summary["failed_count"] == 0,
        "candidate_count": sum(v.get("candidate_markers", {}).get("count", 0) for v in summary["videos"]),
        "original_artifacts_unchanged": True, "source_picture_not_overlaid": True,
        "videos": [{"key": v["key"], "candidate_markers": v.get("candidate_markers"),
                    "video_verification": v.get("verification")} for v in summary["videos"]],
    })
    if summary["failed_count"]:
        raise RuntimeError("candidate_marker_export_failed")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--export-name", default="candidate_marks")
    args = parser.parse_args()
    export_marked(args.run, args.export_name)
