"""Export existing model candidates as complete, time-preserving review MP4s.

The layout and PyAV encoding approach are adapted from the former review exporter.
This module reads only the new repository's artifacts and never invokes a provider.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import groupby
import hashlib
import json
import math
import os
from pathlib import Path
import re

import av
from PIL import Image, ImageDraw, ImageFont

from .io_utils import read_json, sha256_file, write_json
from .results import stage_labels


EXPORTER_VERSION = "review_video_v1"
TIME_BASE = Fraction(1, 90000)
PTS_TOLERANCE_S = float(TIME_BASE) + 1e-9
STAGES = {
    1: ("First-side fold", "Includes pickup and preparation", "#42b7cf"),
    2: ("Both sides folded", "Both sleeves and sides form a long strip", "#9c8cf5"),
    3: ("Cross fold", "The strip is folded into a shorter bundle", "#efb85f"),
    4: ("Move away / finish", "Stage 4 completion has not been evaluated", "#64c396"),
}
BACKGROUND = "#10161e"
MUTED = "#9cabb9"
UNKNOWN = "#46525e"


def candidate_layout(result: dict, frame_count: int):
    """Return fixed slots and labels; every incomplete partition stays unknown."""
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    slots = [None, None, None]
    invalid = False
    stored = result.get("slots", {})
    if not isinstance(stored, dict):
        stored, invalid = {}, True
    for stage in (1, 2, 3):
        slot = stored.get(f"stage{stage}")
        if slot is None:
            continue
        if not isinstance(slot, dict):
            invalid = True
            continue
        value = slot.get("end_frame")
        if type(value) is not int or not 0 <= value < frame_count - 1:
            invalid = True
            continue
        slots[stage - 1] = value
    invalid |= bool(set(stored) - {"stage1", "stage2", "stage3"})
    known = [value for value in slots if value is not None]
    invalid |= any(right <= left for left, right in zip(known, known[1:]))
    labels = stage_labels(frame_count, stored)
    if invalid:
        labels[:] = -1
    segments, cursor = [], 0
    for stage, frames in groupby(labels.tolist()):
        length = sum(1 for _ in frames)
        segments.append({"stage": stage, "start_frame": cursor, "end_frame": cursor + length - 1})
        cursor += length
    return slots, labels, segments, invalid


def _font(size: int, bold: bool = False):
    """Use English labels with a known Latin font, avoiding missing CJK glyphs."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu-sans-fonts") / name,
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _short(text: str, width: int, font):
    if font.getlength(text) <= width:
        return text
    while text and font.getlength(text + "...") > width:
        text = text[:-1]
    return text + "..."


def _template(metadata: dict, result: dict, slots: list, segments: list, axis: list):
    canvas = Image.new("RGB", (1280, 720), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title, body, small = _font(25, True), _font(19), _font(15)
    header = f"MODEL CANDIDATE REVIEW | {metadata.get('run_id', 'review_video_v1')} | {result.get('spec_version', 'reverse_node_v1')}"
    draw.text((24, 10), _short(header, 1232, small), font=small, fill=MUTED)
    draw.text((24, 34), _short(metadata["key"], 945, title), font=title, fill="#eef3f7")
    status = str(result.get("status", "unknown"))
    badge = {"complete": "CHAIN COMPLETE", "needs_review": "NEEDS REVIEW", "execution_error": "EXECUTION ERROR"}.get(status, status.upper())
    draw.rounded_rectangle((1000, 25, 1256, 65), 9, fill="#26313c")
    color = "#64c396" if status == "complete" else "#efb85f"
    draw.text((1010, 36), badge, font=_font(17), fill=color)
    draw.rectangle((24, 88, 583, 647), fill="#202a34")
    draw.text((616, 88), "CURRENT STAGE CANDIDATE", font=small, fill=MUTED)
    draw.text((616, 222), "FOUR-STAGE PARTITION", font=small, fill=MUTED)
    for stage, (name, _, color) in STAGES.items():
        y = 248 + (stage - 1) * 48
        draw.rounded_rectangle((616, y, 1256, y + 41), 7, fill="#1b2631")
        draw.rectangle((616, y + 4, 622, y + 37), fill=color)
        draw.text((636, y + 8), f"S{stage}  {name}", font=body, fill="#dce5ed")
        segment = next((row for row in segments if row["stage"] == stage), None)
        span = f"f{segment['start_frame']} - f{segment['end_frame']}" if segment else "unresolved"
        draw.text((1070, y + 12), span, font=small, fill=MUTED)
    draw.text((616, 453), "FIXED END-BOUNDARY SLOTS", font=small, fill=MUTED)
    for i, value in enumerate(slots):
        stored = result.get("slots", {})
        item = stored.get(f"stage{i + 1}") if isinstance(stored, dict) else None
        node_id = str(item.get("node_id", "?")) if isinstance(item, dict) else "?"
        detail = "unresolved" if value is None else f"f{value} / {axis[value]:.2f}s / {node_id}"
        draw.text((616, 480 + i * 28), _short(f"S{i + 1}: {detail}", 630, body), font=body, fill=STAGES[i + 1][2])
    reasons = result.get("review_reasons", result.get("reasons", []))
    if isinstance(reasons, list):
        reason = "; ".join(str(value) for value in reasons) or "none recorded"
    else:
        reason = str(reasons)
    if not reason.isascii():
        reason = "See result.json for review details"
    draw.text((616, 574), "Stage 4 terminal outcome: NOT EVALUATED", font=_font(18), fill=MUTED)
    draw.text((616, 603), _short(f"Review: {reason}", 630, small), font=small, fill=MUTED)
    draw.text((616, 630), "Model candidates / not human-verified labels", font=small, fill="#efb85f")
    n = metadata["frame_count"]
    for row in segments:
        left = round(24 + row["start_frame"] * 1232 / n)
        right = round(24 + (row["end_frame"] + 1) * 1232 / n)
        draw.rectangle((left, 666, right, 683), fill=STAGES.get(row["stage"], ("", "", UNKNOWN))[2])
    for value in slots:
        if value is not None:
            x = round(24 + (value + 1) * 1232 / n)
            draw.line((x, 662, x, 686), fill="#eaf0f7", width=2)
    draw.text((24, 695), "Model candidate timeline | Boundary belongs to preceding stage | Zero-based source frames", font=_font(14), fill=MUTED)
    return canvas


def render_frame(template: Image.Image, picture: Image.Image, frame_id: int, labels, axis: list):
    """Render a single full source frame without cropping or painting over it."""
    canvas = template.copy()
    ratio = min(560 / picture.width, 560 / picture.height)
    picture = picture.resize((max(1, round(picture.width * ratio)), max(1, round(picture.height * ratio))), Image.Resampling.LANCZOS)
    canvas.paste(picture, (24 + (560 - picture.width) // 2, 88 + (560 - picture.height) // 2))
    draw = ImageDraw.Draw(canvas)
    stage = int(labels[frame_id])
    name, detail, color = STAGES.get(stage, ("Unknown stage", "No complete valid boundary chain", "#efb85f"))
    label = f"S{stage} / {name}" if stage > 0 else name
    stage_font = _font(29, True)
    draw.text((616, 112), _short(label, 630, stage_font), font=stage_font, fill=color)
    detail_font = _font(19)
    draw.text((616, 151), _short(detail, 630, detail_font), font=detail_font, fill=MUTED)
    draw.text((616, 183), f"Frame {frame_id} / {len(labels) - 1}    |    {axis[frame_id]:06.2f} s", font=_font(17), fill="#edf4fa")
    if stage > 0:
        y = 248 + (stage - 1) * 48
        draw.rounded_rectangle((616, y, 1256, y + 41), 7, outline=color, width=2)
    x = round(24 + (frame_id + .5) * 1232 / len(labels))
    draw.line((x, 660, x, 688), fill="white", width=3)
    return canvas


def verify_video(path: Path, expected_axis: list[float]) -> dict:
    """Actually decode the encoded video and compare every presentation timestamp."""
    actual = []
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError("Expected one video stream and no audio")
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        size = [stream.codec_context.width, stream.codec_context.height]
        pixel_format = stream.codec_context.format.name
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Encoded frame has no PTS")
            actual.append(float(frame.pts * frame.time_base))
    if len(actual) != len(expected_axis):
        raise ValueError(f"Output frame count {len(actual)} differs from {len(expected_axis)}")
    if not actual or any(b <= a for a, b in zip(actual, actual[1:])):
        raise ValueError("Output PTS must be nonempty and strictly increasing")
    errors = [abs(a - b) for a, b in zip(actual, expected_axis)]
    maximum = max(errors)
    if maximum > PTS_TOLERANCE_S:
        raise ValueError(f"Output relative PTS error {maximum:g}s exceeds {PTS_TOLERANCE_S:g}s")
    if codec != "h264" or size != [1280, 720] or pixel_format != "yuv420p":
        raise ValueError(f"Unexpected output encoding: {codec}, {size}, {pixel_format}")
    return {"passed": True, "frame_count": len(actual), "max_pts_error_s": maximum,
            "pts_tolerance_s": PTS_TOLERANCE_S, "codec": codec, "size": size,
            "pixel_format": pixel_format, "audio_streams": 0, "first_pts_s": actual[0],
            "last_pts_s": actual[-1]}


def export_episode(episode_dir: Path, video_dir: Path, config: dict) -> dict:
    """Export one saved result; refuse existing names and preserve partial failures."""
    episode_dir, video_dir = Path(episode_dir), Path(video_dir)
    metadata = read_json(episode_dir / "input.json")
    result_path = episode_dir / "result.json"
    result = read_json(result_path)
    axis = read_json(episode_dir / "video_axis.json")["timestamps_s"]
    n, key = metadata["frame_count"], metadata["key"]
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
        raise ValueError("Unsafe episode key")
    if result.get("frame_count", n) != n or result.get("key", key) != key:
        raise ValueError("Result identity does not match input")
    if (len(axis) != n or not axis or any(not math.isfinite(value) for value in axis)
            or abs(axis[0]) > 1e-9 or any(b <= a for a, b in zip(axis, axis[1:]))):
        raise ValueError("Invalid relative source PTS axis")
    settings = config.get("video_export", {})
    if settings.get("label_source", "model_candidates") != "model_candidates":
        raise ValueError("This export entry point requires model_candidates labels")
    suffix = settings.get("filename_suffix", "candidates")
    if not isinstance(suffix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", suffix):
        raise ValueError("Unsafe export suffix")
    video_path = video_dir / f"{key}__{suffix}.mp4"
    sidecar, temporary = video_path.with_suffix(".json"), video_path.with_suffix(".part.mp4")
    if any(path.exists() for path in (video_path, sidecar, temporary)):
        raise FileExistsError(f"Refusing to overwrite existing export: {video_path}")
    source_path = Path(metadata["source_video_path"])
    source_hash = sha256_file(source_path)
    if metadata.get("source_video_sha256", source_hash) != source_hash:
        raise ValueError("Source video hash differs from prepared input")
    result_hash = sha256_file(result_path)
    slots, labels, segments, invalid = candidate_layout(result, n)
    display_metadata = dict(metadata, run_id=episode_dir.parent.parent.name)
    template = _template(display_metadata, result, slots, segments, axis)
    crf = settings.get("crf", 20)
    if type(crf) is not int or not 0 <= crf <= 51:
        raise ValueError("Video CRF must be an integer in [0, 51]")
    preset = settings.get("preset", "veryfast")
    encoding = {"crf": str(crf), "preset": preset, "tune": "zerolatency"}
    video_dir.mkdir(parents=True, exist_ok=True)
    count, source_start = 0, None
    try:
        with av.open(str(source_path)) as source:
            if len(source.streams.video) != 1:
                raise ValueError("Expected exactly one source video stream")
            if source.streams.audio:
                raise ValueError("Source has audio; preserving audio is not supported")
            source_stream = source.streams.video[0]
            rate = source_stream.average_rate
            source_time_base = str(source_stream.time_base)
            if rate is None or rate <= 0:
                raise ValueError("Source has no valid average frame rate")
            with av.open(str(temporary), "w", format="mp4", options={"movflags": "+faststart"}) as destination:
                stream = destination.add_stream("libx264", rate=rate)
                stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
                stream.time_base = TIME_BASE
                stream.codec_context.time_base = TIME_BASE
                stream.codec_context.thread_count = 2
                stream.options = encoding
                for i, source_frame in enumerate(source.decode(source_stream)):
                    if i >= n:
                        raise ValueError("Source has more frames than prepared input")
                    if source_frame.pts is None:
                        raise ValueError("Source frame has no PTS")
                    source_time = source_frame.pts * source_frame.time_base
                    if source_start is None:
                        source_start = source_time
                    if abs(float(source_time - source_start) - axis[i]) > PTS_TOLERANCE_S:
                        raise ValueError("Source PTS differs from prepared axis")
                    canvas = render_frame(template, source_frame.to_image(), i, labels, axis)
                    frame = av.VideoFrame.from_image(canvas)
                    frame.pts = round((source_time - source_start) / TIME_BASE)
                    frame.time_base = TIME_BASE
                    for packet in stream.encode(frame):
                        destination.mux(packet)
                    count += 1
                for packet in stream.encode():
                    destination.mux(packet)
        if count != n:
            raise ValueError(f"Exported {count} frames, expected {n}")
        verification = verify_video(temporary, axis)
        if sha256_file(result_path) != result_hash or sha256_file(source_path) != source_hash:
            raise ValueError("Source or result changed during export")
        record = {
            "key": key, "video_path": str(video_path), "frame_count": n,
            "source_frame_count": n, "output_frame_count": count, "output_size": [1280, 720],
            "source_video_path": str(source_path), "source_video_sha256": source_hash,
            "result_sha256": result_hash, "label_source": "model_candidates", "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "labels_digest_format": "derived frame labels as signed int8 bytes",
            "exporter_version": EXPORTER_VERSION, "exporter_sha256": sha256_file(Path(__file__)),
            "status": result.get("status", "unknown"), "review_status": result.get("review_status", "unreviewed"),
            "candidate_boundary_slots": slots, "candidate_partition": segments,
            "invalid_boundary_order": invalid, "stage4_terminal_status": "not_evaluated",
            "display_is_model_candidate": True, "training_labels_modified": False,
            "stage_names": {str(stage): values[0] for stage, values in STAGES.items()},
            "frame_convention": "Zero based, inclusive; boundary belongs to preceding stage.",
            "codec": "h264", "pixel_format": "yuv420p", "audio_streams": 0,
            "time_base": str(TIME_BASE), "source_start_pts_s": float(source_start),
            "source_fps": float(rate), "source_time_base": source_time_base,
            "encoding": encoding, "faststart": True, "verification": verification,
            "output_sha256": sha256_file(temporary), "export_status": "success",
        }
        # link() has no-overwrite semantics even if another exporter races this one.
        os.link(temporary, video_path)
        temporary.unlink()
        write_json(sidecar, record)
        return record
    except Exception as error:
        failure = video_path.with_suffix(".failure.json")
        if not failure.exists():
            write_json(failure, {"key": key, "export_status": "failed", "error": f"{type(error).__name__}: {error}",
                                 "partial_video_path": str(temporary), "output_frame_count": count})
        raise


def export_run(run_dir: Path, config: dict) -> dict:
    """Export every frozen episode and write complete success/failure indexes."""
    run_dir = Path(run_dir)
    indexes = [run_dir / "videos" / "exports.json", run_dir / "reports" / "video_verification.json"]
    if any(path.exists() for path in indexes):
        raise FileExistsError("Refusing to overwrite an existing export index; use a new export directory")
    manifest = [json.loads(line) for line in (run_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    records = []
    for entry in manifest:
        key = entry["key"]
        episode_dir = run_dir / "episodes" / key
        try:
            record = export_episode(episode_dir, run_dir / "videos", config)
        except Exception as error:
            record = {"key": key, "export_status": "failed", "error": f"{type(error).__name__}: {error}"}
        records.append(record)
    successes = [row for row in records if row.get("export_status") == "success"]
    summary = {"videos": records, "count": len(records), "successful_count": len(successes),
               "failed_count": len(records) - len(successes),
               "phase1_complete": len(manifest) == 10 and len({row['key'] for row in successes}) == 10}
    write_json(run_dir / "videos" / "exports.json", summary)
    write_json(run_dir / "reports" / "video_verification.json", {
        "videos": [{"key": row["key"], "verification": row.get("verification"),
                    "export_status": row["export_status"], "error": row.get("error")} for row in records],
        "total_verified_frames": sum(row["frame_count"] for row in successes),
        "all_passed": len(successes) == len(manifest) and bool(manifest),
    })
    return summary
