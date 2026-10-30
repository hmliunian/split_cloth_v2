"""Render saved joint predictions without training or changing boundaries."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from functools import lru_cache
import hashlib
import html
import os
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..io_utils import read_json, sha256_file, write_json
from .results import stage_labels

TIME_BASE = Fraction(1, 1_000_000)
VERSION = "joint_review_v1"
COLORS = ["#536076", "#42b7cf", "#9c8cf5", "#efb85f", "#64c396"]
VIEW_NAMES = ["Preparation + first fold", "Second sleeve fold", "Final fold", "Move / place / finish"]


@lru_cache(maxsize=None)
def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _short(text, width=525, size=17):
    text = str(text).encode("ascii", "backslashreplace").decode()
    font = _font(size)
    while text and font.getlength(text) > width:
        text = text[:-4] + "..." if len(text) > 3 else ""
    return text


def _inside(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Prepared data must stay inside the input run")
    return path


def _load_episode(run, meta):
    key = meta["key"]
    if not isinstance(key, str) or Path(key).name != key or key in {"", ".", ".."}:
        raise ValueError("Unsafe episode key")
    result_path = run / "labels" / key / "result.json"
    result = read_json(result_path)
    if result.get("key", key) != key:
        raise ValueError("Result identity differs from manifest")
    array_path = _inside(run, meta["dataset_npz"])
    with np.load(array_path, allow_pickle=False) as arrays:
        axis = arrays["timestamps"].copy()
    n = meta["frame_count"]
    if (type(n) is not int or n < 2 or axis.shape != (n,) or not np.isfinite(axis).all()
            or abs(axis[0]) > 1e-9 or np.any(np.diff(axis) <= 0)):
        raise ValueError("Expected the complete, increasing relative video PTS axis")
    event_frames = [c["source_frame_id"] for c in meta["candidates"]]
    if (any(type(f) is not int or not 0 <= f < n for f in event_frames)
            or any(a >= b for a, b in zip(event_frames, event_frames[1:]))):
        raise ValueError("Candidate frames must retain the original increasing order")
    candidates = {c["source_frame_id"]: c for c in meta["candidates"]}
    frames = result["action_frames"]
    raw = result.get("raw_action_frames", [None] * 4)
    for boundaries in (frames, raw):
        if len(boundaries) != 4:
            raise ValueError("Expected fixed b1-b4 boundary slots")
        known = [f for f in boundaries if f is not None]
        if (any(type(f) is not int or not 0 <= f < n - 1 or f not in candidates for f in known)
                or any(a >= b for a, b in zip(known, known[1:]))):
            raise ValueError("Boundaries must retain ordered nonterminal candidate identities")
    if result.get("display_frames", frames[1:]) != frames[1:]:
        raise ValueError("Display boundaries differ from saved action boundaries")
    for index, frame in enumerate(frames, 1):
        stored = result.get("action_boundaries", {}).get(f"b{index}")
        if stored is not None and (frame is None or stored["frame"] != frame or stored["node_id"] != candidates[frame]["node_id"]):
            raise ValueError("Saved boundary identity differs from candidate")
    labels = {"display_stage_4": stage_labels(n, frames[1:]), "action_stage_5": stage_labels(n, frames)}
    for name, expected in labels.items():
        saved = result_path.parent / f"{name}.npy"
        if saved.exists() and not np.array_equal(np.load(saved, allow_pickle=False), expected):
            raise ValueError(f"Saved {name} labels differ from boundaries")
    return result, axis, labels, {result_path: sha256_file(result_path), array_path: sha256_file(array_path)}


def _track(draw, n, boundaries, y, colors, prefix):
    if any(f is None for f in boundaries):
        draw.rectangle((24, y, 1255, y + 16), fill=COLORS[0])
        return
    for index, (a, b) in enumerate(zip([-1, *boundaries], [*boundaries, n - 1])):
        left, right = round(24 + (a + 1) * 1232 / n), round(24 + (b + 1) * 1232 / n)
        draw.rectangle((left, y, right, y + 16), fill=colors[index])
        if right - left > 28:
            draw.text((left + 3, y), f"{prefix}{index + 1}", font=_font(13), fill="white")


def _template(meta, result, axis):
    canvas = Image.new("RGB", (1280, 720), "#101a2a")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), _short(meta["key"], 1230, 24), font=_font(24), fill="white")
    draw.text((24, 50), "Saved joint predictions | S5 -> S2 search | Original speed | Boundaries remain unconfirmed", font=_font(16), fill="#b9c8df")
    draw.rounded_rectangle((690, 82, 1255, 532), radius=10, fill="#1b2940")
    for i, frame in enumerate(result["action_frames"], 1):
        node = next((c["node_id"] for c in meta["candidates"] if c["source_frame_id"] == frame), "")
        value = "Not determined" if frame is None else f"f{frame} / {axis[frame]:.3f}s / {node}"
        draw.text((710, 205 + (i - 1) * 36), _short(f"b{i}: {value}"), font=_font(17), fill=COLORS[max(1, i - 1)])
    details = [f"Program: {result.get('program_status', 'completed')}",
               f"Review: {result.get('status', 'unknown')}",
               f"Human confirmed: {bool(result.get('human_confirmed', False))}",
               f"Source: {result.get('label_source', 'joint template inference')}",
               f"S5 terminal: {result.get('stage5_terminal_status', 'not_evaluated')}"]
    for i, text in enumerate(details):
        draw.text((710, 357 + i * 25), _short(text), font=_font(15), fill="#b9c8df")
    tracks = [("Accepted V1-V4 (V1 = S1 + S2)", result["action_frames"][1:], 563, COLORS[1:], "V"),
              ("Accepted S1-S5 | S1 free preparation, no flip count", result["action_frames"], 606, COLORS, "S"),
              ("Raw hypothesis WITHOUT rejection / acceptance checks", result.get("raw_action_frames", [None] * 4), 649, COLORS, "S")]
    for title, bounds, y, colors, prefix in tracks:
        draw.text((24, y - 20), title, font=_font(14), fill="#b9c8df")
        _track(draw, len(axis), bounds, y, colors, prefix)
    draw.text((24, 672), "Opening candidates | Gold = accepted boundary | NOW only at the original event frame", font=_font(14), fill="#b9c8df")
    draw.line((24, 702, 1255, 702), fill="#7786a2")
    for candidate in meta["candidates"]:
        x = round(24 + (candidate["source_frame_id"] + 1) * 1232 / len(axis))
        selected = candidate["source_frame_id"] in result["action_frames"]
        draw.line((x, 695, x, 710), fill="#ffd066" if selected else "#7786a2", width=2)
    return canvas


def _render_frame(template, picture, frame_id, axis, labels, candidates):
    canvas = template.copy()
    scale = min(650 / picture.width, 450 / picture.height)
    picture = picture.resize((max(1, round(picture.width * scale)), max(1, round(picture.height * scale))), Image.Resampling.LANCZOS)
    canvas.paste(picture, (24 + (650 - picture.width) // 2, 82 + (450 - picture.height) // 2))
    draw = ImageDraw.Draw(canvas)
    stage = int(labels["display_stage_4"][frame_id])
    title = "V? | Unknown partition" if stage < 0 else f"V{stage} | {VIEW_NAMES[stage - 1]}"
    draw.text((710, 101), _short(title, 525, 22), font=_font(22), fill="white")
    action = int(labels["action_stage_5"][frame_id])
    draw.text((710, 137), f"Accepted action: S{action}" if action > 0 else "Accepted action: unknown", font=_font(18), fill="#d7e4f3")
    draw.text((710, 170), f"Source f{frame_id}/{len(axis) - 1} | {axis[frame_id]:.3f} s", font=_font(18), fill="white")
    event = candidates.get(frame_id)
    detail = (f"NOW {event['node_id']} | {' + '.join(event['opening_arms'])} opens" if event else "No opening event on this frame")
    draw.text((710, 494), _short(detail, 525, 16), font=_font(16), fill="#ffd066" if event else "#b9c8df")
    x = round(24 + (frame_id + .5) * 1232 / len(axis))
    draw.line((x, 560, x, 712), fill="white", width=2)
    return canvas


def verify_video(path, expected_axis):
    """Re-decode every frame; validate output encoding and relative timestamps."""
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError("Expected one output video stream without audio")
        stream = container.streams.video[0]
        codec, size = stream.codec_context.name, [stream.width, stream.height]
        pixel_format = stream.codec_context.pix_fmt
        actual = [float(f.pts * f.time_base) for f in container.decode(stream)]
    if len(actual) != len(expected_axis) or not actual or np.any(np.diff(actual) <= 0):
        raise ValueError("Export must preserve every source frame in presentation order")
    maximum = float(np.max(np.abs(np.asarray(actual) - expected_axis)))
    if maximum > float(TIME_BASE) + 1e-9:
        raise ValueError(f"Output PTS error {maximum:g}s exceeds encoder time base")
    if codec != "h264" or size != [1280, 720] or pixel_format != "yuv420p":
        raise ValueError("Unexpected output codec, dimensions or pixel format")
    return {"passed": True, "frame_count": len(actual), "max_pts_error_s": maximum,
            "pts_tolerance_s": float(TIME_BASE) + 1e-9, "codec": codec, "size": size, "pixel_format": pixel_format}


def _render_one(run, meta, output):
    result, axis, labels, hashes = _load_episode(run, meta)
    source_path = Path(meta["source_video_path"])
    if not source_path.is_absolute():
        source_path = run / source_path
    hashes[source_path] = sha256_file(source_path)
    if meta.get("source_video_sha256", hashes[source_path]) != hashes[source_path]:
        raise ValueError("Source video hash differs from prepared input")
    path = output / "videos" / f"{meta['key']}.mp4"
    temporary = path.with_suffix(".part.mp4")
    if any(p.exists() for p in (path, temporary, path.with_suffix(".json"))):
        raise FileExistsError("Existing video export; use a new export directory")
    template = _template(meta, result, axis)
    candidates = {c["source_frame_id"]: c for c in meta["candidates"]}
    source_axis, source_start = [], None
    encoding = {"crf": "20", "preset": "veryfast", "tune": "zerolatency"}
    with av.open(str(source_path)) as source:
        if len(source.streams.video) != 1:
            raise ValueError("Expected exactly one source video stream")
        if source.streams.audio:
            raise ValueError("Source contains audio; audio-preserving export is not supported")
        source_stream = source.streams.video[0]
        rate = source_stream.average_rate or Fraction(1 / float(np.median(np.diff(axis)))).limit_denominator(1000000)
        source_time_base = source_stream.time_base
        source_tolerance = max(float(source_time_base) / 2, float(TIME_BASE)) + 1e-9
        with av.open(str(temporary), "w", format="mp4", options={"movflags": "+faststart"}) as destination:
            stream = destination.add_stream("libx264", rate=rate)
            stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
            stream.time_base = stream.codec_context.time_base = TIME_BASE
            stream.codec_context.thread_count = 2
            stream.options = encoding
            for i, frame in enumerate(source.decode(source_stream)):
                if i >= len(axis) or frame.pts is None:
                    raise ValueError("Source frame count or PTS differs from prepared input")
                timestamp = frame.pts * frame.time_base
                if source_start is None:
                    source_start = timestamp
                relative = timestamp - source_start
                if abs(float(relative) - axis[i]) > source_tolerance:
                    raise ValueError("Source PTS differs from prepared input")
                source_axis.append(float(relative))
                canvas = _render_frame(template, frame.to_image(), i, axis, labels, candidates)
                rendered = av.VideoFrame.from_image(canvas)
                rendered.pts, rendered.time_base = round(relative / TIME_BASE), TIME_BASE
                for packet in stream.encode(rendered):
                    destination.mux(packet)
            for packet in stream.encode():
                destination.mux(packet)
    if len(source_axis) != len(axis):
        raise ValueError("Source has fewer frames than prepared input")
    verification = verify_video(temporary, np.asarray(source_axis))
    if any(sha256_file(p) != digest for p, digest in hashes.items()):
        raise ValueError("Source, prepared data or saved predictions changed during export")
    snapshots = {0, len(axis) - 1}
    snapshots.update(f + offset for f in result["action_frames"] if f is not None for offset in (0, 1))
    with av.open(str(temporary)) as video:
        for i, frame in enumerate(video.decode(video=0)):
            if i in snapshots:
                frame.to_image().save(output / "checks" / f"{meta['key']}__{i:06d}.jpg")
    record = {"key": meta["key"], "path": str(path.relative_to(output)), "export_status": "success",
              "source_video_path": str(source_path), "source_sha256": hashes[source_path],
              "result_sha256": hashes[run / "labels" / meta["key"] / "result.json"],
              "labels_sha256": {name: hashlib.sha256(values.tobytes()).hexdigest() for name, values in labels.items()},
              "labels_digest_format": "derived frame labels as signed int16 bytes",
              "renderer_version": VERSION, "renderer_sha256": sha256_file(Path(__file__)),
              "action_frames": result["action_frames"], "raw_action_frames": result.get("raw_action_frames"),
              "candidates": meta["candidates"], "status": result.get("status"),
              "label_source": result.get("label_source"), "human_confirmed": result.get("human_confirmed", False),
              "source_frame_count": len(axis), "output_frame_count": len(source_axis),
              "source_start_pts_s": float(source_start), "source_time_base": str(source_time_base),
              "prepared_source_pts_tolerance_s": source_tolerance,
              "max_prepared_source_pts_error_s": float(np.max(np.abs(np.asarray(source_axis) - axis))),
              "source_fps": str(rate), "time_base": str(TIME_BASE), "encoding": encoding,
              "audio_handling": "source checked: no audio", "verification": verification,
              "output_sha256": sha256_file(temporary), "check_frames": sorted(snapshots)}
    os.link(temporary, path)
    temporary.unlink()
    write_json(path.with_suffix(".json"), record)
    return record


def _new_output(run, output, name):
    run = Path(run).resolve()
    output = Path(output).resolve() if output is not None else run / "exports" / name
    if run.is_relative_to(output) or (output.is_relative_to(run) and output.parent != run / "exports"):
        raise ValueError("Exports inside a run must use a new directory under exports/")
    output.mkdir(parents=True, exist_ok=False)
    return run, output


def _gallery(run, output, entries, records):
    rows = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Joint segmentation review</title><style>body{max-width:1200px;margin:24px auto;padding:0 16px;font:16px/1.6 system-ui;background:#101a2a;color:#e4edf8}a{color:#76d8ea}details{background:#1b2940;padding:14px;margin:14px 0;border-radius:8px}video{width:100%}</style><h1>Joint segmentation review</h1><p>Saved program predictions, Codex references and user confirmation are distinct. Export success measures software execution; accuracy has not been evaluated.</p>']
    for meta in entries:
        key = meta["key"]
        result = read_json(run / "labels" / key / "result.json", default={})
        record = records.get(key, {})
        rows.append(f'<details><summary>{html.escape(key)} | {html.escape(str(result.get("status", "unknown")))}</summary><p>b1-b4: {html.escape(str(result.get("action_frames", [None] * 4)))}</p><p>Source: {html.escape(str(result.get("label_source", "unknown")))} | Human confirmed: {bool(result.get("human_confirmed", False))}</p>')
        if record.get("export_status") == "success":
            path = html.escape(record["path"], quote=True)
            rows.append(f'<video controls preload="none" src="{path}"></video><p><a href="{path}">Complete MP4</a></p>')
        elif record:
            rows.append(f'<p>Export failed: {html.escape(record.get("error", "unknown"))}</p>')
        rows.append("</details>")
    (output / "index.html").write_text("\n".join(rows) + "\n</html>\n", encoding="utf-8")


def report_run(run, output=None):
    run, output = _new_output(run, output, "report")
    entries = read_json(run / "manifest.json")
    for meta in entries:
        _load_episode(run, meta)
    _gallery(run, output, entries, {})
    summary = {"episodes": len(entries), "accuracy": None, "report": str(output / "index.html")}
    write_json(output / "summary.json", summary)
    return summary


def render_run(run, workers=2, output=None):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    run, output = _new_output(run, output, "video")
    entries = read_json(run / "manifest.json")
    if not entries or len({m["key"] for m in entries}) != len(entries):
        raise ValueError("Expected a nonempty manifest of distinct episodes")
    for name in ("videos", "checks"):
        (output / name).mkdir()
    manifest_hash = sha256_file(run / "manifest.json")
    records = {}
    def collect(meta, operation):
        try:
            records[meta["key"]] = operation()
        except Exception as error:
            records[meta["key"]] = {"key": meta["key"], "export_status": "failed", "error": f"{type(error).__name__}: {error}"}
    if workers == 1:
        for meta in entries:
            collect(meta, lambda: _render_one(run, meta, output))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(_render_one, run, meta, output): meta for meta in entries}
            for future in as_completed(jobs):
                collect(jobs[future], future.result)
    if sha256_file(run / "manifest.json") != manifest_hash:
        raise ValueError("Input manifest changed during export")
    successes = [r for r in records.values() if r["export_status"] == "success"]
    summary = {"videos": [records[m["key"]] for m in entries], "successful_count": len(successes),
               "failed_count": len(entries) - len(successes), "accuracy": None,
               "frames": sum(r["output_frame_count"] for r in successes), "output": str(output)}
    write_json(output / "video_verification.json", summary)
    _gallery(run, output, entries, records)
    if summary["failed_count"]:
        raise RuntimeError(f"{summary['failed_count']} export failures recorded in {output / 'video_verification.json'}")
    return summary
