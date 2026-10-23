"""Registered fold-cloth inputs and full original-frame/video PTS validation."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import av
import pyarrow.parquet as pq

from .detector import _finite_float, _validate_axis, detect_candidates
from .evidence import extract_evidence
from .io_utils import fingerprint, read_json, sha256_file, write_json

REGISTERED_DATASETS = frozenset({"fold_cloth_v3", "fold_cloth_v4"})


def validate_entry(entry):
    """Require explicit registered task provenance rather than infer a task."""
    if not isinstance(entry, dict):
        raise ValueError("Manifest row must be a JSON object")
    if entry.get("dataset_id") not in REGISTERED_DATASETS or entry.get("task_family") != "fold_cloth":
        raise ValueError("Input must come from a registered fold_cloth dataset")
    episode = entry.get("episode_id", "")
    if not re.fullmatch(r"episode_[0-9]{6}", episode):
        raise ValueError("Invalid episode_id")
    if entry.get("key") != f"{entry['dataset_id']}__{episode}":
        raise ValueError("Manifest key disagrees with dataset/episode")
    index = entry.get("episode_index")
    if isinstance(index, bool) or not isinstance(index, int) or index != int(episode[8:]):
        raise ValueError("episode_index disagrees with episode_id")
    count = entry.get("frame_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("frame_count must be an integer >= 2")
    if _finite_float(entry.get("source_fps"), "source_fps") <= 0:
        raise ValueError("source_fps must be positive")
    if entry.get("camera_key") != "base_0_rgb":
        raise ValueError("Only the verified base_0_rgb camera is supported")
    dataset_root = Path(entry["dataset_root"])
    if not dataset_root.is_absolute() or dataset_root.name != entry["dataset_id"]:
        raise ValueError("dataset_root must be the explicit registered dataset directory")
    chunk = f"chunk-{index // 1000:03d}"
    expected = {"source_video_path": dataset_root / "videos" / chunk / "base_0_rgb" / f"{episode}.mp4",
                "parquet_path": dataset_root / "data" / chunk / f"{episode}.parquet"}
    for field, expected_path in expected.items():
        path = Path(entry[field])
        if not path.is_absolute() or path.resolve() != expected_path.resolve():
            raise ValueError(f"{field} disagrees with registered source layout")
        if not path.resolve().is_relative_to(dataset_root.resolve()):
            raise ValueError(f"{field} escapes dataset_root")
    for field in ("source_video_sha256", "source_parquet_sha256"):
        if not isinstance(entry.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", entry[field]):
            raise ValueError(f"Missing or invalid {field}")
    if entry.get("state_column") not in {None, "state", "observation.state"}:
        raise ValueError("Only measured state columns are allowed")
    return entry


def load_manifest(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            validate_entry(entry)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid manifest row {line_number}: {exc}") from exc
        rows.append(entry)
    if not rows or len({entry["key"] for entry in rows}) != len(rows):
        raise ValueError("Manifest must contain nonempty, unique episode keys")
    return rows


def read_state(path, *, state_column=None, timestamp_column="timestamp", frame_index_column="frame_index"):
    names = pq.read_schema(path).names
    if state_column is None:
        available = [name for name in ("state", "observation.state") if name in names]
        if len(available) != 1:
            raise ValueError("Expected one measured-state column; configure state_column explicitly")
        state_column = available[0]
    if state_column not in {"state", "observation.state"}:
        raise ValueError("Only measured state may be read; action is not feedback")
    columns = [state_column, timestamp_column, frame_index_column]
    if any(name not in names for name in columns):
        raise ValueError(f"Missing required measured-state/timing columns: {columns}")
    table = pq.read_table(path, columns=columns).to_pydict()
    times, indices, _ = _validate_axis(table[timestamp_column], table[frame_index_column], None, None)
    if not times or indices != list(range(len(times))):
        raise ValueError("Measured frame_index must cover consecutive original frames starting at 0")
    states = []
    for row, state in enumerate(table[state_column]):
        if state is None or len(state) != 14:
            raise ValueError(f"Expected 14D measured state at row {row}")
        states.append([_finite_float(value, "state") for value in state])
    return {"states": states, "timestamps": times, "frame_indices": indices, "state_column": state_column}


def decode_video_axis(path):
    """Decode all frames in presentation order and retain their actual PTS."""
    raw_times, raw_pts = [], []
    with av.open(str(path)) as source:
        if len(source.streams.video) != 1:
            raise ValueError("Expected exactly one source video stream")
        stream = source.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate else None
        time_base = str(stream.time_base)
        width, height = stream.width, stream.height
        audio_streams = len(source.streams.audio)
        for frame in source.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Every source frame must have an explicit PTS")
            raw_times.append(float(frame.pts * frame.time_base))
            raw_pts.append(frame.pts)
    if not raw_times or any(not math.isfinite(value) for value in raw_times):
        raise ValueError("Video has no finite decoded PTS axis")
    if any(b <= a for a, b in zip(raw_times, raw_times[1:])):
        raise ValueError("Decoded source PTS must be strictly increasing")
    return {"timestamps_s": [value - raw_times[0] for value in raw_times],
            "frame_count": len(raw_times), "source_start_time_s": raw_times[0],
            "source_start_pts": raw_pts[0], "source_pts": raw_pts,
            "source_time_base": time_base, "source_fps": source_fps,
            "width": width, "height": height, "audio_streams": audio_streams}


def prepare_episode(entry, episode_dir, config):
    """Prepare a new episode atomically enough for explicit safe resume.

    A completed preparation is reusable only when source, configuration and
    saved evidence hashes still match. Other existing directories are retained
    and rejected; callers can select a new run instead of overwriting history.
    """
    validate_entry(entry)
    episode_dir = Path(episode_dir)
    destination = episode_dir.resolve()
    dataset_root = Path(entry["dataset_root"]).resolve()
    if destination.is_relative_to(dataset_root) or dataset_root.is_relative_to(destination):
        raise ValueError("Preparation output must be separate from original dataset")
    implementation = {name: sha256_file(Path(__file__).with_name(name))
                      for name in ("data.py", "detector.py", "evidence.py")}
    preparation_fingerprint = fingerprint({"entry": entry, "config": config,
                                           "implementation_sha256": implementation})
    source_hashes = {"source_video_sha256": sha256_file(Path(entry["source_video_path"])),
                     "source_parquet_sha256": sha256_file(Path(entry["parquet_path"]))}
    if any(source_hashes[field] != entry[field] for field in source_hashes):
        raise ValueError("Source content does not match frozen manifest SHA-256")
    marker = episode_dir / "prepare.json"
    if episode_dir.exists() and any(episode_dir.iterdir()):
        if not marker.is_file():
            raise FileExistsError("Refusing to overwrite an existing unrecognized episode directory")
        previous = read_json(marker)
        if previous.get("fingerprint") != preparation_fingerprint or previous.get("status") != "complete":
            raise FileExistsError("Existing preparation differs or is incomplete; use a new run")
        for relative, digest in previous["artifact_sha256"].items():
            if sha256_file(episode_dir / relative) != digest:
                raise ValueError(f"Prepared artifact changed: {relative}")
        return read_json(episode_dir / "input.json")
    axis = decode_video_axis(entry["source_video_path"])
    if axis["frame_count"] != entry["frame_count"]:
        raise ValueError("Decoded video frame count differs from frozen manifest")
    if axis["source_fps"] is None or not math.isclose(axis["source_fps"], entry["source_fps"], rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Decoded source FPS differs from frozen manifest")
    measured = read_state(entry["parquet_path"], state_column=entry.get("state_column"),
                          timestamp_column=entry.get("timestamp_column", "timestamp"),
                          frame_index_column=entry.get("frame_index_column", "frame_index"))
    candidates = detect_candidates(measured["states"], measured["timestamps"], measured["frame_indices"],
                                   source_timestamps=axis["timestamps_s"], config=config.get("detector", {}))
    episode_dir.mkdir(parents=True, exist_ok=True)
    write_json(marker, {"status": "preparing", "fingerprint": preparation_fingerprint})
    info = {**entry, **source_hashes, "state_column": measured["state_column"],
            "state_dimension": 14, "state_rows": len(measured["states"]),
            "alignment": candidates["alignment"], "alignment_verified": True,
            "candidate_count": len(candidates["candidates"]),
            "max_logical_requests": sum(item["source_frame_id"] < axis["frame_count"] - 1 for item in candidates["candidates"]),
            "preparation_fingerprint": preparation_fingerprint}
    write_json(episode_dir / "input.json", info)
    write_json(episode_dir / "video_axis.json", axis)
    write_json(episode_dir / "candidates.json", candidates)
    extract_evidence(entry["source_video_path"], axis["timestamps_s"], candidates["candidates"], episode_dir, config)
    artifacts = [path for path in episode_dir.rglob("*") if path.is_file() and path != marker]
    write_json(marker, {"status": "complete", "fingerprint": preparation_fingerprint,
                        "artifact_sha256": {str(path.relative_to(episode_dir)): sha256_file(path) for path in sorted(artifacts)}})
    return info
