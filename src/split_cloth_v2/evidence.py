"""Exact candidate frames and chronological fixed local context images."""
from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
import math

import av
from PIL import Image

from .io_utils import sha256_file, write_json


def select_context_frames(timestamps, candidate_frame, offsets_s):
    """Clip targets to the source axis, deduplicate and retain the exact node."""
    if not timestamps or any(not math.isfinite(float(t)) for t in timestamps):
        raise ValueError("A finite nonempty PTS axis is required")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("PTS must be strictly increasing")
    if isinstance(candidate_frame, bool) or not isinstance(candidate_frame, int) or not 0 <= candidate_frame < len(timestamps):
        raise ValueError("Candidate must identify an exact original frame")
    offsets = [float(value) for value in offsets_s]
    if not offsets or not all(math.isfinite(value) for value in offsets) or 0. not in offsets:
        raise ValueError("Finite context offsets must include 0")
    selected = {candidate_frame}
    origin = timestamps[candidate_frame]
    for offset in offsets:
        target = min(max(origin + offset, timestamps[0]), timestamps[-1])
        right = bisect_left(timestamps, target)
        choices = [index for index in (right - 1, right) if 0 <= index < len(timestamps)]
        selected.add(min(choices, key=lambda index: (abs(timestamps[index] - target), index)))
    return [{"source_frame_id": index, "timestamp_s": timestamps[index],
             "role": "CANDIDATE" if index == candidate_frame else "BEFORE" if index < candidate_frame else "AFTER",
             "relative_time_s": timestamps[index] - origin}
            for index in sorted(selected)]


def extract_evidence(video_path, timestamps, candidates, episode_dir, config):
    """Decode selected frames once; preserve full source field of view."""
    episode_dir = Path(episode_dir)
    offsets = config.get("context_offsets_s", [-1., -.5, 0., .5, 1.])
    image_config = {"max_edge": 768, "jpeg_quality": 90, **config.get("evidence", {})}
    max_edge = image_config["max_edge"]
    quality = image_config["jpeg_quality"]
    if isinstance(max_edge, bool) or not isinstance(max_edge, int) or max_edge < 1:
        raise ValueError("evidence.max_edge must be a positive integer")
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ValueError("evidence.jpeg_quality must be an integer in [1,100]")
    nodes = {}
    for candidate in candidates:
        frame = candidate["source_frame_id"]
        selected = select_context_frames(timestamps, frame, offsets)
        if not math.isclose(candidate["timestamp_s"], timestamps[frame], rel_tol=0., abs_tol=1e-9):
            raise ValueError("Candidate timestamp disagrees with its original frame")
        if candidate["node_id"] in nodes:
            raise ValueError("Duplicate candidate node_id")
        nodes[candidate["node_id"]] = {"source_frame_id": frame, "images": selected}
    selected_ids = {item["source_frame_id"] for node in nodes.values() for item in node["images"]}
    image_dir = episode_dir / "evidence" / "frames"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_records = {}
    with av.open(str(video_path)) as source:
        for index, frame in enumerate(source.decode(video=0)):
            if index not in selected_ids:
                continue
            image = frame.to_image().convert("RGB")
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            target = image_dir / f"frame_{index:06d}.jpg"
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite evidence image: {target}")
            temporary = target.with_suffix(".part.jpg")
            image.save(temporary, format="JPEG", quality=quality)
            temporary.rename(target)
            image_records[index] = {"path": str(target.relative_to(episode_dir)),
                                    "sha256": sha256_file(target),
                                    "width": image.width, "height": image.height}
    if set(image_records) != selected_ids:
        raise ValueError("Source video ended before all evidence frames were decoded")
    for node in nodes.values():
        for item in node["images"]:
            item.update(image_records[item["source_frame_id"]])
    result = {"context_policy": "fixed_local_v1", "context_offsets_s": offsets,
              "image_config": image_config, "nodes": nodes}
    write_json(episode_dir / "evidence" / "index.json", result)
    return result
