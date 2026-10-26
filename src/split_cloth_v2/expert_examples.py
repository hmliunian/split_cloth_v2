"""Validate and freeze stage references without altering their image bytes."""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from .io_utils import read_json, sha256_file, write_json

STAGES = ("stage1", "stage2", "stage3")
_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def inspect_expert_examples(mapping: dict, config_dir: Path) -> dict:
    """Read source assets only while loading a new configuration."""
    if not isinstance(mapping, dict) or set(mapping) != set(STAGES):
        raise ValueError("expert_examples_require_exactly_stage1_stage2_stage3")
    sources = {}
    for stage in STAGES:
        value = mapping[stage]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid_expert_example_path:{stage}")
        source = (Path(config_dir) / value).resolve()
        try:
            data = source.read_bytes()
            with Image.open(BytesIO(data)) as image:
                image.load()
                mime_type = Image.MIME.get(image.format)
                width, height = image.size
            if mime_type not in _SUFFIXES:
                raise ValueError("unsupported_expert_example_image_format")
            source_hash = hashlib.sha256(data).hexdigest()
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid_expert_example_image:{stage}") from exc
        sources[stage] = {"source_path": str(source), "source_sha256": source_hash,
                          "width": width, "height": height, "mime_type": mime_type}
    return sources


def _expected_index(config: dict) -> dict:
    sources = config.get("expert_example_sources")
    if not isinstance(sources, dict) or set(sources) != set(STAGES):
        raise ValueError("invalid_frozen_expert_example_sources")
    stages = {}
    for stage in STAGES:
        source = sources[stage]
        if (not isinstance(source, dict) or set(source) != {"source_path", "source_sha256", "width", "height", "mime_type"}
                or source.get("mime_type") not in _SUFFIXES
                or config.get("expert_examples", {}).get(stage) != source.get("source_path")):
            raise ValueError("invalid_frozen_expert_example_sources")
        suffix = _SUFFIXES[source["mime_type"]]
        stages[stage] = {**source, "role": "EXPERT_EXAMPLE", "stage": stage,
                         "path": f"expert_examples/s{stage[-1]}{suffix}",
                         "sha256": source["source_sha256"]}
    return {"image_processing": {"mode": "original_bytes"}, "stages": stages}


def _frozen_path(run: Path, item: dict) -> Path:
    path = (run / item["path"]).resolve()
    if not path.is_relative_to(run):
        raise ValueError("frozen_expert_example_path_outside_run")
    return path


def freeze_expert_examples(run: str | Path, config: dict) -> dict | None:
    """Copy references once; an existing frozen index must already match."""
    if config.get("spec_version") != "reverse_node_v2":
        return None
    run = Path(run).resolve()
    index_path = run / "expert_examples/index.json"
    if index_path.exists():
        return verify_expert_examples(run, config)
    index = _expected_index(config)
    if not index_path.resolve().is_relative_to(run):
        raise ValueError("frozen_expert_example_path_outside_run")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    for item in index["stages"].values():
        destination = _frozen_path(run, item)
        if destination.exists():
            if sha256_file(destination) != item["sha256"]:
                raise ValueError("frozen_expert_example_hash_mismatch")
            continue
        data = Path(item["source_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["source_sha256"]:
            raise ValueError("expert_example_source_changed_during_prepare")
        with destination.open("xb") as handle:
            handle.write(data)
    write_json(index_path, index)
    return verify_expert_examples(run, config)


def verify_expert_examples(run: str | Path, config: dict) -> dict | None:
    """Verify all frozen references without reading mutable source files."""
    if config.get("spec_version") != "reverse_node_v2":
        return None
    run = Path(run).resolve()
    index_path = run / "expert_examples/index.json"
    if not index_path.resolve().is_relative_to(run):
        raise ValueError("frozen_expert_example_path_outside_run")
    if not index_path.is_file():
        raise ValueError("frozen_expert_example_index_missing")
    index = read_json(index_path)
    if index != _expected_index(config):
        raise ValueError("frozen_expert_example_index_changed")
    for item in index["stages"].values():
        path = _frozen_path(run, item)
        if not path.is_file():
            raise ValueError("frozen_expert_example_missing")
        if sha256_file(path) != item["sha256"]:
            raise ValueError("frozen_expert_example_hash_mismatch")
    return index


def get_expert_example(run: str | Path, config: dict, target_stage: int) -> dict:
    """Return the one target reference with an absolute frozen image path."""
    if type(target_stage) is not int or target_stage not in (1, 2, 3):
        raise ValueError("invalid_target_stage")
    run = Path(run).resolve()
    index = verify_expert_examples(run, config)
    if index is None:
        raise ValueError("expert_examples_require_reverse_node_v2")
    item = index["stages"][f"stage{target_stage}"]
    return {**item, "path": str(_frozen_path(run, item))}
