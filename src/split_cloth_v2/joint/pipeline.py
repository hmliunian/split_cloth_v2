"""Freeze inputs and dispatch the shared fitting and inference implementation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import time

import numpy as np

from ..io_utils import fingerprint, read_json, sha256_file, write_json
from . import ALGORITHM
from .features import Episode
from .results import make_result, stage_labels
from .search import predict_reverse
from .templates import fit_templates, load_templates

def _new_run(output, sources):
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output)
                              for p in sources):
        raise ValueError("Output must be a new directory outside read-only inputs")
    return output


def _inputs(input_run, config):
    root = Path(input_run).resolve()
    manifest = root / "manifest.json"
    entries = read_json(manifest)
    if not isinstance(entries, list) or not entries:
        raise ValueError("Expected a nonempty manifest")
    episodes, hashes, seen = [], {str(manifest): sha256_file(manifest)}, set()
    if any("detector_sha256" not in meta for meta in entries):
        detector_path = root / "detector_config.json"
        detector = read_json(detector_path)
        # Frozen six-axis runs predate explicit layout keys in detector configs.
        detector.setdefault("state_dimensions", config["layout"]["state_dimensions"])
        detector.setdefault("state_indices", config["layout"]["gripper_indices"])
        if detector != config["detector"]:
            raise ValueError("Frozen detector configuration differs from requested configuration")
        hashes[str(detector_path)] = sha256_file(detector_path)
    for meta in entries:
        key = meta["key"]
        if key in seen:
            raise ValueError("Manifest episode keys must be distinct")
        seen.add(key)
        relative = Path(meta["dataset_npz"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("dataset_npz must stay inside the input run")
        with np.load(path, allow_pickle=False) as arrays:
            episode = Episode(meta, arrays, config)
        hashes[str(path)] = sha256_file(path)
        episodes.append((episode, path))
    return root, episodes, hashes


def _freeze(output, config, hashes, purpose):
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "input_sha256.json", hashes)
    package = Path(__file__).resolve().parent.parent
    sources = [p for p in package.rglob("*") if p.suffix in {".py", ".cpp"}]
    write_json(output / "protocol.json", {
        "algorithm": ALGORITHM, "purpose": purpose, "config_sha256": fingerprint(config),
        "source_sha256": {str(p.relative_to(package)): sha256_file(p) for p in sources},
        "search_order": [5, 4, 3, 2], "prefix": "free_unscored_preparation",
        "accuracy": None, "human_confirmed": False,
    })


def _verify_inputs(hashes):
    for name, digest in hashes.items():
        if sha256_file(name) != digest:
            raise ValueError(f"Input changed during execution: {name}")


def fit_run(config, input_run, references_path, output_run):
    root = Path(input_run).resolve()
    references_path = Path(references_path).resolve()
    output = _new_run(output_run, [root, references_path, Path(config["dataset_root"]).resolve()])
    _, inputs, hashes = _inputs(root, config)
    hashes[str(references_path)] = sha256_file(references_path)
    references = read_json(references_path)
    bundle = fit_templates([e for e, _ in inputs], references, config)
    bundle["input_sha256"] = hashes
    bundle["references_sha256"] = hashes[str(references_path)]
    _freeze(output, config, hashes, "demonstration_template_fit")
    shutil.copy2(references_path, output / "references.json")
    write_json(output / "templates.json", bundle)
    _verify_inputs(hashes)
    summary = {"episodes": len(inputs), "templates": str(output / "templates.json"),
               "template_sha256": sha256_file(output / "templates.json"), "accuracy": None}
    write_json(output / "summary.json", summary)
    return summary


def infer_run(config, input_run, output_run, template_path=None, *, training_replay=False):
    root = Path(input_run).resolve()
    template_path = template_path or config["template"]["path"]
    if not template_path:
        raise ValueError("A frozen template bundle is required; run fit first")
    template_path = Path(template_path).resolve()
    output = _new_run(output_run, [root, template_path, Path(config["dataset_root"]).resolve()])
    _, models = load_templates(template_path, config)
    _, inputs, hashes = _inputs(root, config)
    training_keys = set(models[0]["training_keys"])
    overlap = sorted({e.key for e, _ in inputs} & training_keys)
    if overlap and not training_replay:
        raise ValueError("Inference overlaps demonstrations; use --training-replay explicitly")
    hashes[str(template_path)] = sha256_file(template_path)
    purpose = "demonstration_replay" if training_replay else "inference"
    _freeze(output, config, hashes, purpose)
    shutil.copy2(template_path, output / "templates.json")
    (output / "data").mkdir()
    (output / "labels").mkdir()
    manifest, predictions = [], {}
    start = time.monotonic()
    for rank, (episode, path) in enumerate(inputs, 1):
        meta = {**episode.meta, "sample_rank": rank, "sample_count": len(inputs),
                "dataset_npz": f"data/{episode.key}.npz"}
        if meta.get("source_video_path"):
            meta["source_video_path"] = str((root / meta["source_video_path"]).resolve())
        shutil.copy2(path, output / meta["dataset_npz"])
        manifest.append(meta)
        predicted = {i: predict_reverse(episode, model, model["calibration"])
                     for i, model in models.items()}
        result = make_result(meta, episode.t, predicted[0], predicted.get(1),
                             acceptance_policy=config["template"]["acceptance_policy"])
        result["run_purpose"] = purpose
        result["feature_sampling"] = episode.feature_sampling
        folder = output / "labels" / episode.key
        folder.mkdir()
        for name, ends in (("action_stage_5", result["action_frames"]),
                           ("display_stage_4", result["display_frames"])):
            np.save(folder / f"{name}.npy", stage_labels(episode.n, ends))
        write_json(folder / "result.json", result)
        predictions[episode.key] = result
    write_json(output / "manifest.json", manifest)
    write_json(output / "predictions.json", predictions)
    _verify_inputs(hashes)
    artifacts = [output / "predictions.json", *(output / "labels").rglob("*")]
    write_json(output / "prediction_sha256.json", {
        str(p.relative_to(output)): sha256_file(p) for p in artifacts if p.is_file()})
    summary = {"episodes": len(inputs), "frames": sum(e.n for e, _ in inputs),
               "candidates": sum(len(e.meta["candidates"]) for e, _ in inputs),
               "elapsed_s": time.monotonic() - start, "purpose": purpose,
               "overlapping_training_keys": overlap,
               "status_counts": dict(Counter(p["status"] for p in predictions.values())),
               "five_stage_structural_complete": sum(
                   all(f is not None for f in p["action_frames"]) for p in predictions.values()),
               "accuracy": None, "human_confirmed": False}
    write_json(output / "summary.json", summary)
    return summary
