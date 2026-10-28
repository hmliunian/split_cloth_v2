"""Freeze a balanced sample from the two registered folding datasets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random

from split_cloth_v2.data import REGISTERED_DATASETS, read_state, validate_entry
from split_cloth_v2.io_utils import now, sha256_file, write_json


def materialize(item):
    root, info, episode = item
    index = episode["episode_index"]
    episode_id = f"episode_{index:06d}"
    parameters = {"episode_chunk": index // info["chunks_size"], "episode_index": index,
                  "video_key": "base_0_rgb"}
    video = root / info["video_path"].format(**parameters)
    parquet = root / info["data_path"].format(**parameters)
    entry = {"key": f"{root.name}__{episode_id}", "dataset_id": root.name, "dataset_root": str(root),
             "episode_id": episode_id, "episode_index": index, "frame_count": episode["length"],
             "source_fps": info["fps"], "camera_key": "base_0_rgb", "task_family": "fold_cloth",
             "source_video_path": str(video), "parquet_path": str(parquet),
             "source_video_sha256": sha256_file(video), "source_parquet_sha256": sha256_file(parquet),
             "state_column": "state", "timestamp_column": "timestamp", "frame_index_column": "frame_index"}
    validate_entry(entry)
    state = read_state(parquet, state_column="state")
    if len(state["states"]) != episode["length"]:
        raise ValueError(f"state_length_mismatch: {entry['key']}")
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    output, data = args.output.resolve(), args.dataset_root.resolve()
    project = Path(__file__).resolve().parents[1]
    runs = (project / "runs").resolve()
    if not output.is_relative_to(runs) or output == runs:
        raise ValueError("sample_output_must_be_inside_repository_runs")
    if args.per_dataset < 1:
        raise ValueError("per_dataset_must_be_positive")
    if output.exists():
        raise FileExistsError("Selection output already exists; use a new directory")
    groups, populations = [], []
    for dataset_id in sorted(REGISTERED_DATASETS):
        root = data / dataset_id
        info = json.loads((root / "meta/info.json").read_text())
        episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        if len(episodes) != info["total_episodes"] or len({row["episode_index"] for row in episodes}) != len(episodes):
            raise ValueError(f"inconsistent_episode_population: {dataset_id}")
        if any(row["tasks"] not in (["Fold the clothes."], ["Fold the shirts."]) for row in episodes):
            raise ValueError(f"non_folding_task_metadata: {dataset_id}")
        if info["features"]["state"]["shape"] != [14] or info["chunks_size"] != 1000:
            raise ValueError(f"unsupported_state_or_layout: {dataset_id}")
        seed_hash = hashlib.sha256(f"{args.seed}:{dataset_id}".encode()).hexdigest()
        chosen = random.Random(int(seed_hash, 16)).sample(sorted(episodes, key=lambda row: row["episode_index"]), args.per_dataset)
        groups.append([(root, info, row) for row in chosen])
        populations.append({"dataset_id": dataset_id, "total_episodes": len(episodes), "seed_sha256": seed_hash,
                            "sample_episode_indices": [row["episode_index"] for row in chosen],
                            "metadata_sha256": {name: sha256_file(root / "meta" / name)
                                                for name in ("info.json", "episodes.jsonl", "tasks.jsonl")}})
    with ThreadPoolExecutor(max_workers=16) as executor:
        entries = list(executor.map(materialize, [item for group in zip(*groups) for item in group]))
    output.mkdir(parents=True)
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for entry in entries))
    write_json(output / "selection.json", {"created_at": now(), "seed": args.seed, "sample_count": len(entries),
               "per_dataset": args.per_dataset, "populations": populations,
               "method": "Independent SHA256(seed:dataset) random samples without replacement; interleaved sources; no outcome filtering",
               "manifest_sha256": sha256_file(manifest), "script_sha256": sha256_file(__file__)})
    print(json.dumps({"sample_count": len(entries), "total_frames": sum(entry["frame_count"] for entry in entries),
                      "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
