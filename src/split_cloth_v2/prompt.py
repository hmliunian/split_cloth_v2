"""Build one fixed-node request without annotations or prior model answers."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from .io_utils import fingerprint, read_json, sha256_file

PROMPT_VERSION = "node_shape_prompt_v1"


def system_text() -> str:
    return files("split_cloth_v2").joinpath("assets/node_shape_prompt_v1.txt").read_text(encoding="utf-8").strip()


def build_request(node: dict, target_stage: int, episode_dir: Path, config: dict) -> dict:
    episode_dir = Path(episode_dir).resolve()
    if type(target_stage) is not int or target_stage not in (1, 2, 3):
        raise ValueError("invalid_target_stage")
    evidence = read_json(episode_dir / "evidence/index.json")["nodes"][node["node_id"]]
    images = []
    for item in evidence["images"]:
        path = (episode_dir / item["path"]).resolve()
        if not path.is_relative_to(episode_dir):
            raise ValueError("evidence_path_outside_episode")
        actual_hash = sha256_file(path)
        if actual_hash != item["sha256"]:
            raise ValueError("evidence_hash_mismatch")
        images.append({**item, "path": str(path), "sha256": actual_hash})
    candidates = [item for item in images if item["role"] == "CANDIDATE"]
    if len(candidates) != 1 or candidates[0]["source_frame_id"] != node["source_frame_id"]:
        raise ValueError("candidate_image_mismatch")
    if any(a["timestamp_s"] >= b["timestamp_s"] for a, b in zip(images, images[1:])):
        raise ValueError("context_not_chronological")
    return {
        "prompt_version": PROMPT_VERSION,
        "node_id": node["node_id"],
        "target_stage": target_stage,
        "system_text": config.get("prompt_text", system_text()),
        "user_text": f"当前候选：{node['node_id']}\n当前目标：stage{target_stage}\n以下图片按时间正序排列；CANDIDATE是唯一待判断帧。",
        "images": images,
        "provider": config["provider"],
        "spec_version": config["spec_version"],
        "config_fingerprint": fingerprint(config),
    }
