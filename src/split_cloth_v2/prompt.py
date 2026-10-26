"""Build one fixed-node request without annotations or prior model answers."""
from __future__ import annotations

import hashlib
import math
from importlib.resources import files
from pathlib import Path

from .io_utils import fingerprint, read_json, sha256_file

PROMPT_VERSION = "node_shape_prompt_v3"


def system_text() -> str:
    return files("split_cloth_v2").joinpath(f"assets/{PROMPT_VERSION}.txt").read_text(encoding="utf-8").strip()


def build_request(node: dict, target_stage: int, episode_dir: Path, config: dict) -> dict:
    episode_dir = Path(episode_dir).resolve()
    if type(target_stage) is not int or target_stage not in (1, 2, 3):
        raise ValueError("invalid_target_stage")
    evidence_index = read_json(episode_dir / "evidence/index.json")
    evidence = evidence_index["nodes"][node["node_id"]]
    is_v2 = config["spec_version"] == "reverse_node_v2"
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
    if is_v2:
        _validate_local_images(node, images, evidence_index, config)
        prompt_text = config.get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text.strip() or not config.get("prompt_version"):
            raise ValueError("missing_frozen_prompt")
        if hashlib.sha256(prompt_text.encode("utf-8")).hexdigest() != config.get("prompt_sha256"):
            raise ValueError("frozen_prompt_hash_mismatch")
    else:
        prompt_text = config.get("prompt_text", system_text())
    request = {
        "prompt_version": config["prompt_version"] if is_v2 else PROMPT_VERSION,
        "node_id": node["node_id"],
        "target_stage": target_stage,
        "system_text": prompt_text,
        "user_text": f"当前候选：{node['node_id']}\n当前目标：stage{target_stage}\n返回JSON中的target_stage字段必须是整数{target_stage}。\n以下图片按时间正序排列；CANDIDATE是唯一待判断帧。",
        "images": images,
        "provider": config["provider"],
        "spec_version": config["spec_version"],
        "config_fingerprint": fingerprint(config),
    }
    if is_v2:
        from .expert_examples import get_expert_example

        run = episode_dir.parent.parent
        example = get_expert_example(run, config, target_stage)
        request["images"] = [example, *images]
        request["expert_examples"] = read_json(run / "expert_examples/index.json")
        request["context_policy"] = config["context_policy"]
        request["context_offsets_s"] = config["context_offsets_s"]
        request["image_config"] = evidence_index.get("image_config", {})
        request["user_text"] = (
            f"当前候选：{node['node_id']}\n当前目标：stage{target_stage}\n"
            f"返回JSON中的target_stage字段必须是整数{target_stage}。\n"
            "先提供一张当前阶段的 EXPERT_EXAMPLE，再提供本轨迹当前候选及其前后局部上下文。\n"
            "本轨迹图像按时间正序排列，只有 CANDIDATE 是本次待判断节点。"
        )
        if len(request["images"]) > config["provider"].get("max_images", 5):
            raise ValueError("image_budget_exceeded")
    return request


def _validate_local_images(node: dict, images: list[dict], evidence_index: dict, config: dict) -> None:
    """Check the exact candidate and the source-axis roles before adding an example."""
    if evidence_index.get("context_policy") != config.get("context_policy") or evidence_index.get("context_offsets_s") != config.get("context_offsets_s"):
        raise ValueError("context_sampling_config_mismatch")
    frame_ids = [item["source_frame_id"] for item in images]
    if any(type(frame) is not int or frame < 0 for frame in frame_ids) or frame_ids != sorted(set(frame_ids)):
        raise ValueError("context_source_frames_not_ordered_unique")
    for item in images:
        frame = item["source_frame_id"]
        expected_role = "CANDIDATE" if frame == node["source_frame_id"] else "BEFORE" if frame < node["source_frame_id"] else "AFTER"
        timestamp, relative = item["timestamp_s"], item["relative_time_s"]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (timestamp, relative)):
            raise ValueError("context_time_not_finite")
        if item["role"] != expected_role:
            raise ValueError("context_role_mismatch")
        if not math.isclose(relative, timestamp - node["timestamp_s"], rel_tol=0., abs_tol=1e-9):
            raise ValueError("context_relative_time_mismatch")
        if frame == node["source_frame_id"] and (relative != 0 or not math.isclose(timestamp, node["timestamp_s"], rel_tol=0., abs_tol=1e-9)):
            raise ValueError("candidate_time_mismatch")
