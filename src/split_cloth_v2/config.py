"""Load explicit settings and freeze the effective inference contract."""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from .prompt import PROMPT_VERSION, system_text


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration_must_be_an_object")
    if config.get("spec_version") != "reverse_node_v1":
        raise ValueError("unsupported_spec_version")
    if config.get("candidate_policy") != "any_open" or config.get("search_order") != [3, 2, 1]:
        raise ValueError("unsupported_candidate_or_search_policy")
    if config.get("max_inflight_episodes") != 1:
        raise ValueError("first_version_requires_serial_episodes")
    offsets = config.get("context_offsets_s")
    if not isinstance(offsets, list) or not offsets or any(type(x) not in (int, float) or not math.isfinite(x) for x in offsets):
        raise ValueError("invalid_context_offsets")
    if offsets != sorted(set(offsets)) or 0 not in offsets:
        raise ValueError("context_offsets_must_be_sorted_unique_and_include_zero")
    provider = config.get("provider")
    if not isinstance(provider, dict) or not provider.get("base_url", "").startswith("https://") or not provider.get("model_id"):
        raise ValueError("invalid_provider_config")
    forbidden = {"api_key", "authorization", "token", "password", "secret"}
    if any(key.lower() in forbidden for key in provider):
        raise ValueError("credentials_must_not_be_embedded_in_configuration")
    for name, default in (("max_attempts", 3), ("max_output_tokens", 512), ("max_images", 5)):
        value = provider.get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"invalid_provider_{name}")
    if len(offsets) > provider.get("max_images", 5):
        raise ValueError("context_exceeds_image_budget")
    config["prompt_text"] = system_text()
    config["prompt_version"] = PROMPT_VERSION
    return config
