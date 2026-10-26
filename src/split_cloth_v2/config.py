"""Load explicit settings and freeze the effective inference contract."""
from __future__ import annotations

import math
import hashlib
from pathlib import Path

import yaml

from .expert_examples import inspect_expert_examples
from .prompt import PROMPT_VERSION, system_text


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration_must_be_an_object")
    if config.get("spec_version") not in ("reverse_node_v1", "reverse_node_v2"):
        raise ValueError("unsupported_spec_version")
    v2 = config["spec_version"] == "reverse_node_v2"
    if config.get("candidate_policy") != "any_open" or config.get("search_order") != [3, 2, 1]:
        raise ValueError("unsupported_candidate_or_search_policy")
    if config.get("max_inflight_episodes") != 1:
        raise ValueError("first_version_requires_serial_episodes")
    offsets = config.get("context_offsets_s")
    if not isinstance(offsets, list) or not offsets or any(type(x) not in (int, float) or not math.isfinite(x) for x in offsets):
        raise ValueError("invalid_context_offsets")
    if offsets != sorted(set(offsets)) or 0 not in offsets:
        raise ValueError("context_offsets_must_be_sorted_unique_and_include_zero")
    if v2 and config.get("context_policy") != "fixed_local_v1":
        raise ValueError("unsupported_context_policy")
    provider = config.get("provider")
    if not isinstance(provider, dict) or not provider.get("base_url", "").startswith("https://") or not provider.get("model_id"):
        raise ValueError("invalid_provider_config")
    forbidden = {"api_key", "authorization", "token", "password", "secret"}
    if any(key.lower() in forbidden for key in provider):
        raise ValueError("credentials_must_not_be_embedded_in_configuration")
    image_budget_default = 6 if v2 else 5
    for name, default in (("max_attempts", 3), ("max_output_tokens", 512), ("max_images", image_budget_default)):
        value = provider.get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"invalid_provider_{name}")
    if len(offsets) + int(v2) > provider.get("max_images", image_budget_default):
        raise ValueError("context_exceeds_image_budget")
    if v2:
        prompt_file = config.get("prompt_file")
        if not isinstance(prompt_file, str) or not prompt_file.strip():
            raise ValueError("prompt_file_required_for_v2")
        prompt_path = (path.parent / prompt_file).resolve()
        try:
            prompt_bytes = prompt_path.read_bytes()
            prompt_text = prompt_bytes.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError("invalid_prompt_file") from exc
        if not prompt_text.strip():
            raise ValueError("prompt_file_must_not_be_empty")
        if not isinstance(config.get("prompt_version"), str) or not config["prompt_version"].strip():
            raise ValueError("prompt_version_required_for_v2")
        config["prompt_file"] = str(prompt_path)
        config["prompt_text"] = prompt_text
        config["prompt_sha256"] = hashlib.sha256(prompt_bytes).hexdigest()
        provider.setdefault("max_images", image_budget_default)
        sources = inspect_expert_examples(config.get("expert_examples"), path.parent)
        config["expert_examples"] = {stage: item["source_path"] for stage, item in sources.items()}
        config["expert_example_sources"] = sources
    else:
        config["prompt_text"] = system_text()
        config["prompt_version"] = PROMPT_VERSION
    return config
