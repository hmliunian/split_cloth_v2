"""Validate external instructions, reference mappings and immutable run assets."""
import hashlib
from pathlib import Path

from PIL import Image
import pytest
import yaml

from split_cloth_v2.config import load_config
from split_cloth_v2.expert_examples import (
    freeze_expert_examples, get_expert_example, verify_expert_examples,
)
from split_cloth_v2.io_utils import fingerprint, read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def v2_config(tmp_path):
    directory = tmp_path / "config"
    (directory / "images").mkdir(parents=True)
    configuration = yaml.safe_load((ROOT / "configs/reverse_node_v2.yaml").read_text())
    # Keep CRLF, leading spaces and trailing newlines exactly as supplied.
    prompt = "  固定模型指令\r\n第二行\r\n\r\n".encode("utf-8")
    (directory / "prompt.txt").write_bytes(prompt)
    for number in (1, 2, 3):
        Image.new("RGB", (11 + number, 9), (number * 60, 0, 0)).save(directory / f"images/s{number}.png")
    path = directory / "run.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path, configuration, prompt


def test_v2_paths_are_relative_to_yaml_and_prompt_bytes_are_preserved(v2_config, tmp_path, monkeypatch):
    path, _, prompt = v2_config
    monkeypatch.chdir(tmp_path)
    config = load_config(path)
    assert config["prompt_file"] == str(path.parent / "prompt.txt")
    assert config["prompt_text"].encode("utf-8") == prompt
    assert config["prompt_sha256"] == hashlib.sha256(prompt).hexdigest()
    assert config["prompt_version"] == "node_completion_prompt_v1"
    assert config["provider"]["max_images"] == 6
    for stage, source in config["expert_example_sources"].items():
        assert source["source_path"] == config["expert_examples"][stage]
        assert source["source_sha256"] == sha256_file(source["source_path"])
        assert source["mime_type"] == "image/png"


@pytest.mark.parametrize("content", [b"", b"  \n\t", b"\xff"])
def test_v2_rejects_empty_or_invalid_utf8_prompt(v2_config, content):
    path, _, _ = v2_config
    (path.parent / "prompt.txt").write_bytes(content)
    with pytest.raises(ValueError, match="prompt"):
        load_config(path)


def test_v2_never_falls_back_when_prompt_file_is_missing(v2_config):
    path, _, _ = v2_config
    (path.parent / "prompt.txt").unlink()
    with pytest.raises(ValueError, match="invalid_prompt_file"):
        load_config(path)


@pytest.mark.parametrize("problem", ["missing_stage", "extra_stage", "missing_file", "invalid_image", "insufficient_budget"])
def test_v2_rejects_incomplete_reference_contract_before_requests(v2_config, problem):
    path, configuration, _ = v2_config
    if problem == "missing_stage":
        del configuration["expert_examples"]["stage2"]
    elif problem == "extra_stage":
        configuration["expert_examples"]["stage4"] = "images/s1.png"
    elif problem == "missing_file":
        (path.parent / "images/s2.png").unlink()
    elif problem == "invalid_image":
        (path.parent / "images/s2.png").write_text("not an image")
    else:
        configuration["provider"]["max_images"] = 5
    path.write_text(yaml.safe_dump(configuration))
    with pytest.raises(ValueError):
        load_config(path)


def test_freeze_copies_original_bytes_and_survives_deleted_sources(v2_config, tmp_path):
    path, _, _ = v2_config
    config = load_config(path)
    run = tmp_path / "runs/test"
    index = freeze_expert_examples(run, config)
    for number in (1, 2, 3):
        stage = f"stage{number}"
        item = index["stages"][stage]
        original = Path(config["expert_examples"][stage])
        assert (run / item["path"]).read_bytes() == original.read_bytes()
        assert item["sha256"] == item["source_sha256"]
        original.unlink()
    before = (run / "expert_examples/index.json").read_bytes()
    assert verify_expert_examples(run, config) == index
    assert freeze_expert_examples(run, config) == index
    assert (run / "expert_examples/index.json").read_bytes() == before
    for number in (1, 2, 3):
        image = get_expert_example(run, config, number)
        assert image["stage"] == f"stage{number}" and image["role"] == "EXPERT_EXAMPLE"
        assert Path(image["path"]).is_absolute()
        assert not {"source_frame_id", "timestamp_s", "relative_time_s", "node_id"} & image.keys()


@pytest.mark.parametrize("changed", ["bytes", "mapping", "metadata", "deleted"])
def test_any_frozen_reference_change_is_rejected_without_overwrite(v2_config, tmp_path, changed):
    path, _, _ = v2_config
    config = load_config(path)
    run = tmp_path / "run"
    index = freeze_expert_examples(run, config)
    image_path = run / index["stages"]["stage1"]["path"]
    if changed == "bytes":
        image_path.write_bytes(b"changed image")
    elif changed == "deleted":
        image_path.unlink()
    else:
        if changed == "mapping":
            index["stages"]["stage1"], index["stages"]["stage2"] = index["stages"]["stage2"], index["stages"]["stage1"]
        else:
            index["stages"]["stage1"]["width"] += 1
        write_json(run / "expert_examples/index.json", index)
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="frozen_expert_example"):
        verify_expert_examples(run, config)
    with pytest.raises(ValueError, match="frozen_expert_example"):
        freeze_expert_examples(run, config)
    assert {p: p.read_bytes() for p in run.rglob("*") if p.is_file()} == before


def test_freeze_rejects_source_changed_after_configuration_load(v2_config, tmp_path):
    path, _, _ = v2_config
    config = load_config(path)
    Path(config["expert_examples"]["stage1"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source_changed_during_prepare"):
        freeze_expert_examples(tmp_path / "run", config)
    assert not (tmp_path / "run/expert_examples/index.json").exists()


def test_v2_fingerprints_include_prompt_content_and_reference_mapping(v2_config):
    path, configuration, _ = v2_config
    original = load_config(path)
    prompt_path = path.parent / "prompt.txt"
    prompt_path.write_bytes(prompt_path.read_bytes() + b"extra instructions")
    changed_prompt = load_config(path)
    assert fingerprint(changed_prompt) != fingerprint(original)
    configuration["expert_examples"]["stage1"], configuration["expert_examples"]["stage2"] = configuration["expert_examples"]["stage2"], configuration["expert_examples"]["stage1"]
    path.write_text(yaml.safe_dump(configuration))
    assert fingerprint(load_config(path)) != fingerprint(changed_prompt)


def test_v1_stays_compatible_and_v2_preserves_baseline_detector_and_inference_parameters(tmp_path):
    baseline = load_config(ROOT / "configs/default.yaml")
    v2 = load_config(ROOT / "configs/reverse_node_v2.yaml")
    assert baseline["spec_version"] == "reverse_node_v1"
    assert baseline["prompt_version"] == "node_shape_prompt_v3"
    assert "expert_examples" not in baseline and "prompt_file" not in baseline
    assert freeze_expert_examples(tmp_path, baseline) is None
    assert verify_expert_examples(tmp_path, baseline) is None
    assert baseline["detector"] == v2["detector"]
    assert baseline["context_offsets_s"] == v2["context_offsets_s"]
    assert {k: v for k, v in baseline["provider"].items() if k != "max_images"} == {k: v for k, v in v2["provider"].items() if k != "max_images"}
