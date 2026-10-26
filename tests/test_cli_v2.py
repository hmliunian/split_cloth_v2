"""Verify frozen v2 assets through the complete offline CLI lifecycle."""
from pathlib import Path

from PIL import Image
import pytest
import yaml

from split_cloth_v2 import cli
from split_cloth_v2.io_utils import read_json, sha256_file, write_json
from split_cloth_v2.provider import MockProvider
from test_cli import cli_project, _prepare


def _use_v2(project):
    root = project["project"]
    prompt = root / "prompt.txt"
    prompt.write_bytes("固定提示词\r\n保留原文。\r\n".encode("utf-8"))
    images = root / "images"
    images.mkdir()
    for stage in range(1, 4):
        Image.new("RGB", (24, 16), (stage * 60, 20, 30)).save(images / f"s{stage}.png")
    config = project["configuration"]
    config.update(spec_version="reverse_node_v2", prompt_file="prompt.txt",
                  prompt_version="node_completion_prompt_v1",
                  expert_examples={f"stage{s}": f"images/s{s}.png" for s in (1, 2, 3)})
    config["provider"]["max_images"] = 6
    project["config"].write_text(yaml.safe_dump(config))
    return prompt, images


def test_v2_cli_freezes_exact_prompt_and_images_and_resumes_without_source_assets(cli_project, monkeypatch):
    project = cli_project
    prompt, images = _use_v2(project)
    original_prompt = prompt.read_bytes()
    assert _prepare(project) == 0
    run = project["run"]
    assert (run / "prompt.txt").read_bytes() == original_prompt
    record = read_json(run / "run.json")
    assert record["config"]["prompt_sha256"] == sha256_file(run / "prompt.txt")
    assert read_json(run / "reports/prepare_summary.json")["max_images"] == 12
    prompt.unlink()
    for image in images.glob("*.png"):
        image.unlink()
    assert cli.main(["annotate", "--run", str(run)]) == 0
    episode = run / "episodes" / project["entry"]["key"]
    result = read_json(episode / "result.json")
    assert result["spec_version"] == "reverse_node_v2"
    assert result["status"] == "complete"
    calls_before = {path: path.read_bytes() for path in (episode / "calls").rglob("*.json")}
    def reject_repeated_judgment(*args, **kwargs):
        pytest.fail("A completed v2 run must replay saved decisions")
    monkeypatch.setattr(MockProvider, "judge", reject_repeated_judgment)
    assert cli.main(["annotate", "--run", str(run)]) == 0
    assert calls_before == {path: path.read_bytes() for path in calls_before}
    assert cli.main(["export-video", "--run", str(run)]) == 0
    index = read_json(run / "reports/request_images.json")
    assert len(index["requests"]) == 3
    assert {item["images"][0]["stage"] for item in index["requests"]} == {"stage1", "stage2", "stage3"}
    assert all(item["images"][0]["role"] == "EXPERT_EXAMPLE" for item in index["requests"])
    summary = read_json(run / "reports/summary.json")
    assert summary["spec_version"] == "reverse_node_v2"
    assert summary["episodes"][0]["video_verified"]
    assert summary["model_accuracy"] is None
    assert not summary["p1_delivery_complete"]


@pytest.mark.parametrize("changed", ["prompt_bytes", "example_bytes", "example_mapping"])
def test_v2_cli_refuses_frozen_asset_tampering_before_provider_use(cli_project, monkeypatch, changed):
    project = cli_project
    _use_v2(project)
    assert _prepare(project) == 0
    run = project["run"]
    if changed == "prompt_bytes":
        # Even a trailing whitespace change alters the fixed model input.
        path = run / "prompt.txt"
        path.write_bytes(path.read_bytes() + b" ")
    elif changed == "example_bytes":
        (run / "expert_examples/s1.png").write_bytes(b"modified")
    else:
        path = run / "expert_examples/index.json"
        index = read_json(path)
        index["stages"]["stage1"], index["stages"]["stage2"] = index["stages"]["stage2"], index["stages"]["stage1"]
        write_json(path, index)
    def reject_provider_construction(*args, **kwargs):
        pytest.fail("Frozen asset tampering must be rejected before provider construction")
    monkeypatch.setattr(cli, "MockProvider", reject_provider_construction)
    assert cli.main(["annotate", "--run", str(run)]) == 1
    assert not (run / "episodes" / project["entry"]["key"] / "calls").exists()


def test_real_cli_accepts_explicit_repository_local_credential(cli_project, monkeypatch):
    project = cli_project
    _use_v2(project)
    credential = project["project"] / ".secrets/dashscope_api_key.txt"
    credential.parent.mkdir()
    credential.write_text("offline-dummy-credential")
    received = []
    def offline_provider(*, credential_file):
        received.append(credential_file)
        return MockProvider({"n0000": "match", "n0001": "match", "n0002": "match"})
    monkeypatch.setattr(cli, "DashScopeProvider", offline_provider)
    assert _prepare(project, mock=False, decisions=False) == 0
    assert cli.main(["annotate", "--run", str(project["run"]), "--credential-file", str(credential)]) == 0
    assert received == [str(credential)]
    for path in project["run"].rglob("*.json"):
        assert "offline-dummy-credential" not in path.read_text()
