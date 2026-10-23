"""Exercise the file-producing CLI with real short media and offline decisions."""
import json
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from split_cloth_v2 import cli
from split_cloth_v2.io_utils import read_json, sha256_file, write_json
from split_cloth_v2.provider import MockProvider
from test_data import _input


@pytest.fixture
def cli_project(tmp_path, monkeypatch):
    entry = _input(tmp_path)
    table = pq.read_table(entry["parquet_path"]).to_pydict()
    for index, state in enumerate(table["state"]):
        state[13] = float(8 <= index < 13 or 18 <= index < 23 or index >= 28)
    pq.write_table(pa.table(table), entry["parquet_path"])
    entry["source_parquet_sha256"] = sha256_file(entry["parquet_path"])
    project = tmp_path / "project"
    (project / "docs").mkdir(parents=True)
    manifest = project / "docs/phase1_manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")
    actual_root = Path(__file__).resolve().parents[1]
    configuration = yaml.safe_load((actual_root / "configs/default.yaml").read_text())
    configuration["context_offsets_s"] = [-.1, 0., .1]
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump(configuration))
    decisions = project / "decisions.json"
    write_json(decisions, {"n0000": "match", "n0001": "match", "n0002": "match"})
    run = project / "runs" / "cli_test"
    monkeypatch.setattr(cli, "PROJECT_ROOT", project)
    # Any accidental use of a real provider in explicitly mocked runs fails.
    def reject_paid_provider(*args, **kwargs):
        pytest.fail("A mock CLI run attempted to construct a paid provider")
    monkeypatch.setattr(cli, "DashScopeProvider", reject_paid_provider)
    return {"entry": entry, "project": project, "manifest": manifest,
            "config": config_path, "configuration": configuration,
            "decisions": decisions, "run": run}


def _prepare(project, *, mock=True, decisions=True):
    arguments = ["prepare", "--manifest", str(project["manifest"]), "--config", str(project["config"]),
                 "--run", str(project["run"])]
    if mock:
        arguments.append("--mock")
    if decisions:
        arguments.extend(["--mock-decisions", str(project["decisions"])])
    return cli.main(arguments)


def test_cli_mock_complete_chain_exports_full_video_and_resumes_without_new_calls(cli_project):
    project = cli_project
    entry, run = project["entry"], project["run"]
    original = {name: sha256_file(entry[name]) for name in ("source_video_path", "parquet_path")}
    assert _prepare(project) == 0
    input_before = (run / "episodes" / entry["key"] / "input.json").read_bytes()
    assert _prepare(project) == 0
    assert (run / "episodes" / entry["key"] / "input.json").read_bytes() == input_before
    assert cli.main(["annotate", "--run", str(run)]) == 0
    episode = run / "episodes" / entry["key"]
    result = read_json(episode / "result.json")
    assert result["status"] == "complete"
    assert [result["slots"][stage]["end_frame"] for stage in ("stage1", "stage2", "stage3")] == [8, 18, 28]
    calls_before = {path.relative_to(episode): path.read_bytes() for path in (episode / "calls").rglob("*.json")}
    assert cli.main(["annotate", "--run", str(run)]) == 0
    assert {path.relative_to(episode): path.read_bytes() for path in (episode / "calls").rglob("*.json")} == calls_before
    assert cli.main(["export-video", "--run", str(run)]) == 0
    assert cli.main(["report", "--run", str(run)]) == 0
    summary = read_json(run / "reports/summary.json")
    assert summary["execution_mode"] == "mock"
    assert summary["p1_delivery_complete"] is False
    assert summary["model_accuracy"] is None
    assert summary["logical_calls"] == 3 and summary["http_attempts"] == 0
    assert summary["episodes"][0]["review_status"] == "unreviewed"
    assert summary["episodes"][0]["video_verification"]["passed"] is True
    video = Path(summary["episodes"][0]["video_path"])
    with av.open(str(video)) as source:
        times = [float(frame.pts * frame.time_base) for frame in source.decode(video=0)]
    assert len(times) == entry["frame_count"]
    assert times == pytest.approx([index / 20 for index in range(entry["frame_count"])])
    assert {name: sha256_file(entry[name]) for name in original} == original
    assert (run / "reports/phase1_report.md").is_file()


def test_cli_uncertain_result_still_exports_an_unknown_review_video(cli_project):
    project = cli_project
    assert _prepare(project, decisions=False) == 0
    assert cli.main(["annotate", "--run", str(project["run"])]) == 0
    assert cli.main(["export-video", "--run", str(project["run"])]) == 0
    summary = read_json(project["run"] / "reports/summary.json")
    episode = summary["episodes"][0]
    assert episode["status"] == "needs_review" and episode["has_uncertainty"]
    assert list(episode["boundaries"].values()) == [None, None, None]
    assert episode["video_path"] and Path(episode["video_path"]).is_file()
    assert summary["p1_delivery_complete"] is False


@pytest.mark.parametrize("changed", ["config", "code", "manifest"])
def test_prepare_refuses_to_reuse_a_run_after_identity_changes(cli_project, monkeypatch, changed):
    project = cli_project
    assert _prepare(project) == 0
    run_before = (project["run"] / "run.json").read_bytes()
    if changed == "config":
        project["configuration"]["context_offsets_s"] = [-.2, 0., .2]
        project["config"].write_text(yaml.safe_dump(project["configuration"]))
    elif changed == "code":
        monkeypatch.setattr(cli, "source_digest", lambda: "changed-source-digest")
    else:
        project["manifest"].write_text(json.dumps({**project["entry"], "frame_count": 39}) + "\n")
    assert _prepare(project) == 1
    assert (project["run"] / "run.json").read_bytes() == run_before


@pytest.mark.parametrize("changed", ["code", "frozen_manifest", "frozen_prompt", "frozen_config"])
def test_annotate_rejects_modified_frozen_inputs_before_provider_use(cli_project, monkeypatch, changed):
    project = cli_project
    assert _prepare(project) == 0
    run = project["run"]
    if changed == "code":
        monkeypatch.setattr(cli, "source_digest", lambda: "changed-source-digest")
    elif changed == "frozen_manifest":
        path = run / "manifest.jsonl"
        path.write_text(json.dumps({**project["entry"], "frame_count": 39}) + "\n")
    elif changed == "frozen_prompt":
        (run / "prompt.txt").write_text("different instructions")
    else:
        record = read_json(run / "run.json")
        record["config"]["context_offsets_s"] = [-.2, 0., .2]
        write_json(run / "run.json", record)
    assert cli.main(["annotate", "--run", str(run)]) == 1
    assert not (run / "episodes" / project["entry"]["key"] / "calls").exists()


def test_cli_refuses_dataset_as_run_directory(cli_project):
    project = cli_project
    project["run"] = Path(project["entry"]["dataset_root"]) / "run"
    assert _prepare(project) == 1
    assert not project["run"].exists()


def test_real_mode_small_experiment_cannot_claim_full_p1(cli_project, monkeypatch):
    project = cli_project
    # Keep transport offline while exercising real-mode completion reporting.
    monkeypatch.setattr(cli, "DashScopeProvider", lambda **kwargs: MockProvider({"n0000": "match", "n0001": "match", "n0002": "match"}))
    assert _prepare(project, mock=False, decisions=False) == 0
    assert cli.main(["annotate", "--run", str(project["run"])]) == 0
    assert cli.main(["export-video", "--run", str(project["run"])]) == 0
    summary = read_json(project["run"] / "reports/summary.json")
    assert summary["p1_delivery_complete"] is False


def test_export_cli_returns_failure_when_any_episode_could_not_be_exported(cli_project):
    project = cli_project
    assert _prepare(project) == 0
    # Annotation has not run, so the export must report the missing result.
    assert cli.main(["export-video", "--run", str(project["run"])]) == 1
    summary = read_json(project["run"] / "videos/exports.json")
    assert summary["failed_count"] == 1 and summary["successful_count"] == 0


def test_report_does_not_advertise_a_deleted_video(cli_project):
    project = cli_project
    assert _prepare(project) == 0
    assert cli.main(["annotate", "--run", str(project["run"])]) == 0
    assert cli.main(["export-video", "--run", str(project["run"])]) == 0
    summary = read_json(project["run"] / "reports/summary.json")
    Path(summary["episodes"][0]["video_path"]).unlink()
    assert cli.main(["report", "--run", str(project["run"])]) == 0
    summary = read_json(project["run"] / "reports/summary.json")
    assert summary["episodes"][0]["video_path"] is None
    assert summary["p1_delivery_complete"] is False
