"""Exercise bounded episode concurrency, replay, and real process video exports."""
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import threading

import pytest
import yaml

from split_cloth_v2 import cli
from split_cloth_v2.config import load_config
from split_cloth_v2.io_utils import read_json, sha256_file
from split_cloth_v2.provider import MockProvider
from split_cloth_v2.video_export import export_run
from scripts.export_candidate_marks import export_marked_episode
from test_cli import cli_project, _prepare
from test_cli_v2 import _use_v2

ROOT = Path(__file__).resolve().parents[1]


def _batch(project, count=4):
    _use_v2(project)
    first = project["entry"]
    entries = [first]
    for offset in range(1, count):
        index = first["episode_index"] + offset
        episode_id = f"episode_{index:06d}"
        entry = {**first, "episode_index": index, "episode_id": episode_id,
                 "key": f"{first['dataset_id']}__{episode_id}"}
        for field in ("source_video_path", "parquet_path"):
            source = Path(first[field])
            destination = source.with_name(episode_id + source.suffix)
            shutil.copyfile(source, destination)
            entry[field] = str(destination)
        entries.append(entry)
    project["manifest"].write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    project["configuration"]["max_inflight_episodes"] = 2
    project["configuration"]["video_export"]["max_workers"] = 2
    project["config"].write_text(yaml.safe_dump(project["configuration"]))
    assert _prepare(project) == 0
    return entries, project["run"]


def test_parallel_search_preserves_per_episode_order_and_replays_without_calls(cli_project, monkeypatch):
    entries, run = _batch(cli_project)
    barrier, lock = threading.Barrier(2), threading.Lock()
    active, calls = set(), defaultdict(list)
    peak = 0
    original = MockProvider.judge

    def judge(self, request, call_dir):
        nonlocal peak
        key = call_dir.parent.parent.name
        with lock:
            assert key not in active, "A single episode must never issue concurrent judgments"
            active.add(key)
            peak = max(peak, len(active))
            calls[key].append((request["node_id"], request["target_stage"]))
        try:
            if request["target_stage"] == 3:
                barrier.wait(timeout=10)
            return original(self, request, call_dir)
        finally:
            with lock:
                active.remove(key)

    monkeypatch.setattr(MockProvider, "judge", judge)
    assert cli.main(["annotate", "--run", str(run)]) == 0
    assert peak == 2
    for entry in entries:
        assert calls[entry["key"]] == [("n0002", 3), ("n0001", 2), ("n0000", 1)]
    execution = read_json(run / "reports/annotation_execution.json")
    assert execution["peak_parallel_episodes"] == 2
    assert [task["key"] for task in execution["tasks"]] == [entry["key"] for entry in entries]
    before = {path: path.read_bytes() for path in (run / "episodes").glob("*/calls/*/*.json")}

    def reject_repeated_call(*args, **kwargs):
        pytest.fail("A parallel resume must use the saved decisions")

    monkeypatch.setattr(MockProvider, "judge", reject_repeated_call)
    assert cli.main(["annotate", "--run", str(run)]) == 0
    assert before == {path: path.read_bytes() for path in before}
    assert len(list((run / "reports/executions").glob("annotate_*.json"))) == 2
    assert cli.main(["export-video", "--run", str(run)]) == 0
    exports = read_json(run / "videos/exports.json")
    assert exports["all_video_exports_complete"] is True
    assert exports["execution"]["peak_parallel_episodes"] == 2
    assert len({record["worker_pid"] for record in exports["videos"]}) == 2
    assert all(record["worker_pid"] != os.getpid() for record in exports["videos"])
    assert [record["key"] for record in exports["videos"]] == [entry["key"] for entry in entries]
    assert all(record["verification"]["passed"] for record in exports["videos"])
    assert all(entry["video_verified"] for entry in read_json(run / "reports/summary.json")["episodes"])
    marked = export_run(run, read_json(run / "run.json")["config"], export_dir=run / "exports/markers",
                        episode_exporter=export_marked_episode)
    assert marked["all_video_exports_complete"] is True
    assert marked["execution"]["peak_parallel_episodes"] == 2
    assert all(record["candidate_markers"]["count"] == 3 for record in marked["videos"])
    assert before == {path: path.read_bytes() for path in before}


def test_parallel_failures_do_not_hide_other_episode_results(cli_project):
    entries, run = _batch(cli_project)
    invalid = run / "episodes" / entries[1]["key"]
    (invalid / "candidates.json").write_bytes((invalid / "candidates.json").read_bytes() + b" ")
    assert cli.main(["annotate", "--run", str(run)]) == 1
    assert not (invalid / "calls").exists()
    summary = read_json(run / "reports/summary.json")
    assert [item["status"] for item in summary["episodes"]] == ["complete", "execution_error", "complete", "complete"]
    assert "Prepared artifact changed" in summary["episodes"][1]["execution_error"]
    protected = {path: sha256_file(path) for path in (run / "episodes").rglob("*") if path.is_file()}
    assert cli.main(["export-video", "--run", str(run)]) == 1
    exports = read_json(run / "videos/exports.json")
    assert exports["successful_count"] == 3 and exports["failed_count"] == 1
    assert exports["videos"][1]["key"] == entries[1]["key"]
    assert exports["videos"][1]["export_status"] == "failed"
    assert all(digest == sha256_file(path) for path, digest in protected.items())


@pytest.mark.parametrize("field", ["max_inflight_episodes", "render"])
@pytest.mark.parametrize("value", [True, 0, -1, 17, 1.5, "16"])
def test_invalid_parallelism_is_rejected_before_any_work(tmp_path, field, value):
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    if field == "render":
        config["video_export"]["max_workers"] = value
    else:
        config[field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="must_be_integer_1_to_16"):
        load_config(path)


def test_parallel16_configuration_preserves_the_main_inference_contract():
    baseline = load_config(ROOT / "configs/reverse_node_v2_stage3_shape_guard.yaml")
    parallel = load_config(ROOT / "configs/reverse_node_v2_parallel16.yaml")
    assert parallel.pop("max_inflight_episodes") == 16
    baseline.pop("max_inflight_episodes")
    assert parallel["video_export"].pop("max_workers") == 16
    assert parallel == baseline
