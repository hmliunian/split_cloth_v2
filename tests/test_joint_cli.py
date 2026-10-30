from pathlib import Path

import numpy as np
import pytest
import yaml

from split_cloth_v2.cli import main
from split_cloth_v2.io_utils import fingerprint, read_json, sha256_file, write_json
from split_cloth_v2.joint.config import load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs/joint/fold_cloth_v3_6axis.yaml"


def _source(tmp_path, *, training_key=False):
    settings = yaml.safe_load(CONFIG.read_text())
    settings["dataset_root"] = str(tmp_path / "raw_dataset")
    settings["template"]["path"] = str(CONFIG.parent / settings["template"]["path"])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    config = load_config(path)
    training = read_json(config["template"]["path"])["models"]["0"]["training_keys"]
    key = training[0] if training_key else "fold_cloth_v3__cli_test"
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    np.savez(source / "data" / "state.npz", joint=np.zeros((60, 14)), timestamps=np.arange(60) / 20)
    write_json(source / "manifest.json", [{
        "key": key, "dataset_id": "fold_cloth_v3", "task_family": "fold_cloth",
        "episode_id": "cli_test", "frame_count": 60, "state_dimensions": 14,
        "state_representation": "measured_joint", "layout": config["layout"],
        "dataset_npz": "data/state.npz",
        "detector_sha256": fingerprint(config["detector"]), "candidates": [],
    }])
    return path, source, key


def test_prepare_infer_report_preserve_sources_and_unknown_boundaries(tmp_path):
    config, source, key = _source(tmp_path)
    protected = [source / "manifest.json", source / "data/state.npz"]
    before = {p: sha256_file(p) for p in protected}
    prepared, inferred = tmp_path / "prepared", tmp_path / "inferred"
    assert main(["prepare", "--config", str(config), "--manifest", str(source / "manifest.json"), "--run", str(prepared)]) == 0
    assert read_json(prepared / "manifest.json")[0]["candidates"] == []
    assert main(["infer", "--config", str(config), "--input-run", str(prepared), "--run", str(inferred)]) == 0
    result_path = inferred / "labels" / key / "result.json"
    result = read_json(result_path)
    assert result["action_frames"] == [None] * 4
    assert result["status"] == "no_accepted_suffix"
    assert result["program_status"] == "completed"
    for name in ("action_stage_5", "display_stage_4"):
        assert np.all(np.load(result_path.parent / f"{name}.npy") == -1)
    result_digest = sha256_file(result_path)
    assert main(["report", "--run", str(inferred)]) == 0
    assert (inferred / "exports/report/index.html").is_file()
    assert sha256_file(result_path) == result_digest
    assert {p: sha256_file(p) for p in protected} == before
    assert read_json(inferred / "summary.json")["accuracy"] is None


def test_training_overlap_requires_explicit_replay_and_records_purpose(tmp_path, capsys):
    config, source, key = _source(tmp_path, training_key=True)
    output = tmp_path / "inferred"
    args = ["infer", "--config", str(config), "--input-run", str(source), "--run", str(output)]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "--training-replay" in capsys.readouterr().err
    assert not output.exists()
    assert main([*args, "--training-replay"]) == 0
    summary = read_json(output / "summary.json")
    assert summary["purpose"] == "demonstration_replay"
    assert summary["overlapping_training_keys"] == [key]
    assert read_json(output / "protocol.json")["purpose"] == "demonstration_replay"


def test_cli_rejects_output_inside_input_and_existing_outputs(tmp_path, capsys):
    config, source, _ = _source(tmp_path)
    for output in (source / "child", source):
        with pytest.raises(SystemExit) as error:
            main(["infer", "--config", str(config), "--input-run", str(source), "--run", str(output)])
        assert error.value.code == 2
        assert "new directory outside read-only inputs" in capsys.readouterr().err
    assert not (source / "child").exists()


def test_cli_no_longer_exposes_vlm_annotation(capsys):
    with pytest.raises(SystemExit) as error:
        main(["annotate"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
