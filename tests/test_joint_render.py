from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image
import pytest

from split_cloth_v2.io_utils import read_json, sha256_file, write_json
from split_cloth_v2.joint.render import _load_episode, render_run, report_run
from split_cloth_v2.joint.results import stage_labels


def _prepared_run(tmp_path, *, audio=False):
    run = tmp_path / "input"
    (run / "data").mkdir(parents=True)
    (run / "labels" / "episode").mkdir(parents=True)
    axis = np.array([0, .04, .09, .12, .19, .24, .28, .35, .41, .44, .50, .58])
    source = run / "source.mp4"
    with av.open(str(source), "w") as target:
        stream = target.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1_000_000)
        stream.options = {"tune": "zerolatency"}
        if audio:
            sound = target.add_stream("aac", rate=48000)
            sound.layout = "mono"
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024), dtype=np.float32), format="fltp", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
            for packet in sound.encode(frame):
                target.mux(packet)
            for packet in sound.encode():
                target.mux(packet)
        for index, timestamp in enumerate(axis):
            picture = Image.new("RGB", (160, 120), (100 + index, 40, 60))
            frame = av.VideoFrame.from_image(picture)
            frame.pts, frame.time_base = round(timestamp * 1e6), Fraction(1, 1_000_000)
            for packet in stream.encode(frame):
                target.mux(packet)
        for packet in stream.encode():
            target.mux(packet)
    np.savez(run / "data" / "episode.npz", timestamps=axis)
    frames = [None, 3, 6, 9]
    meta = {"key": "episode", "dataset_npz": "data/episode.npz", "frame_count": len(axis),
            "source_video_path": str(source), "source_video_sha256": sha256_file(source),
            "candidates": [{"node_id": f"n{i}", "source_frame_id": frame,
                            "timestamp_s": float(axis[frame]), "opening_arms": ["left"]}
                           for i, frame in enumerate([1, 3, 6, 9])]}
    result = {"key": "episode", "action_frames": frames, "display_frames": frames[1:],
              "raw_action_frames": [1, 3, 6, 9], "label_source": "reverse_joint_template_inference",
              "status": "partial_suffix_needs_review", "human_confirmed": False}
    write_json(run / "manifest.json", [meta])
    write_json(run / "labels" / "episode" / "result.json", result)
    for name, boundaries in [("action_stage_5", frames), ("display_stage_4", frames[1:])]:
        np.save(run / "labels" / "episode" / f"{name}.npy", stage_labels(len(axis), boundaries))
    return run, meta, axis


def test_complete_vfr_video_export_preserves_source_and_unknown_slots(tmp_path):
    run, meta, axis = _prepared_run(tmp_path)
    protected = [Path(meta["source_video_path"]), run / "manifest.json", run / "labels" / "episode" / "result.json"]
    before = {p: sha256_file(p) for p in protected}
    result, _, labels, _ = _load_episode(run, meta)
    assert result["action_frames"][0] is None
    assert np.all(labels["action_stage_5"] == -1)
    assert labels["display_stage_4"].tolist() == [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    summary = render_run(run, workers=1)
    assert summary["successful_count"] == 1 and summary["failed_count"] == 0
    record = summary["videos"][0]
    assert record["verification"]["passed"]
    assert record["output_frame_count"] == len(axis)
    assert record["verification"]["max_pts_error_s"] < 1e-6
    assert record["action_frames"] == [None, 3, 6, 9]
    assert {p: sha256_file(p) for p in protected} == before
    output = run / "exports" / "video"
    assert (output / "index.html").exists()
    for frame in [0, 3, 4, 6, 7, 9, 10, len(axis) - 1]:
        assert (output / "checks" / f"episode__{frame:06d}.jpg").exists()
    with pytest.raises(FileExistsError):
        render_run(run, workers=1)


@pytest.mark.parametrize("problem", ["shifted_pts", "audio"])
def test_export_failures_are_recorded_without_final_video(tmp_path, problem):
    run, _, axis = _prepared_run(tmp_path, audio=problem == "audio")
    if problem == "shifted_pts":
        axis[4:] += .01
        np.savez(run / "data" / "episode.npz", timestamps=axis)
    with pytest.raises(RuntimeError, match="export failures recorded"):
        render_run(run, workers=1)
    output = run / "exports" / "video"
    summary = read_json(output / "video_verification.json")
    assert summary["failed_count"] == 1
    assert not (output / "videos" / "episode.mp4").exists()
    message = "Source PTS" if problem == "shifted_pts" else "audio"
    assert message in summary["videos"][0]["error"]


def test_report_reads_saved_labels_without_source_media_or_inference(tmp_path):
    run, meta, _ = _prepared_run(tmp_path)
    Path(meta["source_video_path"]).unlink()
    summary = report_run(run)
    assert summary["episodes"] == 1 and summary["accuracy"] is None
    assert "None, 3, 6, 9" in Path(summary["report"]).read_text()
    with pytest.raises(ValueError, match="under exports"):
        report_run(run, run / "labels" / "replacement")
