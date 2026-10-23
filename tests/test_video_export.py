"""Offline checks of review-video semantics, encoding, and input protection."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
import pytest

from split_cloth_v2.video_export import (
    PTS_TOLERANCE_S,
    STAGES,
    UNKNOWN,
    _template,
    candidate_layout,
    export_episode,
    export_run,
    render_frame,
    verify_video,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_video(path, count=12, offset=Fraction(2), audio=False, axis=None):
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, 90000)
        stream.codec_context.time_base = Fraction(1, 90000)
        stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
        audio_stream = container.add_stream("aac", rate=48000) if audio else None
        for index in range(count):
            image = Image.new("RGB", (64, 48), (20 * index % 255, 90, 170))
            frame = av.VideoFrame.from_image(image)
            frame.time_base = Fraction(1, 90000)
            relative = Fraction(str(axis[index])) if axis is not None else Fraction(index, 20)
            frame.pts = round((offset + relative) * 90000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            frame = av.AudioFrame.from_ndarray(np.zeros((2, 1024), dtype=np.float32), format="fltp", layout="stereo")
            frame.sample_rate, frame.time_base, frame.pts = 48000, Fraction(1, 48000), 0
            for packet in audio_stream.encode(frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)


def setup_episode(tmp_path, *, slots=(2, 5, 8), status="complete", audio=False):
    source = tmp_path / "source.mp4"
    make_video(source, audio=audio)
    directory = tmp_path / "run" / "episodes" / "fold_cloth_v3__episode_000001"
    key = directory.name
    result = {"key": key, "frame_count": 12, "status": status,
              "slots": {f"stage{i}": None if value is None else {"node_id": f"n{i:04d}", "end_frame": value}
                        for i, value in enumerate(slots, 1)},
              "review_status": "unreviewed", "training_ready": False}
    write_json(directory / "input.json", {"key": key, "frame_count": 12,
               "source_video_path": str(source), "source_video_sha256": digest(source)})
    write_json(directory / "video_axis.json", {"frame_count": 12, "timestamps_s": [i / 20 for i in range(12)]})
    write_json(directory / "result.json", result)
    write_json(directory / "review.json", {"review_status": "confirmed", "private_note": "preserve"})
    np.save(directory / "stage_candidates.npy", np.full(12, 42, dtype=np.int8))
    np.save(directory / "stage_reviewed.npy", np.full(12, 24, dtype=np.int8))
    return directory, tmp_path / "run" / "videos", result


def test_boundary_ownership_and_highlight_switch():
    n = 12
    result = {"slots": {f"stage{i}": {"end_frame": value, "node_id": f"n{i}"}
                         for i, value in enumerate((2, 5, 8), 1)}, "status": "complete"}
    slots, labels, segments, invalid = candidate_layout(result, n)
    assert not invalid
    assert labels.tolist() == [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3
    template = _template({"key": "test", "frame_count": n}, result, slots, segments, [i / 20 for i in range(n)])
    at_boundary = render_frame(template, Image.new("RGB", (64, 48), "red"), 2, labels, [i / 20 for i in range(n)])
    after_boundary = render_frame(template, Image.new("RGB", (64, 48), "red"), 3, labels, [i / 20 for i in range(n)])
    assert at_boundary.getpixel((900, 248)) == tuple(bytes.fromhex(STAGES[1][2][1:]))
    assert after_boundary.getpixel((900, 296)) == tuple(bytes.fromhex(STAGES[2][2][1:]))
    assert after_boundary.getpixel((900, 248)) != at_boundary.getpixel((900, 248))
    assert at_boundary.getpixel((300, 350)) == (255, 0, 0)


@pytest.mark.parametrize("slots,invalid", [((None, 5, 8), False), ((2, None, 8), False), ((5, 2, 8), True), ((2, 5, 11), True)])
def test_incomplete_or_conflicting_partition_is_entirely_unknown(slots, invalid):
    result = {"slots": {f"stage{i}": None if value is None else {"end_frame": value}
                         for i, value in enumerate(slots, 1)}}
    fixed, labels, segments, is_invalid = candidate_layout(result, 12)
    assert labels.tolist() == [-1] * 12
    assert segments == [{"stage": -1, "start_frame": 0, "end_frame": 11}]
    assert is_invalid is invalid
    if slots[2] == 8:
        assert fixed[2] == 8


def test_export_preserves_nonzero_source_pts_all_frames_and_input_artifacts(tmp_path):
    directory, destination, _ = setup_episode(tmp_path)
    protected = [directory / name for name in ("result.json", "review.json", "stage_candidates.npy", "stage_reviewed.npy")]
    before = {str(path): digest(path) for path in protected}
    record = export_episode(directory, destination, {})
    assert record["frame_count"] == 12
    assert record["source_start_pts_s"] == pytest.approx(2.0)
    assert record["verification"]["passed"]
    assert record["verification"]["max_pts_error_s"] <= PTS_TOLERANCE_S
    assert record["candidate_boundary_slots"] == [2, 5, 8]
    assert record["label_source"] == "model_candidates"
    assert record["review_status"] == "unreviewed"
    output = Path(record["video_path"])
    assert output.is_file() and output.with_suffix(".json").is_file()
    assert not output.with_suffix(".part.mp4").exists()
    assert before == {str(path): digest(path) for path in protected}
    raw = output.read_bytes()
    assert raw.find(b"moov") < raw.find(b"mdat")
    verification = verify_video(output, [i / 20 for i in range(12)])
    assert verification["codec"] == "h264"
    assert verification["size"] == [1280, 720]
    assert verification["pixel_format"] == "yuv420p"
    assert verification["first_pts_s"] == 0


def test_unknown_result_still_exports_full_video(tmp_path):
    directory, destination, _ = setup_episode(tmp_path, slots=(None, None, 8), status="needs_review")
    record = export_episode(directory, destination, {})
    assert record["status"] == "needs_review"
    assert record["candidate_boundary_slots"] == [None, None, 8]
    assert record["candidate_partition"] == [{"stage": -1, "start_frame": 0, "end_frame": 11}]
    assert record["verification"]["frame_count"] == 12


def test_repeat_export_refuses_overwrite(tmp_path):
    directory, destination, _ = setup_episode(tmp_path)
    record = export_episode(directory, destination, {})
    output = Path(record["video_path"])
    previous = digest(output)
    with pytest.raises(FileExistsError):
        export_episode(directory, destination, {})
    assert digest(output) == previous


def test_pts_mismatch_leaves_partial_and_failure_without_publishing(tmp_path):
    directory, destination, _ = setup_episode(tmp_path)
    axis_path = directory / "video_axis.json"
    axis = json.loads(axis_path.read_text())
    axis["timestamps_s"][3] += .001
    write_json(axis_path, axis)
    with pytest.raises(ValueError, match="Source PTS differs"):
        export_episode(directory, destination, {})
    base = destination / f"{directory.name}__candidates"
    assert base.with_suffix(".part.mp4").is_file()
    assert base.with_suffix(".failure.json").is_file()
    assert not base.with_suffix(".mp4").exists()


def test_audio_is_explicitly_rejected(tmp_path):
    directory, destination, _ = setup_episode(tmp_path, audio=True)
    with pytest.raises(ValueError, match="Source has audio"):
        export_episode(directory, destination, {})
    assert not list(destination.glob("*.mp4"))
    assert len(list(destination.glob("*.failure.json"))) == 1


def test_run_summary_counts_failures_and_cannot_call_one_video_phase1_complete(tmp_path):
    directory, _, _ = setup_episode(tmp_path)
    run = directory.parent.parent
    manifest = [{"key": directory.name}, {"key": "fold_cloth_v4__episode_000002"}]
    (run / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in manifest))
    summary = export_run(run, {})
    assert summary["count"] == 2
    assert summary["successful_count"] == 1
    assert summary["failed_count"] == 1
    assert not summary["phase1_complete"]
    verification = json.loads((run / "reports" / "video_verification.json").read_text())
    assert verification["total_verified_frames"] == 12
    assert not verification["all_passed"]


def test_nonuniform_pts_are_preserved_instead_of_reconstructed_from_fps(tmp_path):
    directory, destination, _ = setup_episode(tmp_path)
    source = tmp_path / "irregular.mp4"
    axis = [0, .05, .1, .2, .25, .3, .45, .5, .6, .65, .7, .8]
    make_video(source, offset=Fraction(3), axis=axis)
    metadata_path = directory / "input.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(source_video_path=str(source), source_video_sha256=digest(source))
    write_json(metadata_path, metadata)
    write_json(directory / "video_axis.json", {"frame_count": 12, "timestamps_s": axis})
    record = export_episode(directory, destination, {})
    assert record["verification"]["last_pts_s"] == pytest.approx(.8)
    assert record["verification"]["max_pts_error_s"] <= PTS_TOLERANCE_S


def test_repeat_run_refuses_to_replace_historical_indexes(tmp_path):
    directory, _, _ = setup_episode(tmp_path)
    run = directory.parent.parent
    (run / "manifest.jsonl").write_text(json.dumps({"key": directory.name}) + "\n")
    export_run(run, {})
    paths = [run / "videos" / "exports.json", run / "reports" / "video_verification.json"]
    before = {str(path): digest(path) for path in paths}
    with pytest.raises(FileExistsError, match="export index"):
        export_run(run, {})
    assert before == {str(path): digest(path) for path in paths}
