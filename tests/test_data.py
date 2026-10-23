from fractions import Fraction
import json

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from split_cloth_v2.data import decode_video_axis, load_manifest, prepare_episode, read_state
from split_cloth_v2.io_utils import read_json, sha256_file


def _input(tmp_path, count=40):
    root = tmp_path / "dataset" / "fold_cloth_v3"
    video = root / "videos" / "chunk-000" / "base_0_rgb" / "episode_000001.mp4"
    parquet = root / "data" / "chunk-000" / "episode_000001.parquet"
    video.parent.mkdir(parents=True)
    parquet.parent.mkdir(parents=True)
    with av.open(str(video), "w") as output:
        stream = output.add_stream("libx264", rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 3, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 20)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    states = [[0.] * 14 for _ in range(count)]
    for state in states[count // 2:]:
        state[13] = 1.
    pq.write_table(pa.table({"state": states, "timestamp": [i / 20 for i in range(count)],
                            "frame_index": list(range(count))}), parquet)
    return {"key": "fold_cloth_v3__episode_000001", "dataset_id": "fold_cloth_v3",
            "dataset_root": str(root), "episode_id": "episode_000001", "episode_index": 1,
            "source_video_path": str(video), "parquet_path": str(parquet),
            "frame_count": count, "source_fps": 20, "camera_key": "base_0_rgb",
            "source_video_sha256": sha256_file(video), "source_parquet_sha256": sha256_file(parquet),
            "task_family": "fold_cloth", "state_column": "state"}


def test_full_preparation_validates_input_writes_evidence_and_resumes(tmp_path):
    entry = _input(tmp_path)
    episode = tmp_path / "run" / "episodes" / entry["key"]
    config = {"context_offsets_s": [-.5, 0., .5]}
    info = prepare_episode(entry, episode, config)
    assert info["alignment_verified"] and info["state_dimension"] == 14
    assert info["candidate_count"] == info["max_logical_requests"] == 1
    nodes = read_json(episode / "candidates.json")["candidates"]
    assert [(node["source_frame_id"], node["opening_arms"]) for node in nodes] == [(20, ["right"])]
    assert read_json(episode / "video_axis.json")["timestamps_s"] == pytest.approx([i / 20 for i in range(40)])
    assert prepare_episode(entry, episode, config) == info
    assert sha256_file(entry["source_video_path"]) == entry["source_video_sha256"]
    assert sha256_file(entry["parquet_path"]) == entry["source_parquet_sha256"]
    with pytest.raises(FileExistsError, match="differs"):
        prepare_episode(entry, episode, {"context_offsets_s": [-1., 0., 1.]})


def test_preparation_refuses_changed_sources_history_and_dataset_output(tmp_path):
    entry = _input(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        prepare_episode(entry, tmp_path / "dataset" / "fold_cloth_v3" / "results", {})
    existing = tmp_path / "history"
    existing.mkdir()
    (existing / "review.json").write_text('{"human": true}')
    with pytest.raises(FileExistsError):
        prepare_episode(entry, existing, {})
    assert (existing / "review.json").read_text() == '{"human": true}'
    changed = {**entry, "source_video_sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_episode(changed, tmp_path / "output", {})


def test_manifest_rejects_nonfold_and_duplicate_episodes(tmp_path):
    entry = _input(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")
    assert load_manifest(manifest) == [entry]
    manifest.write_text(json.dumps({**entry, "task_family": "hang_cloth"}) + "\n")
    with pytest.raises(ValueError, match="registered fold_cloth"):
        load_manifest(manifest)
    manifest.write_text((json.dumps(entry) + "\n") * 2)
    with pytest.raises(ValueError, match="unique"):
        load_manifest(manifest)


def test_state_column_ambiguity_and_command_substitution_are_rejected(tmp_path):
    path = tmp_path / "state.parquet"
    values = {"state": [[0.] * 14] * 3, "observation.state": [[0.] * 14] * 3,
              "action": [[1.] * 14] * 3, "timestamp": [0., .05, .1], "frame_index": [0, 1, 2]}
    pq.write_table(pa.table(values), path)
    with pytest.raises(ValueError, match="explicitly"):
        read_state(path)
    with pytest.raises(ValueError, match="action"):
        read_state(path, state_column="action")
    assert read_state(path, state_column="state")["state_column"] == "state"


@pytest.mark.parametrize("kind", ["dimension", "nonfinite", "gap", "reverse_time"])
def test_bad_state_values_and_time_axes_are_rejected(tmp_path, kind):
    values = {"state": [[0.] * 14 for _ in range(3)], "timestamp": [0., .05, .1], "frame_index": [0, 1, 2]}
    if kind == "dimension":
        values["state"][1] = [0.] * 16
    elif kind == "nonfinite":
        values["state"][1][6] = float("inf")
    elif kind == "gap":
        values["frame_index"] = [0, 2, 3]
    else:
        values["timestamp"] = [.1, .05, 0.]
    path = tmp_path / "state.parquet"
    pq.write_table(pa.table(values), path)
    with pytest.raises(ValueError):
        read_state(path)


def test_wrong_manifest_frame_count_rejects_before_writing_output(tmp_path):
    entry = _input(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="frame count"):
        prepare_episode({**entry, "frame_count": 39}, output, {})
    assert not output.exists()


def test_changed_prepared_evidence_is_not_silently_reused(tmp_path):
    entry = _input(tmp_path)
    output = tmp_path / "output"
    prepare_episode(entry, output, {})
    image = next((output / "evidence" / "frames").glob("*.jpg"))
    image.write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact changed"):
        prepare_episode(entry, output, {})


def test_nonzero_source_pts_are_normalized_without_rebuilding_axis(tmp_path):
    video = tmp_path / "offset.mp4"
    with av.open(str(video), "w") as output:
        stream = output.add_stream("libx264", rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for pts in [100, 101, 103, 104]:
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 20)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    axis = decode_video_axis(video)
    assert axis["source_start_time_s"] == pytest.approx(5.)
    assert axis["timestamps_s"] == pytest.approx([0., .05, .15, .2])
