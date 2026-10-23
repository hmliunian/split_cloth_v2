from fractions import Fraction

import av
import numpy as np
from PIL import Image
import pytest

from split_cloth_v2.evidence import extract_evidence, select_context_frames
from split_cloth_v2.io_utils import sha256_file


def test_context_uses_pts_clips_deduplicates_and_keeps_exact_candidate():
    selected = select_context_frames([0., .13, .31, .59, .93], 2, [-1., -.5, 0., .5, 1.])
    assert [item["source_frame_id"] for item in selected] == [0, 2, 4]
    assert [item["role"] for item in selected] == ["BEFORE", "CANDIDATE", "AFTER"]
    assert selected[1]["relative_time_s"] == 0.
    assert [item["role"] for item in select_context_frames([0., .1], 0, [-1., 0., 1.])] == ["CANDIDATE", "AFTER"]


def test_offsets_must_include_candidate_and_be_finite():
    with pytest.raises(ValueError, match="include 0"):
        select_context_frames([0., .1], 0, [-1., 1.])
    with pytest.raises(ValueError):
        select_context_frames([0., .1], 0, [0., float("nan")])


def test_extracted_candidate_is_the_requested_source_frame(tmp_path):
    video = tmp_path / "source.mp4"
    with av.open(str(video), "w") as output:
        stream = output.add_stream("libx264", rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(10):
            pixels = np.full((32, 32, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 20)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    source_hash = sha256_file(video)
    episode = tmp_path / "episode"
    result = extract_evidence(video, [i / 20 for i in range(10)],
                              [{"node_id": "n0000", "source_frame_id": 4, "timestamp_s": .2}],
                              episode, {"context_offsets_s": [-.1, 0., .1]})
    images = result["nodes"]["n0000"]["images"]
    assert [item["source_frame_id"] for item in images] == [2, 4, 6]
    candidate = images[1]
    assert candidate["role"] == "CANDIDATE"
    assert abs(np.asarray(Image.open(episode / candidate["path"])).mean() - 80) < 4
    assert candidate["sha256"] == sha256_file(episode / candidate["path"])
    assert sha256_file(video) == source_hash
    with pytest.raises(FileExistsError):
        extract_evidence(video, [i / 20 for i in range(10)],
                         [{"node_id": "n0000", "source_frame_id": 4, "timestamp_s": .2}],
                         episode, {"context_offsets_s": [-.1, 0., .1]})
