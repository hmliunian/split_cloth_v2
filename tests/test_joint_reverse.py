from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/joint_reverse_suffix"))
import itertools
import numpy as np
from reverse_model import reverse_decode, stage_labels

def test_reverse_best_two_match_exhaustive_search():
    rng = np.random.default_rng(1729)
    for candidate_count in [4, 5, 6, 8]:
        size = candidate_count + 2
        for trial in range(5):
            costs = rng.uniform(.01, 7, size=(4, size, size))
            if trial:
                costs[rng.uniform(size=costs.shape) < .15] = np.inf
            oracle = []
            for a, b, c, d in itertools.combinations(range(1, size - 1), 4):
                path = [(5, d, size - 1), (4, c, d), (3, b, c), (2, a, b)]
                score = sum(costs[s - 2, i, j] for s, i, j in path)
                if np.isfinite(score):
                    oracle.append((score, path))
            if not oracle:
                continue
            actual, layers, stages = reverse_decode(costs)
            expected = sorted(oracle, key=lambda x: x[0])[:2]
            assert stages == [5, 4, 3, 2]
            assert [x['stage'] for x in layers] == ['S5', 'S4', 'S3', 'S2']
            assert [p for score, p in actual] == [p for score, p in expected]
            assert np.allclose([score for score, p in actual], [score for score, p in expected])

def test_reverse_search_escapes_greedy_dead_end():
    costs = np.full((4, 7, 7), np.inf)
    costs[3, 5, 6] = 0  # Locally best S5 has no matching preceding S4.
    costs[3, 4, 6] = 1
    costs[2, 3, 4] = costs[1, 2, 3] = costs[0, 1, 2] = 0
    paths, layers, stages = reverse_decode(costs)
    assert stages == [5, 4, 3, 2]
    assert paths[0] == (1.0, [(5, 4, 6), (4, 3, 4), (3, 2, 3), (2, 1, 2)])

def test_missing_earlier_stage_keeps_only_supported_suffix():
    costs = np.full((4, 6, 6), np.inf)
    costs[3, 4, 5] = costs[2, 3, 4] = costs[1, 2, 3] = .2
    paths, layers, stages = reverse_decode(costs)
    assert stages == [5, 4, 3]
    assert [s for s, i, j in paths[0][1]] == stages
    assert layers[-1]['feasible_start_candidates'] == 0
    costs[3] = np.inf
    assert reverse_decode(costs)[0] == []

def test_video_end_and_start_sentinels_cannot_be_stage_boundaries():
    costs = np.full((4, 6, 6), np.inf)
    costs[3, 0, 5] = costs[3, 5, 5] = -100
    assert reverse_decode(costs)[0] == []

def test_boundary_frame_belongs_to_preceding_stage_and_unknown_stays_unknown():
    assert stage_labels(10, [1, 3, 6, 8]).tolist() == [1, 1, 2, 2, 3, 3, 3, 4, 4, 5]
    assert np.all(stage_labels(10, [None, 3, 6, 8]) == -1)

def _episode_fixture():
    q = np.arange(80 * 14, dtype=float).reshape(80, 14) / 1000
    times = np.arange(80) / 20
    meta = {'key': 'fold_cloth_v3__episode_test', 'dataset_id': 'fold_cloth_v3',
            'task_family': 'fold_cloth', 'frame_count': 80, 'candidates': []}
    return meta, {'joint': q, 'timestamps': times}

def test_frozen_templates_and_empty_candidate_episode():
    from run import CODE, load_templates, validate_episode
    from model import Episode
    from reverse_model import predict_reverse
    bundle, models = load_templates(CODE / 'templates_v3_ten.json')
    meta, arrays = _episode_fixture()
    validate_episode(meta, arrays)
    result = predict_reverse(Episode(meta, arrays), models[0], models[0]['calibration'])
    assert result['status'] == 'no_accepted_suffix'
    assert result['action_frames'] == [None] * 4 and result['s1_repeat_count'] is None

def test_rejects_other_tasks_and_shifted_or_nonfinite_state():
    import pytest
    from run import validate_episode
    meta, arrays = _episode_fixture()
    with pytest.raises(ValueError, match='Only fold_cloth_v3'):
        validate_episode({**meta, 'task_family': 'fold_and_hang'}, arrays)
    with pytest.raises(ValueError, match='20 FPS'):
        validate_episode(meta, {**arrays, 'timestamps': arrays['timestamps'] + .05})
    bad = arrays['joint'].copy()
    bad[4, 3] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        validate_episode(meta, {**arrays, 'joint': bad})

def test_native_dtw_requires_full_finite_sequences():
    import pytest
    from dtw import dtw
    a = np.zeros((32, 2))
    assert dtw(a, a) == 0
    b = a.copy()
    b[-1] = 10
    assert dtw(a, b) > 0  # The final template endpoint cannot be skipped.
    with pytest.raises(ValueError):
        dtw(a, np.zeros((31, 2)))
    b[0] = np.nan
    with pytest.raises(ValueError):
        dtw(a, b)

def test_existing_output_and_input_directory_are_never_overwritten(tmp_path):
    import pytest
    from run import CODE, infer_run
    source = tmp_path / 'source'
    source.mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('existing output')
    with pytest.raises(ValueError, match='new directory'):
        infer_run(source, output, CODE / 'templates_v3_ten.json')
    with pytest.raises(ValueError, match='new directory'):
        infer_run(source, source / 'child', CODE / 'templates_v3_ten.json')
    assert sentinel.read_text() == 'existing output'
