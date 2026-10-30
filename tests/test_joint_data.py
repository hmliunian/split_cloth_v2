from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from split_cloth_v2.joint.config import load_config, validate_config
from split_cloth_v2.joint.data import prepare_run, validate_episode
from split_cloth_v2.joint.adapters.mcap import align_nearest

CONFIGS = Path(__file__).resolve().parents[1] / 'configs' / 'joint'


def config(axis=6):
    return load_config(CONFIGS / ('fold_cloth_v3_6axis.yaml' if axis == 6 else 'fold_cloth_v3_pushcar_7axis.yaml'))


def fixture(axis=6):
    cfg = config(axis)
    n = 80
    times = np.arange(n) / (20 if axis == 6 else 30)
    arrays = {'joint': np.zeros((n, cfg['layout']['state_dimensions'])), 'timestamps': times}
    if axis == 7:
        clock = np.rint(times * 1e9).astype(np.int64) + 1_000_000_000
        arrays.update(source_video_clock_ns=clock, feedback_sample_rows=np.tile(np.arange(n)[:, None], (1, 4)),
                      feedback_time_errors_ns=np.zeros((n, 4), dtype=np.int64))
    return cfg, {'key': 'fold_example', 'dataset_id': cfg['dataset_id'], 'frame_count': n, 'candidates': [],
                 'state_representation': 'measured_joint', 'layout': cfg['layout']}, arrays


def test_shared_configs_resolve_paths_against_config_file(tmp_path):
    cfg = config()
    cfg['dataset_root'] = 'source'
    cfg['template']['path'] = 'models/example.json'
    path = tmp_path / 'local.yaml'
    path.write_text(yaml.safe_dump(cfg))
    resolved = load_config(path)
    assert resolved['dataset_root'] == str(tmp_path / 'source')
    assert resolved['template']['path'] == str(tmp_path / 'models/example.json')
    assert config(7)['layout']['joint_indices'][-1] == 14


@pytest.mark.parametrize('change', ['drop_joint', 'action', 'eef', 'other_dataset'])
def test_config_rejects_layout_or_feedback_substitution(change):
    cfg = config()
    if change == 'drop_joint':
        cfg['layout']['joint_indices'].pop()
    elif change == 'other_dataset':
        cfg['dataset_id'] = 'fold_and_hang'
    else:
        cfg['source']['feedback_field'] = change
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_nearest_sample_alignment_keeps_rows_signed_error_and_earlier_tie():
    selected, rows, errors = align_nearest([0, 10, 20], [[1], [2], [3]], [5, 11, 19], tolerance_ns=5)
    assert rows.tolist() == [0, 1, 2]
    assert errors.tolist() == [-5, -1, 1]
    assert selected[:, 0].tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match='gap exceeds'):
        align_nearest([0, 10], [[1], [2]], [16], tolerance_ns=5)
    with pytest.raises(ValueError, match='increasing'):
        align_nearest([0, 0], [[1], [2]], [0, 1])


def test_validation_preserves_actual_7axis_clock_and_requires_provenance():
    cfg, meta, arrays = fixture(7)
    validate_episode(meta, arrays, cfg)
    with pytest.raises(ValueError, match='provenance'):
        validate_episode(meta, {k: arrays[k] for k in ('joint', 'timestamps')}, cfg)
    broken = dict(arrays, feedback_time_errors_ns=np.full((80, 4), 50_000_001, dtype=np.int64))
    with pytest.raises(ValueError, match='alignment limit'):
        validate_episode(meta, broken, cfg)
    with pytest.raises(ValueError, match='source clock'):
        validate_episode(meta, dict(arrays, timestamps=arrays['timestamps'] * .99), cfg)


def test_validation_rejects_foreign_or_terminal_candidates():
    cfg, meta, arrays = fixture()
    candidate = {'node_id': 'n1', 'source_frame_id': 79, 'timestamp_s': 3.95, 'opening_arms': ['left']}
    with pytest.raises(ValueError, match='nonterminal'):
        validate_episode(dict(meta, candidates=[candidate]), arrays, cfg)
    candidate.update(source_frame_id=20, timestamp_s=1.)
    validate_episode(dict(meta, candidates=[candidate]), arrays, cfg)
    with pytest.raises(ValueError, match='unique identities'):
        validate_episode(dict(meta, candidates=[candidate, candidate]), arrays, cfg)
    with pytest.raises(ValueError, match='never EEF'):
        validate_episode(dict(meta, state_representation='eef'), arrays, cfg)


def test_prepare_preserves_frozen_candidates_with_provenance_and_protects_inputs(tmp_path):
    cfg, meta, arrays = fixture()
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'data').mkdir()
    np.savez(source / 'data' / 'episode.npz', **arrays)
    meta['dataset_npz'] = 'data/episode.npz'
    manifest = source / 'manifest.json'
    manifest.write_text(json.dumps([meta]))
    detector = source / 'detector_config.json'
    detector.write_text(json.dumps(cfg['detector']))
    before = manifest.read_bytes(), (source / meta['dataset_npz']).read_bytes()
    output = tmp_path / 'prepared'
    rows = prepare_run(cfg, manifest, output)
    assert rows[0]['candidates'] == []
    assert rows[0]['detector_sha256']
    assert (output / 'inputs' / 'fold_example.json').exists()
    assert before == (manifest.read_bytes(), (source / meta['dataset_npz']).read_bytes())
    with pytest.raises(ValueError, match='new directory'):
        prepare_run(cfg, manifest, output)
    with pytest.raises(ValueError, match='input run'):
        prepare_run(cfg, manifest, source / 'new')
    changed = dict(cfg['detector'], settled_samples=4)
    detector.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='detector configuration'):
        prepare_run(cfg, manifest, tmp_path / 'tampered')


def test_bare_dimensions_do_not_establish_measured_feedback():
    cfg, meta, arrays = fixture()
    del meta['state_representation']
    del meta['layout']
    with pytest.raises(ValueError, match='declare measured_joint'):
        validate_episode(meta, arrays, cfg)


def test_prepare_rejects_bad_key_before_creating_output(tmp_path):
    cfg = config(7)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'key': '../escape'}]))
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='plain filenames'):
        prepare_run(cfg, manifest, output)
    assert not output.exists()
