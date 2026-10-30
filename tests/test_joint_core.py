from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from split_cloth_v2.io_utils import write_json
from split_cloth_v2.joint.config import load_config
from split_cloth_v2.joint.features import Episode, dense_features
from split_cloth_v2.joint.results import make_result, stage_labels
from split_cloth_v2.joint.search import predict_reverse
from split_cloth_v2.joint.templates import fit_templates, load_templates


def config(axis=7):
    name = 'fold_cloth_v3_pushcar_7axis.yaml' if axis == 7 else 'fold_cloth_v3_6axis.yaml'
    return load_config(Path(__file__).resolve().parents[1] / 'configs/joint' / name)


def episodes_and_references(cfg):
    times = np.arange(160) / 20
    episodes, references = [], {}
    for number in range(10):
        key = f'demo_{number}'
        joint = np.sin(times[:, None] * np.linspace(.3, 1.7, cfg['layout']['state_dimensions']))
        candidates = [{'node_id': f'n{i}', 'source_frame_id': frame,
                       'timestamp_s': float(times[frame]), 'opening_arms': ['left']}
                      for i, frame in enumerate([19, 49, 79, 109])]
        meta = {'key': key, 'episode_id': key, 'dataset_id': cfg['dataset_id'],
                'frame_count': len(times), 'candidates': candidates,
                'state_representation': 'measured_joint', 'layout': cfg['layout']}
        arrays = {'joint': joint, 'timestamps': times,
                  'source_video_clock_ns': np.rint(times * 1e9).astype(np.int64),
                  'feedback_sample_rows': np.tile(np.arange(len(times))[:, None], (1, 4)),
                  'feedback_time_errors_ns': np.zeros((len(times), 4), dtype=np.int64)}
        episodes.append(Episode(meta, arrays, cfg))
        references[key] = {'source': 'Codex', 'human_confirmed': False,
                           'selection_reason': 'clear complete folding demonstration',
                           'boundaries': {f'b{i + 1}': {'frame': node['source_frame_id'],
                                          'node_id': node['node_id'], 'timestamp_s': node['timestamp_s'],
                                          'reason': 'observed action transition', 'evidence': ['candidate-frame.jpg']}
                                          for i, node in enumerate(candidates)}}
    return episodes, references


@pytest.mark.parametrize('axis,dimensions', [(6, 100), (7, 116)])
def test_features_preserve_every_joint_and_original_time_axis(axis, dimensions):
    cfg = config(axis)
    times = np.arange(80) / 20
    joint = np.zeros((80, cfg['layout']['state_dimensions']))
    phi, blocks = dense_features(joint, times, cfg)
    assert phi.shape == (80, dimensions) and sum(blocks) == dimensions
    joint[:, cfg['layout']['joint_indices'][-1]] = np.linspace(0, 1, 80)
    changed, _ = dense_features(joint, times, cfg)
    assert not np.array_equal(phi, changed)


def test_fit_roundtrip_preserves_candidates_and_rejects_mismatched_templates(tmp_path):
    cfg = config()
    episodes, references = episodes_and_references(cfg)
    bundle = fit_templates(episodes, references, cfg)
    path = tmp_path / 'templates.json'
    write_json(path, bundle)
    loaded, models = load_templates(path, cfg)
    assert loaded['references'] == references
    assert models[0]['weights'].shape == (116,)
    predicted = predict_reverse(episodes[0], models[0], models[0]['calibration'])
    assert predicted['action_frames'] == [19, 49, 79, 109]
    mismatch = deepcopy(cfg)
    mismatch['features']['smoothing_s'] = .3
    with pytest.raises(ValueError, match='do not match'):
        load_templates(path, mismatch)
    mismatch = deepcopy(cfg)
    mismatch['alignment']['hard_limit_ms'] = 60
    with pytest.raises(ValueError, match='do not match'):
        load_templates(path, mismatch)
    with pytest.raises(ValueError, match='do not match'):
        load_templates(path, config(6))


def test_fit_excludes_unknown_intervals_and_never_synthesizes_references():
    cfg = config()
    episodes, references = episodes_and_references(cfg)
    references[episodes[0].key]['boundaries']['b1'] = None
    bundle = fit_templates(episodes, references, cfg)
    assert len(bundle['models']['0']['calibration']['S2']['residuals']) == 9
    assert bundle['references'][episodes[0].key]['boundaries']['b1'] is None
    for reference in references.values():
        reference['boundaries']['b1'] = None
    with pytest.raises(ValueError, match='no supported S2'):
        fit_templates(episodes, references, cfg)


def test_fit_rejects_duplicate_sources_and_moved_candidates():
    cfg = config()
    episodes, references = episodes_and_references(cfg)
    episodes[1].meta['episode_id'] = episodes[0].meta['episode_id']
    with pytest.raises(ValueError, match='distinct source'):
        fit_templates(episodes, references, cfg)
    episodes[1].meta['episode_id'] = episodes[1].key
    references[episodes[0].key]['boundaries']['b1']['frame'] += 1
    with pytest.raises(ValueError, match='existing nonterminal candidate'):
        fit_templates(episodes, references, cfg)


def test_acceptance_only_removes_b1_and_four_stage_labels_remain_usable():
    cfg = config()
    episodes, _ = episodes_and_references(cfg)
    episode = episodes[0]
    primary = {'action_frames': [19, 49, 79, 109], 'display_frames': [49, 79, 109]}
    alternate = {'action_frames': [None, 49, 79, 109], 'raw_action_frames': [None] * 4,
                 'objective': 0., 'matched_suffix_stages': [5, 4, 3]}
    result = make_result(episode.meta, episode.t, primary, alternate, 'dual_initialization_b1')
    assert result['action_frames'] == [None, 49, 79, 109]
    assert np.all(stage_labels(episode.n, result['action_frames']) == -1)
    assert set(stage_labels(episode.n, result['display_frames'])) == {1, 2, 3, 4}
    assert result['action_boundaries']['b2']['node_id'] == 'n1'
    assert result['missing_reasons']['b1'] == 'initializations_disagree'
