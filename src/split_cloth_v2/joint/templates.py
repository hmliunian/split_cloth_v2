"""Validated frozen templates and fitting from explicit candidate references."""
from copy import deepcopy
import numpy as np

from ..io_utils import fingerprint, read_json
from . import ALGORITHM
from .dtw import dtw


def template_contract(config):
    return {'layout': config['layout'], 'features': config['features'],
            'source': config['source'], 'alignment': config['alignment'],
            'detector_sha256': fingerprint(config['detector']),
            'acceptance_policy': config['template']['acceptance_policy'],
            'min_duration_s': config['template']['min_duration_s']}


def load_templates(path, config):
    bundle = read_json(path)
    if bundle.get('format') != 'joint_reverse_templates_v1' or bundle.get('algorithm') != ALGORITHM:
        raise ValueError('Expected a frozen reverse-suffix template bundle')
    if bundle.get('dataset') != config['dataset_id'] or bundle.get('contract') != template_contract(config):
        raise ValueError('Template layout, features, detector or acceptance policy do not match configuration')
    dimensions = 8 * len(config['layout']['joint_indices']) + 4
    if bundle.get('state_dimensions') != config['layout']['state_dimensions'] or bundle.get('primary_initialization') != 0:
        raise ValueError('Invalid template layout or primary initialization')
    models = {}
    for key, values in bundle['models'].items():
        model = {**values, 'weights': np.asarray(values['weights'], dtype=float),
                 'templates': [None] + [np.asarray(t, dtype=float) for t in values['templates'][1:]]}
        if model['weights'].shape != (dimensions,) or not np.isfinite(model['weights']).all() or np.any(model['weights'] <= 0):
            raise ValueError('Invalid feature weights')
        if len(model['templates']) != 5 or any(t.shape != (32, dimensions) or not np.isfinite(t).all() for t in model['templates'][1:]):
            raise ValueError('Invalid S2-S5 templates')
        if not np.isfinite(model['min_duration_s']) or model['min_duration_s'] != config['template']['min_duration_s']:
            raise ValueError('Invalid minimum action duration')
        for stage in range(2, 6):
            threshold = model['calibration'][f'S{stage}']['threshold']
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError('Invalid action rejection threshold')
        models[int(key)] = model
    expected = {0, 1} if config['template']['acceptance_policy'] == 'dual_initialization_b1' else {0}
    if set(models) != expected or any(set(m['training_keys']) != set(models[0]['training_keys']) for m in models.values()):
        raise ValueError('Template initializations or training sources are inconsistent')
    return bundle, models


def _reference_frames(episode, reference):
    if reference.get('source') != 'Codex' or reference.get('human_confirmed') is not False:
        raise ValueError('References must explicitly distinguish Codex observation from user confirmation')
    if not reference.get('selection_reason'):
        raise ValueError('Each demonstration needs its selection reason')
    nodes = {node['source_frame_id']: node for node in episode.meta['candidates']}
    frames = []
    for slot in ['b1', 'b2', 'b3', 'b4']:
        value = reference['boundaries'][slot]
        if value is None:
            frames.append(None)
            continue
        frame = value['frame']
        if isinstance(frame, bool) or not isinstance(frame, int) or frame not in nodes:
            raise ValueError('Reference boundary must be an existing nonterminal candidate')
        if value.get('node_id') != nodes[frame]['node_id'] or not np.isfinite(value['timestamp_s']) or abs(value['timestamp_s'] - episode.t[frame]) > 1e-6:
            raise ValueError('Reference candidate identity or timestamp differs from frozen input')
        if not value.get('reason') or not value.get('evidence'):
            raise ValueError('Each reference boundary needs an observation reason and evidence')
        frames.append(frame)
    known = [frame for frame in frames if frame is not None]
    if any(a >= b for a, b in zip(known, known[1:])):
        raise ValueError('Reference boundary slots must be strictly increasing')
    return frames


def fit_templates(episodes, references, config):
    if config['template']['acceptance_policy'] != 'explicit_boundaries':
        raise ValueError('Fitting requires the explicit-boundary acceptance policy')
    if len(episodes) != 10 or len({e.key for e in episodes}) != 10 or set(references) != {e.key for e in episodes}:
        raise ValueError('Template fitting requires exactly ten distinct demonstrations and matching references')
    if any(e.meta['dataset_id'] != config['dataset_id'] or template_contract(e.config) != template_contract(config) for e in episodes):
        raise ValueError('Demonstration layout or feature configuration does not match fitting configuration')
    # Source identities prevent two exports of the same recorded episode counting twice.
    identities = [e.meta.get('source_mcap_sha256', e.meta.get('episode_id', e.key)) for e in episodes]
    if len(set(identities)) != 10:
        raise ValueError('Demonstrations must come from ten distinct source episodes')
    frames = {e.key: _reference_frames(e, references[e.key]) for e in episodes}
    features = np.concatenate([e.phi for e in episodes])
    block_sizes = episodes[0].blocks
    block_scale = np.concatenate([np.full(size, np.sqrt(size * 10)) for size in block_sizes])
    weights = 1 / np.maximum(features.std(axis=0), .2) / block_scale
    minimum = config['template']['min_duration_s']
    samples = {stage: [] for stage in range(2, 6)}
    for episode in episodes:
        intervals = episode.intervals(weights, minimum)
        ends = [*frames[episode.key], episode.n - 1]
        indices = {int(frame): i for i, frame in enumerate(episode.v)}
        for stage in range(2, 6):
            start, end = ends[stage - 2:stage]
            if start is None or end is None:
                continue
            interval = intervals.get((indices[start], indices[end]))
            if interval is None:
                raise ValueError(f'{episode.key} S{stage}: reference interval is too short to score')
            samples[stage].append({'key': episode.key, 'start_frame': start + 1, 'end_frame': end,
                                   'duration_s': float(interval['duration']), 'seq': interval['seq']})
    templates, calibration, provenance = [None], {}, {}
    for stage, rows in samples.items():
        if not rows:
            raise ValueError(f'Template building incomplete: no supported S{stage} reference intervals')
        median = np.median(np.stack([row['seq'] for row in rows]), axis=0)
        chosen = min(rows, key=lambda row: np.mean((row['seq'] - median) ** 2))
        templates.append(chosen['seq'].tolist())
        residuals = [{**{key: value for key, value in row.items() if key != 'seq'},
                      'mean_dtw': dtw(row['seq'], chosen['seq']), 'origin': 'Codex_candidate_reference'} for row in rows]
        calibration[f'S{stage}'] = {'threshold': float(2 * np.quantile([row['mean_dtw'] for row in residuals], .95) + .05),
                                    'residuals': residuals, 'rule': 'training residual P95 * 2 + 0.05'}
        provenance[f'S{stage}'] = {key: value for key, value in chosen.items() if key != 'seq'}
    return {'format': 'joint_reverse_templates_v1', 'algorithm': ALGORITHM,
            'dataset': config['dataset_id'], 'state_dimensions': config['layout']['state_dimensions'],
            'contract': deepcopy(template_contract(config)), 'primary_initialization': 0,
            'human_confirmed': False, 'references': deepcopy(references), 'references_sha256': fingerprint(references),
            'training_candidates_sha256': fingerprint({e.key: e.meta['candidates'] for e in episodes}),
            'feature_sampling': {e.key: e.feature_sampling for e in episodes},
            'training_sources': [deepcopy(e.meta) for e in episodes],
            'models': {'0': {'weights': weights.tolist(), 'templates': templates,
                             'min_duration_s': minimum, 'training_keys': [e.key for e in episodes],
                             'initialization': 0, 'calibration': calibration, 'template_sources': provenance}}}
