"""Explicit measured-feedback layouts and shared algorithm configuration."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

DATASETS = {'fold_cloth_v3': 6, 'fold_cloth_v3_pushcar_batch_20260923': 7}


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    validate_config(config)
    config = deepcopy(config)
    config['dataset_root'] = str((path.parent / config['dataset_root']).resolve())
    if config['template'].get('path'):
        config['template']['path'] = str((path.parent / config['template']['path']).resolve())
    return config


def validate_config(config):
    if not isinstance(config, dict) or config.get('dataset_id') not in DATASETS:
        raise ValueError('Only registered fold_cloth measured-feedback datasets are supported')
    n = DATASETS[config['dataset_id']]
    layout = config.get('layout', {})
    expected = {'state_dimensions': 2 * (n + 1), 'joints_per_arm': n,
                'joint_indices': list(range(n)) + list(range(n + 1, 2 * n + 1)),
                'gripper_indices': {'left': n, 'right': 2 * n + 1}}
    if any(layout.get(key) != value for key, value in expected.items()):
        raise ValueError('Layout must declare every measured joint and both gripper columns in source order')
    if layout.get('feedback') != 'measured_joint' or layout.get('joint_unit') != 'radian':
        raise ValueError('Only measured joint feedback in radians is supported')
    if not layout.get('gripper_unit'):
        raise ValueError('Gripper units must be declared')
    names = layout.get('joint_names', {})
    if any(len(names.get(arm, [])) != n or len(set(names[arm])) != n for arm in ('left', 'right')):
        raise ValueError('Each arm must declare distinct joint names in source order')
    if not config.get('dataset_root') or config.get('reader') not in {'prepared_npz', 'mcap'}:
        raise ValueError('A dataset root and supported reader are required')
    source = config.get('source', {})
    if source.get('feedback_field') != 'joint' or not source.get('camera'):
        raise ValueError('Measured feedback field and source camera must be explicit')
    detector = config.get('detector', {})
    if detector.get('state_dimensions') != expected['state_dimensions'] or detector.get('state_indices') != expected['gripper_indices']:
        raise ValueError('Detector and measured layout must agree')
    if detector.get('node_policy') != 'any_open' or set(detector.get('opening_directions', {})) != {'left', 'right'}:
        raise ValueError('Detector must declare independent openings of either arm')
    if any(v not in {'higher', 'lower'} for v in detector['opening_directions'].values()):
        raise ValueError('Invalid gripper opening direction')
    features = config.get('features', {})
    if features.get('progress_points') != 32 or features.get('window_s') != [.5, 1.]:
        raise ValueError('Shared feature contract requires 32 progress points and 0.5/1 second windows')
    if any(not isinstance(features.get(k), (float, int)) or not np.isfinite(features[k]) or features[k] <= 0 for k in ('smoothing_s', 'sample_hz')):
        raise ValueError('Feature time scales must be finite and positive')
    template = config.get('template', {})
    if template.get('acceptance_policy') not in {'dual_initialization_b1', 'explicit_boundaries'}:
        raise ValueError('An explicit template acceptance policy is required')
    duration = template.get('min_duration_s')
    if not isinstance(duration, (int, float)) or not np.isfinite(duration) or duration <= 0:
        raise ValueError('Minimum action duration must be finite and positive')
    alignment = config.get('alignment', {})
    if alignment.get('method') not in {'frozen_original_axis', 'nearest_measured_sample'}:
        raise ValueError('Unsupported measured-feedback alignment policy')
    if config['reader'] == 'mcap':
        if alignment.get('method') != 'nearest_measured_sample' or alignment.get('ties') != 'earlier':
            raise ValueError('MCAP alignment must choose nearest recorded samples, ties earlier')
        hard, report = alignment.get('hard_limit_ms'), alignment.get('report_above_ms')
        if not isinstance(hard, (int, float)) or not isinstance(report, (int, float)) or not 0 <= report <= hard or not np.isfinite(hard):
            raise ValueError('Invalid feedback alignment limits')
        topics = source.get('feedback_topics', {})
        required = {f'{arm}_{part}' for arm in ('left', 'right') for part in ('arm', 'gripper')}
        if set(topics) != required or any(not topic.startswith('/observation/robot/') for topic in topics.values()):
            raise ValueError('Four execution-robot measured feedback topics are required')
