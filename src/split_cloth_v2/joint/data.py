"""Validate and freeze measured trajectories on their original video axis."""
from pathlib import Path

import numpy as np

from split_cloth_v2.detector import detect_candidates
from split_cloth_v2.io_utils import fingerprint, read_json, sha256_file, write_json
from .config import validate_config


def validate_episode(meta, arrays, config):
    layout = config['layout']
    if meta.get('dataset_id') != config['dataset_id'] or meta.get('task_family', 'fold_cloth') != 'fold_cloth':
        raise ValueError('Only configured fold_cloth measured-feedback trajectories are supported')
    key = meta.get('key')
    if not isinstance(key, str) or not key or Path(key).name != key or key in {'.', '..'}:
        raise ValueError('Episode keys must be plain filenames')
    source = meta.get('joint_source', {})
    legacy_source = (config['dataset_id'] == 'fold_cloth_v3' and bool(source.get('path'))
                     and bool(source.get('sha256')) and source.get('dimensions') == layout['state_dimensions']
                     and source.get('joint_count_per_arm') == layout['joints_per_arm'])
    declared_source = meta.get('state_representation') == 'measured_joint' and meta.get('layout') == layout
    if meta.get('state_representation', 'measured_joint') != 'measured_joint' or not (legacy_source or declared_source):
        raise ValueError('Input must declare measured_joint feedback, never EEF or action')
    if meta.get('state_dimensions', layout['state_dimensions']) != layout['state_dimensions']:
        raise ValueError('Input layout does not match configured measured feedback')
    if source and (source.get('dimensions') != layout['state_dimensions'] or source.get('joint_count_per_arm') != layout['joints_per_arm']):
        raise ValueError('Frozen joint source mapping does not match configured layout')
    if 'layout' in meta and meta['layout'] != layout:
        raise ValueError('Input joint layout does not match configuration')
    q, times = np.asarray(arrays['joint']), np.asarray(arrays['timestamps'])
    n = meta.get('frame_count')
    if isinstance(n, bool) or not isinstance(n, int) or n < 3 or q.shape != (n, layout['state_dimensions']) or times.shape != (n,):
        raise ValueError('Need one measured joint row per original video frame')
    if not np.isfinite(q).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Nonfinite state or nonmonotonic timestamps')
    if abs(float(times[0])) > 1e-9:
        raise ValueError('Original video timestamps must be normalized to their first frame')
    fps = config['alignment'].get('frozen_fps')
    if fps and not np.allclose(np.diff(times), 1 / fps, rtol=0, atol=1e-6):
        raise ValueError(f'The frozen model requires the original {fps:g} FPS video axis')
    frames, identities = [], []
    for candidate in meta.get('candidates', []):
        frame = candidate.get('source_frame_id')
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < n - 1:
            raise ValueError('Boundaries must be real nonterminal candidate frames')
        arms = candidate.get('opening_arms', [])
        if not arms or len(set(arms)) != len(arms) or not set(arms) <= {'left', 'right'}:
            raise ValueError('Candidates must be measured openings of either arm')
        stamp = candidate.get('timestamp_s')
        if not isinstance(stamp, (float, int)) or not np.isfinite(stamp) or abs(stamp - times[frame]) > .0025:
            raise ValueError('Candidate timestamps do not match the original video axis')
        identity = candidate.get('node_id')
        if not isinstance(identity, str) or not identity:
            raise ValueError('Candidate identities must be nonempty strings')
        frames.append(frame)
        identities.append(identity)
    if any(a >= b for a, b in zip(frames, frames[1:])) or len(set(identities)) != len(identities):
        raise ValueError('Candidates must have unique identities and strictly increasing frames')
    if 'detector_sha256' in meta and meta['detector_sha256'] != fingerprint(config['detector']):
        raise ValueError('Frozen candidate detector configuration does not match')
    if 'source_frame_ids' in arrays and not np.array_equal(arrays['source_frame_ids'], np.arange(n)):
        raise ValueError('Original source frames must remain consecutive and ordered')
    alignment_fields = {'feedback_sample_rows', 'feedback_time_errors_ns', 'source_video_clock_ns'}
    if config['reader'] == 'mcap' and not alignment_fields <= set(arrays):
        raise ValueError('MCAP inputs must retain per-frame measured feedback alignment provenance')
    if 'feedback_sample_rows' in arrays:
        rows = np.asarray(arrays['feedback_sample_rows'])
        errors = np.asarray(arrays['feedback_time_errors_ns'])
        clock = np.asarray(arrays['source_video_clock_ns'])
        if rows.shape != (n, 4) or errors.shape != (n, 4) or clock.shape != (n,):
            raise ValueError('Feedback alignment must retain four source samples per original frame')
        if not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0) or np.any(np.diff(rows, axis=0) < 0):
            raise ValueError('Invalid measured feedback source rows')
        if not np.issubdtype(errors.dtype, np.integer) or np.any(np.diff(clock) <= 0):
            raise ValueError('Invalid measured feedback source clock/errors')
        if not np.allclose((clock - clock[0]) / 1e9, times, rtol=0, atol=1e-9):
            raise ValueError('Video timestamps disagree with original source clock')
        limit = config['alignment'].get('hard_limit_ms', 50) * 1e6
        if np.max(np.abs(errors)) > limit:
            raise ValueError('Measured feedback/video gap exceeds configured alignment limit')
        if 'feedback_source_timestamps_ns' in arrays and not np.array_equal(arrays['feedback_source_timestamps_ns'], clock[:, None] + errors):
            raise ValueError('Measured source timestamps disagree with alignment errors')


def _candidate_identity(candidates):
    return [{k: c[k] for k in ('node_id', 'source_frame_id', 'timestamp_s', 'opening_arms')} for c in candidates]


def prepare_run(config, input_manifest_path, output_run):
    """Freeze a JSON list of prepared-NPZ or MCAP sources into a new run."""
    validate_config(config)
    manifest = Path(input_manifest_path).resolve()
    if manifest.is_dir():
        manifest /= 'manifest.json'
    output = Path(output_run).resolve()
    source_root = Path(config['dataset_root']).resolve()
    if output.exists() or output.is_relative_to(source_root) or source_root.is_relative_to(output) or manifest.is_relative_to(output):
        raise ValueError('Output must be a new directory outside the read-only source dataset')
    if output.is_relative_to(manifest.parent) and (manifest.parent / 'data').is_dir():
        raise ValueError('Output must be a new directory outside the read-only input run')
    entries = read_json(manifest)
    if not isinstance(entries, list) or not entries or len({e['key'] for e in entries}) != len(entries):
        raise ValueError('Expected a nonempty manifest with distinct episodes')
    for entry in entries:
        key = entry['key']
        if not isinstance(key, str) or not key or Path(key).name != key or key in {'.', '..'}:
            raise ValueError('Episode keys must be plain filenames')
        if entry.get('dataset_id', config['dataset_id']) != config['dataset_id'] or entry.get('task_family', 'fold_cloth') != 'fold_cloth':
            raise ValueError('Only configured fold_cloth sources are supported')
        fields = ('source_mcap', 'source_metadata') if config['reader'] == 'mcap' else ('dataset_npz',)
        if any((manifest.parent / entry[field]).resolve().is_relative_to(output) for field in fields):
            raise ValueError('Output cannot contain or replace an input source')
    output.mkdir(parents=True)
    for folder in ('data', 'inputs'):
        (output / folder).mkdir()
    write_json(output / 'config.json', config)
    write_json(output / 'detector_config.json', config['detector'])
    inputs = {str(manifest): sha256_file(manifest)}
    results = []
    for entry in entries:
        if config['reader'] == 'mcap':
            from .adapters.mcap import read_episode
            meta = dict(entry)
            for field in ('source_mcap', 'source_metadata'):
                meta[field] = str((manifest.parent / meta[field]).resolve())
                inputs[meta[field]] = sha256_file(meta[field])
            meta, arrays = read_episode(meta, config, output)
        else:
            source = (manifest.parent / entry['dataset_npz']).resolve()
            if not source.is_relative_to(manifest.parent):
                raise ValueError('dataset_npz must stay inside the read-only input run')
            inputs[str(source)] = sha256_file(source)
            with np.load(source, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            meta = dict(entry)
            validate_episode(meta, arrays, config)
            if meta.get('source_video_path'):
                meta['source_video_path'] = str((manifest.parent / meta['source_video_path']).resolve())
        times = arrays['timestamps']
        if config['reader'] == 'prepared_npz' and 'candidates' in meta:
            detector_path = manifest.parent / 'detector_config.json'
            if detector_path.exists():
                frozen = read_json(detector_path)
                frozen.setdefault('state_dimensions', config['layout']['state_dimensions'])
                frozen.setdefault('state_indices', config['layout']['gripper_indices'])
                if fingerprint(frozen) != fingerprint(config['detector']):
                    raise ValueError('Frozen candidate detector configuration does not match')
                inputs[str(detector_path)] = sha256_file(detector_path)
            elif meta.get('detector_sha256') != fingerprint(config['detector']):
                raise ValueError('Frozen candidates require detector configuration provenance')
            candidates = meta['candidates']
            detection = {'candidates': candidates, 'candidate_import': 'frozen_prepared_input',
                         'source_manifest_sha256': inputs[str(manifest)],
                         'detector_sha256': fingerprint(config['detector'])}
        else:
            detection = detect_candidates(arrays['joint'].tolist(), times.tolist(), list(range(len(times))),
                                          source_timestamps=times.tolist(), config=config['detector'])
            candidates = [c for c in detection['candidates'] if c['source_frame_id'] < len(times) - 1]
            if 'candidates' in meta and _candidate_identity(meta['candidates']) != _candidate_identity(candidates):
                raise ValueError(f'Frozen candidates differ from configured detector: {meta["key"]}')
        meta.update({'candidates': meta.get('candidates', candidates), 'layout': config['layout'],
                     'state_representation': 'measured_joint', 'state_dimensions': config['layout']['state_dimensions'],
                     'detector_sha256': fingerprint(config['detector']), 'dataset_npz': f'data/{meta["key"]}.npz'})
        arrays['source_frame_ids'] = np.arange(len(times), dtype=np.int64)
        validate_episode(meta, arrays, config)
        np.savez_compressed(output / meta['dataset_npz'], **arrays)
        write_json(output / 'inputs' / f'{meta["key"]}.json', {'detection': detection, 'meta': meta})
        results.append(meta)
    for path, digest in inputs.items():
        if sha256_file(path) != digest:
            raise ValueError(f'Input changed during preparation: {path}')
    write_json(output / 'manifest.json', results)
    write_json(output / 'input_sha256.json', inputs)
    write_json(output / 'prepare_summary.json', {'episodes': len(results), 'frames': sum(m['frame_count'] for m in results),
               'candidates': sum(len(m['candidates']) for m in results), 'config_sha256': fingerprint(config),
               'detector_sha256': sha256_file(Path(__file__).parents[1] / 'detector.py'),
               'accuracy': None, 'external_vlm_calls': 0})
    return results
