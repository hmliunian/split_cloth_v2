"""Portable offline entry point for the frozen v3 reverse-suffix model."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from collections import Counter
from pathlib import Path
import argparse
import json
import shutil
import time
import numpy as np
from model import Episode
from reverse_model import predict_reverse, stage_labels
from run_io import read, save, sha

CODE = Path(__file__).resolve().parent

def load_templates(path):
    bundle = read(path)
    if bundle.get('format') != 'joint_reverse_templates_v1' or bundle.get('state_dimensions') != 14:
        raise ValueError('Expected a joint_reverse_templates_v1 bundle for measured 14D joint state')
    if bundle.get('dataset') != 'fold_cloth_v3' or bundle.get('primary_initialization') != 0:
        raise ValueError('This frozen experiment supports fold_cloth_v3 with primary initialization 0')
    models = {}
    for init in [0, 1]:
        m = bundle['models'][str(init)]
        model = {**m, 'weights': np.array(m['weights'], dtype=float),
                 'templates': [None] + [np.array(t, dtype=float) for t in m['templates'][1:]]}
        dimensions = len(model['weights'])
        if dimensions != 100 or not np.isfinite(model['weights']).all() or np.any(model['weights'] <= 0):
            raise ValueError('Invalid feature weights')
        if len(model['templates']) != 5 or any(t.shape != (32, dimensions) or not np.isfinite(t).all() for t in model['templates'][1:]):
            raise ValueError('Invalid S2-S5 templates')
        if not np.isfinite(model['min_duration_s']) or model['min_duration_s'] <= 0:
            raise ValueError('Invalid minimum action duration')
        for stage in ['S2', 'S3', 'S4', 'S5']:
            limit = model['calibration'][stage]['threshold']
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError('Invalid action rejection threshold')
        models[init] = model
    if set(models[0]['training_keys']) != set(models[1]['training_keys']):
        raise ValueError('Initialization training sets differ')
    return bundle, models

def validate_episode(meta, arrays):
    if meta.get('dataset_id') != 'fold_cloth_v3' or meta.get('task_family', 'fold_cloth') != 'fold_cloth':
        raise ValueError('Only fold_cloth_v3 folding trajectories are supported')
    key = meta['key']
    if not key or Path(key).name != key or key in {'.', '..'}:
        raise ValueError('Episode keys must be plain filenames')
    q, times = arrays['joint'], arrays['timestamps']
    n = meta['frame_count']
    if n < 3 or q.shape != (n, 14) or times.shape != (n,):
        raise ValueError('Need one measured 14D joint row per original video frame')
    if not np.isfinite(q).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Nonfinite state or nonmonotonic timestamps')
    if abs(float(times[0])) > 1e-9 or not np.allclose(np.diff(times), .05, rtol=0, atol=1e-6):
        raise ValueError('The frozen model requires the original normalized 20 FPS video axis')
    frames, node_ids = [], []
    for c in meta['candidates']:
        frame = c['source_frame_id']
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < n - 1:
            raise ValueError('Boundaries must be real nonterminal candidate frames')
        if not c['opening_arms'] or not set(c['opening_arms']) <= {'left', 'right'}:
            raise ValueError('Candidates must be measured openings of either arm')
        if abs(c['timestamp_s'] - float(times[frame])) > .0025:
            raise ValueError('Candidate timestamps do not match the original video axis')
        frames.append(frame)
        node_ids.append(c['node_id'])
    if any(a >= b for a, b in zip(frames, frames[1:])) or len(set(node_ids)) != len(node_ids):
        raise ValueError('Candidates must have unique identities and strictly increasing frames')

def make_result(meta, timestamps, primary, alternate):
    frames = primary['action_frames'].copy()
    stable = frames[0] is not None and frames[0] == alternate['action_frames'][0]
    if not stable:
        frames[0] = None
    nodes = {c['source_frame_id']: c for c in meta['candidates']}
    boundaries = {f'b{i+1}': None if f is None else {
        'frame': f, 'timestamp_s': float(timestamps[f]), 'node_id': nodes[f]['node_id'],
        'source': 'reverse_joint_suffix_template', 'human_confirmed': False,
    } for i, f in enumerate(frames)}
    return {**primary, 'key': meta['key'], 'program_status': 'completed',
            'action_frames': frames, 'action_boundaries': boundaries,
            'display_boundaries': {f'b{i+2}': f for i, f in enumerate(frames[1:])},
            'b1_stable_between_initializations': bool(stable),
            'action_stage_5_status': 'inferred_complete_unconfirmed' if all(f is not None for f in frames) else 'incomplete',
            's1_subsegments': [], 's1_decomposition_complete': False,
            'alternative_initialization': {k: alternate[k] for k in ['action_frames', 'raw_action_frames', 'objective', 'matched_suffix_stages']},
            'human_confirmed': False, 'codex_review_status': 'not_requested'}

def infer_run(input_run, output_run, template_path):
    input_run, output_run, template_path = Path(input_run).resolve(), Path(output_run).resolve(), Path(template_path).resolve()
    if output_run.exists() or output_run.is_relative_to(input_run) or input_run.is_relative_to(output_run):
        raise ValueError('Output must be a new directory outside the read-only input run')
    bundle, models = load_templates(template_path)
    original_manifest = input_run / 'manifest.json'
    entries = read(original_manifest)
    if not entries or len({e['key'] for e in entries}) != len(entries):
        raise ValueError('Expected a nonempty manifest with distinct episodes')
    if {e['key'] for e in entries} & set(models[0]['training_keys']):
        raise ValueError('Inference episodes must not overlap the frozen training ten')
    prepared = []
    hashes = {str(original_manifest): sha(original_manifest), str(template_path): sha(template_path)}
    for rank, meta in enumerate(entries, 1):
        relative = Path(meta['dataset_npz'])
        source = (input_run / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(input_run):
            raise ValueError('dataset_npz must stay inside the prepared input run')
        with np.load(source, allow_pickle=False) as arrays:
            validate_episode(meta, arrays)
        hashes[str(source)] = sha(source)
        prepared.append(({**meta, 'sample_rank': rank, 'sample_count': len(entries),
                          'dataset_npz': f'data/{meta["key"]}.npz'}, source))
    output_run.mkdir(parents=True)
    for name in ['data', 'labels', 'videos', 'plots', 'source_snapshot']:
        (output_run / name).mkdir()
    shutil.copy2(template_path, output_run / 'templates.json')
    for path in CODE.iterdir():
        if path.suffix in {'.py', '.cpp'}:
            shutil.copy2(path, output_run / 'source_snapshot' / path.name)
    for meta, source in prepared:
        shutil.copy2(source, output_run / meta['dataset_npz'])
    save(output_run / 'manifest.json', [m for m, source in prepared])
    save(output_run / 'input_sha256.json', hashes)
    save(output_run / 'protocol.json', {
        'algorithm': 'reverse_ordered_suffix_dtw_v1', 'search_order': [5, 4, 3, 2],
        'prefix': 'free_unscored_preparation', 'flip_count_inferred': False,
        'templates_sha256': sha(template_path), 'training_keys': models[0]['training_keys'],
        'input_manifest_sha256': sha(original_manifest), 'new_codex_review': False,
        'external_vlm_calls': 0, 'accuracy': None,
        'source_sha256': {p.name: sha(p) for p in (output_run / 'source_snapshot').iterdir()},
    })
    results = {}
    start = time.monotonic()
    for meta, source in prepared:
        with np.load(output_run / meta['dataset_npz'], allow_pickle=False) as arrays:
            episode = Episode(meta, arrays)
        predictions = {i: predict_reverse(episode, model, model['calibration']) for i, model in models.items()}
        result = make_result(meta, episode.t, predictions[0], predictions[1])
        folder = output_run / 'labels' / episode.key
        folder.mkdir()
        for name, ends in [('action_stage_5', result['action_frames']), ('display_stage_4', result['display_frames'])]:
            np.save(folder / f'{name}.npy', stage_labels(episode.n, ends))
        save(folder / 'result.json', result)
        results[episode.key] = result
        print('reverse inferred', meta['sample_rank'], episode.key, result['action_frames'], flush=True)
    save(output_run / 'predictions.json', results)
    for name, digest in hashes.items():
        if sha(name) != digest:
            raise ValueError(f'Input changed during inference: {name}')
    paths = [output_run / 'predictions.json'] + [p for p in (output_run / 'labels').rglob('*') if p.is_file()]
    save(output_run / 'prediction_sha256.json', {str(p.relative_to(output_run)): sha(p) for p in paths})
    summary = {'episodes': len(entries), 'frames': sum(e['frame_count'] for e in entries),
               'candidates': sum(len(e['candidates']) for e in entries), 'elapsed_s': time.monotonic() - start,
               'status_counts': dict(Counter(p['status'] for p in results.values())),
               'five_stage_structural_complete': sum(all(f is not None for f in p['action_frames']) for p in results.values()),
               'new_codex_review': False, 'accuracy': None, 'external_vlm_calls': 0}
    save(output_run / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--templates', type=Path, default=CODE / 'templates_v3_ten.json')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    infer_run(args.input_run, args.run, args.templates)
    if args.render:
        from render import render_run
        render_run(args.run.resolve(), args.workers)

if __name__ == '__main__':
    main()
