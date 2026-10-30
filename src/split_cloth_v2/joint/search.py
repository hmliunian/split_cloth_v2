"""Candidate-constrained reverse matching: S5 -> S4 -> S3 -> S2.

S1 is the remaining free prefix. No flip template, repeat count, random
forest head, video pixels, or absolute position-in-video prior is scored.
The original forward-time feature sequences and templates are preserved;
only candidate search proceeds backward.
"""
import numpy as np
from .dtw import dtw


def best_two(items):
    output, seen = [], set()
    for cost, path in sorted(items, key=lambda item: item[0]):
        token = tuple(path)
        if np.isfinite(cost) and token not in seen:
            seen.add(token)
            output.append((float(cost), path))
            if len(output) == 2:
                break
    return output

STAGES = (5, 4, 3, 2)

def reverse_decode(costs):
    """Exact best-two paths for the longest feasible ordered suffix.

costs[stage-2, i, j] scores frames (v[i], v[j]]. v[0] is the
start sentinel, v[-1] the video end; i must be a real candidate.
Two best suffixes per start candidate suffice for exact top-two decoding.
"""
    if costs.ndim != 3 or costs.shape[0] != 4 or costs.shape[1] != costs.shape[2] or costs.shape[1] < 2:
        raise ValueError('Expected a 4-by-K-by-K interval cost matrix')
    size = costs.shape[1]
    states = {size - 1: [(0.0, [])]}
    layers = []
    completed = []
    for stage in STAGES:
        next_states = {}
        for i in range(size - 2, 0, -1):
            choices = []
            for j, suffixes in states.items():
                if i >= j or not np.isfinite(costs[stage - 2, i, j]):
                    continue
                for score, path in suffixes:
                    choices.append((score + costs[stage - 2, i, j],
                                    path + [(stage, i, j)]))
            choices = best_two(choices)
            if choices:
                next_states[i] = choices
        layers.append({'stage': f'S{stage}', 'feasible_start_candidates': len(next_states),
                       'retained_paths': sum(len(v) for v in next_states.values())})
        if not next_states:
            break
        states = next_states
        completed.append(stage)
    solutions = best_two([p for paths in states.values() for p in paths]) if completed else []
    return solutions, layers, completed

def segment_costs(episode, model):
    segments = episode.intervals(model['weights'], model['min_duration_s'])
    costs = np.full((4, len(episode.v), len(episode.v)), np.inf)
    for (i, j), segment in segments.items():
        for stage in range(2, 6):
            costs[stage - 2, i, j] = dtw(segment['seq'], model['templates'][stage - 1])
    return segments, costs

def path_frames(episode, path):
    frames = [None] * 4
    for stage, i, j in path:
        frames[stage - 2] = int(episode.v[i])
    return frames

def predict_reverse(episode, model, calibration):
    segments, costs = segment_costs(episode, model)
    raw, raw_layers, _ = reverse_decode(costs)
    allowed = costs.copy()
    for stage in range(2, 6):
        allowed[stage - 2, costs[stage - 2] > calibration[f'S{stage}']['threshold']] = np.inf
    accepted, layers, completed = reverse_decode(allowed)
    chosen = accepted[0][1] if accepted else []
    diagnostic = raw[0][1] if raw else []
    frames = path_frames(episode, chosen)
    steps = []
    for stage, i, j in chosen:
        steps.append({'stage': f'S{stage}', 'start_candidate_frame': int(episode.v[i]),
                      'start_frame': int(episode.v[i] + 1), 'end_frame': int(episode.v[j]),
                      'mean_dtw': float(costs[stage - 2, i, j]),
                      'threshold': calibration[f'S{stage}']['threshold'],
                      'duration_s': segments[i, j]['duration'], 'human_confirmed': False})
    raw_segments = []
    if diagnostic:
        prefix_end = diagnostic[-1][1]
        raw_segments.append({'stage': 'PREP', 'start_frame': 0, 'end_frame': int(episode.v[prefix_end]),
                             'scored': False, 'meaning': 'free prefix; no flip-count inference'})
        for stage, i, j in reversed(diagnostic):
            raw_segments.append({'stage': f'A{stage}', 'start_frame': int(episode.v[i] + 1),
                                 'end_frame': int(episode.v[j]), 'mean_cost': float(costs[stage - 2, i, j]),
                                 'duration_s': segments[i, j]['duration']})
    status = ('inferred_needs_review' if len(completed) == 4 else
              'partial_suffix_needs_review' if completed else 'no_accepted_suffix')
    return {'status': status, 'action_frames': frames, 'display_frames': frames[1:],
            'raw_action_frames': path_frames(episode, diagnostic),
            'objective': accepted[0][0] if accepted else None,
            'alternative_objective': accepted[1][0] if len(accepted) > 1 else None,
            'raw_objective': raw[0][0] if raw else None,
            'reverse_search_order': list(STAGES), 'matched_suffix_stages': completed,
            'reverse_search_layers': layers, 'raw_search_layers': raw_layers,
            'reverse_steps': steps, 'segments': raw_segments,
            'paths': [{'objective': value, 'steps': path} for value, path in accepted],
            'raw_paths': [{'objective': value, 'steps': path} for value, path in raw],
            's1_mode': 'free_unscored_prefix', 's1_repeat_count': None,
            'label_source': 'reverse_joint_template_inference',
            'human_confirmed': False, 'codex_review_status': 'not_requested',
            'stage5_terminal_status': 'not_evaluated'}
