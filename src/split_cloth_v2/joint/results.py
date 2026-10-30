"""Fixed boundary slots, explicit acceptance and original-frame labels."""
import numpy as np


def stage_labels(frame_count, boundaries):
    labels = np.full(frame_count, -1, dtype=np.int16)
    known = [value for value in boundaries if value is not None]
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or
           not 0 <= value < frame_count - 1 for value in known):
        raise ValueError('Boundaries must be nonterminal integer frames')
    if any(a >= b for a, b in zip(known, known[1:])):
        raise ValueError('Boundary slots must be strictly increasing')
    if len(known) == len(boundaries):
        for stage, (a, b) in enumerate(zip([-1, *boundaries], [*boundaries, frame_count - 1]), 1):
            labels[a + 1:b + 1] = stage
    return labels


def make_result(meta, timestamps, primary, alternate=None, acceptance_policy='explicit_boundaries'):
    frames = primary['action_frames'].copy()
    stable = None
    if acceptance_policy == 'dual_initialization_b1':
        if alternate is None:
            raise ValueError('Dual initialization acceptance requires the alternate prediction')
        stable = frames[0] is not None and frames[0] == alternate['action_frames'][0]
        if not stable:
            frames[0] = None
    elif acceptance_policy != 'explicit_boundaries':
        raise ValueError('Unknown template acceptance policy')
    if len(frames) != 4:
        raise ValueError('Expected fixed b1-b4 slots')
    stage_labels(meta['frame_count'], frames)
    nodes = {node['source_frame_id']: node for node in meta['candidates']}
    if any(frame is not None and frame not in nodes for frame in frames):
        raise ValueError('Predicted boundary is not a frozen candidate')
    reasons = {f'b{i + 1}': None if frame is not None else
               ('initializations_disagree' if i == 0 and stable is False and
                primary['action_frames'][0] is not None else 'no_accepted_ordered_suffix')
               for i, frame in enumerate(frames)}
    boundaries = {f'b{i + 1}': None if frame is None else {
        'frame': frame, 'timestamp_s': float(timestamps[frame]), 'node_id': nodes[frame]['node_id'],
        'source': 'reverse_joint_suffix_template', 'human_confirmed': False,
    } for i, frame in enumerate(frames)}
    return {**primary, 'key': meta['key'], 'program_status': 'completed',
            'action_frames': frames, 'display_frames': frames[1:],
            'action_boundaries': boundaries,
            'display_boundaries': {f'b{i + 2}': frame for i, frame in enumerate(frames[1:])},
            'missing_reasons': reasons, 'acceptance_policy': acceptance_policy,
            'b1_stable_between_initializations': stable,
            'action_stage_5_status': 'inferred_complete_unconfirmed' if all(frame is not None for frame in frames) else 'incomplete',
            'alternative_initialization': None if alternate is None else {
                key: alternate[key] for key in ['action_frames', 'raw_action_frames', 'objective', 'matched_suffix_stages']},
            'human_confirmed': False, 'codex_review_status': 'not_requested'}
