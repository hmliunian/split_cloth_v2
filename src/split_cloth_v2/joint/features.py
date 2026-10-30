"""Shared joint features and complete candidate-to-candidate intervals."""
import numpy as np
from scipy.ndimage import uniform_filter1d


def norm(values, floor):
    low, high = np.quantile(values, [.05, .95], axis=0)
    return (values - low) / np.maximum(high - low, floor)


def _smoothing_samples(timestamps, config):
    return max(1, round(config['features']['smoothing_s'] / float(np.median(np.diff(timestamps)))))


def dense_features(joint, timestamps, config):
    layout, settings = config['layout'], config['features']
    pos = np.unwrap(joint[:, layout['joint_indices']], axis=0)
    grip = np.clip(norm(joint[:, [layout['gripper_indices'][arm] for arm in ['left', 'right']]], 1e-4), 0, 1)
    # Round the declared duration to samples on this original video axis.
    # At 20 Hz, 0.25 s exactly reproduces the frozen five-sample filter.
    samples = _smoothing_samples(timestamps, config)
    velocity = np.gradient(uniform_filter1d(pos, size=samples, axis=0, mode='nearest'),
                           timestamps, axis=0)
    blocks = [np.sin(pos), np.cos(pos), norm(pos, .2), grip, velocity,
              np.gradient(grip, timestamps, axis=0)]
    for seconds in settings['window_s']:
        for sign in [-1, 1]:
            target = np.clip(timestamps + sign * seconds, timestamps[0], timestamps[-1])
            shifted = np.column_stack([np.interp(target, timestamps, pos[:, k])
                                       for k in range(pos.shape[1])])
            blocks.append(sign * (shifted - pos))
    return np.column_stack(blocks), [block.shape[1] for block in blocks]


class Episode:
    def __init__(self, meta, arrays, config):
        from .data import validate_episode
        validate_episode(meta, arrays, config)
        self.meta, self.key, self.config = meta, meta['key'], config
        self.t = np.asarray(arrays['timestamps'], dtype=float)
        self.n = len(self.t)
        self.end_t = self.t[-1] + np.median(np.diff(self.t))
        self.frames = np.array([c['source_frame_id'] for c in meta['candidates']], dtype=int)
        self.v = np.r_[-1, self.frames, self.n - 1]
        self.phi, self.blocks = dense_features(np.asarray(arrays['joint']), self.t, config)
        self.feature_sampling = {'smoothing_samples': _smoothing_samples(self.t, config),
                                 'median_frame_interval_s': float(np.median(np.diff(self.t))),
                                 'rule': 'max(1, round(smoothing_s / median_frame_interval_s)); nearest-even rounding',
                                 'interval_sampling_hz': config['features']['sample_hz']}

    def intervals(self, weights, min_duration):
        grid = np.arange(self.t[0], self.end_t, 1 / self.config['features']['sample_hz'])
        sampled = np.column_stack([np.interp(grid, self.t, self.phi[:, k])
                                   for k in range(self.phi.shape[1])]) * weights
        endpoints = np.r_[self.t[self.v[:-1] + 1], self.end_t]
        bins = np.searchsorted(grid, endpoints, side='left')
        bins[-1] = len(grid)
        segments = {}
        for i in range(1, len(self.v) - 1):
            for j in range(i + 1, len(self.v)):
                duration = endpoints[j] - endpoints[i]
                count = int(bins[j] - bins[i])
                if duration + 1e-9 < min_duration or count < 2:
                    continue
                data = sampled[bins[i]:bins[j]]
                progress = np.linspace(0, len(data) - 1, 32)
                seq = np.column_stack([np.interp(progress, np.arange(len(data)), data[:, k])
                                       for k in range(data.shape[1])])
                segments[i, j] = {'seq': seq, 'count': count, 'duration': duration}
        return segments
