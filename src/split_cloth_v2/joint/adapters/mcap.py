"""MCAP measured feedback reader with explicit nearest-sample alignment."""
from fractions import Fraction
from pathlib import Path
import tomllib

import av
import numpy as np

from split_cloth_v2.io_utils import read_json, sha256_file
from .mcap_io import summary, messages, scalar, field, indirect, text_field, joint_values, timestamp


def align_nearest(clock, values, video_clock, tolerance_ns=50_000_000):
    """Choose recorded samples, retaining row identity and signed clock error."""
    clock, video_clock = np.asarray(clock, dtype=np.int64), np.asarray(video_clock, dtype=np.int64)
    q = np.asarray(values, dtype=float)
    if not np.isfinite(tolerance_ns) or tolerance_ns < 0:
        raise ValueError('Alignment tolerance must be finite and nonnegative')
    if clock.ndim != 1 or video_clock.ndim != 1 or len(clock) < 1 or len(video_clock) < 1:
        raise ValueError('Nonempty measured and video clocks are required')
    if np.any(np.diff(clock) <= 0) or np.any(np.diff(video_clock) <= 0):
        raise ValueError('Strictly increasing measured and video timestamps required')
    if q.ndim != 2 or len(q) != len(clock) or not np.isfinite(q).all():
        raise ValueError('Finite measured values must match their source clock')
    after = np.searchsorted(clock, video_clock).clip(0, len(clock) - 1)
    before = (after - 1).clip(0)
    rows = np.where(np.abs(clock[before] - video_clock) <= np.abs(clock[after] - video_clock), before, after)
    errors = clock[rows] - video_clock
    if np.max(np.abs(errors)) > tolerance_ns:
        raise ValueError(f'Measured feedback/video gap exceeds {tolerance_ns / 1e6:g} ms')
    return q[rows], rows, errors


def _video(payloads, clock, destination):
    """Decode every source presentation frame and preserve its relative PTS."""
    axis = clock - clock[0]
    temporary = destination.with_suffix('.partial.mp4')
    if destination.exists() or temporary.exists():
        raise ValueError('Source-video conversion output must be new')
    decoder = av.CodecContext.create('h264', 'r')
    decoded = 0
    with av.open(str(temporary), 'w', options={'movflags': '+faststart'}) as target:
        rate = Fraction(1_000_000_000, int(np.median(np.diff(clock)))).limit_denominator(10000)
        stream = target.add_stream('libx264', rate=rate)
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1_000_000)
        stream.codec_context.thread_count = 2
        stream.options = {'crf': '18', 'preset': 'veryfast'}

        def emit(frame):
            nonlocal decoded
            if decoded >= len(clock) or frame.pts != round(int(axis[decoded]) / 1000):
                raise ValueError('H264 presentation order disagrees with source clock')
            if decoded == 0:
                stream.width, stream.height, stream.pix_fmt = frame.width, frame.height, 'yuv420p'
            frame.time_base = Fraction(1, 1_000_000)
            for packet in stream.encode(frame):
                target.mux(packet)
            decoded += 1

        for i, data in enumerate(payloads):
            packet = av.Packet(data)
            packet.pts = packet.dts = round(int(axis[i]) / 1000)
            packet.time_base = Fraction(1, 1_000_000)
            for frame in decoder.decode(packet):
                emit(frame)
        for frame in decoder.decode(None):
            emit(frame)
        for packet in stream.encode():
            target.mux(packet)
    with av.open(str(temporary)) as check:
        out_times = np.asarray([float(f.pts * f.time_base) for f in check.decode(video=0)])
    if decoded != len(clock) or len(out_times) != len(clock):
        raise ValueError('Converted source video lost or added frames')
    error = float(np.max(np.abs(out_times - axis / 1e9)))
    if error > 1e-6:
        raise ValueError('Converted source video differs from original PTS by more than one microsecond')
    temporary.rename(destination)
    return {'frames': decoded, 'max_pts_error_s': error, 'time_base': '1/1000000',
            'codec': 'libx264', 'crf': 18, 'preset': 'veryfast', 'audio': 'MCAP selected video topic has no audio'}


def read_episode(meta, config, output):
    source, layout = config['source'], config['layout']
    path, metadata_path = Path(meta['source_mcap']), Path(meta['source_metadata'])
    original = {str(p): sha256_file(p) for p in (path, metadata_path)}
    root = Path(config['dataset_root']).resolve()
    if any(not p.resolve().is_relative_to(root) for p in (path, metadata_path)):
        raise ValueError('MCAP and episode metadata must belong to the configured source dataset')
    cfg = tomllib.loads(read_json(metadata_path)['config_toml'])
    devices = {d['name']: d for d in cfg['devices']}
    for arm in ('left', 'right'):
        channels = devices.get(f'robot/{arm}', {}).get('channels', [])
        arms = [c for c in channels if c['channel_type'] == 'arm']
        if len(arms) != 1 or arms[0].get('dof') != layout['joints_per_arm']:
            raise ValueError('Episode execution-robot metadata has an incompatible arm DOF')
    topics = [source['feedback_topics'][f'{arm}_{part}'] for arm in ('left', 'right') for part in ('arm', 'gripper')]
    head = source['camera']
    info = summary(path)
    selected = {cid: ch for cid, ch in info['channels'].items() if ch['topic'] in [head, *topics]}
    if len(selected) != 5 or {ch['topic'] for ch in selected.values()} != {head, *topics}:
        raise ValueError('MCAP requires one video and four measured-feedback channels')
    for ch in selected.values():
        expected = 'foxglove.CompressedVideo' if ch['topic'] == head else 'foxglove.JointStates'
        schema = info['schemas'][ch['schema_id']]
        if ch['encoding'] != 'flatbuffer' or schema['encoding'] != 'flatbuffer' or ch['schema'] != expected:
            raise ValueError('Unsupported measured-feedback/video MCAP schema or encoding')
        objects = {obj['name']: obj for obj in schema['objects']}
        if expected not in objects:
            raise ValueError('MCAP schema root does not match channel type')
        fields = {f['name']: f['id'] for f in objects[expected]['fields']}
        required = {'timestamp': 0, 'data': 2, 'format': 3} if ch['topic'] == head else {'timestamp': 0, 'joints': 1}
        if any(fields.get(name) != index for name, index in required.items()):
            raise ValueError('MCAP schema lacks required measured-feedback/video fields')
    clocks = {topic: [] for topic in [head, *topics]}
    values = {topic: [] for topic in topics}
    payloads, joint_names = [], {}
    for msg in messages(path, info['summary_offset']):
        cid = scalar(msg, 0, 'H')
        if cid not in selected:
            continue
        topic, body = selected[cid]['topic'], msg[22:]
        stamp = timestamp(body)
        if stamp != scalar(msg, 6, 'Q'):
            raise ValueError('Embedded and MCAP log timestamps disagree')
        clocks[topic].append(stamp)
        if topic == head:
            root_table = scalar(body, 0, 'I')
            if text_field(body, field(body, root_table, 3)) != 'h264':
                raise ValueError('Only explicitly encoded H264 source packets are supported')
            data = indirect(body, field(body, root_table, 2))
            payloads.append(bytes(body[data + 4:data + 4 + scalar(body, data, 'I')]))
        else:
            names, positions = joint_values(body)
            arm = 'left' if topic in topics[:2] else 'right'
            expected_names = source['gripper_names'] if '/gripper/' in topic else layout['joint_names'][arm]
            if names != expected_names or len(positions) != len(expected_names):
                raise ValueError('MCAP joint names/order or feedback dimensions disagree with configuration')
            joint_names[topic] = names
            values[topic].append(positions)
    if any(len(clocks[ch['topic']]) != ch['messages'] for ch in selected.values()):
        raise ValueError('MCAP decoded channel counts disagree with source summary')
    clock = np.asarray(clocks[head], dtype=np.int64)
    if len(clock) < 3 or np.any(np.diff(clock) <= 0):
        raise ValueError('Original video clock must have at least three strictly increasing frames')
    parts, rows, errors, stamps, alignment = [], [], [], [], {}
    policy = config['alignment']
    for topic in topics:
        measured, selected_rows, signed_errors = align_nearest(clocks[topic], values[topic], clock, round(policy['hard_limit_ms'] * 1e6))
        parts.append(measured)
        rows.append(selected_rows)
        errors.append(signed_errors)
        stamps.append(np.asarray(clocks[topic], dtype=np.int64)[selected_rows])
        alignment[topic] = {'raw_samples': len(clocks[topic]), 'joint_names': joint_names[topic],
            'max_abs_error_ms': float(np.max(np.abs(signed_errors)) / 1e6),
            'frames_above_report_limit': np.where(np.abs(signed_errors) > policy['report_above_ms'] * 1e6)[0].tolist()}
    times = (clock - clock[0]) / 1e9
    arrays = {'joint': np.column_stack(parts), 'timestamps': times, 'source_video_clock_ns': clock,
              'feedback_sample_rows': np.column_stack(rows), 'feedback_time_errors_ns': np.column_stack(errors),
              'feedback_source_timestamps_ns': np.column_stack(stamps)}
    video_folder = Path(output) / 'source_videos'
    video_folder.mkdir(exist_ok=True)
    destination = video_folder / f'{meta["key"]}.mp4'
    video_check = _video(payloads, clock, destination)
    if any(sha256_file(p) != digest for p, digest in original.items()):
        raise ValueError('Raw MCAP or episode metadata changed during preparation')
    meta = {**meta, 'dataset_id': config['dataset_id'], 'task_family': 'fold_cloth', 'frame_count': len(times),
            'source_video_path': str(destination), 'source_video_sha256': sha256_file(destination),
            'source_mcap_sha256': original[str(path)], 'source_metadata_sha256': original[str(metadata_path)],
            'camera_key': head, 'source_fps': float(1e9 / np.median(np.diff(clock))),
            'alignment_stats': alignment, 'alignment_policy': policy, 'prepared_video_validation': video_check,
            'mcap_schema_sha256': {ch['topic']: info['schemas'][ch['schema_id']]['sha256'] for ch in selected.values()}}
    return meta, arrays
