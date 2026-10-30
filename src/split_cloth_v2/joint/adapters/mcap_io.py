"""Read-only MCAP and embedded FlatBuffer decoding for measured execution-robot feedback.

Reads schema descriptors and validates channel counts before alignment.
No runtime dependency on the previous diagnostic run.
"""
import ctypes
import ctypes.util
import hashlib
import struct
MAGIC = b'\x89MCAP0\r\n'
_zstd_library = ctypes.util.find_library('zstd')
if not _zstd_library:
    raise RuntimeError('MCAP zstd chunks require the system libzstd library')
ZSTD = ctypes.CDLL(_zstd_library)
ZSTD.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
ZSTD.ZSTD_decompress.restype = ctypes.c_size_t
ZSTD.ZSTD_isError.argtypes = [ctypes.c_size_t]
ZSTD.ZSTD_isError.restype = ctypes.c_uint

def scalar(b, o, fmt):
    return struct.unpack_from('<' + fmt, b, o)[0]

def string(b, o):
    n = scalar(b, o, 'I')
    return (bytes(b[o + 4:o + 4 + n]).decode(), o + 4 + n)

def field(b, t, k):
    v = t - scalar(b, t, 'i')
    loc = 4 + 2 * k
    offset = scalar(b, v + loc, 'H') if loc < scalar(b, v, 'H') else 0
    return t + offset if offset else 0

def indirect(b, a):
    return a + scalar(b, a, 'I') if a else 0

def text_field(b, a):
    a = indirect(b, a)
    return string(b, a)[0] if a else ''

def table_vector(b, a):
    a = indirect(b, a)
    return [indirect(b, a + 4 + 4 * i) for i in range(scalar(b, a, 'I'))] if a else []

def schema_description(b):
    root = scalar(b, 0, 'I')
    objects = []
    for obj in table_vector(b, field(b, root, 0)):
        fields = []
        for fld in table_vector(b, field(b, obj, 1)):
            typ = indirect(b, field(b, fld, 1))
            types = []
            for k, fmt, default in [(0, 'B', 0), (1, 'B', 0), (2, 'i', -1)]:
                a = field(b, typ, k)
                types.append(scalar(b, a, fmt) if a else default)
            fields.append({'name': text_field(b, field(b, fld, 0)), 'id': scalar(b, field(b, fld, 2), 'H') if field(b, fld, 2) else 0, 'base_type_element_index': types})
        objects.append({'name': text_field(b, field(b, obj, 0)), 'is_struct': bool(scalar(b, field(b, obj, 2), 'B')) if field(b, obj, 2) else False, 'fields': fields})
    return objects

def header(f):
    return struct.unpack('<BQ', f.read(9))

def summary(path):
    schemas, channels, counts = ({}, {}, {})
    with path.open('rb') as f:
        if not f.read(8) == MAGIC:
            raise ValueError('Invalid MCAP record')
        f.seek(-37, 2)
        tail = f.read(37)
        if not (tail[-8:] == MAGIC and struct.unpack_from('<BQ', tail) == (2, 20)):
            raise ValueError('Invalid MCAP record')
        start = scalar(tail, 9, 'Q')
        f.seek(start)
        while True:
            op, n = header(f)
            b = f.read(n)
            if not len(b) == n:
                raise ValueError('Invalid MCAP record')
            if op == 3:
                sid = scalar(b, 0, 'H')
                name, o = string(b, 2)
                enc, o = string(b, o)
                data = b[o + 4:]
                if not len(data) == scalar(b, o, 'I'):
                    raise ValueError('Invalid MCAP record')
                schemas[sid] = {'name': name, 'encoding': enc, 'sha256': hashlib.sha256(data).hexdigest(), 'objects': schema_description(data)}
            elif op == 4:
                cid, sid = struct.unpack_from('<HH', b)
                topic, o = string(b, 4)
                enc, o = string(b, o)
                channels[cid] = {'topic': topic, 'schema_id': sid, 'encoding': enc}
            elif op == 11:
                counts = dict(struct.iter_unpack('<HQ', b[46:]))
                ts1, ts2 = struct.unpack_from('<QQ', b, 26)
                duration = (ts2 - ts1) / 1000000000.0
            elif op == 2:
                break
    for cid, ch in channels.items():
        ch['messages'] = counts[cid]
        ch['schema'] = schemas[ch['schema_id']]['name']
    return {'path': str(path), 'summary_offset': start, 'duration_s': duration, 'schemas': schemas, 'channels': channels}

def messages(path, stop):
    with path.open('rb') as f:
        f.seek(8)
        while f.tell() < stop:
            op, n = header(f)
            if op not in (5, 6):
                f.seek(n, 1)
                continue
            b = f.read(n)
            if not len(b) == n:
                raise ValueError('Invalid MCAP record')
            if op == 5:
                yield b
                continue
            raw_size = scalar(b, 16, 'Q')
            compression, o = string(b, 28)
            size = scalar(b, o, 'Q')
            compressed = b[o + 8:]
            if not len(compressed) == size:
                raise ValueError('Invalid MCAP record')
            if compression == 'zstd':
                output = ctypes.create_string_buffer(raw_size)
                actual = ZSTD.ZSTD_decompress(output, raw_size, compressed, size)
                if not (not ZSTD.ZSTD_isError(actual) and actual == raw_size):
                    raise ValueError('Invalid MCAP record')
                raw = output.raw
            else:
                if not (compression == '' and len(compressed) == raw_size):
                    raise ValueError('Invalid MCAP record')
                raw = compressed
            view, o = (memoryview(raw), 0)
            while o < len(view):
                inner, size = struct.unpack_from('<BQ', view, o)
                o += 9
                if not o + size <= len(view):
                    raise ValueError('Invalid MCAP record')
                if inner == 5:
                    yield view[o:o + size]
                o += size
            if not o == len(view):
                raise ValueError('Invalid MCAP record')

def joint_values(b):
    root = scalar(b, 0, 'I')
    joints = table_vector(b, field(b, root, 1))
    names = [text_field(b, field(b, j, 0)) for j in joints]
    positions = [scalar(b, field(b, j, 1), 'd') if field(b, j, 1) else 0.0 for j in joints]
    return (names, positions)

def timestamp(b):
    root = scalar(b, 0, 'I')
    a = field(b, root, 0)
    sec, nsec = struct.unpack_from('<II', b, a)
    return sec * 1000000000 + nsec
