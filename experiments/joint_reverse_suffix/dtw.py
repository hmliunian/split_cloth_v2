"""Load the bundled DTW kernel, compiling it locally without network access."""
from functools import lru_cache
from pathlib import Path
import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import numpy as np

@lru_cache(maxsize=1)
def _native():
    source = Path(__file__).with_suffix('.cpp')
    version = hashlib.sha256(source.read_bytes() + platform.machine().encode()).hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / 'split_cloth_dtw_cache' / version
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / 'libdtw.so'
    if not library.exists():
        compiler = shutil.which('g++')
        if compiler is None:
            raise RuntimeError('A local g++ compiler is required to build the bundled DTW kernel')
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / 'libdtw.so'
            subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC',
                            str(source), '-o', str(output)], check=True, capture_output=True)
            os.replace(output, library)
    native = ctypes.CDLL(str(library))
    native.dtw32.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    native.dtw32.restype = ctypes.c_double
    return native

def dtw(x, y):
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape or x.shape[0] != 32 or not x.shape[1]:
        raise ValueError('DTW requires two equally shaped 32-by-D arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('DTW inputs must be finite')
    return _native().dtw32(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                          y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), x.shape[1])
