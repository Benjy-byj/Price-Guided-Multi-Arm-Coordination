from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


ArrayModule = Any


def get_array_module(name: str) -> ArrayModule:
    backend = name.lower()
    if backend == "numpy":
        return np
    if backend == "cupy":
        if cp is None:
            raise RuntimeError(
                "CuPy backend requested but CuPy is not installed. "
                "Install `cupy-cuda12x` in a GPU-enabled environment first."
            )
        try:
            device_count = int(cp.cuda.runtime.getDeviceCount())
        except Exception as exc:
            raise RuntimeError(f"CuPy backend requested but GPU probing failed: {exc}") from exc
        if device_count <= 0:
            raise RuntimeError("CuPy backend requested but no CUDA device is visible.")
        return cp
    raise ValueError(f"Unsupported array backend: {name}")


def to_numpy(array: Any) -> np.ndarray:
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def scalar_to_float(value: Any) -> float:
    if cp is not None and isinstance(value, cp.ndarray):
        return float(cp.asnumpy(value).item())
    arr = np.asarray(value)
    return float(arr.item() if arr.shape == () else arr)
