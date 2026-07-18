"""Trajectory smoothing filters for recorded episodes.

Replay always has the whole episode available, so the default filters here are
**non-causal, zero-lag**: a cubic smoothing B-spline (the representation from
"B-spline Policy", arXiv:2607.09648) when SciPy is present, and a zero-phase
Gaussian otherwise. A causal One-Euro filter is offered for the case where only
past samples are available. Only numpy is required; SciPy methods degrade to the
Gaussian fallback with a note when SciPy is missing.
"""
from __future__ import annotations

import math

import numpy as np

try:  # SciPy is optional; spline/savgol light up when it is installed.
    from scipy.interpolate import make_smoothing_spline
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _HAS_SCIPY = False

METHODS = ("spline", "gaussian", "savgol", "moving_average", "one_euro", "none")
_CHANNELS = ("leader", "observation", "action")


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window) | 1)  # force odd
    if window <= 1:
        return arr.astype(float)
    pad = window // 2
    kernel = np.ones(window) / window
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        padded = np.pad(arr[:, j], pad, mode="edge")
        out[:, j] = np.convolve(padded, kernel, mode="valid")
    return out


def _gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(1e-3, float(sigma))
    radius = max(1, int(round(sigma * 3)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        padded = np.pad(arr[:, j], radius, mode="edge")
        out[:, j] = np.convolve(padded, kernel, mode="valid")
    return out


def _one_euro(arr: np.ndarray, fps: float, mincutoff: float, beta: float, dcutoff: float = 1.0) -> np.ndarray:
    frames, joints = arr.shape
    out = np.empty_like(arr, dtype=float)
    dt = 1.0 / max(1e-3, float(fps))

    def alpha(cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * max(1e-6, cutoff))
        return 1.0 / (1.0 + tau / dt)

    for j in range(joints):
        x_prev = float(arr[0, j])
        dx_prev = 0.0
        out[0, j] = x_prev
        for t in range(1, frames):
            x = float(arr[t, j])
            dx = (x - x_prev) / dt
            a_d = alpha(dcutoff)
            dx_hat = a_d * dx + (1 - a_d) * dx_prev
            cutoff = mincutoff + beta * abs(dx_hat)
            a = alpha(cutoff)
            x_hat = a * x + (1 - a) * x_prev
            out[t, j] = x_hat
            x_prev, dx_prev = x_hat, dx_hat
    return out


def _spline(arr: np.ndarray, lam: float) -> np.ndarray:
    x = np.arange(arr.shape[0], dtype=float)
    out = np.empty_like(arr, dtype=float)
    lam_arg = None if lam is None or lam <= 0 else float(lam)
    for j in range(arr.shape[1]):
        spline = make_smoothing_spline(x, arr[:, j], lam=lam_arg)
        out[:, j] = spline(x)
    return out


def _savgol(arr: np.ndarray, window: int, polyorder: int = 3) -> np.ndarray:
    window = max(polyorder + 2, int(window) | 1)
    window = min(window, arr.shape[0] if arr.shape[0] % 2 == 1 else arr.shape[0] - 1)
    if window <= polyorder:
        return arr.astype(float)
    return savgol_filter(arr, window, polyorder, axis=0, mode="interp")


def smooth_columns(arr: np.ndarray, method: str, strength: float, fps: float) -> tuple[np.ndarray, str]:
    """Smooth ``[T, J]`` columns while preserving exact episode endpoints."""
    method = str(method or "spline").lower()
    strength = max(0.0, float(strength))
    if arr.shape[0] < 3 or method == "none":
        return arr.astype(float), "none"
    if method in {"spline", "savgol"} and not _HAS_SCIPY:
        method = "gaussian"  # graceful fallback; reported to the user
    smoothed: np.ndarray
    if method == "spline":
        smoothed, effective = _spline(arr, lam=strength), "spline"
    elif method == "savgol":
        smoothed, effective = _savgol(arr, window=int(round(strength * 10)) + 3), "savgol"
    elif method == "gaussian":
        smoothed, effective = _gaussian(arr, sigma=max(0.5, strength * 2.0)), "gaussian"
    elif method == "moving_average":
        smoothed, effective = _moving_average(arr, window=int(round(strength * 6)) + 1), "moving_average"
    elif method == "one_euro":
        smoothed, effective = _one_euro(
            arr, fps=fps, mincutoff=1.0 / max(0.1, strength), beta=0.007), "one_euro"
    else:
        smoothed, effective = _gaussian(arr, sigma=max(0.5, strength * 2.0)), "gaussian"
    smoothed = np.asarray(smoothed, dtype=float).copy()
    smoothed[0, :] = arr[0, :]
    smoothed[-1, :] = arr[-1, :]
    return smoothed, effective


def jerk_rms(arr: np.ndarray) -> float:
    """RMS of the third time-difference across all joints — a shakiness proxy."""
    if arr.shape[0] < 4:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(arr, n=3, axis=0) ** 2)))


def smooth_episode(raw_frames: list, joint_names: list[str], method: str, strength: float,
                   fps: float, preview_source: str, preview_joint: str) -> dict:
    """Smooth every channel of a recorded episode and return channels + preview data."""
    def stack(field: str) -> np.ndarray:
        return np.array(
            [[float((frame.get(field) or {}).get(name, 0.0)) for name in joint_names]
             for frame in raw_frames],
            dtype=float,
        )

    raw = {channel: stack(channel) for channel in _CHANNELS}
    smoothed: dict[str, np.ndarray] = {}
    effective = str(method or "spline").lower()
    for channel, arr in raw.items():
        smoothed[channel], effective = smooth_columns(arr, method, strength, fps)

    source = preview_source if preview_source in _CHANNELS else "action"
    joint_index = joint_names.index(preview_joint) if preview_joint in joint_names else 0
    return {
        "method": effective,
        "requested_method": str(method or "spline").lower(),
        "channels": {channel: values.tolist() for channel, values in smoothed.items()},
        "preview": {
            "source": source,
            "joint": joint_names[joint_index] if joint_names else "",
            "raw": raw[source][:, joint_index].tolist() if joint_names else [],
            "smoothed": smoothed[source][:, joint_index].tolist() if joint_names else [],
        },
        "jerk_raw": jerk_rms(raw[source]),
        "jerk_smoothed": jerk_rms(smoothed[source]),
        "scipy": _HAS_SCIPY,
    }
