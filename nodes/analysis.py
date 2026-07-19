"""Self-calibrating episode analysis — the data-driven core of ``EpisodeEvaluator``.

Nothing here is task-specific or hard-coded. Every threshold, the gripper joint,
the open/closed bands, the "engaged" state, and the segment boundaries are all
*derived from the episode's own statistics*. The only task knowledge that ever
enters the pipeline is a success criterion the caller supplies explicitly
(a rule expression, a reference distribution, or a vision verdict) — this module
never decides on its own what "success" means.

Pure functions, numpy-only, no I/O except :func:`load_episode_matrices` reading a
saved ``data.parquet``. That keeps the whole thing unit-testable against a
synthesised episode with no robot and no hardware.
"""
from __future__ import annotations

from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - guarded like the rest of the package
    np = None

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


def _require() -> None:
    missing = [name for name, mod in (("numpy", np), ("pyarrow", pq)) if mod is None]
    if missing:
        raise RuntimeError("episode analysis needs: " + ", ".join(missing))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_episode_matrices(data_path: str) -> dict[str, Any]:
    """Read a saved ``data.parquet`` into aligned ``[T, J]`` matrices.

    Returns observation/action/leader arrays plus timestamps, joint names, and
    the recorded task string. No interpretation — just the raw signal.
    """
    _require()
    table = pq.read_table(data_path)
    names = table.column_names

    def _matrix(col: str) -> "np.ndarray":
        return np.asarray(table.column(col).to_pylist(), dtype=float)

    obs = _matrix("observation.state")
    joint_count = obs.shape[1] if obs.ndim == 2 else 0
    task_col = table.column("task").to_pylist() if "task" in names else []
    task = next((str(t) for t in task_col if str(t).strip()), "")
    return {
        "timestamp": np.asarray(table.column("timestamp").to_pylist(), dtype=float),
        "observation": obs,
        "action": _matrix("action") if "action" in names else np.empty((len(obs), 0)),
        "leader": _matrix("leader.state") if "leader.state" in names else np.empty((len(obs), 0)),
        "joint_count": joint_count,
        "task": task,
    }


# --------------------------------------------------------------------------- #
# Self-calibration
# --------------------------------------------------------------------------- #
def _otsu_separation(x: "np.ndarray") -> tuple[float, float]:
    """Best 2-cluster split of a 1-D signal. Returns ``(separation_ratio, threshold)``.

    ``separation_ratio`` is between-class / total variance in ``[0, 1]``: high for
    a cleanly bimodal joint (a gripper snapping open/closed), low for a smoothly
    swept arm joint. Fully non-parametric — no assumed threshold.
    """
    x = np.asarray(x, dtype=float)
    total = float(x.var())
    if x.size < 4 or total <= 0.0:
        return 0.0, float(x.mean()) if x.size else 0.0
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        return 0.0, float((lo + hi) / 2)
    best_ratio, best_t = 0.0, float((lo + hi) / 2)
    for t in np.linspace(lo, hi, 64):
        a, b = x[x <= t], x[x > t]
        if a.size == 0 or b.size == 0:
            continue
        between = (a.size / x.size) * (b.size / x.size) * (a.mean() - b.mean()) ** 2
        ratio = between / total
        if ratio > best_ratio:
            best_ratio, best_t = float(ratio), float(t)
    return best_ratio, best_t


def discover_gripper_joint(obs: "np.ndarray") -> dict[str, Any]:
    """Pick the most bimodal joint as the gripper — data-driven, not by name/position.

    Confidence blends how clean the winner's split is with how far it stands
    above the runner-up, so an ambiguous arm-only episode reports low confidence
    instead of a false pick.
    """
    _require()
    if obs.ndim != 2 or obs.shape[1] == 0:
        return {"index": -1, "threshold": 0.0, "separation": 0.0, "confidence": 0.0}
    scored = [(_otsu_separation(obs[:, j]), j) for j in range(obs.shape[1])]
    scored.sort(key=lambda item: item[0][0], reverse=True)
    (top_sep, top_t), top_j = scored[0]
    runner = scored[1][0][0] if len(scored) > 1 else 0.0
    margin = top_sep - runner
    confidence = float(max(0.0, min(1.0, top_sep)) * (0.5 + 0.5 * min(1.0, margin / (top_sep + 1e-9))))
    return {"index": int(top_j), "threshold": float(top_t), "separation": float(top_sep),
            "confidence": confidence}


def _percentile_threshold(values: "np.ndarray", pct: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, pct))


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def derive_signals(matrices: dict[str, Any], gripper: dict[str, Any]) -> dict[str, Any]:
    """Compute evidence signals. Every cutoff is a percentile of this episode's data."""
    _require()
    obs = matrices["observation"]
    action = matrices["action"]
    frames = int(obs.shape[0])
    gi = int(gripper["index"])
    signals: dict[str, Any] = {"frames": frames, "gripper_index": gi}
    if frames < 2 or gi < 0:
        signals["insufficient_data"] = True
        return signals

    # Arm = every joint except the discovered gripper.
    arm_cols = [j for j in range(obs.shape[1]) if j != gi]
    arm = obs[:, arm_cols] if arm_cols else obs[:, :0]
    arm_vel = np.linalg.norm(np.diff(arm, axis=0), axis=1) if arm.size else np.zeros(frames - 1)

    # "Moving" is defined against this episode's own velocity distribution.
    move_cut = _percentile_threshold(arm_vel, 60.0)
    moving = arm_vel > move_cut  # length T-1

    thr = gripper["threshold"]
    obs_state = (obs[:, gi] > thr).astype(int)  # achieved gripper aperture, binarised

    # Slip = the achieved gripper disagreeing with the *commanded* gripper for a
    # sustained run. This uses the action channel, so it needs no guess about
    # which aperture means "closed" — an intended open/close moves both together,
    # only a failure to hold (or an unintended release) diverges them.
    grip_slip, slip_frames, cmd_state = False, [], []
    if action.shape == obs.shape and action.size:
        cmd_state = (action[:, gi] > thr).astype(int)
        mismatch = cmd_state != obs_state
        slip_frames = [f for run in _runs(mismatch) if len(run) >= 3 for f in run]
        grip_slip = len(slip_frames) > 0

    # Tracking error / stall: action vs achieved, noise floor from quiescent frames.
    tracking_error: dict[int, float] = {}
    stall_frames: list[int] = []
    if action.shape == obs.shape and action.size:
        gap = np.abs(action - obs)
        per_joint_max = gap.max(axis=0)
        tracking_error = {int(j): float(per_joint_max[j]) for j in range(gap.shape[1])}
        quiescent = np.concatenate(([True], ~moving)) if moving.size else np.ones(frames, bool)
        noise_floor = _percentile_threshold(gap[quiescent].max(axis=1), 90.0) if quiescent.any() else 0.0
        stall_cut = max(noise_floor * 2.0, _percentile_threshold(gap.max(axis=1), 95.0))
        stall_frames = np.nonzero(gap.max(axis=1) > stall_cut)[0].tolist()

    signals.update({
        "arm_velocity": arm_vel.tolist(),
        "move_cutoff": float(move_cut),
        "gripper_series": obs[:, gi].tolist(),
        "gripper_state": obs_state.tolist(),
        "gripper_command": list(cmd_state) if len(cmd_state) else [],
        "grip_slip": grip_slip,
        "slip_frames": slip_frames,
        "tracking_error": tracking_error,
        "stall_frames": stall_frames,
        "settle_frames": (np.nonzero(~moving)[0] + 1).tolist(),
    })
    signals["segments"] = segment_episode(obs_state, moving, arm)
    return signals


def _runs(mask: "np.ndarray") -> list[list[int]]:
    """Contiguous index runs where ``mask`` is truthy."""
    _require()
    runs: list[list[int]] = []
    current: list[int] = []
    for i, on in enumerate(np.asarray(mask, dtype=bool).tolist()):
        if on:
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def segment_episode(state: "np.ndarray", moving: "np.ndarray", arm: "np.ndarray") -> list[dict[str, Any]]:
    """Discover segments from change-points; tag each by *observed evidence only*.

    Tags are neutral descriptors — ``grip-change`` (the gripper aperture flips),
    ``move`` (sustained arm motion), ``settle`` (quiescent) — never task phases
    like "grasp" or "lift", which would smuggle in an assumed task. Count and
    order are whatever the data shows.
    """
    _require()
    frames = int(state.shape[0])
    if frames < 2:
        return []
    move_full = np.concatenate(([moving[0]], moving)) if moving.size else np.zeros(frames, bool)
    boundaries = {0, frames}
    boundaries.update((np.nonzero(np.diff(state) != 0)[0] + 1).tolist())
    boundaries.update((np.nonzero(np.diff(move_full.astype(int)) != 0)[0] + 1).tolist())
    marks = sorted(boundaries)

    segments: list[dict[str, Any]] = []
    for start, end in zip(marks, marks[1:]):
        seg_moving = bool(move_full[start:end].mean() >= 0.5)
        grip_change = start > 0 and int(state[start - 1]) != int(round(float(state[start:end].mean())))
        travel = float(np.linalg.norm(arm[end - 1] - arm[start])) if arm.size and end > start else 0.0
        dominant = int(np.argmax(np.abs(arm[end - 1] - arm[start]))) if arm.size and end > start else -1
        tag = "grip-change" if grip_change else "move" if seg_moving else "settle"
        segments.append({
            "start_frame": int(start), "end_frame": int(end), "tag": tag,
            "moving": seg_moving, "gripper_state": int(round(float(state[start:end].mean()))),
            "travel": travel, "dominant_joint": dominant,
        })
    return segments


# --------------------------------------------------------------------------- #
# Success — always caller-supplied, never invented here
# --------------------------------------------------------------------------- #
_ALLOWED_EXPR_NAMES = {"grip_slip", "stall_frames", "settle_frames", "slip_frames",
                       "tracking_error", "segments", "frames"}


def judge_by_expression(signals: dict[str, Any], expression: str) -> dict[str, Any]:
    """Evaluate a user boolean expression over the derived signals. The definition
    of success lives entirely in the caller's string, not in this code."""
    ns = {k: signals.get(k) for k in _ALLOWED_EXPR_NAMES}
    ns["n_segments"] = len(signals.get("segments") or [])
    ns["tags"] = [s.get("tag") for s in (signals.get("segments") or [])]
    ns["max_tracking_error"] = max((signals.get("tracking_error") or {}).values(), default=0.0)
    try:
        verdict = bool(eval(expression, {"__builtins__": {}}, ns))  # noqa: S307 - user-owned local rule
    except Exception as exc:  # noqa: BLE001
        return {"success": None, "confidence": 0.0, "reason": f"rule error: {exc}", "rater": "expression"}
    return {"success": verdict, "confidence": 1.0,
            "reason": f"rule `{expression}` -> {verdict}", "rater": "expression"}


def judge_by_reference(signals: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Score against a distribution of known-good demonstrations.

    ``reference`` carries per-feature ``{mean, std}`` built from labelled-good
    episodes (so the definition of success comes from *data*, not code). We flag
    an episode as failed when it is an outlier on any monitored feature.
    """
    _require()
    feats = _reference_features(signals)
    stats = dict(reference.get("features") or {})
    if not stats:
        return {"success": None, "confidence": 0.0,
                "reason": "no reference features supplied", "rater": "reference"}
    z_by: dict[str, float] = {}
    for name, value in feats.items():
        ref = stats.get(name)
        if not ref:
            continue
        std = float(ref.get("std") or 0.0)
        delta = abs(value - float(ref.get("mean") or 0.0))
        # Zero-variance reference + a differing value is maximally anomalous, not z=0.
        z_by[name] = delta / std if std > 1e-9 else (0.0 if delta <= 1e-9 else 1e6)
    if not z_by:
        return {"success": None, "confidence": 0.0,
                "reason": "no overlapping features with reference", "rater": "reference"}
    worst = max(z_by.values())
    cut = float(reference.get("z_threshold") or 3.0)
    success = worst <= cut
    confidence = float(max(0.0, min(1.0, abs(worst - cut) / cut)))
    driver = max(z_by, key=z_by.get)
    return {"success": success, "confidence": confidence,
            "reason": f"z(worst={driver})={worst:.2f} vs {cut:.1f}", "rater": "reference",
            "z_scores": z_by}


def _reference_features(signals: dict[str, Any]) -> dict[str, float]:
    """Compact numeric fingerprint of an episode for reference comparison."""
    return {
        "grip_slip": float(bool(signals.get("grip_slip"))),
        "n_stall": float(len(signals.get("stall_frames") or [])),
        "n_segments": float(len(signals.get("segments") or [])),
        "max_tracking_error": float(max((signals.get("tracking_error") or {}).values(), default=0.0)),
        "n_move": float(sum(1 for s in (signals.get("segments") or []) if s.get("tag") == "move")),
    }


def build_reference(feature_rows: list[dict[str, float]], z_threshold: float = 3.0) -> dict[str, Any]:
    """Aggregate per-episode fingerprints of known-good demos into a reference."""
    _require()
    if not feature_rows:
        return {"features": {}, "z_threshold": z_threshold, "n": 0}
    keys = set().union(*(row.keys() for row in feature_rows))
    features = {}
    for key in keys:
        col = np.asarray([float(row.get(key, 0.0)) for row in feature_rows], dtype=float)
        features[key] = {"mean": float(col.mean()), "std": float(col.std())}
    return {"features": features, "z_threshold": float(z_threshold), "n": len(feature_rows)}
