"""Trajectory smoother: offline zero-lag filtering of a recorded episode.

Injects a synthetic noisy episode (a clean sine plus jitter) and checks that the
smoother reduces shakiness, mints a new replay token, and that the token streams
cleanly through the publisher. No hardware, ROS, or network required.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

import blacknode  # noqa: F401 - triggers package discovery
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_dataset import filters, runtime as rt

_CLIENTS = Path(__file__).resolve().parents[1] / "clients"
sys.path.insert(0, str(_CLIENTS))
_spec = importlib.util.spec_from_file_location("blacknode_ws", _CLIENTS / "blacknode_ws.py")
blacknode_ws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blacknode_ws)

_JOINTS = ["a", "b"]
_FRAMES = 120


def _noisy_frames():
    frames = []
    for i in range(_FRAMES):
        clean = math.sin(i * 0.08)
        jitter = 0.25 * math.sin(i * 2.3) * (1 if i % 2 else -1)  # high-freq shake
        value = clean + jitter
        frames.append({
            "kind": "blacknode.episode-frame", "frame_index": i, "frames": _FRAMES,
            "timestamp": i / 60.0, "joint_names": _JOINTS,
            "action": {"a": value, "b": value * 0.5},
            "observation": {"a": value, "b": value * 0.5},
            "leader": {"a": value, "b": value * 0.5},
            "cameras": {},
        })
    return frames


@pytest.fixture
def episode(monkeypatch):
    frames = _noisy_frames()
    original = rt.replay_frame

    def fake_replay_frame(token, index):
        # smoothed tokens carry in-memory frames; serve those via the real code path
        with rt._lock:
            session = dict(rt._replay_sessions.get(str(token or "")) or {})
        if session.get("smoothed_frames") is not None:
            return original(token, index)
        return dict(frames[min(max(0, int(index)), len(frames) - 1)])

    monkeypatch.setattr(rt, "replay_frame", fake_replay_frame)
    with rt._lock:
        rt._replay_sessions["raw"] = {
            "frames": _FRAMES, "fps": 60, "units": "radians",
            "joint_names": _JOINTS, "task": "synthetic",
        }
    yield
    rt.stop_runtime_services()
    with rt._lock:
        for key in [k for k in rt._replay_sessions if k in {"raw"} or rt._replay_sessions[k].get("source_token") == "raw"]:
            rt._replay_sessions.pop(key, None)


def test_smoother_node_registered():
    assert "TrajectorySmoother" in _NODE_REGISTRY
    assert _NODE_REGISTRY["TrajectorySmoother"]._bn_category == "Dataset"


@pytest.mark.parametrize("method", ["spline", "gaussian", "savgol", "moving_average", "one_euro"])
def test_smoothing_reduces_jerk_and_mints_token(episode, method):
    info = rt.register_smoothed_replay("raw", method, strength=1.0, preview_source="action")
    assert info["token"] and info["token"] != "raw"
    # the synthetic signal is deliberately shaky; every filter should calm it
    assert info["jerk_reduction_pct"] > 10.0
    # the new token serves smoothed frames of the same length
    frame = rt.replay_frame(info["token"], 3)
    assert frame is not None
    assert frame["joint_names"] == _JOINTS
    assert "smoothing" in frame


def test_smoothed_stream_streams_through_publisher(episode):
    info = rt.register_smoothed_replay("raw", "spline", strength=1.0)
    handle = {"kind": "blacknode.replay-stream", "token": info["token"]}
    status = rt.start_stream(run_id="sm", stream=handle, host="127.0.0.1", port=0,
                             fps=60, rate=1.0, loop=True, source="action", units="radians",
                             sync_to_browser=False)
    assert status["streaming"] is True
    stream = blacknode_ws.connect(status["stream_url"], timeout=5.0)
    try:
        frame = stream.recv_json()
        assert frame is not None
        assert frame["joint_names"] == _JOINTS
        assert len(frame["positions"]) == 2
    finally:
        stream.close()
    rt.control_stream("sm", "stop")


def test_parameter_update_only_recomputes_smoother_and_hot_swaps_publisher(episode):
    first = rt.apply_configured_smoother(
        "smoother-node", "gaussian", 0.5,
        stream={"kind": "blacknode.replay-stream", "token": "raw"},
    )
    first_token = first["stream"]["token"]
    status = rt.start_stream(
        run_id="hot-swap", stream=first["stream"], host="127.0.0.1", port=0,
        fps=60, rate=1.0, loop=True, source="action", units="radians",
        sync_to_browser=True,
    )
    original_url = status["stream_url"]
    publisher = rt._publishers["hot-swap"]
    original_thread = publisher.thread
    stream = blacknode_ws.connect(original_url, timeout=5.0)
    try:
        assert stream.recv_json()["kind"] == "blacknode.stream-schema"
        rt.publish_replay_event("raw", 20, "seek")
        before = stream.recv_json()
        assert before["frame_index"] == 20
        assert before["smoothing"] == {"method": "gaussian", "strength": 0.5}

        updated = rt.apply_configured_smoother("smoother-node", "gaussian", 2.0)
        updated_token = updated["stream"]["token"]

        assert updated_token != first_token
        assert publisher.token == updated_token
        assert publisher.thread is original_thread
        assert publisher.status()["stream_url"] == original_url
        assert "updated 1 running publisher" in updated["report"]
        assert "refreshed the current pose" in updated["report"]
        with rt._lock:
            assert first_token not in rt._replay_sessions

        frame = stream.recv_json()
        assert frame["frame_index"] == 20
        assert frame["smoothing"] == {"method": "gaussian", "strength": 2.0}
        assert frame["playback_event"] == "smoother_update"
    finally:
        stream.close()
        rt.control_stream("hot-swap", "stop")


def test_direct_filter_math_is_zero_lag_and_shorter_jerk():
    import numpy as np
    arr = np.array([[math.sin(i * 0.1) + (0.2 if i % 2 else -0.2)] for i in range(80)])
    smoothed, effective = filters.smooth_columns(arr, "gaussian", strength=1.0, fps=60)
    assert smoothed.shape == arr.shape  # zero-lag: same length, no trimming
    assert filters.jerk_rms(smoothed) < filters.jerk_rms(arr)
