"""End-to-end replay stream transport test using synthetic frames only.

Exercises the stdlib WebSocket broadcast server, the stdlib subscriber client,
and the publisher walk loop without hardware, ROS, or the network. A synthetic
replay session is injected and ``replay_frame`` is stubbed, so no dataset files
are read.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

import blacknode  # noqa: F401 - triggers package discovery
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_dataset import runtime as rt

# The dependency-free subscriber lives under clients/, outside the node alias.
_CLIENTS = Path(__file__).resolve().parents[1] / "clients"
sys.path.insert(0, str(_CLIENTS))
_spec = importlib.util.spec_from_file_location("blacknode_ws", _CLIENTS / "blacknode_ws.py")
blacknode_ws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blacknode_ws)


def _install_session(token: str, frames: int, joint_names: list[str]) -> None:
    with rt._lock:
        rt._replay_sessions[token] = {
            "frames": frames, "fps": 60, "units": "radians",
            "joint_names": joint_names, "task": "synthetic",
        }


def _fake_replay_frame(token, index):
    names = ["a", "b"]
    return {
        "kind": "blacknode.episode-frame", "schema_version": 1,
        "frame_index": int(index), "frames": 5, "timestamp": index / 60.0,
        "joint_names": names,
        "action": {"a": float(index), "b": float(index) * 2},
        "observation": {"a": 0.0, "b": 0.0},
        "leader": {"a": 0.0, "b": 0.0},
        "cameras": {},
    }


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(rt, "replay_frame", _fake_replay_frame)
    _install_session("tok", frames=5, joint_names=["a", "b"])
    yield
    rt.stop_runtime_services()
    with rt._lock:
        rt._replay_sessions.pop("tok", None)


def test_publisher_node_is_registered():
    assert "ReplayStreamPublisher" in _NODE_REGISTRY
    assert _NODE_REGISTRY["ReplayStreamPublisher"]._bn_category == "Dataset"


def test_publisher_streams_frames_to_a_subscriber(stubbed):
    status = rt.start_replay_stream(run_id="t1", token="tok", host="127.0.0.1", port=0,
                                    fps=60, rate=1.0, loop=True, source="action", units="radians")
    assert status["streaming"] is True
    url = status["stream_url"]
    assert url.startswith("ws://127.0.0.1:")

    stream = blacknode_ws.connect(url, timeout=5.0)
    try:
        frame = stream.recv_json()
        assert frame is not None
        assert frame["joint_names"] == ["a", "b"]
        # positions is the ordered 'action' source: b == 2 * a
        assert frame["positions"][1] == frame["positions"][0] * 2
        assert frame["source"] == "action"
        assert frame["units"] == "radians"
        # subscriber becomes visible to the publisher
        deadline = time.monotonic() + 2.0
        while rt.control_replay_stream("t1", "status")["clients"] < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert rt.control_replay_stream("t1", "status")["clients"] >= 1
    finally:
        stream.close()

    stopped = rt.control_replay_stream("t1", "stop")
    assert stopped["streaming"] is False


def test_start_without_session_reports_error(stubbed):
    with pytest.raises(ValueError):
        rt.start_replay_stream(run_id="missing", token="nope", host="127.0.0.1", port=0,
                               fps=30, rate=1.0, loop=False, source="action", units="radians")


def test_stop_runtime_services_stops_publishers(stubbed):
    rt.start_replay_stream(run_id="t2", token="tok", host="127.0.0.1", port=0,
                           fps=60, rate=2.0, loop=True, source="observation", units="radians")
    assert rt.runtime_status()["active"] is True
    result = rt.stop_runtime_services()
    assert result["stopped"]["streams"] >= 1
    assert rt.control_replay_stream("t2", "status")["streaming"] is False
