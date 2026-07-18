"""End-to-end stream transport test using synthetic frames only.

Exercises the stdlib WebSocket broadcast server, the stdlib subscriber client,
and the publisher walk loop without hardware, ROS, or the outside network. A
synthetic replay session is injected and ``replay_frame`` is stubbed, and a local
HTTP server stands in for a live sample-stream, so no dataset files are read.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def _replay_handle(token: str = "tok") -> dict:
    return {"kind": "blacknode.replay-stream", "token": token}


def test_publisher_node_is_registered():
    assert "StreamPublisher" in _NODE_REGISTRY
    assert _NODE_REGISTRY["StreamPublisher"]._bn_category == "Dataset"


def test_publisher_streams_replay_to_a_subscriber(stubbed):
    status = rt.start_stream(run_id="t1", stream=_replay_handle(), host="127.0.0.1", port=0,
                             fps=60, rate=1.0, loop=True, source="action", units="radians",
                             sync_to_browser=False)
    assert status["streaming"] is True
    assert status["mode"] == "replay"
    url = status["stream_url"]
    assert url.startswith("ws://127.0.0.1:")

    stream = blacknode_ws.connect(url, timeout=5.0)
    try:
        frame = stream.recv_json()
        assert frame is not None
        assert frame["joint_names"] == ["a", "b"]
        assert frame["positions"][1] == frame["positions"][0] * 2
        assert frame["source"] == "action"
        deadline = time.monotonic() + 2.0
        while rt.control_stream("t1", "status")["clients"] < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert rt.control_stream("t1", "status")["clients"] >= 1
    finally:
        stream.close()

    stopped = rt.control_stream("t1", "stop")
    assert stopped["streaming"] is False


def test_unrecognized_stream_handle_reports_error(stubbed):
    with pytest.raises(ValueError):
        rt.start_stream(run_id="bad", stream={"nope": 1}, host="127.0.0.1", port=0,
                        fps=30, rate=1.0, loop=False, source="action", units="radians")


def test_stop_runtime_services_stops_publishers(stubbed):
    rt.start_stream(run_id="t2", stream=_replay_handle(), host="127.0.0.1", port=0,
                    fps=60, rate=2.0, loop=True, source="observation", units="radians",
                    sync_to_browser=False)
    assert rt.runtime_status()["active"] is True
    result = rt.stop_runtime_services()
    assert result["stopped"]["streams"] >= 1
    assert rt.control_stream("t2", "status")["streaming"] is False


class _SampleHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps({
            "kind": "blacknode.teleoperation-sample",
            "captured_at_ns": 1, "joint_names": ["a", "b"],
            "action": {"a": 1.0, "b": 2.0}, "observation": {"a": 1.0, "b": 2.0},
            "leader": {"a": 1.0, "b": 2.0}, "armed": True, "live": True,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):  # silence test server logging
        pass


def test_publisher_streams_a_live_sample_stream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SampleHandler)
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/sample"
    try:
        status = rt.start_stream(run_id="live", stream={"kind": "blacknode.sample-stream", "url": url},
                                 host="127.0.0.1", port=0, fps=30, rate=1.0, loop=False,
                                 source="action", units="radians")
        assert status["mode"] == "sample"
        stream = blacknode_ws.connect(status["stream_url"], timeout=5.0)
        try:
            frame = stream.recv_json()
            assert frame is not None
            assert frame["joint_names"] == ["a", "b"]
            assert frame["positions"] == [1.0, 2.0]
        finally:
            stream.close()
    finally:
        rt.control_stream("live", "stop")
        server.shutdown()


def test_browser_synchronized_publisher_waits_then_emits_play_and_seek(stubbed):
    status = rt.start_stream(run_id="browser", stream=_replay_handle(), host="127.0.0.1", port=0,
                             fps=60, rate=1.0, loop=True, source="action", units="radians")
    assert status["sync_to_browser"] is True
    stream = blacknode_ws.connect(status["stream_url"], timeout=5.0)
    try:
        schema = stream.recv_json()
        assert schema["kind"] == "blacknode.stream-schema"
        assert schema["joint_names"] == ["a", "b"]
        assert schema["positions"] == []
        assert schema["trajectory"] == [[float(index), float(index) * 2] for index in range(5)]

        stream._sock.settimeout(0.15)
        with pytest.raises(socket.timeout):
            stream.recv_json()

        stream._sock.settimeout(2.0)
        emitted = rt.publish_replay_event("tok", 3, "seek")
        assert emitted["publishers"] == 1
        frame = stream.recv_json()
        assert frame["frame_index"] == 3
        assert frame["positions"] == [3.0, 6.0]
        assert frame["playback_event"] == "seek"
    finally:
        stream.close()
        rt.control_stream("browser", "stop")


def test_maya_window_persists_axis_direction_mapping_and_gets_schema():
    source = (_CLIENTS / "maya_stream.py").read_text(encoding="utf-8")
    assert "Get joints / Connect" in source
    assert 'globals().get("_state")' in source
    assert "previous_thread.join(timeout=1.0)" in source
    assert 'previous_socket.close()' in source
    assert "blacknode.stream-schema" in source
    assert "cmds.optionVar" in source
    assert 'for value in ("X", "Y", "Z")' in source
    assert 'label="-1"' in source
    assert 'label="Path"' in source
    assert "onCommand=lambda" in source
    assert "offCommand=lambda" in source
    assert "changeCommand=lambda enabled" not in source
    assert "cmds.xform(node, query=True, worldSpace=True, translation=True)" in source
    assert "cmds.curve" in source
    assert "editPoint=points" in source
    assert '("overrideColorRGB", (1.0, 0.0, 0.0))' in source
    assert '("lineWidth", (4.0,))' in source
    assert "_reject_discontinuity_outliers" in source
    assert "_build_full_trajectory_paths" in source
    assert "_trajectory_sample_indices" in source
    assert "_on_path_changed" in source
    assert "full episode path ready" in source
    assert "path has no world-space movement" in source
    assert "_drain_pending" in source
    assert '"latest_frame"' in source
    assert "def _run(url, state):" in source
    assert "stale dropped" in source
    assert "cmds.scriptJob(idleEvent=_drain_pending" in source
    assert "_set_stream_status" in source
    assert "executeInMainThreadWithResult" not in source
    assert "_capture_debug_paths" not in source
    assert "Clear debug paths" in source
    for client in ("maya_client.py", "ros2_bridge.py", "isaac_lab_client.py"):
        assert 'frame.get("kind") == "blacknode.stream-schema"' in (
            _CLIENTS / client
        ).read_text(encoding="utf-8")


def test_maya_path_filter_drops_isolated_spike_but_keeps_sustained_motion():
    source = (_CLIENTS / "maya_stream.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"_distance", "_median", "_reject_discontinuity_outliers"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "maya_path_filter", "exec"), namespace)
    reject = namespace["_reject_discontinuity_outliers"]

    spike = [(0, 0, 0), (1, 0, 0), (1000, 1000, 0), (2, 0, 0), (3, 0, 0)]
    assert reject(spike) == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                             (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    sustained = [(0, 0, 0), (1, 0, 0), (100, 0, 0), (101, 0, 0), (102, 0, 0)]
    assert reject(sustained) == [tuple(float(value) for value in point) for point in sustained]


def test_maya_path_sampling_covers_full_range_with_exact_endpoints():
    source = (_CLIENTS / "maya_stream.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_trajectory_sample_indices")
    function.args.defaults = [ast.Constant(value=600)]
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "maya_path_sampling", "exec"), namespace)

    indices = namespace["_trajectory_sample_indices"](10_000)
    assert indices[0] == 0
    assert indices[-1] == 9_999
    assert len(indices) <= 600
    assert indices == sorted(set(indices))


def test_isaac_sim_script_editor_client_is_self_contained_and_safe_for_unmatched_dofs():
    source = (_CLIENTS / "isaac_sim_stream.py").read_text(encoding="utf-8")
    assert "show_blacknode_isaac_window" in source
    assert "omni.ui as ui" in source
    assert "isaacsim.core.experimental.prims import Articulation" in source
    assert "set_dof_position_targets" in source
    assert "articulation.set_dof_position_targets(targets)" in source
    assert 'frame.get("kind") == "blacknode.stream-schema"' in source
    assert "targets = current.reshape(-1)" in source
    assert "ROS 2 and terminal processes are not required" in source
    assert "Use selected prim" in source
    assert "type its path" in source
    assert "Dataset joint mapping" in source
    assert 'ui.Button("Discover Joints"' in source
    assert 'ui.Button("Apply Drive Settings"' in source
    assert 'ui.Label("Stiffness"' in source
    assert 'ui.Label("Damping"' in source
    assert 'ui.Label("Max Force"' in source
    assert "UsdPhysics.DriveAPI.Apply" in source
    assert '"drive_settings": dict(_drive_settings)' in source
    assert "per-joint home calibration" in source
    assert "_joint_mapping" in source
    assert "numeric * float(mapping.get(\"sign\", 1.0))" in source
    assert 'float(mapping.get("scale", 1.0))' in source
    assert 'ui.Label("Motion Scale"' in source
    assert "Joint Angle" in source
    assert "Angle Nudge" in source
    assert "_sync_joint_angle_sliders" in source
    assert "display_actual = (actual - base) * factor" in source
    assert "slider.model.set_value(display_actual)" in source
    assert "_set_joint_angle" in source
    assert 'angle must be finite' in source
    assert 'ui.FloatSlider(min=lower * factor, max=upper * factor' in source
    assert "ui.FloatField(angle_slider.model" in source
    assert "Set Home Pose" in source
    assert "_set_home_pose" in source
    assert "_go_home_pose" in source
    assert 'ui.Button("Go Home"' in source
    assert 'ui.Button("Calibrate"' in source
    assert 'ui.Button("Add to Home Pose"' in source
    assert 'ui.Button("Save Calibration File"' in source
    assert 'ui.Button("Load Calibration File"' in source
    assert "blacknode.isaac-articulation-calibration" in source
    assert '"home_pose"' in source
    assert "_begin_home_calibration" in source
    assert "_cancel_home_calibration" in source
    assert "_add_to_home_pose" in source
    assert "previous_home + delta" in source
    assert "replay motion is held while applying nudges" in source
    assert "_zero_angle_sliders" in source
    assert "get_dof_limits" in source
    assert "Angle units" in source
    assert "set_string(_PREF_KEY" in source
    assert "_reapply_current_frame" in source
    assert "_state[\"last_frame\"] = dict(frame)" in source
    assert "sim_home" in source
    assert "dataset_home" in source
    assert 'width=1360, height=720' in source
    assert 'ui.ScrollingFrame(height=360)' in source
