"""Self-contained, generic Maya window for a Blacknode StreamPublisher stream.

Works for any robot: the joint names are read from the stream itself (the
dataset), so nothing here is robot-specific. A minimal WebSocket client is
inlined, so plain exec works with no sys.path setup:

    exec(open(r"E:\\F\\PROJECTS\\NVDIA\\Blacknode\\packages\\blacknode-dataset\\clients\\maya_stream.py").read())
    show_blacknode_stream_window()

Click Get joints / Connect: the publisher immediately sends the dataset joint
schema, without moving the rig. Map each joint to a Maya attribute, choose its
X/Y/Z axis and +/- direction, and optionally change the scale magnitude. The
mapping is saved in Maya preferences and restored next time. Poses are applied
only while Dataset Browser playback runs or its timeline is scrubbed. Values
are radians by default; rotate.* attributes are converted to degrees for you.
Unmapped/blank rows are ignored. Enable Path on any mapped joint to draw a
world-space spline through that rig node's positions across the episode.
"""
from __future__ import annotations

import base64
import json
import math
import os
import socket
import ssl
import struct
import threading
from urllib.parse import urlparse

import maya.cmds as cmds
import maya.utils

_URL = "bnStreamUrl"
_STATUS = "bnStreamStatus"
_JOINTS_COL = "bnStreamJoints"
_MAPPING_OPTION = "blacknodeDatasetJointMappingV1"
_PATH_GROUP = "blacknodeDatasetDebugPaths"

JOINT_MAP: dict[str, dict] = {}   # joint name -> {"attr": ..., "scale": ...}, built from the UI
_rows: dict[str, tuple] = {}      # joint name -> (attr, axis, sign, magnitude, path checkbox)
_path_points: dict[str, dict[int, tuple[float, float, float]]] = {}
_path_curves: dict[str, str] = {}
_state = {"sock": None, "thread": None, "frames": 0, "err": "", "running": False,
          "joints": None, "trajectory": None, "building_paths": False,
          "latest_frame": None, "pending_schema": None, "dropped": 0,
          "pump_job": None, "lock": threading.Lock()}


# ---------------- minimal inlined WebSocket text client (stdlib only) ----------------
def _ws_connect(url, timeout=10):
    u = urlparse(url)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "wss" else 80)
    path = (u.path or "/") + (("?" + u.query) if u.query else "")
    sock = socket.create_connection((host, port), timeout=timeout)
    if u.scheme == "wss":
        sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                  f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                  "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(1024)
        if not chunk:
            raise ConnectionError("handshake failed")
        buf += chunk
    if b" 101 " not in buf.split(b"\r\n", 1)[0]:
        raise ConnectionError("server refused the WebSocket upgrade")
    sock.settimeout(None)
    return sock, buf.split(b"\r\n\r\n", 1)[1]


def _ws_frames(sock, rest):
    buf = rest

    def read(n):
        nonlocal buf
        while len(buf) < n:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        out, buf = buf[:n], buf[n:]
        return out

    while True:
        b0, b1 = read(2)
        opcode = b0 & 0x0F
        masked = b1 & 0x80
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", read(8))[0]
        mask = read(4) if masked else b""
        data = read(length)
        if masked:
            data = bytes(x ^ mask[i % 4] for i, x in enumerate(data))
        if opcode == 0x8:  # close
            return
        if opcode == 0x1:  # text
            yield data.decode("utf-8")
# ---------------- end WS client ----------------


def _load_mapping():
    if not cmds.optionVar(exists=_MAPPING_OPTION):
        return {}
    try:
        value = json.loads(cmds.optionVar(query=_MAPPING_OPTION) or "{}")
        return value if isinstance(value, dict) else {}
    except Exception:  # noqa: BLE001 - a damaged preference should not block the client
        return {}


def _save_mapping():
    cmds.optionVar(stringValue=(_MAPPING_OPTION, json.dumps(JOINT_MAP, sort_keys=True)))


def _sync_axis_from_attr(name):
    row = _rows.get(name)
    if not row:
        return
    attr = cmds.textField(row[0], query=True, text=True).strip()
    if attr and attr[-1:].upper() in {"X", "Y", "Z"}:
        cmds.optionMenu(row[1], edit=True, value=attr[-1:].upper())
    apply_mapping(rebuild_paths=True)


def apply_mapping(rebuild_paths=False):
    """Read the joint rows in the window into JOINT_MAP."""
    for name in _path_curves:
        _set_path_visibility(name, False)
    JOINT_MAP.clear()
    for name, (attr_ctrl, axis_ctrl, sign_ctrl, magnitude_ctrl, path_ctrl) in _rows.items():
        attr = cmds.textField(attr_ctrl, query=True, text=True).strip()
        if not attr:
            continue
        axis = cmds.optionMenu(axis_ctrl, query=True, value=True)
        if attr[-1:].upper() in {"X", "Y", "Z"}:
            attr = attr[:-1] + axis
            cmds.textField(attr_ctrl, edit=True, text=attr)
        sign = -1.0 if cmds.optionMenu(sign_ctrl, query=True, value=True) == "-1" else 1.0
        magnitude = max(0.0, float(cmds.floatField(magnitude_ctrl, query=True, value=True)))
        JOINT_MAP[name] = {"attr": attr, "axis": axis, "sign": int(sign),
                           "magnitude": magnitude, "scale": sign * magnitude,
                           "debug_path": bool(cmds.checkBox(path_ctrl, query=True, value=True))}
        _set_path_visibility(name, JOINT_MAP[name]["debug_path"])
    _save_mapping()
    _refresh_debug_paths(force=True)
    if rebuild_paths and _state.get("trajectory") and not _state.get("building_paths"):
        _build_full_trajectory_paths(_state["trajectory"])


def _on_path_changed(name, enabled):
    """Apply a Path checkbox immediately and report what Maya built."""
    apply_mapping(rebuild_paths=False)
    enabled = bool(enabled)
    if not enabled:
        _set_path_visibility(name, False)
        _set_status(f"{name} path hidden")
        return
    target = JOINT_MAP.get(name) or {}
    node = str(target.get("attr") or "").rsplit(".", 1)[0]
    if not node or not cmds.objExists(node):
        _set_status(f"Path error: {name} has no valid mapped Maya node")
        return
    trajectory = _state.get("trajectory")
    if not trajectory or not trajectory.get("trajectory"):
        _set_status(f"{name} path enabled - waiting for the complete episode trajectory")
        return
    result = _build_full_trajectory_paths(trajectory)
    if name in result.get("built", []):
        _set_status(f"{name} full episode path ready")
    elif name in result.get("stationary", []):
        _set_status(f"{name} path has no world-space movement on mapped node {node}")
    else:
        detail = (result.get("errors") or {}).get(name, "could not create curve")
        _set_status(f"Path error for {name}: {detail}")


def _build_rows(names):
    if not cmds.columnLayout(_JOINTS_COL, exists=True):
        return
    for child in (cmds.columnLayout(_JOINTS_COL, query=True, childArray=True) or []):
        cmds.deleteUI(child)
    _rows.clear()
    saved_mapping = _load_mapping()
    for name in names:
        saved = dict(saved_mapping.get(name) or {})
        attr = str(saved.get("attr") or f"{name}.rotateZ")
        axis = str(saved.get("axis") or (attr[-1:].upper() if attr[-1:].upper() in {"X", "Y", "Z"} else "Z"))
        scale = float(saved.get("scale", 1.0))
        cmds.rowLayout(numberOfColumns=6, parent=_JOINTS_COL,
                       columnWidth6=(100, 175, 44, 44, 58, 48),
                       columnAttach=[(1, "both", 2), (2, "both", 2), (3, "both", 2),
                                     (4, "both", 2), (5, "both", 2), (6, "both", 2)])
        cmds.text(label=name, align="left")
        attr_ctrl = cmds.textField(text=attr,
                                   annotation="Maya node.attribute this joint drives; blank = ignore")
        axis_ctrl = cmds.optionMenu(annotation="axis suffix",
                                    changeCommand=lambda *_: apply_mapping(rebuild_paths=True))
        for value in ("X", "Y", "Z"):
            cmds.menuItem(label=value)
        cmds.optionMenu(axis_ctrl, edit=True, value=axis)
        sign_ctrl = cmds.optionMenu(annotation="direction",
                                    changeCommand=lambda *_: apply_mapping(rebuild_paths=True))
        cmds.menuItem(label="+1")
        cmds.menuItem(label="-1")
        cmds.optionMenu(sign_ctrl, edit=True, value="-1" if scale < 0 else "+1")
        magnitude_ctrl = cmds.floatField(value=abs(scale), minValue=0.0, precision=4,
                                         changeCommand=lambda *_: apply_mapping(rebuild_paths=True),
                                         annotation="scale magnitude")
        path_ctrl = cmds.checkBox(label="Path", value=bool(saved.get("debug_path", False)),
                                  changeCommand=lambda enabled, joint=name: _on_path_changed(joint, enabled),
                                  annotation="Show this rig node's world-space episode path")
        cmds.setParent("..")
        _rows[name] = (attr_ctrl, axis_ctrl, sign_ctrl, magnitude_ctrl, path_ctrl)
        cmds.textField(attr_ctrl, edit=True, changeCommand=lambda *_, joint=name: _sync_axis_from_attr(joint))
    apply_mapping()


def _curve_name(name):
    safe = "".join(character if character.isalnum() or character == "_" else "_" for character in name)
    return f"bnPath_{safe}"


def _set_path_visibility(name, visible):
    curve = _path_curves.get(name)
    if curve and cmds.objExists(curve):
        cmds.setAttr(f"{curve}.visibility", bool(visible))


def _rebuild_debug_path(name):
    points = [_path_points[name][index] for index in sorted(_path_points.get(name, {}))]
    points = _reject_discontinuity_outliers(points)
    points = [point for index, point in enumerate(points) if index == 0 or point != points[index - 1]]
    if len(points) < 2:
        return False
    curve = _path_curves.get(name)
    degree = min(3, len(points) - 1)
    if curve and cmds.objExists(curve):
        curve = cmds.curve(curve, replace=True, degree=degree, editPoint=points)
    else:
        if not cmds.objExists(_PATH_GROUP):
            cmds.group(empty=True, name=_PATH_GROUP)
        curve = cmds.curve(name=_curve_name(name), degree=degree, editPoint=points)
        cmds.parent(curve, _PATH_GROUP)
        _path_curves[name] = curve
    shape = (cmds.listRelatives(curve, shapes=True, fullPath=True) or [None])[0]
    if shape:
        for attribute, values in (
            ("overrideEnabled", (1,)),
            ("overrideRGBColors", (1,)),
            ("overrideColorRGB", (1.0, 0.0, 0.0)),
            ("lineWidth", (4.0,)),
        ):
            plug = f"{shape}.{attribute}"
            if cmds.objExists(plug):
                try:
                    cmds.setAttr(plug, *values)
                except Exception:  # noqa: BLE001 - keep the curve if one display option is unsupported
                    pass
    _set_path_visibility(name, bool((JOINT_MAP.get(name) or {}).get("debug_path")))
    return True


def _refresh_debug_paths(force=False):
    enabled = [name for name, target in JOINT_MAP.items() if target.get("debug_path")]
    if not enabled:
        return
    if not force:
        return
    selection = cmds.ls(selection=True, long=True) or []
    result = {"built": [], "stationary": [], "errors": {}}
    try:
        for name in enabled:
            try:
                bucket = "built" if _rebuild_debug_path(name) else "stationary"
                result[bucket].append(name)
            except Exception as exc:  # noqa: BLE001 - debug drawing must not interrupt rig replay
                result["errors"][name] = f"{type(exc).__name__}: {exc}"
    finally:
        selection = [node for node in selection if cmds.objExists(node)]
        if selection:
            cmds.select(selection, replace=True)
        else:
            cmds.select(clear=True)
    return result


def _distance(a, b):
    return math.sqrt(sum((float(left) - float(right)) ** 2 for left, right in zip(a, b)))


def _median(values):
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def _reject_discontinuity_outliers(points):
    """Drop isolated world-space spikes while preserving sustained robot motion."""
    valid = [tuple(float(value) for value in point[:3]) for point in points
             if len(point) >= 3 and all(math.isfinite(float(value)) for value in point[:3])]
    if len(valid) < 4:
        return valid
    steps = [_distance(valid[index - 1], valid[index]) for index in range(1, len(valid))]
    typical = _median(steps)
    deviations = [abs(step - typical) for step in steps]
    threshold = max(typical * 8.0, typical + 8.0 * _median(deviations), 1e-4)
    filtered = [valid[0]]
    for index in range(1, len(valid) - 1):
        previous, point, following = filtered[-1], valid[index], valid[index + 1]
        isolated_spike = (_distance(previous, point) > threshold
                          and _distance(point, following) > threshold
                          and _distance(previous, following) <= threshold * 2.0)
        if not isolated_spike:
            filtered.append(point)
    filtered.append(valid[-1])
    return filtered


def _apply_positions(names, positions, units):
    radians = str(units or "radians").startswith("rad")
    for name, value in zip(names, positions):
        target = JOINT_MAP.get(name)
        if not target:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        out = numeric * float(target.get("scale", 1.0)) + float(target.get("offset", 0.0))
        if radians and ".rotate" in target["attr"].lower():
            out = math.degrees(out)
        try:
            cmds.setAttr(target["attr"], out)
        except Exception:  # noqa: BLE001 - keep evaluating past a missing attr
            pass


def _build_full_trajectory_paths(message):
    """Evaluate every streamed episode pose once and build complete Maya paths."""
    trajectory = list(message.get("trajectory") or [])
    names = list(message.get("joint_names") or [])
    enabled = {name for name, target in JOINT_MAP.items() if target.get("debug_path")}
    empty = {"built": [], "stationary": [], "errors": {}}
    if not trajectory or not names or not enabled or _state.get("building_paths"):
        return empty
    _state["building_paths"] = True
    _state["trajectory"] = message
    saved_values = {}
    selection = cmds.ls(selection=True, long=True) or []
    undo_enabled = bool(cmds.undoInfo(query=True, state=True))
    try:
        cmds.undoInfo(stateWithoutFlush=False)
        clear_debug_paths()
        for target in JOINT_MAP.values():
            attr = str(target.get("attr") or "")
            if attr and cmds.objExists(attr):
                try:
                    saved_values[attr] = cmds.getAttr(attr)
                except Exception:  # noqa: BLE001
                    pass
        for frame_index, positions in enumerate(trajectory):
            _apply_positions(names, positions, message.get("units"))
            for name in enabled:
                target = JOINT_MAP.get(name) or {}
                node = str(target.get("attr") or "").rsplit(".", 1)[0]
                if not node or not cmds.objExists(node):
                    continue
                try:
                    position = cmds.xform(node, query=True, worldSpace=True, translation=True)
                    _path_points.setdefault(name, {})[frame_index] = tuple(
                        float(value) for value in position[:3])
                except Exception:  # noqa: BLE001 - one invalid rig node must not block other paths
                    continue
        result = _refresh_debug_paths(force=True)
    finally:
        for attr, value in saved_values.items():
            try:
                cmds.setAttr(attr, value)
            except Exception:  # noqa: BLE001
                pass
        selection = [node for node in selection if cmds.objExists(node)]
        cmds.select(selection, replace=True) if selection else cmds.select(clear=True)
        if undo_enabled:
            cmds.undoInfo(stateWithoutFlush=True)
        _state["building_paths"] = False
    return result


def clear_debug_paths(*_):
    """Delete generated curves and forget sampled positions for the current replay."""
    _path_points.clear()
    _path_curves.clear()
    if cmds.objExists(_PATH_GROUP):
        cmds.delete(_PATH_GROUP)


def _apply(frame):
    _apply_positions(frame.get("joint_names") or [], frame.get("positions") or [], frame.get("units"))


def _tick(frame):
    names = frame.get("joint_names") or []
    if names and names != _state.get("joints"):
        _state["joints"] = names
        _build_rows(names)  # joints come from the dataset, not a hardcoded list
    if frame.get("kind") in {"blacknode.stream-schema", "blacknode.stream-trajectory"}:
        _state["trajectory"] = frame
        if any(target.get("debug_path") for target in JOINT_MAP.values()):
            result = _build_full_trajectory_paths(frame)
            built = len(result.get("built", []))
            stationary = len(result.get("stationary", []))
            errors = len(result.get("errors", {}))
            suffix = f"{built} path(s) ready, {stationary} stationary, {errors} error(s)"
        else:
            suffix = "enable a Path checkbox to build a full episode path"
        _set_status(f"{len(names)} joints loaded - {suffix}")
        return
    _apply(frame)
    _set_status(f"streaming latest - {_state['frames']} received - {_state['dropped']} stale dropped - "
                f"{len(JOINT_MAP)}/{len(names)} joints mapped")


def _drain_pending():
    """Apply only the newest network frame from Maya's main thread."""
    with _state["lock"]:
        schema = _state.get("pending_schema")
        frame = _state.get("latest_frame")
        _state["pending_schema"] = None
        _state["latest_frame"] = None
    if schema is not None:
        _tick(schema)
    if frame is not None:
        _tick(frame)


def _set_status(label):
    if cmds.text(_STATUS, exists=True):
        cmds.text(_STATUS, edit=True, label=label)


def _run(url):
    try:
        sock, rest = _ws_connect(url)
        _state["sock"] = sock
        _state["err"] = ""
        for text in _ws_frames(sock, rest):
            if not _state["running"]:
                break
            frame = json.loads(text)
            with _state["lock"]:
                if frame.get("kind") in {"blacknode.stream-schema", "blacknode.stream-trajectory"}:
                    _state["pending_schema"] = frame
                else:
                    _state["frames"] += 1
                    if _state.get("latest_frame") is not None:
                        _state["dropped"] += 1
                    _state["latest_frame"] = frame
    except Exception as exc:  # noqa: BLE001 - surfaced in the window
        _state["err"] = str(exc)
    finally:
        _state["running"] = False
        try:
            _state["sock"].close()
        except Exception:  # noqa: BLE001
            pass
        maya.utils.executeDeferred(_set_status, f"error: {_state['err']}" if _state["err"] else "stopped")


def start_blacknode_stream(url="ws://127.0.0.1:8765"):
    if _state["running"]:
        cmds.warning("blacknode: already streaming; stop first")
        return
    clear_debug_paths()
    _state.update(frames=0, err="", running=True, joints=None, trajectory=None, building_paths=False,
                  latest_frame=None, pending_schema=None, dropped=0)
    _state["thread"] = threading.Thread(target=_run, args=(url,), daemon=True, name="blacknode-stream")
    _state["thread"].start()
    _set_status(f"connecting to {url}")


def stop_blacknode_stream():
    _state["running"] = False
    with _state["lock"]:
        _state["latest_frame"] = None
        _state["pending_schema"] = None
    try:
        _state["sock"].close()
    except Exception:  # noqa: BLE001
        pass
    _set_status("stopped")


def show_blacknode_stream_window():
    win = "bnStreamWin"
    if cmds.window(win, exists=True):
        cmds.deleteUI(win)
    cmds.window(win, title="Blacknode Dataset Replay", widthHeight=(590, 540))
    main = cmds.columnLayout(adjustableColumn=True, rowSpacing=6, columnOffset=("both", 10))
    cmds.text(label="")
    cmds.textFieldGrp(_URL, label="URL", text="ws://127.0.0.1:8765", columnWidth2=(36, 320))
    cmds.rowLayout(numberOfColumns=2, columnWidth2=(265, 265),
                   columnAttach=[(1, "both", 3), (2, "both", 3)])
    cmds.button(label="Get joints / Connect", height=30,
                command=lambda *_: start_blacknode_stream(cmds.textFieldGrp(_URL, query=True, text=True)))
    cmds.button(label="Stop", height=30, command=lambda *_: stop_blacknode_stream())
    cmds.setParent("..")
    cmds.text(_STATUS, label="stopped", align="left")
    cmds.frameLayout(label="Dataset joint Maya attribute  axis  direction  scale  debug path",
                     collapsable=False, marginHeight=4)
    cmds.scrollLayout(height=300)
    cmds.columnLayout(_JOINTS_COL, adjustableColumn=True, rowSpacing=3)
    cmds.setParent(main)
    cmds.button(label="Clear debug paths", height=26, command=clear_debug_paths,
                annotation="Delete all generated episode path curves")
    _state["pump_job"] = cmds.scriptJob(idleEvent=_drain_pending, parent=win, protected=True)
    cmds.showWindow(win)
