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
Unmapped/blank rows are ignored.
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

JOINT_MAP: dict[str, dict] = {}   # joint name -> {"attr": ..., "scale": ...}, built from the UI
_rows: dict[str, tuple] = {}      # joint name -> (attr field, axis menu, sign menu, magnitude field)
_state = {"sock": None, "thread": None, "frames": 0, "err": "", "running": False, "joints": None}


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
    apply_mapping()


def apply_mapping():
    """Read the joint rows in the window into JOINT_MAP."""
    JOINT_MAP.clear()
    for name, (attr_ctrl, axis_ctrl, sign_ctrl, magnitude_ctrl) in _rows.items():
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
                           "magnitude": magnitude, "scale": sign * magnitude}
    _save_mapping()


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
        cmds.rowLayout(numberOfColumns=5, parent=_JOINTS_COL,
                       columnWidth5=(110, 190, 48, 48, 62),
                       columnAttach=[(1, "both", 2), (2, "both", 2), (3, "both", 2),
                                     (4, "both", 2), (5, "both", 2)])
        cmds.text(label=name, align="left")
        attr_ctrl = cmds.textField(text=attr,
                                   annotation="Maya node.attribute this joint drives; blank = ignore")
        axis_ctrl = cmds.optionMenu(annotation="axis suffix", changeCommand=lambda *_: apply_mapping())
        for value in ("X", "Y", "Z"):
            cmds.menuItem(label=value)
        cmds.optionMenu(axis_ctrl, edit=True, value=axis)
        sign_ctrl = cmds.optionMenu(annotation="direction", changeCommand=lambda *_: apply_mapping())
        cmds.menuItem(label="+1")
        cmds.menuItem(label="-1")
        cmds.optionMenu(sign_ctrl, edit=True, value="-1" if scale < 0 else "+1")
        magnitude_ctrl = cmds.floatField(value=abs(scale), minValue=0.0, precision=4,
                                         changeCommand=lambda *_: apply_mapping(),
                                         annotation="scale magnitude")
        cmds.setParent("..")
        _rows[name] = (attr_ctrl, axis_ctrl, sign_ctrl, magnitude_ctrl)
        cmds.textField(attr_ctrl, edit=True, changeCommand=lambda *_, joint=name: _sync_axis_from_attr(joint))
    apply_mapping()


def _apply(frame):
    names = frame.get("joint_names") or []
    positions = frame.get("positions") or []
    radians = str(frame.get("units") or "radians").startswith("rad")
    for name, value in zip(names, positions):
        target = JOINT_MAP.get(name)
        if not target:
            continue
        out = float(value) * float(target.get("scale", 1.0)) + float(target.get("offset", 0.0))
        if radians and ".rotate" in target["attr"].lower():
            out = math.degrees(out)
        try:
            cmds.setAttr(target["attr"], out)
        except Exception:  # noqa: BLE001 - keep streaming past a missing attr
            pass


def _tick(frame):
    names = frame.get("joint_names") or []
    if names and names != _state.get("joints"):
        _state["joints"] = names
        _build_rows(names)  # joints come from the dataset, not a hardcoded list
    if frame.get("kind") == "blacknode.stream-schema":
        _set_status(f"{len(names)} joints loaded - mapping restored - waiting for Browser play/seek")
        return
    _apply(frame)
    _set_status(f"streaming - {_state['frames']} frames - {len(JOINT_MAP)}/{len(names)} joints mapped")


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
            if frame.get("kind") != "blacknode.stream-schema":
                _state["frames"] += 1
            maya.utils.executeInMainThreadWithResult(_tick, frame)
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
    _state.update(frames=0, err="", running=True, joints=None)
    _state["thread"] = threading.Thread(target=_run, args=(url,), daemon=True, name="blacknode-stream")
    _state["thread"].start()
    _set_status(f"connecting to {url}")


def stop_blacknode_stream():
    _state["running"] = False
    try:
        _state["sock"].close()
    except Exception:  # noqa: BLE001
        pass
    _set_status("stopped")


def show_blacknode_stream_window():
    win = "bnStreamWin"
    if cmds.window(win, exists=True):
        cmds.deleteUI(win)
    cmds.window(win, title="Blacknode Dataset Replay", widthHeight=(560, 500))
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
    cmds.frameLayout(label="Dataset joint → Maya attribute · axis · direction · scale", collapsable=False, marginHeight=4)
    cmds.scrollLayout(height=290)
    cmds.columnLayout(_JOINTS_COL, adjustableColumn=True, rowSpacing=3)
    cmds.setParent(main)
    cmds.showWindow(win)
