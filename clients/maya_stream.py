"""Self-contained, generic Maya window for a Blacknode StreamPublisher stream.

Works for any robot: the joint names are read from the stream itself (the
dataset), so nothing here is robot-specific. A minimal WebSocket client is
inlined, so plain exec works with no sys.path setup:

    exec(open(r"E:\\F\\PROJECTS\\NVDIA\\Blacknode\\packages\\blacknode-dataset\\clients\\maya_stream.py").read())
    show_blacknode_stream_window()

Click Connect: the window discovers the joints from the incoming frames and adds
one row per joint (joint name -> Maya attribute + scale). Fill in your rig's
attributes and edits apply live. Values are radians by default; rotate.*
attributes are converted to degrees for you. Unmapped/blank rows are ignored.
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

JOINT_MAP: dict[str, dict] = {}   # joint name -> {"attr": ..., "scale": ...}, built from the UI
_rows: dict[str, tuple] = {}      # joint name -> (attr textField, scale textField)
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


def apply_mapping():
    """Read the joint rows in the window into JOINT_MAP."""
    JOINT_MAP.clear()
    for name, (attr_ctrl, scale_ctrl) in _rows.items():
        attr = cmds.textField(attr_ctrl, query=True, text=True).strip()
        if not attr:
            continue
        try:
            scale = float(cmds.textField(scale_ctrl, query=True, text=True) or "1")
        except ValueError:
            scale = 1.0
        JOINT_MAP[name] = {"attr": attr, "scale": scale}


def _build_rows(names):
    if not cmds.columnLayout(_JOINTS_COL, exists=True):
        return
    for child in (cmds.columnLayout(_JOINTS_COL, query=True, childArray=True) or []):
        cmds.deleteUI(child)
    _rows.clear()
    for name in names:
        cmds.rowLayout(numberOfColumns=3, parent=_JOINTS_COL,
                       columnWidth3=(120, 170, 44), columnAlign3=("left", "left", "left"),
                       columnAttach=[(1, "both", 2), (2, "both", 2), (3, "both", 2)])
        cmds.text(label=name, align="left")
        attr_ctrl = cmds.textField(text=f"{name}.rotateZ",
                                   annotation="Maya node.attribute this joint drives; blank = ignore")
        scale_ctrl = cmds.textField(text="1", annotation="multiply the joint value")
        cmds.setParent("..")
        _rows[name] = (attr_ctrl, scale_ctrl)
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
            _state["frames"] += 1
            maya.utils.executeInMainThreadWithResult(_tick, json.loads(text))
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
    cmds.window(win, title="Blacknode Stream", widthHeight=(400, 460))
    main = cmds.columnLayout(adjustableColumn=True, rowSpacing=6, columnOffset=("both", 10))
    cmds.text(label="")
    cmds.textFieldGrp(_URL, label="URL", text="ws://127.0.0.1:8765", columnWidth2=(36, 320))
    cmds.rowLayout(numberOfColumns=3, columnWidth3=(126, 126, 126),
                   columnAttach=[(1, "both", 3), (2, "both", 3), (3, "both", 3)])
    cmds.button(label="▶ Connect", height=30,
                command=lambda *_: start_blacknode_stream(cmds.textFieldGrp(_URL, query=True, text=True)))
    cmds.button(label="■ Stop", height=30, command=lambda *_: stop_blacknode_stream())
    cmds.button(label="Apply mapping", height=30, command=lambda *_: apply_mapping())
    cmds.setParent("..")
    cmds.text(_STATUS, label="stopped", align="left")
    cmds.frameLayout(label="Joints (from dataset)", collapsable=False, marginHeight=4)
    cmds.scrollLayout(height=250)
    cmds.columnLayout(_JOINTS_COL, adjustableColumn=True, rowSpacing=3)
    cmds.setParent(main)
    cmds.showWindow(win)
