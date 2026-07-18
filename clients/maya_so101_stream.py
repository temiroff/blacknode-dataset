"""Self-contained Maya window for a Blacknode StreamPublisher stream.

Paste-and-run in Maya's Script Editor (Python), exactly like a standalone tool —
no imports of other files and no sys.path setup, because a minimal WebSocket
client is inlined (this is why plain exec works here but not for maya_client.py):

    exec(open(r"E:\\F\\PROJECTS\\NVDIA\\Blacknode\\packages\\blacknode-dataset\\clients\\maya_so101_stream.py").read())
    show_so101_stream_window()

Edit JOINT_MAP below for your rig (stream joint name -> Maya attribute). Values
are radians by default; rotate.* attributes are converted to degrees for you.
You can also skip the window: start_so101_stream("ws://127.0.0.1:8765") / stop_so101_stream().
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

# --- edit for your rig: stream joint name -> {"attr": ..., "scale": ?, "offset": ?} ---
JOINT_MAP = {
    "shoulder_pan":  {"attr": "so101_shoulder_pan.rotateY"},
    "shoulder_lift": {"attr": "so101_shoulder_lift.rotateX", "scale": -1.0},
    "elbow_flex":    {"attr": "so101_elbow.rotateX"},
    "wrist_flex":    {"attr": "so101_wrist_flex.rotateX"},
    "wrist_roll":    {"attr": "so101_wrist_roll.rotateZ"},
    "gripper":       {"attr": "so101_gripper.translateZ", "scale": 0.01},
}

_STATUS = "bnSo101Status"
_state = {"sock": None, "thread": None, "frames": 0, "err": "", "running": False}


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
    _apply(frame)
    if cmds.text(_STATUS, exists=True):
        cmds.text(_STATUS, edit=True, label=f"streaming - {_state['frames']} frames")


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
            # Maya is not thread-safe: apply + status update on the main thread.
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


def start_so101_stream(url="ws://127.0.0.1:8765"):
    if _state["running"]:
        cmds.warning("blacknode: already streaming; stop first")
        return
    _state.update(frames=0, err="", running=True)
    _state["thread"] = threading.Thread(target=_run, args=(url,), daemon=True, name="blacknode-so101")
    _state["thread"].start()
    _set_status(f"connecting to {url}")


def stop_so101_stream():
    _state["running"] = False
    try:
        _state["sock"].close()
    except Exception:  # noqa: BLE001
        pass
    _set_status("stopped")


def show_so101_stream_window():
    win = "bnSo101StreamWin"
    if cmds.window(win, exists=True):
        cmds.deleteUI(win)
    cmds.window(win, title="Blacknode SO-101 Stream", widthHeight=(340, 120), sizeable=False)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=8, columnOffset=("both", 10))
    cmds.text(label="")
    url_field = cmds.textFieldGrp(label="URL", text="ws://127.0.0.1:8765", columnWidth2=(36, 270))
    cmds.rowLayout(numberOfColumns=2, columnWidth2=(150, 150), columnAttach=[(1, "both", 4), (2, "both", 4)])
    cmds.button(label="▶ Start", height=30,
                command=lambda *_: start_so101_stream(cmds.textFieldGrp(url_field, query=True, text=True)))
    cmds.button(label="■ Stop", height=30, command=lambda *_: stop_so101_stream())
    cmds.setParent("..")
    cmds.text(_STATUS, label="stopped", align="left")
    cmds.showWindow(win)
