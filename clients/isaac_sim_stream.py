"""Self-contained Blacknode replay window for the Isaac Sim Script Editor.

Open Window > Script Editor in Isaac Sim and run:

    exec(open(r"E:\\F\\PROJECTS\\NVDIA\\Blacknode\\packages\\blacknode-dataset\\clients\\isaac_sim_stream.py").read())
    show_blacknode_isaac_window()

Enter the StreamPublisher URL and the articulation-root prim path, then click
Connect. Dataset joints are matched to Isaac Sim DOFs by exact name. Unmatched
DOFs retain their current targets. The WebSocket uses only Python's standard
library; ROS 2 and terminal processes are not required.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import queue
import socket
import ssl
import struct
import threading
from pathlib import Path
from urllib.parse import urlparse

import omni.kit.app
import omni.timeline
import omni.ui as ui

_window = None
_url_model = None
_prim_model = None
_status_label = None
_joint_stack = None
_calibrate_button = None
_joint_mapping = {}
_articulation = None
_dof_lookup = {}
_calibration = {}
_dataset_home = {}
_dof_limits = {}
_joint_sliders = {}
_manual_targets = {}
_stall_state = {}
_joint_prim_paths = {}
_drive_settings = {}
_nudge_base = {}


def _discover_joint_prims(root_path, dof_names):
    """Find USD joint prims below the articulation root by basename/DOF name."""
    found = {}
    try:
        import omni.usd
        from pxr import Usd
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(str(root_path))
        if not root:
            return found
        wanted = {str(name).lower(): str(name) for name in dof_names}
        for prim in Usd.PrimRange(root):
            if "joint" not in str(prim.GetTypeName()).lower() and "joint" not in str(prim.GetName()).lower():
                continue
            base = str(prim.GetName()).lower()
            for key, dof in wanted.items():
                if base == key or base == key + "_joint" or base.endswith("_" + key):
                    found[dof] = str(prim.GetPath())
        return found
    except Exception:
        return found


def _drive_type_for_prim(prim):
    return "linear" if "prismatic" in str(prim.GetTypeName()).lower() else "angular"


def _read_joint_drive(dof):
    """Read authored USD drive values for one discovered articulation DOF."""
    prim_path = _joint_prim_paths.get(str(dof))
    if not prim_path:
        return {}
    try:
        import omni.usd
        from pxr import UsdPhysics
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return {}
        drive = UsdPhysics.DriveAPI.Get(prim, _drive_type_for_prim(prim))
        values = {}
        for key, getter in (("stiffness", drive.GetStiffnessAttr),
                            ("damping", drive.GetDampingAttr),
                            ("max_force", drive.GetMaxForceAttr)):
            value = getter().Get()
            if value is not None and math.isfinite(float(value)):
                values[key] = float(value)
        return values
    except Exception:
        return {}


def _apply_joint_drive(joint_name):
    """Apply one row's saved stiffness, damping and force to its USD drive."""
    mapping = _joint_mapping.get(str(joint_name)) or {}
    dof = mapping.get("dof")
    prim_path = _joint_prim_paths.get(dof)
    settings = _drive_settings.get(str(joint_name)) or {}
    if not dof or not prim_path:
        raise ValueError(f"{joint_name}: no discovered USD joint prim for DOF {dof or '(ignore)'}")
    import omni.usd
    from pxr import UsdPhysics
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"{joint_name}: USD joint prim is unavailable: {prim_path}")
    drive = UsdPhysics.DriveAPI.Apply(prim, _drive_type_for_prim(prim))
    if "stiffness" in settings:
        drive.CreateStiffnessAttr().Set(float(settings["stiffness"]))
    if "damping" in settings:
        drive.CreateDampingAttr().Set(float(settings["damping"]))
    if "max_force" in settings:
        drive.CreateMaxForceAttr().Set(float(settings["max_force"]))
    return prim_path


def _set_drive_value(joint_name, field, value):
    try:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"{field} must be a finite value >= 0")
        _drive_settings.setdefault(str(joint_name), {})[str(field)] = numeric
        path = _apply_joint_drive(str(joint_name))
        _save_preferences()
        _set_status(f"drive updated: {joint_name} · {field}={numeric:g} · {path}")
    except Exception as exc:  # noqa: BLE001 - shown in the Isaac window
        _set_status(f"error: drive setting was not applied: {exc}")


def _apply_all_drive_settings():
    applied = 0
    errors = []
    for name in list(_drive_settings):
        try:
            _apply_joint_drive(name)
            applied += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    _save_preferences()
    if errors:
        _set_status(f"error: applied {applied} drive(s); {errors[0]}")
    else:
        _set_status(f"drive settings applied to {applied} joint(s)")


def _discover_joints_from_ui():
    root = _prim_model.as_string if _prim_model is not None else ""
    names = list(_state.get("dof_names") or [])
    _joint_prim_paths.clear()
    _joint_prim_paths.update(_discover_joint_prims(root, names))
    if _joint_prim_paths:
        _set_status(f"discovered {len(_joint_prim_paths)}/{len(names)} USD joint prim(s)")
        if _state.get("schema_joints"):
            _build_joint_rows(_state["schema_joints"], names)
    else:
        _set_status(f"error: no USD joint prims found below {root}")
_unit_model = None
_PREF_KEY = "/persistent/blacknode/dataset_isaac_stream"
_CALIBRATION_KIND = "blacknode.isaac-articulation-calibration"
_prefs = {}
_state = {
    "running": False, "generation": 0, "sock": None, "thread": None,
    "queue": queue.Queue(maxsize=2), "status": "stopped", "frames": 0,
    "schema_joints": [],
    "dof_names": [], "rows_ready": False,
    "dataset_home": {}, "articulation": None, "angle_unit": "degrees", "last_frame": None,
    "nudge_mode": False, "updating_sliders": False,
}


def _load_preferences():
    global _prefs
    try:
        import carb.settings
        raw = carb.settings.get_settings().get_as_string(_PREF_KEY)
        _prefs = json.loads(raw) if raw else {}
    except Exception:
        _prefs = {}
    return _prefs


def _save_preferences():
    payload = dict(_prefs)
    payload.update({
        "url": _url_model.as_string if _url_model is not None else payload.get("url", "ws://127.0.0.1:8765"),
        "prim_path": _prim_model.as_string if _prim_model is not None else payload.get("prim_path", "/World/Robot"),
        "angle_unit": _state["angle_unit"],
        "mapping": {name: {"dof": value.get("dof"), "sign": value.get("sign", 1.0),
                           "scale": value.get("scale", 1.0)}
                    for name, value in _joint_mapping.items()},
        "calibration": dict(_calibration),
        "drive_settings": dict(_drive_settings),
        "angles": dict(payload.get("angles") or {}),
    })
    _prefs.clear(); _prefs.update(payload)
    try:
        import carb.settings
        carb.settings.get_settings().set_string(_PREF_KEY, json.dumps(payload, sort_keys=True))
    except Exception:
        pass
    _save_calibration_artifact(silent=True)


def _calibration_artifact_path():
    prim_path = (_prim_model.as_string if _prim_model is not None
                 else str(_prefs.get("prim_path") or "/World/Robot"))
    raw_name = prim_path.rstrip("/").split("/")[-1] or "robot"
    robot_name = "".join(char if char.isalnum() or char in "-_" else "_"
                         for char in raw_name).strip("_") or "robot"
    return Path.home() / ".blacknode" / "calibrations" / "isaac" / f"{robot_name}.json"


def _save_calibration_artifact(silent=False):
    """Write a portable Blacknode calibration artifact for Isaac and RL use."""
    try:
        path = _calibration_artifact_path()
        if not _calibration and path.exists() and silent:
            return path
        payload = {
            "kind": _CALIBRATION_KIND,
            "schema_version": 1,
            "robot_id": path.stem,
            "articulation_root": (_prim_model.as_string if _prim_model is not None
                                  else str(_prefs.get("prim_path") or "")),
            "joint_units": "radians",
            "display_angle_unit": _state.get("angle_unit", "degrees"),
            "mapping": {name: dict(value) for name, value in _joint_mapping.items()},
            "home_pose": {name: dict(value) for name, value in _calibration.items()},
            "drive_settings": {name: dict(value) for name, value in _drive_settings.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        if not silent:
            _set_status(f"saved Blacknode calibration: {path}")
        return path
    except Exception as exc:  # noqa: BLE001
        if not silent:
            _set_status(f"error: calibration file was not saved: {exc}")
        return None


def _load_calibration_artifact():
    try:
        path = _calibration_artifact_path()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != _CALIBRATION_KIND or int(payload.get("schema_version") or 0) != 1:
            raise ValueError("unsupported Blacknode Isaac calibration file")
        _joint_mapping.clear(); _joint_mapping.update({
            str(name): dict(value) for name, value in dict(payload.get("mapping") or {}).items()})
        _calibration.clear(); _calibration.update({
            str(name): dict(value) for name, value in dict(payload.get("home_pose") or {}).items()})
        _drive_settings.clear(); _drive_settings.update({
            str(name): dict(value) for name, value in dict(payload.get("drive_settings") or {}).items()})
        _state["angle_unit"] = str(payload.get("display_angle_unit") or _state["angle_unit"])
        _save_preferences()
        if _state.get("schema_joints") and _state.get("dof_names"):
            _build_joint_rows(_state["schema_joints"], _state["dof_names"])
        if _articulation is not None:
            _apply_all_drive_settings()
        _set_status(f"loaded Blacknode calibration: {path}")
    except Exception as exc:  # noqa: BLE001
        _set_status(f"error: calibration file was not loaded: {exc}")


def _ws_connect(url, timeout=10):
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    path = (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
    sock = socket.create_connection((host, port), timeout=timeout)
    if parsed.scheme == "wss":
        sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                  f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                  "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = sock.recv(1024)
        if not chunk:
            raise ConnectionError("WebSocket handshake failed")
        buffer += chunk
    if b" 101 " not in buffer.split(b"\r\n", 1)[0]:
        raise ConnectionError("server refused the WebSocket upgrade")
    sock.settimeout(None)
    return sock, buffer.split(b"\r\n\r\n", 1)[1]


def _ws_frames(sock, rest):
    buffer = rest

    def read(size):
        nonlocal buffer
        while len(buffer) < size:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("stream closed")
            buffer += chunk
        result, buffer = buffer[:size], buffer[size:]
        return result

    while True:
        first, second = read(2)
        opcode, masked, length = first & 0x0F, second & 0x80, second & 0x7F
        if length == 126:
            length = struct.unpack(">H", read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", read(8))[0]
        mask = read(4) if masked else b""
        data = read(length)
        if masked:
            data = bytes(value ^ mask[index % 4] for index, value in enumerate(data))
        if opcode == 0x8:
            return
        if opcode == 0x1:
            yield data.decode("utf-8")


def _set_status(text):
    _state["status"] = str(text)
    if _status_label is not None:
          _status_label.text = _state["status"]
          low = str(text).lower()
          # Use explicit RGBA floats; packed integer colors are interpreted
          # differently by some Isaac Sim/omni.ui versions (red can appear
          # purple because of channel ordering).
          color = (0.88, 0.72, 0.30, 1.0)
          if "error" in low or "failed" in low or "invalid" in low:
              color = (1.0, 0.16, 0.16, 1.0)
          elif "streaming" in low or "ready" in low or "connected" in low or "home" in low:
              color = (0.25, 0.86, 0.48, 1.0)
          _status_label.style = {"color": color}


def _replace_queued_frame(frame):
    try:
        while True:
            _state["queue"].get_nowait()
    except queue.Empty:
        pass
    try:
        _state["queue"].put_nowait(frame)
    except queue.Full:
        pass


def _receive(url, generation):
    try:
        sock, rest = _ws_connect(url)
        if generation != _state["generation"]:
            sock.close()
            return
        _state["sock"] = sock
        _state["status"] = f"connected to {url}; waiting for Browser play/seek"
        for text in _ws_frames(sock, rest):
            if not _state["running"] or generation != _state["generation"]:
                break
            frame = json.loads(text)
            if frame.get("kind") == "blacknode.stream-schema":
                _state["schema_joints"] = list(frame.get("joint_names") or [])
                trajectory = list(frame.get("trajectory") or [])
                if trajectory:
                    _dataset_home.clear()
                    _dataset_home.update(dict(zip(_state["schema_joints"], trajectory[0])))
                    _state["dataset_home"] = dict(_dataset_home)
                continue
            _replace_queued_frame(frame)
    except Exception as exc:  # noqa: BLE001 - displayed in the Isaac Sim window
        if generation == _state["generation"]:
            _state["status"] = f"error: {type(exc).__name__}: {exc}"
    finally:
        if generation == _state["generation"]:
            _state["running"] = False
        try:
            _state["sock"].close()
        except Exception:  # noqa: BLE001
            pass


def _numpy(value):
    return value.numpy() if hasattr(value, "numpy") else value


def _apply_frame(articulation, frame, dof_lookup):
    import numpy as np

    if _state.get("nudge_mode"):
        return 0
    current = np.asarray(_numpy(articulation.get_dof_positions()), dtype=np.float32).copy()
    targets = current.reshape(-1)
    matched = 0
    for name, value in zip(frame.get("joint_names") or [], frame.get("positions") or []):
        mapping = _joint_mapping.get(str(name)) or {}
        dof_name = mapping.get("dof") or str(name)
        index = dof_lookup.get(dof_name)
        numeric = float(value)
        if index is None or not math.isfinite(numeric):
            continue
        calibration = _calibration.get(str(name)) or {}
        if calibration.get("calibrated"):
            targets[index] = (float(calibration["sim_home"])
                              + float(mapping.get("sign", 1.0))
                              * float(mapping.get("scale", 1.0))
                              * (numeric - float(calibration["dataset_home"])))
        else:
            targets[index] = (numeric * float(mapping.get("sign", 1.0))
                              * float(mapping.get("scale", 1.0)))
        matched += 1
    if matched:
        articulation.set_dof_position_targets(targets)
        _state["last_frame"] = dict(frame)
    return matched


def _reapply_current_frame():
    frame = _state.get("last_frame")
    if frame is not None and _articulation is not None and _dof_lookup:
        _apply_frame(_articulation, frame, _dof_lookup)


def _sync_joint_angle_sliders(articulation):
    """Show measured Isaac joint positions without issuing new commands."""
    if _state.get("nudge_mode") or _state.get("updating_sliders"):
        return
    import numpy as np
    positions = np.asarray(_numpy(articulation.get_dof_positions()), dtype=np.float32).reshape(-1)
    factor = 180.0 / math.pi if _state["angle_unit"] == "degrees" else 1.0
    _state["updating_sliders"] = True
    try:
        for name, slider in _joint_sliders.items():
            if name in _manual_targets:
                continue
            dof = (_joint_mapping.get(name) or {}).get("dof")
            index = _dof_lookup.get(dof)
            if index is not None and index < len(positions):
                slider.model.set_value(float(positions[index]) * factor)
    finally:
        _state["updating_sliders"] = False


async def _prepare_and_pump(prim_path, generation):
    try:
        from isaacsim.core.experimental.prims import Articulation

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
        await omni.kit.app.get_app().next_update_async()
        articulation = Articulation(prim_path)
        await omni.kit.app.get_app().next_update_async()
        if not articulation.is_physics_tensor_entity_valid():
            raise RuntimeError(f"{prim_path} is not a valid initialized articulation root")
        dof_names = list(articulation.dof_names)
        global _articulation, _dof_lookup
        _articulation = articulation
        _state["articulation"] = articulation
        _state["dof_names"] = dof_names
        dof_lookup = {name: index for index, name in enumerate(dof_names)}
        _dof_lookup = dict(dof_lookup)
        _joint_prim_paths.clear()
        _joint_prim_paths.update(_discover_joint_prims(prim_path, dof_names))
        if _drive_settings:
            _apply_all_drive_settings()
        _dof_limits.clear()
        try:
            lower, upper = articulation.get_dof_limits()
            for name, lo, hi in zip(dof_names, list(_numpy(lower)), list(_numpy(upper))):
                _dof_limits[name] = (float(lo), float(hi))
        except Exception:
            pass
        if _state.get("schema_joints"):
            _build_joint_rows(_state["schema_joints"], dof_names)
        _set_status(f"ready: {len(dof_names)} Isaac DOFs; waiting for Blacknode")
        while _state["running"] and generation == _state["generation"]:
            _set_status(_state["status"])
            if _state.get("schema_joints") and not _state.get("rows_ready"):
                _build_joint_rows(_state["schema_joints"], dof_names)
            frame = None
            try:
                while True:
                    frame = _state["queue"].get_nowait()
            except queue.Empty:
                pass
            if frame is not None:
                if _state.get("nudge_mode"):
                    _state["status"] = "CALIBRATING: replay motion is held while applying nudges"
                else:
                    matched = _apply_frame(articulation, frame, dof_lookup)
                    _manual_targets.clear()
                    _stall_state.clear()
                    _state["frames"] += 1
                    streamed = len(frame.get("joint_names") or [])
                    _state["status"] = (f"streaming frame {frame.get('frame_index', '?')} - "
                                        f"{matched}/{streamed} joints matched")
            await omni.kit.app.get_app().next_update_async()
            # If physics cannot move a manually commanded joint (for example a
            # collision stop), tighten that slider to the reached position.
            if _manual_targets:
                import numpy as np
                positions = np.asarray(_numpy(articulation.get_dof_positions()), dtype=np.float32).reshape(-1)
                for name, target in list(_manual_targets.items()):
                    dof = (_joint_mapping.get(name) or {}).get("dof")
                    index = dof_lookup.get(dof)
                    if index is None:
                        continue
                    actual = float(positions[index])
                    error = abs(target - actual)
                    if error <= 0.005:
                        _manual_targets.pop(name, None)
                        _stall_state.pop(name, None)
                        continue
                    state = _stall_state.setdefault(name, {"best_error": error, "ticks": 0})
                    # Detect progress toward the command instead of requiring a
                    # perfectly motionless joint; contacts often produce jitter.
                    if error < float(state["best_error"]) - 5e-4:
                        state["best_error"] = error
                        state["ticks"] = 0
                    elif error > 0.01:
                        state["ticks"] += 1
                    if state["ticks"] >= 15:
                        lo, hi = _dof_limits.get(dof, (-math.pi, math.pi))
                        if target > actual: hi = min(hi, actual)
                        else: lo = max(lo, actual)
                        _dof_limits[dof] = (lo, hi)
                        slider = _joint_sliders.get(name)
                        if slider is not None:
                            factor = 180.0 / math.pi if _state["angle_unit"] == "degrees" else 1.0
                            base = (float(_nudge_base.get(name, 0.0))
                                    if _state.get("nudge_mode") else 0.0)
                            display_lo = (lo - base) * factor
                            display_hi = (hi - base) * factor
                            display_actual = (actual - base) * factor
                            _state["updating_sliders"] = True
                            try:
                                slider.model.set_min(display_lo)
                                slider.model.set_max(display_hi)
                                slider.model.set_value(display_actual)
                                _prefs.setdefault("angles", {})[str(name)] = display_actual
                            finally:
                                _state["updating_sliders"] = False
                            _save_preferences()
                        _set_status(f"error: {name} stopped at a physical limit ({actual:.3f} rad)")
                        _manual_targets.pop(name, None); _stall_state.pop(name, None)
            _sync_joint_angle_sliders(articulation)
        _set_status(_state["status"])
    except Exception as exc:  # noqa: BLE001 - displayed in the Isaac Sim window
        if generation == _state["generation"]:
            message = f"error: {type(exc).__name__}: {exc}"
            _state["running"] = False
            _state["generation"] += 1
            try:
                _state["sock"].close()
            except Exception:  # noqa: BLE001
                pass
            _set_status(message)


def start_blacknode_isaac_stream(url, prim_path):
    stop_blacknode_isaac_stream()
    try:
        while True:
            _state["queue"].get_nowait()
    except queue.Empty:
        pass
    url, prim_path = str(url or "").strip(), str(prim_path or "").strip()
    if not url.startswith(("ws://", "wss://")):
        _set_status("error: URL must start with ws:// or wss://")
        return
    if not prim_path.startswith("/"):
        _set_status("error: articulation path must start with /")
        return
    _state["generation"] += 1
    generation = _state["generation"]
    _state.update(running=True, frames=0, schema_joints=[], dof_names=[], rows_ready=False,
                  dataset_home={}, articulation=None, angle_unit=str(_prefs.get("angle_unit") or "degrees"),
                  last_frame=None, nudge_mode=False, updating_sliders=False)
    _nudge_base.clear()
    _manual_targets.clear(); _stall_state.clear()
    _joint_mapping.clear()
    _joint_mapping.update({str(name): dict(value) for name, value in
                           (_prefs.get("mapping") or {}).items()})
    _calibration.clear(); _calibration.update(dict(_prefs.get("calibration") or {})); _dataset_home.clear()
    _drive_settings.clear(); _drive_settings.update(dict(_prefs.get("drive_settings") or {}))
    _save_preferences()
    _state["thread"] = threading.Thread(
        target=_receive, args=(url, generation), daemon=True, name="blacknode-isaac-stream")
    _state["thread"].start()
    asyncio.ensure_future(_prepare_and_pump(prim_path, generation))
    _set_status(f"connecting to {url}")


def stop_blacknode_isaac_stream():
    _state["running"] = False
    _set_calibrate_active(False)
    _nudge_base.clear()
    _manual_targets.clear(); _stall_state.clear()
    _state["generation"] += 1
    try:
        _state["sock"].close()
    except Exception:  # noqa: BLE001
        pass
    _set_status("stopped")


def _combo_value(model, item):
    try:
        value_model = model.get_item_value_model(item) if item is not None else model.get_item_value_model()
        value = str(value_model.as_string)
        if value:
            return value
    except Exception:  # noqa: BLE001 - tolerate small omni.ui API differences
        pass
    # Isaac versions differ: some callbacks provide an item index while the
    # model exposes only the selected index. Decode that form explicitly.
    try:
        index = int(model.get_item_value_model().as_int)
        return str(index)
    except Exception:  # noqa: BLE001
        try:
            return str(model.as_string)
        except Exception:
            return ""


def _set_joint_angle(name, angle):
    """Command one mapped Isaac DOF to a live angle in radians."""
    if _state.get("updating_sliders"):
        return
    mapping = _joint_mapping.get(str(name)) or {}
    dof = mapping.get("dof")
    if _articulation is None or dof not in _dof_lookup:
        _set_status(f"{name}: connect to a valid articulation and select an Isaac DOF first")
        return
    try:
        import numpy as np
        display_angle = float(angle)
        if not math.isfinite(display_angle):
            raise ValueError("angle must be finite")
        angle = math.radians(display_angle) if _state["angle_unit"] == "degrees" else display_angle
        if _state.get("nudge_mode"):
            angle = float(_nudge_base.get(str(name), 0.0)) + angle
        lower, upper = _dof_limits.get(dof, (-math.pi, math.pi))
        if not math.isfinite(lower): lower = -math.pi
        if not math.isfinite(upper): upper = math.pi
        angle = min(max(angle, lower), upper)
        targets = np.asarray(_numpy(_articulation.get_dof_positions()), dtype=np.float32).copy().reshape(-1)
        targets[_dof_lookup[dof]] = angle
        _articulation.set_dof_position_targets(targets)
        _manual_targets[str(name)] = angle
        _stall_state.pop(str(name), None)
        _prefs.setdefault("angles", {})[str(name)] = display_angle
        _save_preferences()
        mode = "home nudge" if _state.get("nudge_mode") else "live angle"
        _set_status(f"{name}: {mode} {display_angle:.3f} {_state['angle_unit']}")
    except Exception as exc:  # noqa: BLE001
        _set_status(f"{name}: angle was not applied: {exc}")


def _set_home_pose():
    if _articulation is None:
        _set_status("connect to a valid articulation first")
        return
    try:
        import numpy as np
        targets = np.asarray(_numpy(_articulation.get_dof_positions()), dtype=np.float32).copy().reshape(-1)
        captured = 0
        for name, mapping in _joint_mapping.items():
            dof = mapping.get("dof")
            if dof not in _dof_lookup:
                continue
            calibration = _calibration.setdefault(name, {})
            calibration.update({"dataset_home": float(_dataset_home.get(name, 0.0)),
                                "sim_home": float(targets[_dof_lookup[dof]]), "calibrated": True})
            captured += 1
        _set_status(f"set home pose for {captured} joint(s)")
        _save_preferences()
    except Exception as exc:  # noqa: BLE001
        _set_status(f"home pose failed: {exc}")


def _zero_angle_sliders():
    _state["updating_sliders"] = True
    try:
        for name, slider in _joint_sliders.items():
            slider.model.set_value(0.0)
            _prefs.setdefault("angles", {})[str(name)] = 0.0
    finally:
        _state["updating_sliders"] = False


def _set_calibrate_active(active):
    _state["nudge_mode"] = bool(active)
    if _calibrate_button is not None:
        _calibrate_button.text = "CALIBRATING" if active else "Calibrate"
        _calibrate_button.style = ({
            "Button": {"background_color": ui.color(255, 140, 20, 255),
                       "color": ui.color(15, 15, 15, 255)},
            "Button:hovered": {"background_color": ui.color(255, 180, 45, 255)},
        } if active else {})


def _begin_home_calibration():
    if _state.get("nudge_mode"):
        _cancel_home_calibration()
        return
    if _articulation is None:
        _set_status("connect to a valid articulation first")
        return
    try:
        import numpy as np
        positions = np.asarray(_numpy(_articulation.get_dof_positions()), dtype=np.float32).reshape(-1)
        _nudge_base.clear()
        for name, mapping in _joint_mapping.items():
            dof = mapping.get("dof")
            if dof in _dof_lookup:
                _nudge_base[name] = float(positions[_dof_lookup[dof]])
        _set_calibrate_active(True)
        _prefs["angles"] = {str(name): 0.0 for name in _joint_mapping}
        if _state.get("schema_joints") and _state.get("dof_names"):
            _build_joint_rows(_state["schema_joints"], _state["dof_names"])
        _zero_angle_sliders()
        _save_preferences()
        _set_status(f"calibration nudge mode: zeroed {len(_nudge_base)} joint slider(s)")
    except Exception as exc:  # noqa: BLE001
        _set_status(f"error: calibration mode failed: {exc}")


def _cancel_home_calibration():
    """Leave nudge mode without committing its current offsets to home."""
    try:
        import numpy as np
        positions = (np.asarray(_numpy(_articulation.get_dof_positions()), dtype=np.float32)
                     .reshape(-1) if _articulation is not None else [])
        factor = 180.0 / math.pi if _state["angle_unit"] == "degrees" else 1.0
        angles = {}
        for name, mapping in _joint_mapping.items():
            dof = mapping.get("dof")
            if dof in _dof_lookup and len(positions) > _dof_lookup[dof]:
                angles[str(name)] = float(positions[_dof_lookup[dof]]) * factor
        _set_calibrate_active(False)
        _nudge_base.clear()
        _prefs["angles"] = angles
        if _state.get("schema_joints") and _state.get("dof_names"):
            _build_joint_rows(_state["schema_joints"], _state["dof_names"])
        _save_preferences()
        _set_status("calibration nudge mode cancelled; home pose was not changed")
    except Exception as exc:  # noqa: BLE001
        _set_status(f"error: could not cancel calibration mode: {exc}")


def _add_to_home_pose():
    if _articulation is None or not _state.get("nudge_mode"):
        _set_status("error: press Calibrate before adding nudges to the home pose")
        return
    try:
        import numpy as np
        positions = np.asarray(_numpy(_articulation.get_dof_positions()), dtype=np.float32).reshape(-1)
        applied = 0
        _prefs["calibration_backup"] = {name: dict(value) for name, value in _calibration.items()}
        for name, mapping in _joint_mapping.items():
            dof = mapping.get("dof")
            if dof not in _dof_lookup:
                continue
            actual = float(positions[_dof_lookup[dof]])
            base = float(_nudge_base.get(name, actual))
            delta = actual - base
            if abs(delta) < 1e-6:
                continue
            calibration = _calibration.setdefault(name, {})
            previous_home = (float(calibration["sim_home"])
                             if calibration.get("calibrated") else base)
            calibration.update({"dataset_home": float(calibration.get(
                                    "dataset_home", _dataset_home.get(name, 0.0))),
                                "sim_home": previous_home + delta, "calibrated": True})
            _nudge_base[name] = actual
            applied += 1
        _zero_angle_sliders()
        _save_preferences()
        _set_status(f"added nudges to home pose for {applied} joint(s); sliders reset to zero")
    except Exception as exc:  # noqa: BLE001
        _set_status(f"error: add to home pose failed: {exc}")


def _go_home_pose():
    if _articulation is None:
        _set_status("connect to a valid articulation first")
        return
    try:
        import numpy as np
        targets = np.asarray(_numpy(_articulation.get_dof_positions()), dtype=np.float32).copy().reshape(-1)
        applied = 0
        for name, calibration in _calibration.items():
            mapping = _joint_mapping.get(name) or {}
            dof = mapping.get("dof")
            if calibration.get("calibrated") and dof in _dof_lookup:
                targets[_dof_lookup[dof]] = float(calibration["sim_home"])
                applied += 1
        _articulation.set_dof_position_targets(targets)
        _set_calibrate_active(False)
        _nudge_base.clear()
        _zero_angle_sliders()
        _save_preferences()
        _set_status(f"went home on {applied} calibrated joint(s)")
    except Exception as exc:  # noqa: BLE001
        _set_status(f"go home failed: {exc}")


def _set_angle_unit(model, item):
    _state["angle_unit"] = "degrees" if _combo_value(model, item) == "Degrees" else "radians"
    _save_preferences()
    if _state.get("schema_joints") and _state.get("dof_names"):
        _build_joint_rows(_state["schema_joints"], _state["dof_names"])


def _set_motion_scale(joint, value):
    try:
        scale = float(value)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("motion scale must be finite and greater than zero")
        _joint_mapping[str(joint)]["scale"] = scale
        _save_preferences()
        _reapply_current_frame()
        _set_status(f"{joint}: motion scale {scale:g}")
    except Exception as exc:  # noqa: BLE001
        _set_status(f"error: {joint}: {exc}")


def _build_joint_rows(dataset_names, dof_names):
    """Build dataset-joint to Isaac-DOF and direction controls in the window."""
    if _joint_stack is None:
        return
    try:
        _joint_stack.clear()
    except Exception:  # noqa: BLE001
        return
    previous = {name: dict(value) for name, value in _joint_mapping.items()}
    previous.update(dict(_prefs.get("mapping") or {}))
    _joint_mapping.clear()
    _joint_sliders.clear()
    choices = ["(ignore)"] + list(dof_names)
    with _joint_stack:
        with ui.HStack(height=24, spacing=6):
            ui.Label("Dataset Joint", width=150)
            ui.Label("Isaac DOF", width=190)
            ui.Label("Direction", width=60)
            ui.Label("Motion Scale", width=95)
            angle_title = "Angle Nudge" if _state.get("nudge_mode") else "Joint Angle"
            ui.Label(f"{angle_title} ({_state['angle_unit']})", width=220)
            ui.Label("Stiffness", width=95)
            ui.Label("Damping", width=95)
            ui.Label("Max Force", width=95)
            ui.Label("USD Joint", width=80)
        for name in dataset_names:
            default = str(name) if str(name) in dof_names else "(ignore)"
            selected = choices.index(default)
            saved = previous.get(str(name)) or {}
            selected_dof = saved.get("dof") if saved.get("dof") in dof_names else (
                None if default == "(ignore)" else default)
            _joint_mapping[str(name)] = {"dof": selected_dof,
                                         "sign": float(saved.get("sign", 1.0)),
                                         "scale": float(saved.get("scale", 1.0))}
            selected = choices.index(selected_dof or "(ignore)")

            def on_dof(model, item, joint=str(name)):
                value = _combo_value(model, item)
                _joint_mapping[joint]["dof"] = None if value == "(ignore)" else value
                _save_preferences()

            def on_direction(model, item, joint=str(name)):
                value = _combo_value(model, item)
                # ComboBox may report either the label or selected index.
                _joint_mapping[joint]["sign"] = -1.0 if value in ("-1", "1") else 1.0
                _save_preferences()
                _reapply_current_frame()

            with ui.HStack(height=26, spacing=6):
                ui.Label(str(name), width=150)
                dof_combo = ui.ComboBox(selected, *choices, width=190)
                direction_combo = ui.ComboBox(0 if _joint_mapping[str(name)]["sign"] > 0 else 1,
                                              "+1", "-1", width=60)
                scale_model = ui.SimpleFloatModel(float(_joint_mapping[str(name)]["scale"]))
                ui.FloatField(scale_model, width=95)
                lower, upper = _dof_limits.get(selected_dof, (-math.pi, math.pi))
                if not math.isfinite(lower): lower = -math.pi
                if not math.isfinite(upper): upper = math.pi
                factor = 180.0 / math.pi if _state["angle_unit"] == "degrees" else 1.0
                if _state.get("nudge_mode"):
                    base = float(_nudge_base.get(str(name), 0.0))
                    lower, upper = lower - base, upper - base
                with ui.HStack(width=220, height=26, spacing=4):
                    angle_slider = ui.FloatSlider(min=lower * factor, max=upper * factor,
                                                  step=0.1 if factor > 1 else 0.01, width=146)
                    # The field shares the slider model, so typed values obey
                    # the same USD/nudge limits and trigger the same command.
                    ui.FloatField(angle_slider.model, width=70)
                _joint_sliders[str(name)] = angle_slider
                saved_angle = (_prefs.get("angles") or {}).get(str(name))
                if saved_angle is not None:
                    try:
                        angle_slider.model.set_value(float(saved_angle))
                    except Exception:
                        pass
                angle_slider.model.add_value_changed_fn(
                    lambda model, joint=str(name): _set_joint_angle(joint, model.as_float))
                authored_drive = _read_joint_drive(selected_dof)
                saved_drive = dict(_drive_settings.get(str(name)) or
                                   (_prefs.get("drive_settings") or {}).get(str(name)) or {})
                drive = {**authored_drive, **saved_drive}
                stiffness_model = ui.SimpleFloatModel(float(drive.get("stiffness", 0.0)))
                damping_model = ui.SimpleFloatModel(float(drive.get("damping", 0.0)))
                force_model = ui.SimpleFloatModel(float(drive.get("max_force", 0.0)))
                ui.FloatField(stiffness_model, width=95)
                ui.FloatField(damping_model, width=95)
                ui.FloatField(force_model, width=95)
                ui.Label("found" if selected_dof in _joint_prim_paths else "missing", width=80,
                         style={"color": ((0.25, 0.86, 0.48, 1.0) if selected_dof in _joint_prim_paths
                                          else (1.0, 0.16, 0.16, 1.0))})
                stiffness_model.add_value_changed_fn(
                    lambda model, joint=str(name): _set_drive_value(joint, "stiffness", model.as_float))
                damping_model.add_value_changed_fn(
                    lambda model, joint=str(name): _set_drive_value(joint, "damping", model.as_float))
                force_model.add_value_changed_fn(
                    lambda model, joint=str(name): _set_drive_value(joint, "max_force", model.as_float))
            dof_combo.model.add_item_changed_fn(on_dof)
            direction_combo.model.add_item_changed_fn(on_direction)
            scale_model.add_value_changed_fn(
                lambda model, joint=str(name): _set_motion_scale(joint, model.as_float))
    _state["rows_ready"] = True


def _use_selected_prim():
    try:
        import omni.usd
        paths = list(omni.usd.get_context().get_selection().get_selected_prim_paths())
        if paths:
            _prim_model.set_value(paths[0])
            _set_status(f"selected articulation path: {paths[0]}")
        else:
            _set_status("select an articulation root prim in the Stage first")
    except Exception as exc:  # noqa: BLE001
        _set_status(f"could not read Stage selection: {exc}")


def show_blacknode_isaac_window():
    global _window, _url_model, _prim_model, _status_label, _joint_stack, _unit_model, _calibrate_button
    if _window is not None:
        _window.visible = False
    _load_preferences()
    # Leave enough room for the mapping panel and keep the action row visible.
    _window = ui.Window("Blacknode Dataset Replay", width=1360, height=720)
    _state["angle_unit"] = str(_prefs.get("angle_unit") or "degrees")
    _url_model = ui.SimpleStringModel(str(_prefs.get("url") or "ws://127.0.0.1:8765"))
    _prim_model = ui.SimpleStringModel(str(_prefs.get("prim_path") or "/World/Robot"))
    with _window.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("Blacknode WebSocket", height=20)
            ui.StringField(_url_model, height=24)
            ui.Label("Robot articulation root prim", height=20)
            ui.StringField(_prim_model, height=24)
            with ui.HStack(height=30, spacing=8):
                ui.Button("Connect", clicked_fn=lambda: start_blacknode_isaac_stream(
                    _url_model.as_string, _prim_model.as_string))
                ui.Button("Stop", clicked_fn=stop_blacknode_isaac_stream)
                ui.Button("Set Home Pose", clicked_fn=_set_home_pose)
                _calibrate_button = ui.Button("Calibrate", clicked_fn=_begin_home_calibration)
                ui.Button("Add to Home Pose", clicked_fn=_add_to_home_pose)
                ui.Button("Go Home", clicked_fn=_go_home_pose)
                ui.Button("Apply Drive Settings", clicked_fn=_apply_all_drive_settings)
            with ui.HStack(height=26, spacing=6):
                ui.Button("Use selected prim", clicked_fn=_use_selected_prim)
                ui.Button("Discover Joints", clicked_fn=_discover_joints_from_ui)
                ui.Button("Save Calibration File", clicked_fn=_save_calibration_artifact)
                ui.Button("Load Calibration File", clicked_fn=_load_calibration_artifact)
                ui.Label("Angle units", width=80)
                _unit_model = ui.ComboBox(1 if _state["angle_unit"] == "degrees" else 0,
                                           "Radians", "Degrees", width=100)
                _unit_model.model.add_item_changed_fn(_set_angle_unit)
            ui.Label("Select a root in the Stage and click Use selected prim, or type its path.",
                     word_wrap=True, height=28)
            ui.Label("Dataset joint mapping · direction · per-joint home calibration · USD drive stiffness/damping/max force", height=22)
            with ui.ScrollingFrame(height=360):
                _joint_stack = ui.VStack(spacing=2)
            _status_label = ui.Label("stopped", word_wrap=True, height=34)
    _window.visible = True
