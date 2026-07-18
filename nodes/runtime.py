"""Managed, crash-recoverable episode recording runtime."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import filters, storage, wsserver


def dashboard(status: dict[str, Any]) -> str:
    color = "#22c55e" if status.get("recording") else "#f59e0b" if status.get("paused") else "#64748b"
    label = "RECORDING" if status.get("recording") else "PAUSED" if status.get("paused") else "STOPPED"
    error = html.escape(str(status.get("last_error") or "ready"))
    preview = str(status.get("camera_preview") or "")
    camera = html.escape(str(status.get("preview_camera") or "camera"))
    preview_markup = (
        f'<image x="24" y="72" width="600" height="450" href="{preview}" preserveAspectRatio="xMidYMid meet"/>'
        if preview.startswith("data:image/") else
        '<rect x="24" y="72" width="600" height="450" rx="12" fill="#111827"/>'
        '<text x="324" y="292" fill="#64748b" font-family="sans-serif" font-size="18" '
        'text-anchor="middle">Waiting for first camera frame</text>'
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="546" viewBox="0 0 960 546">'
        '<rect width="960" height="546" rx="16" fill="#0f172a"/>'
        f'<circle cx="36" cy="36" r="10" fill="{color}"/><text x="56" y="43" fill="#e2e8f0" '
        f'font-family="sans-serif" font-size="22" font-weight="700">{label}</text>'
        f'<text x="936" y="42" fill="#94a3b8" font-family="sans-serif" font-size="14" '
        f'text-anchor="end">{camera}</text>{preview_markup}'
        f'<text x="660" y="104" fill="#94a3b8" font-family="sans-serif" font-size="13">EPISODE</text>'
        f'<text x="660" y="132" fill="#e2e8f0" font-family="monospace" font-size="22">'
        f'{int(status.get("episode_index") or 0)}</text>'
        f'<text x="660" y="180" fill="#94a3b8" font-family="sans-serif" font-size="13">CAPTURED</text>'
        f'<text x="660" y="208" fill="#e2e8f0" font-family="monospace" font-size="22">'
        f'{int(status.get("frame_count") or 0)} frames</text>'
        f'<text x="660" y="256" fill="#94a3b8" font-family="sans-serif" font-size="13">TIME · RATE</text>'
        f'<text x="660" y="284" fill="#e2e8f0" font-family="monospace" font-size="19">'
        f'{float(status.get("duration_seconds") or 0):.1f}s · {float(status.get("capture_rate_hz") or 0):.1f} fps</text>'
        f'<text x="660" y="332" fill="#94a3b8" font-family="sans-serif" font-size="13">SOURCE AGE</text>'
        f'<text x="660" y="360" fill="#e2e8f0" font-family="monospace" font-size="16">robot '
        f'{float(status.get("robot_age_ms") or 0):.0f} ms</text>'
        f'<text x="660" y="386" fill="#e2e8f0" font-family="monospace" font-size="16">camera '
        f'{float(status.get("camera_age_ms") or 0):.0f} ms</text>'
        f'<text x="660" y="434" fill="#94a3b8" font-family="sans-serif" font-size="13">DROPPED</text>'
        f'<text x="660" y="462" fill="#e2e8f0" font-family="monospace" font-size="22">'
        f'{int(status.get("dropped_frames") or 0)}</text>'
        f'<text x="24" y="535" fill="#fca5a5" font-family="sans-serif" font-size="13">{error}</text></svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - local stream handle is user input
        return dict(json.loads(response.read().decode("utf-8")))


def _get_image(handle: dict[str, Any], timeout: float) -> tuple[bytes, dict[str, Any]]:
    url = str(handle.get("snapshot_url") or "").strip()
    if not url:
        raise ValueError("camera stream is missing snapshot_url")
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - local stream handle is user input
        content = response.read()
        return content, {
            "sequence": int(response.headers.get("X-Blacknode-Frame-Sequence") or 0),
            "captured_at_ns": int(response.headers.get("X-Blacknode-Captured-At-Ns") or 0),
            "media_type": str(response.headers.get_content_type() or "image/jpeg"),
        }


def _stream_url(handle: dict[str, Any]) -> str:
    if handle.get("kind") != "blacknode.sample-stream":
        raise ValueError("robot_stream must be a blacknode.sample-stream handle")
    url = str(handle.get("url") or "").strip()
    if not url:
        raise ValueError("robot stream is missing url")
    return url


def _camera_handles(camera_stream: dict[str, Any], camera_streams: list[Any]) -> list[dict[str, Any]]:
    handles = [dict(item) for item in camera_streams if isinstance(item, dict)]
    if camera_stream:
        handles.insert(0, dict(camera_stream))
    seen: set[str] = set()
    result = []
    for handle in handles:
        if handle.get("kind") != "blacknode.frame-stream":
            raise ValueError("camera streams must be blacknode.frame-stream handles")
        stream_id = str(handle.get("stream_id") or "camera").strip() or "camera"
        if stream_id in seen:
            continue
        seen.add(stream_id)
        handle["stream_id"] = stream_id
        result.append(handle)
    if not result:
        raise ValueError("at least one camera stream is required")
    return result


@dataclass
class EpisodeRecorder:
    run_id: str
    dataset_path: Path
    robot_stream: dict[str, Any]
    cameras: list[dict[str, Any]]
    require_armed: bool
    stale_after: float
    request_timeout: float
    work_path: Path
    episode_index: int
    fps: int
    task: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    thread: threading.Thread | None = None
    frame_count: int = 0
    dropped_frames: int = 0
    consecutive_errors: int = 0
    last_error: str = ""
    started_ns: int = field(default_factory=time.time_ns)
    stopped: bool = False
    camera_preview: str = ""
    preview_camera: str = ""
    robot_age_ms: float = 0.0
    camera_age_ms: float = 0.0
    capture_times: deque[float] = field(default_factory=lambda: deque(maxlen=30))

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"blacknode-dataset-{self.run_id}")
        self.thread.start()

    def _validate_robot(self, sample: dict[str, Any], now_ns: int) -> None:
        if sample.get("kind") != "blacknode.teleoperation-sample":
            raise ValueError("sample stream did not return a blacknode.teleoperation-sample")
        captured = int(sample.get("captured_at_ns") or 0)
        age = max(0, now_ns - captured) / 1e9 if captured else float("inf")
        if not captured or age > self.stale_after:
            detail = "missing capture timestamp" if not captured else f"{age:.2f}s > {self.stale_after:.2f}s"
            raise ValueError(f"robot sample is stale ({detail})")
        self.robot_age_ms = age * 1000.0
        if self.require_armed and not bool(sample.get("armed")):
            raise ValueError("teleoperation is not armed")
        if not bool(sample.get("live")):
            raise ValueError("teleoperation sample is not live")
        joint_names = list(sample.get("joint_names") or [])
        if not joint_names or any(name not in sample.get("observation", {}) for name in joint_names):
            raise ValueError("robot sample has no ordered observation joints")
        if any(name not in sample.get("action", {}) for name in joint_names):
            raise ValueError("robot sample action does not match joint_names")

    def _capture(self) -> None:
        sample = _get_json(_stream_url(self.robot_stream), self.request_timeout)
        now_ns = time.time_ns()
        self._validate_robot(sample, now_ns)
        images: dict[str, bytes] = {}
        camera_meta: dict[str, Any] = {}
        for handle in self.cameras:
            name = str(handle["stream_id"])
            image, meta = _get_image(handle, self.request_timeout)
            captured = int(meta.get("captured_at_ns") or 0)
            received_ns = time.time_ns()
            age = max(0, received_ns - captured) / 1e9 if captured else float("inf")
            if not captured or age > self.stale_after:
                detail = "missing capture timestamp" if not captured else f"{age:.2f}s > {self.stale_after:.2f}s"
                raise ValueError(f"camera {name} frame is stale ({detail})")
            if not image:
                raise ValueError(f"camera {name} returned an empty frame")
            images[name] = image
            camera_meta[name] = meta
            if not self.camera_preview:
                self.preview_camera = name
            if name == self.preview_camera:
                media_type = str(meta.get("media_type") or "image/jpeg")
                self.camera_preview = f"data:{media_type};base64,{base64.b64encode(image).decode('ascii')}"
                self.camera_age_ms = age * 1000.0
        frame_index = self.frame_count
        storage.append_frame(self.work_path, {
            "frame_index": frame_index,
            "timestamp": frame_index / float(self.fps),
            "recorded_at_ns": now_ns,
            "task": self.task,
            "robot": sample,
            "cameras": camera_meta,
        }, images)
        with self.lock:
            self.frame_count += 1
            self.capture_times.append(time.monotonic())
            self.consecutive_errors = 0
            self.last_error = ""

    def _loop(self) -> None:
        period = 1.0 / float(self.fps)
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                self.stop_event.wait(0.05)
                deadline = time.monotonic()
                continue
            try:
                self._capture()
            except Exception as exc:  # noqa: BLE001 - surfaced in recorder status
                with self.lock:
                    self.dropped_frames += 1
                    self.consecutive_errors += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    if self.consecutive_errors >= 3:
                        self.pause_event.set()
            deadline += period
            self.stop_event.wait(max(0.0, deadline - time.monotonic()))
            if deadline < time.monotonic() - period:
                deadline = time.monotonic()
        self.stopped = True

    def status(self) -> dict[str, Any]:
        with self.lock:
            frames = self.frame_count
            dropped = self.dropped_frames
            error = self.last_error
            capture_times = list(self.capture_times)
            preview = self.camera_preview
            preview_camera = self.preview_camera
            robot_age_ms = self.robot_age_ms
            camera_age_ms = self.camera_age_ms
        capture_rate = 0.0
        if len(capture_times) > 1 and capture_times[-1] > capture_times[0]:
            capture_rate = (len(capture_times) - 1) / (capture_times[-1] - capture_times[0])
        alive = bool(self.thread and self.thread.is_alive())
        return {
            "run_id": self.run_id,
            "running": alive,
            "recording": alive and not self.pause_event.is_set(),
            "paused": self.pause_event.is_set(),
            "episode_index": self.episode_index,
            "frame_count": frames,
            "dropped_frames": dropped,
            "duration_seconds": frames / float(self.fps),
            "dataset_path": str(self.dataset_path),
            "work_path": str(self.work_path),
            "last_error": error,
            "capture_rate_hz": capture_rate,
            "camera_preview": preview,
            "preview_camera": preview_camera,
            "robot_age_ms": robot_age_ms,
            "camera_age_ms": camera_age_ms,
        }

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            # One capture can be inside the robot request followed by every
            # camera request. Saving must never race that final append.
            timeout = max(2.0, self.request_timeout * (len(self.cameras) + 1) + 1.0)
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                raise RuntimeError("recorder did not stop before the capture-request deadline; episode remains recoverable")


_recorders: dict[str, EpisodeRecorder] = {}
_recorder_configs: dict[str, dict[str, Any]] = {}
_replay_media: dict[str, Path] = {}
_replay_sessions: dict[str, dict[str, Any]] = {}
_replay_tables: dict[str, Any] = {}
_publishers: dict[str, "StreamPublisher"] = {}
_lock = threading.RLock()


def configure_recorder(*, run_id: str, dataset: dict[str, Any], robot_stream: dict[str, Any],
                       camera_stream: dict[str, Any], camera_streams: list[Any], require_armed: bool,
                       stale_after: float, request_timeout: float) -> bool:
    """Retain resolved stream handles so UI controls never recook the graph."""
    run_id = str(run_id or "episode_recorder").strip() or "episode_recorder"
    try:
        storage.resolve_dataset_path(dataset)
        _stream_url(robot_stream)
        _camera_handles(camera_stream, camera_streams)
    except (TypeError, ValueError):
        return False
    with _lock:
        _recorder_configs[run_id] = {
            "run_id": run_id,
            "dataset": dict(dataset),
            "robot_stream": dict(robot_stream),
            "camera_stream": dict(camera_stream),
            "camera_streams": list(camera_streams),
            "require_armed": bool(require_armed),
            "stale_after": float(stale_after),
            "request_timeout": float(request_timeout),
        }
    return True


def start_recorder(*, run_id: str, dataset: dict[str, Any], robot_stream: dict[str, Any],
                   camera_stream: dict[str, Any], camera_streams: list[Any], require_armed: bool,
                   stale_after: float, request_timeout: float) -> dict[str, Any]:
    run_id = str(run_id or "episode_recorder").strip() or "episode_recorder"
    with _lock:
        existing = _recorders.get(run_id)
        if existing and existing.thread and existing.thread.is_alive():
            return existing.status()
        path = storage.resolve_dataset_path(dataset)
        for active in _recorders.values():
            if active.dataset_path == path and active.thread and active.thread.is_alive():
                raise ValueError(f"dataset already has an active recorder: {active.run_id}")
        manifest = storage.load_manifest(path)
        _stream_url(robot_stream)
        cameras = _camera_handles(camera_stream, camera_streams)
        work, episode_index, _ = storage.begin_episode(path, run_id)
        recorder = EpisodeRecorder(
            run_id=run_id, dataset_path=path, robot_stream=dict(robot_stream), cameras=cameras,
            require_armed=bool(require_armed), stale_after=max(0.05, float(stale_after)),
            request_timeout=max(0.05, float(request_timeout)), work_path=work,
            episode_index=episode_index, fps=int(manifest["fps"]),
            task=str(manifest.get("task") or ""),
        )
        _recorders[run_id] = recorder
        recorder.start()
        return recorder.status()


def control_recorder(run_id: str, action: str) -> dict[str, Any]:
    with _lock:
        recorder = _recorders.get(run_id)
    if recorder is None:
        return {"run_id": run_id, "running": False, "recording": False, "paused": False,
                "frame_count": 0, "dropped_frames": 0, "last_error": "recorder is not running"}
    action = action.lower()
    if action == "pause":
        recorder.pause_event.set()
    elif action == "resume":
        recorder.consecutive_errors = 0
        recorder.last_error = ""
        recorder.pause_event.clear()
    elif action in {"stop", "save", "finalize", "discard"}:
        recorder.stop()
        if action in {"save", "finalize"}:
            episode = storage.save_episode(recorder.dataset_path, recorder.run_id)
            saved_path = recorder.dataset_path / "episodes" / f"episode-{int(episode['episode_index']):06d}"
            result = {**recorder.status(), "saved": True, "saved_path": str(saved_path), "episode": episode}
        elif action == "discard":
            result = {**recorder.status(), "discarded": storage.discard_episode(recorder.dataset_path, recorder.run_id)}
        else:
            result = {**recorder.status(), "recoverable": True}
        with _lock:
            _recorders.pop(run_id, None)
        return result
    return recorder.status()


def recover_episode(dataset: dict[str, Any], run_id: str, action: str) -> dict[str, Any]:
    """Finalize or discard a journal left by Stop All or a prior process."""
    path = storage.resolve_dataset_path(dataset)
    action = action.lower()
    if action in {"save", "finalize"}:
        episode = storage.save_episode(path, run_id)
        saved_path = path / "episodes" / f"episode-{int(episode['episode_index']):06d}"
        return {"run_id": run_id, "running": False, "recording": False, "paused": False,
                "frame_count": int(episode["frames"]), "dropped_frames": 0,
                "duration_seconds": float(episode["duration_seconds"]), "saved": True,
                "saved_path": str(saved_path), "episode": episode}
    if action == "discard":
        discarded = storage.discard_episode(path, run_id)
        return {"run_id": run_id, "running": False, "recording": False, "paused": False,
                "frame_count": 0, "dropped_frames": 0, "discarded": discarded,
                "last_error": "" if discarded else "incomplete episode was not found"}
    raise ValueError("recovery supports save, finalize, or discard")


def recoverable_episode_status(dataset: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    """Return the persisted journal state after a recorder or server stop."""
    path = storage.resolve_dataset_path(dataset)
    work = storage.incomplete_episode_path(path, run_id)
    journal_path = work / "episode.json"
    if not journal_path.exists():
        return None
    episode = json.loads(journal_path.read_text(encoding="utf-8"))
    frames = int(episode.get("frames") or 0)
    fps = max(1, int(episode.get("fps") or 1))
    return {
        "run_id": run_id,
        "running": False,
        "recording": False,
        "paused": False,
        "recoverable": True,
        "episode_index": int(episode.get("episode_index") or 0),
        "frame_count": frames,
        "dropped_frames": 0,
        "duration_seconds": frames / float(fps),
        "dataset_path": str(path),
        "work_path": str(work),
        "last_error": (
            "incomplete episode is recoverable; discard it before recording again"
            if frames == 0 else
            "incomplete episode is recoverable; save or discard it before recording again"
        ),
    }


def recorder_outputs(status: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    state = "recording" if status.get("recording") else "paused" if status.get("paused") else "stopped"
    return {
        "running": bool(status.get("running")),
        "recording": bool(status.get("recording")),
        "paused": bool(status.get("paused")),
        "episode_index": int(status.get("episode_index") or 0),
        "frame_count": int(status.get("frame_count") or 0),
        "dropped_frames": int(status.get("dropped_frames") or 0),
        "duration_seconds": float(status.get("duration_seconds") or 0.0),
        "dataset": dict(dataset),
        "status": status,
        "dashboard": dashboard(status),
        "report": f"episode recorder {state}: {status.get('frame_count', 0)} frames"
                  + (f"; {status['last_error']}" if status.get("last_error") else ""),
    }


def register_replay_media(path: Path) -> str:
    resolved = Path(path).resolve()
    if resolved.suffix.lower() != ".mp4" or not resolved.is_file():
        raise ValueError(f"replay video is unavailable: {resolved}")
    stat = resolved.stat()
    identity = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    with _lock:
        _replay_media[token] = resolved
    return token


def register_episode_replay(replay: dict[str, Any]) -> str:
    video_path = storage.browser_video_path(replay)
    data_path = Path(str(replay.get("data_path") or "")).resolve()
    token = register_replay_media(video_path)
    if not data_path.is_file() or data_path.suffix.lower() != ".parquet":
        raise ValueError(f"replay frame data is unavailable: {data_path}")
    with _lock:
        _replay_sessions[token] = {
            "data_path": data_path,
            "dataset_path": Path(str(replay.get("dataset_path") or "")).resolve(),
            "episode_index": int(replay.get("episode_index") or 0),
            "joint_names": list(replay.get("joint_names") or []),
            "frames": int(replay.get("frames") or 0),
            "fps": int(replay.get("fps") or 0),
            "units": str(replay.get("units") or "radians"),
            "camera": str(replay.get("camera") or ""),
            "task": str(replay.get("task") or ""),
        }
    return token


def replay_media_path(token: str) -> Path | None:
    with _lock:
        path = _replay_media.get(str(token or ""))
    return path if path is not None and path.is_file() else None


def replay_frame(token: str, frame_index: int) -> dict[str, Any] | None:
    with _lock:
        session = dict(_replay_sessions.get(str(token or "")) or {})
        table = _replay_tables.get(str(token or ""))
    if not session:
        return None
    smoothed = session.get("smoothed_frames")
    if smoothed is not None:
        if not smoothed:
            return None
        index = min(max(0, int(frame_index)), len(smoothed) - 1)
        return dict(smoothed[index])
    if storage.pq is None:
        raise RuntimeError("pyarrow is required to read episode frame data")
    if table is None:
        table = storage.pq.read_table(session["data_path"])
        with _lock:
            _replay_tables[str(token)] = table
    if table.num_rows <= 0:
        return None
    index = min(max(0, int(frame_index)), table.num_rows - 1)
    row = {name: values[0] for name, values in table.slice(index, 1).to_pydict().items()}
    joint_names = list(session.get("joint_names") or [])

    def joints(column: str) -> dict[str, float]:
        values = list(row.get(column) or [])
        return {name: float(value) for name, value in zip(joint_names, values)}

    cameras: dict[str, dict[str, int]] = {}
    for name, value in row.items():
        if not name.startswith("camera."):
            continue
        _, camera, field_name = name.split(".", 2)
        cameras.setdefault(camera, {})[field_name] = int(value or 0)
    return {
        "kind": "blacknode.episode-frame",
        "schema_version": 1,
        "frame_index": index,
        "frames": int(table.num_rows),
        "timestamp": float(row.get("timestamp") or 0.0),
        "recorded_at_ns": int(row.get("recorded_at_ns") or 0),
        "sample_sequence": int(row.get("sample_sequence") or 0),
        "captured_at_ns": int(row.get("captured_at_ns") or 0),
        "task": str(row.get("task") or session.get("task") or ""),
        "joint_names": joint_names,
        "leader": joints("leader.state"),
        "observation": joints("observation.state"),
        "action": joints("action"),
        "cameras": cameras,
    }


def trim_replay_episode(token: str, frame_index: int, side: str) -> dict[str, Any]:
    """Trim a saved episode at the selected replay frame without commanding hardware."""
    token = str(token or "")
    with _lock:
        session = dict(_replay_sessions.get(token) or {})
        active_paths = {
            recorder.dataset_path for recorder in _recorders.values()
            if recorder.thread and recorder.thread.is_alive()
        }
    if not session:
        raise ValueError("replay selection expired; refresh the dataset browser")
    dataset_path = Path(session["dataset_path"])
    if dataset_path in active_paths:
        raise ValueError("cannot trim a dataset while it has an active recorder")
    frames = int(session.get("frames") or 0)
    index = min(max(0, int(frame_index)), max(0, frames - 1))
    side = str(side or "").strip().lower()
    if side == "before":
        start, end = index, frames - 1
    elif side == "after":
        start, end = 0, index
    else:
        raise ValueError("trim side must be 'before' or 'after'")
    result = storage.trim_episode(dataset_path, int(session["episode_index"]), start, end)
    data_path = Path(session["data_path"])
    with _lock:
        stale_tokens = [key for key, item in _replay_sessions.items() if Path(item.get("data_path") or "") == data_path]
        for stale in stale_tokens:
            _replay_sessions.pop(stale, None)
            _replay_tables.pop(stale, None)
            _replay_media.pop(stale, None)
    return result


def register_smoothed_replay(source_token: str, method: str, strength: float,
                             preview_source: str = "action", preview_joint: str = "") -> dict[str, Any]:
    """Smooth a selected episode offline and register a new replay token for it.

    Read-only: it re-reads the recorded frames, filters each joint trajectory,
    and stores an in-memory smoothed episode that ``replay_frame`` serves under a
    fresh token. Wire this node's ``stream`` output into StreamPublisher (or
    anything that consumes a replay stream). Never commands hardware.
    """
    source_token = str(source_token or "")
    with _lock:
        session = dict(_replay_sessions.get(source_token) or {})
        media = _replay_media.get(source_token)
    if not session:
        raise ValueError("replay selection expired; select an episode in the Dataset Browser first")
    frames = int(session.get("frames") or 0)
    joint_names = list(session.get("joint_names") or [])
    if frames <= 0 or not joint_names:
        raise ValueError("selected episode has no frames or joints to smooth")
    raw_frames = [replay_frame(source_token, index) for index in range(frames)]
    raw_frames = [frame for frame in raw_frames if frame is not None]
    if not raw_frames:
        raise ValueError("could not read frames from the selected episode")
    fps = float(session.get("fps") or 30) or 30.0
    result = filters.smooth_episode(raw_frames, joint_names, method, float(strength), fps,
                                    preview_source, preview_joint)
    channels = result["channels"]
    smoothed_frames: list[dict[str, Any]] = []
    for index, frame in enumerate(raw_frames):
        new_frame = dict(frame)
        for channel in ("leader", "observation", "action"):
            new_frame[channel] = {name: float(channels[channel][index][col])
                                  for col, name in enumerate(joint_names)}
        new_frame["smoothing"] = {"method": result["method"], "strength": float(strength)}
        smoothed_frames.append(new_frame)
    new_token = hashlib.sha256(
        f"{source_token}|{result['method']}|{strength}|smoothed".encode("utf-8")).hexdigest()[:24]
    with _lock:
        if media is not None:
            _replay_media[new_token] = media
        _replay_sessions[new_token] = {
            **session,
            "smoothed_frames": smoothed_frames,
            "frames": len(smoothed_frames),
            "source_token": source_token,
            "method": result["method"],
            "strength": float(strength),
        }
    reduction = 0.0
    if result["jerk_raw"] > 1e-12:
        reduction = max(0.0, (1.0 - result["jerk_smoothed"] / result["jerk_raw"]) * 100.0)
    return {
        "token": new_token,
        "frames": len(smoothed_frames),
        "fps": fps,
        "joint_names": joint_names,
        "units": str(session.get("units") or "radians"),
        "method": result["method"],
        "requested_method": result["requested_method"],
        "strength": float(strength),
        "scipy": bool(result["scipy"]),
        "jerk_reduction_pct": reduction,
        "preview": result["preview"],
    }


def _sparkline(points: list[float], lo: float, span: float, x0: float, width: float,
               y0: float, height: float, color: str) -> str:
    if len(points) < 2:
        return ""
    step = width / (len(points) - 1)
    coords = " ".join(
        f"{x0 + i * step:.1f},{y0 + height - (value - lo) / span * height:.1f}"
        for i, value in enumerate(points)
    )
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>'


def smoother_dashboard(info: dict[str, Any]) -> str:
    preview = dict(info.get("preview") or {})
    raw = [float(value) for value in (preview.get("raw") or [])]
    smoothed = [float(value) for value in (preview.get("smoothed") or [])]
    method = html.escape(str(info.get("method") or "none"))
    joint = html.escape(str(preview.get("joint") or ""))
    source = html.escape(str(preview.get("source") or "action"))
    reduction = float(info.get("jerk_reduction_pct") or 0.0)
    combined = (raw + smoothed) or [0.0, 1.0]
    lo, hi = min(combined), max(combined)
    span = (hi - lo) or 1.0
    raw_line = _sparkline(raw, lo, span, 40, 880, 96, 150, "#f87171")
    smooth_line = _sparkline(smoothed, lo, span, 40, 880, 96, 150, "#22c55e")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="320" viewBox="0 0 960 320">'
        '<rect width="960" height="320" rx="16" fill="#0f172a"/>'
        f'<text x="40" y="46" fill="#e2e8f0" font-family="sans-serif" font-size="22" font-weight="700">'
        f'TRAJECTORY SMOOTHER</text>'
        f'<text x="920" y="46" fill="#94a3b8" font-family="monospace" font-size="16" text-anchor="end">'
        f'{method}</text>'
        f'<text x="40" y="78" fill="#94a3b8" font-family="sans-serif" font-size="13">'
        f'{source} · {joint} — raw vs smoothed</text>'
        '<rect x="40" y="96" width="880" height="150" rx="8" fill="#020617"/>'
        f'{raw_line}{smooth_line}'
        '<rect x="40" y="270" width="14" height="14" fill="#f87171"/>'
        '<text x="62" y="282" fill="#94a3b8" font-family="sans-serif" font-size="13">recorded (raw)</text>'
        '<rect x="220" y="270" width="14" height="14" fill="#22c55e"/>'
        '<text x="242" y="282" fill="#94a3b8" font-family="sans-serif" font-size="13">smoothed</text>'
        f'<text x="920" y="282" fill="#e2e8f0" font-family="monospace" font-size="16" text-anchor="end">'
        f'jerk -{reduction:.0f}%</text>'
        '</svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def smoother_outputs(info: dict[str, Any]) -> dict[str, Any]:
    method = str(info.get("method") or "none")
    requested = str(info.get("requested_method") or method)
    note = ""
    if requested != method and requested in {"spline", "savgol"} and not info.get("scipy"):
        note = f"; {requested} needs SciPy, used {method} instead"
    report = (f"smoothed {int(info.get('frames') or 0)} frames with {method} "
              f"(strength {float(info.get('strength') or 0):g}); "
              f"jerk -{float(info.get('jerk_reduction_pct') or 0):.0f}% on "
              f"{info.get('preview', {}).get('joint', '')}{note}")
    label = f"smoothed · {method}"
    return {
        "stream": make_replay_stream(
            str(info.get("token") or ""), label=label,
            frames=int(info.get("frames") or 0), fps=float(info.get("fps") or 0.0),
            units=str(info.get("units") or "radians"),
        ),
        "episode": {
            "frames": int(info.get("frames") or 0),
            "fps": float(info.get("fps") or 0.0),
            "joint_names": list(info.get("joint_names") or []),
            "units": str(info.get("units") or "radians"),
            "method": method,
            "strength": float(info.get("strength") or 0.0),
            "jerk_reduction_pct": float(info.get("jerk_reduction_pct") or 0.0),
        },
        "preview": smoother_dashboard(info),
        "report": report,
    }


def make_replay_stream(token: str, label: str = "", frames: int = 0, fps: float = 0.0,
                       units: str = "radians") -> dict[str, Any]:
    """A stream handle wrapping a recorded-episode replay token."""
    return {
        "kind": "blacknode.replay-stream",
        "token": str(token or ""),
        "label": str(label or ""),
        "frames": int(frames or 0),
        "fps": float(fps or 0.0),
        "units": str(units or "radians"),
    }


def parse_stream(stream: Any) -> tuple[str, str, str]:
    """Return (mode, token, url) for a stream handle, or raise if unrecognized.

    Accepts a ``blacknode.replay-stream`` handle (recorded episode, replayed by
    walking frames) or a ``blacknode.sample-stream`` handle (a live source polled
    for its latest sample), so the publisher can stream anything, not just replay.
    """
    handle = dict(stream) if isinstance(stream, dict) else {}
    kind = str(handle.get("kind") or "")
    if kind == "blacknode.replay-stream" or (not kind and handle.get("token")):
        return "replay", str(handle.get("token") or ""), ""
    if kind == "blacknode.sample-stream" or (not kind and handle.get("url")):
        return "sample", "", str(handle.get("url") or "")
    raise ValueError("connect a stream handle: a DatasetBrowser/TrajectorySmoother 'stream' "
                     "output, or a live 'blacknode.sample-stream'")


@dataclass
class StreamPublisher:
    """Broadcast a stream over a WebSocket to any number of subscribers.

    Two source kinds are supported through the same wire format: a recorded
    replay (walk episode frames via ``replay_frame``) and a live sample-stream
    (poll a URL for its latest sample). Read-only either way — it never opens a
    robot connection or commands motion; the receiving client decides what to do.
    """

    run_id: str
    mode: str
    token: str
    url: str
    host: str
    port: int
    fps: float
    rate: float
    loop: bool
    source: str
    units: str
    server: wsserver.WsBroadcastServer
    frames: int
    joint_names: list[str]
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    thread: threading.Thread | None = None
    frame_index: int = 0
    sent: int = 0
    loops: int = 0
    last_error: str = ""
    started_ns: int = field(default_factory=time.time_ns)
    stopped: bool = False

    def start(self) -> None:
        self.server.start()
        self.thread = threading.Thread(target=self._loop, daemon=True,
                                       name=f"blacknode-stream-{self.run_id}")
        self.thread.start()

    def _broadcast(self, frame: dict[str, Any], index: int) -> None:
        names = list(frame.get("joint_names") or [])
        joints = frame.get(self.source) or {}
        positions = [float(joints.get(name, 0.0)) for name in names]
        payload = {
            **frame,
            "fps": self.fps,
            "rate": self.rate,
            "source": self.source,
            "units": self.units,
            "positions": positions,
            "seq": self.sent,
            "run_id": self.run_id,
        }
        self.server.broadcast(json.dumps(payload))
        with self.lock:
            self.frame_index = index
            self.sent += 1
            self.last_error = ""

    def _poll_sample(self) -> dict[str, Any]:
        sample = _get_json(self.url, 2.0)
        if sample.get("kind") != "blacknode.teleoperation-sample":
            raise ValueError("sample stream did not return a teleoperation sample")
        return {
            "kind": "blacknode.stream-frame",
            "schema_version": 1,
            "frame_index": self.sent,
            "frames": 0,
            "timestamp": float(sample.get("captured_at_ns") or 0) / 1e9,
            "captured_at_ns": int(sample.get("captured_at_ns") or 0),
            "joint_names": list(sample.get("joint_names") or []),
            "leader": dict(sample.get("leader") or {}),
            "observation": dict(sample.get("observation") or {}),
            "action": dict(sample.get("action") or {}),
            "cameras": {},
        }

    def _loop(self) -> None:
        try:
            period = 1.0 / max(1e-3, self.fps * max(0.01, self.rate))
            deadline = time.monotonic()
            index = 0
            while not self.stop_event.is_set():
                if self.mode == "replay":
                    if self.frames <= 0:
                        self.stop_event.wait(0.2)
                        continue
                    frame = replay_frame(self.token, index)
                    if frame is None:
                        with self.lock:
                            self.last_error = "replay selection expired; refresh the Dataset Browser"
                        break
                    self._broadcast(frame, index)
                    index += 1
                    if index >= self.frames:
                        if not self.loop:
                            break
                        index = 0
                        with self.lock:
                            self.loops += 1
                else:  # live sample-stream: poll the latest value
                    try:
                        self._broadcast(self._poll_sample(), self.sent)
                    except Exception as exc:  # noqa: BLE001 - a stale poll must not kill the stream
                        with self.lock:
                            self.last_error = f"{type(exc).__name__}: {exc}"
                deadline += period
                self.stop_event.wait(max(0.0, deadline - time.monotonic()))
                if deadline < time.monotonic() - period:
                    deadline = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - surfaced in publisher status
            with self.lock:
                self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.stopped = True

    def status(self) -> dict[str, Any]:
        with self.lock:
            alive = bool(self.thread and self.thread.is_alive())
            return {
                "run_id": self.run_id,
                "mode": self.mode,
                "running": alive,
                "streaming": alive,
                "stream_url": f"ws://{self.host}:{self.port}",
                "host": self.host,
                "port": self.port,
                "clients": self.server.client_count(),
                "frame_index": self.frame_index,
                "frames": self.frames,
                "sent": self.sent,
                "loops": self.loops,
                "fps": self.fps,
                "rate": self.rate,
                "loop": self.loop,
                "source": self.source,
                "units": self.units,
                "last_error": self.last_error,
            }

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        self.server.stop()


def stream_dashboard(status: dict[str, Any]) -> str:
    running = bool(status.get("streaming") or status.get("running"))
    color = "#22c55e" if running else "#64748b"
    label = "STREAMING" if running else "STOPPED"
    url = html.escape(str(status.get("stream_url") or "ws://—"))
    source = html.escape(str(status.get("source") or "action"))
    units = html.escape(str(status.get("units") or "radians"))
    clients = int(status.get("clients") or 0)
    frame_index = int(status.get("frame_index") or 0)
    frames = max(0, int(status.get("frames") or 0) - 1)
    sent = int(status.get("sent") or 0)
    fps = float(status.get("fps") or 0.0)
    rate = float(status.get("rate") or 1.0)
    error = html.escape(str(status.get("last_error") or "read-only replay · never commands hardware"))
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="360" viewBox="0 0 960 360">'
        '<rect width="960" height="360" rx="16" fill="#0f172a"/>'
        f'<circle cx="36" cy="36" r="10" fill="{color}"/>'
        f'<text x="56" y="43" fill="#e2e8f0" font-family="sans-serif" font-size="22" font-weight="700">{label}</text>'
        f'<text x="936" y="42" fill="#94a3b8" font-family="monospace" font-size="16" text-anchor="end">{url}</text>'
        f'<text x="40" y="112" fill="#94a3b8" font-family="sans-serif" font-size="13">SUBSCRIBERS</text>'
        f'<text x="40" y="146" fill="#e2e8f0" font-family="monospace" font-size="30">{clients}</text>'
        f'<text x="300" y="112" fill="#94a3b8" font-family="sans-serif" font-size="13">FRAME</text>'
        f'<text x="300" y="146" fill="#e2e8f0" font-family="monospace" font-size="30">{frame_index}/{frames}</text>'
        f'<text x="560" y="112" fill="#94a3b8" font-family="sans-serif" font-size="13">SENT</text>'
        f'<text x="560" y="146" fill="#e2e8f0" font-family="monospace" font-size="30">{sent}</text>'
        f'<text x="40" y="212" fill="#94a3b8" font-family="sans-serif" font-size="13">SOURCE · UNITS</text>'
        f'<text x="40" y="242" fill="#e2e8f0" font-family="monospace" font-size="20">{source} · {units}</text>'
        f'<text x="560" y="212" fill="#94a3b8" font-family="sans-serif" font-size="13">RATE</text>'
        f'<text x="560" y="242" fill="#e2e8f0" font-family="monospace" font-size="20">{fps:.1f} fps × {rate:g}</text>'
        f'<text x="40" y="322" fill="#fca5a5" font-family="sans-serif" font-size="13">{error}</text></svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def stream_outputs(status: dict[str, Any]) -> dict[str, Any]:
    running = bool(status.get("streaming") or status.get("running"))
    url = str(status.get("stream_url") or "")
    if running:
        report = (f"streaming {status.get('source', 'action')} -> {url} · "
                  f"{int(status.get('clients') or 0)} subscriber(s) · "
                  f"frame {int(status.get('frame_index') or 0)}/{max(0, int(status.get('frames') or 0) - 1)} · "
                  f"sent {int(status.get('sent') or 0)}")
    else:
        report = "stream stopped" + (f": {status['last_error']}" if status.get("last_error") else "")
    return {
        "stream_url": url,
        "streaming": running,
        "clients": int(status.get("clients") or 0),
        "status": status,
        "dashboard": stream_dashboard(status),
        "report": report,
    }


def start_stream(*, run_id: str, stream: Any, host: str, port: int, fps: float,
                 rate: float, loop: bool, source: str, units: str) -> dict[str, Any]:
    run_id = str(run_id or "stream").strip() or "stream"
    source = str(source or "action").strip().lower()
    if source not in {"action", "observation", "leader"}:
        raise ValueError("source must be action, observation, or leader")
    mode, token, url = parse_stream(stream)
    with _lock:
        existing = _publishers.get(run_id)
        if existing and existing.thread and existing.thread.is_alive():
            return existing.status()
    resolved_fps = float(fps or 0) or 30.0
    resolved_units = str(units or "radians")
    frames, joint_names = 0, []
    if mode == "replay":
        with _lock:
            session = dict(_replay_sessions.get(token) or {})
        if not session:
            raise ValueError("replay selection expired; open the Dataset Browser and select an episode first")
        frames = int(session.get("frames") or 0)
        joint_names = list(session.get("joint_names") or [])
        resolved_fps = float(fps or 0) or float(session.get("fps") or 0) or 30.0
        resolved_units = str(units or session.get("units") or "radians")
    server = wsserver.WsBroadcastServer(host, int(port))
    publisher = StreamPublisher(
        run_id=run_id, mode=mode, token=token, url=url,
        host=str(host or "127.0.0.1"), port=int(port),
        fps=resolved_fps, rate=max(0.01, float(rate or 1.0)), loop=bool(loop), source=source,
        units=resolved_units, server=server, frames=frames, joint_names=joint_names,
    )
    try:
        publisher.start()
    except OSError as exc:
        raise ValueError(f"could not open WebSocket port {port}: {exc}") from exc
    publisher.port = server.port  # may differ from the requested port when 0 was passed
    with _lock:
        _publishers[run_id] = publisher
    return publisher.status()


def control_stream(run_id: str, action: str) -> dict[str, Any]:
    run_id = str(run_id or "stream").strip() or "stream"
    action = str(action or "status").strip().lower()
    with _lock:
        publisher = _publishers.get(run_id)
    if action == "stop":
        if publisher is None:
            return {"run_id": run_id, "running": False, "streaming": False, "clients": 0,
                    "sent": 0, "frames": 0, "stream_url": "", "last_error": "stream is not running"}
        publisher.stop()
        with _lock:
            _publishers.pop(run_id, None)
        return {**publisher.status(), "running": False, "streaming": False}
    if publisher is None:
        return {"run_id": run_id, "running": False, "streaming": False, "clients": 0,
                "frames": 0, "sent": 0, "stream_url": "", "last_error": ""}
    return publisher.status()


def control_configured_recorder(run_id: str, action: str) -> dict[str, Any]:
    """Control a configured recorder without evaluating any graph dependencies."""
    run_id = str(run_id or "episode_recorder").strip() or "episode_recorder"
    action = str(action or "status").strip().lower()
    if action not in {"status", "start", "pause", "resume", "save", "finalize", "stop", "discard"}:
        raise ValueError(f"unsupported recorder action: {action}")
    with _lock:
        config = dict(_recorder_configs.get(run_id) or {})
    if not config:
        raise ValueError("recorder inputs are not configured; run the graph once before using recorder controls")
    dataset = dict(config["dataset"])
    if action == "start":
        status = start_recorder(**config)
    else:
        status = control_recorder(run_id, action)
        if (not status.get("running") and status.get("last_error") == "recorder is not running"
                and action in {"save", "finalize", "discard"}):
            status = recover_episode(dataset, run_id, action)
        elif action == "status" and not status.get("running"):
            status = recoverable_episode_status(dataset, run_id) or status
    return recorder_outputs(status, dataset)


def runtime_status() -> dict[str, Any]:
    with _lock:
        runs = [recorder.status() for recorder in _recorders.values()]
    node_outputs = [{
        "run_id": item["run_id"],
        "node_type": "EpisodeRecorder",
        "outputs": {
            **{key: item[key] for key in (
                "running", "recording", "paused", "episode_index", "frame_count",
                "dropped_frames", "duration_seconds",
            )},
            "status": item,
            "dashboard": dashboard(item),
            "report": f"episode recorder {'recording' if item['recording'] else 'paused'}: {item['frame_count']} frames"
                      + (f"; {item['last_error']}" if item.get("last_error") else ""),
        },
    } for item in runs]
    with _lock:
        publishers = [publisher.status() for publisher in _publishers.values()]
    node_outputs.extend({
        "run_id": item["run_id"],
        "node_type": "StreamPublisher",
        "outputs": stream_outputs(item),
    } for item in publishers)
    active = any(item["running"] for item in runs) or any(item["running"] for item in publishers)
    report = f"{len(runs)} dataset recorder(s) managed"
    if publishers:
        report += f"; {len(publishers)} replay stream(s) publishing"
    return {"ok": True, "active": active, "managed_runs": runs,
            "node_outputs": node_outputs,
            "streams": [], "detached_count": 0, "report": report}


def stop_runtime_services() -> dict[str, Any]:
    with _lock:
        recorders = list(_recorders.values())
        _recorders.clear()
        _recorder_configs.clear()
        publishers = list(_publishers.values())
        _publishers.clear()
    for recorder in recorders:
        recorder.stop()
    for publisher in publishers:
        publisher.stop()
    return {"ok": True, "stopped": {"streams": len(publishers), "managed_runs": len(recorders),
                                      "detached": 0, "cv2_streams": 0, "reasoning_streams": 0},
            "report": f"stopped {len(recorders)} recorder(s) and {len(publishers)} replay stream(s); "
                      "incomplete episodes remain recoverable"}
