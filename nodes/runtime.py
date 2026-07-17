"""Managed, crash-recoverable episode recording runtime."""
from __future__ import annotations

import base64
import html
import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import storage


def dashboard(status: dict[str, Any]) -> str:
    color = "#22c55e" if status.get("recording") else "#f59e0b" if status.get("paused") else "#64748b"
    label = "RECORDING" if status.get("recording") else "PAUSED" if status.get("paused") else "STOPPED"
    error = html.escape(str(status.get("last_error") or "ready"))
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="180" viewBox="0 0 720 180">'
        '<rect width="720" height="180" rx="16" fill="#0f172a"/>'
        f'<circle cx="42" cy="40" r="10" fill="{color}"/><text x="62" y="47" fill="#e2e8f0" '
        f'font-family="sans-serif" font-size="22" font-weight="700">{label}</text>'
        f'<text x="30" y="94" fill="#cbd5e1" font-family="monospace" font-size="18">episode '
        f'{int(status.get("episode_index") or 0)} · {int(status.get("frame_count") or 0)} frames · '
        f'{float(status.get("duration_seconds") or 0):.1f}s</text>'
        f'<text x="30" y="134" fill="#94a3b8" font-family="sans-serif" font-size="14">dropped '
        f'{int(status.get("dropped_frames") or 0)} · {error}</text></svg>'
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

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"blacknode-dataset-{self.run_id}")
        self.thread.start()

    def _validate_robot(self, sample: dict[str, Any], now_ns: int) -> None:
        if sample.get("kind") != "blacknode.teleoperation-sample":
            raise ValueError("sample stream did not return a blacknode.teleoperation-sample")
        captured = int(sample.get("captured_at_ns") or 0)
        if not captured or now_ns - captured > int(self.stale_after * 1e9):
            raise ValueError("robot sample is stale")
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
        now_ns = time.time_ns()
        sample = _get_json(_stream_url(self.robot_stream), self.request_timeout)
        self._validate_robot(sample, now_ns)
        images: dict[str, bytes] = {}
        camera_meta: dict[str, Any] = {}
        for handle in self.cameras:
            name = str(handle["stream_id"])
            image, meta = _get_image(handle, self.request_timeout)
            captured = int(meta.get("captured_at_ns") or 0)
            if not captured or now_ns - captured > int(self.stale_after * 1e9):
                raise ValueError(f"camera {name} frame is stale")
            if not image:
                raise ValueError(f"camera {name} returned an empty frame")
            images[name] = image
            camera_meta[name] = meta
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
        }

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=max(2.0, self.request_timeout * 2.0))


_recorders: dict[str, EpisodeRecorder] = {}
_lock = threading.RLock()


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
            result = {**recorder.status(), "saved": True, "episode": episode}
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
        return {"run_id": run_id, "running": False, "recording": False, "paused": False,
                "frame_count": int(episode["frames"]), "dropped_frames": 0,
                "duration_seconds": float(episode["duration_seconds"]), "saved": True, "episode": episode}
    if action == "discard":
        discarded = storage.discard_episode(path, run_id)
        return {"run_id": run_id, "running": False, "recording": False, "paused": False,
                "frame_count": 0, "dropped_frames": 0, "discarded": discarded,
                "last_error": "" if discarded else "incomplete episode was not found"}
    raise ValueError("recovery supports save, finalize, or discard")


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
    return {"ok": True, "active": any(item["running"] for item in runs), "managed_runs": runs,
            "node_outputs": node_outputs,
            "streams": [], "detached_count": 0, "report": f"{len(runs)} dataset recorder(s) managed"}


def stop_runtime_services() -> dict[str, Any]:
    with _lock:
        recorders = list(_recorders.values())
        _recorders.clear()
    for recorder in recorders:
        recorder.stop()
    return {"ok": True, "stopped": {"streams": 0, "managed_runs": len(recorders), "detached": 0,
                                      "cv2_streams": 0, "reasoning_streams": 0},
            "report": f"stopped {len(recorders)} recorder(s); incomplete episodes remain recoverable"}
