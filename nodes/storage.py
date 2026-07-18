"""Blacknode episode storage and export helpers."""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - dependency warning is surfaced by Blacknode
    cv2 = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None
    pq = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from . import filters


def _smooth_rows(rows: Any, method: str, strength: float, fps: float) -> Any:
    """Smooth a per-episode ``[T, J]`` trajectory before export (zero-lag).

    Accepts and returns a numpy array or a list of rows to match the caller. A
    ``none``/empty method or a missing numpy is a pass-through, so existing
    exports are byte-for-byte unchanged unless smoothing is explicitly chosen.
    """
    method = str(method or "none").lower()
    if method == "none" or np is None:
        return rows
    was_list = not hasattr(rows, "shape")
    smoothed, _ = filters.smooth_columns(np.asarray(rows, dtype=float), method, float(strength), float(fps))
    return smoothed.tolist() if was_list else smoothed

DATASET_KIND = "blacknode.episode-dataset"
DATASET_SCHEMA_VERSION = 1


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    if not normalized:
        raise ValueError("dataset_id is required")
    return normalized


def default_home() -> Path:
    configured = os.getenv("BLACKNODE_DATASET_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".blacknode" / "datasets"


def resolve_dataset_path(dataset: dict[str, Any] | str | Path) -> Path:
    if isinstance(dataset, dict):
        raw = str(dataset.get("path") or "").strip()
    else:
        raw = str(dataset or "").strip()
    if not raw:
        raise ValueError("dataset path is required")
    return Path(raw).expanduser().resolve()


def incomplete_episode_path(path: Path, run_id: str) -> Path:
    """Return the stable journal directory for a recorder run."""
    return path / "incomplete" / _slug(run_id)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "dataset.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("kind") != DATASET_KIND or int(data.get("schema_version") or 0) != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported Blacknode dataset manifest: {manifest_path}")
    return data


def descriptor(path: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    info = manifest or load_manifest(path)
    return {
        "kind": DATASET_KIND,
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": info["dataset_id"],
        "path": str(path),
        "fps": int(info["fps"]),
        "task": str(info.get("task") or ""),
        "episode_count": len(info.get("episodes") or []),
    }


def create_dataset(
    dataset_id: str,
    *,
    root: str = "",
    task: str,
    fps: int,
    robot_type: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = Path(root).expanduser().resolve() if str(root or "").strip() else default_home().resolve()
    path = base / _slug(dataset_id)
    manifest_path = path / "dataset.json"
    if manifest_path.exists():
        manifest = load_manifest(path)
        if int(manifest.get("fps") or 0) != int(fps):
            raise ValueError(f"dataset already exists at {path} with fps={manifest.get('fps')}")
        requested_task = str(task or "").strip()
        if requested_task and requested_task != str(manifest.get("task") or ""):
            raise ValueError(f"dataset already exists at {path} with a different task")
        return descriptor(path, manifest)
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": DATASET_KIND,
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": _slug(dataset_id),
        "created_at": _now(),
        "updated_at": _now(),
        "fps": max(1, int(fps)),
        "task": str(task or "").strip(),
        "robot_type": str(robot_type or "").strip(),
        "metadata": dict(metadata or {}),
        "features": {},
        "episodes": [],
    }
    (path / "episodes").mkdir(exist_ok=True)
    (path / "incomplete").mkdir(exist_ok=True)
    _atomic_json(manifest_path, manifest)
    return descriptor(path, manifest)


def begin_episode(path: Path, run_id: str) -> tuple[Path, int, dict[str, Any]]:
    manifest = load_manifest(path)
    episode_index = len(manifest.get("episodes") or [])
    work = incomplete_episode_path(path, run_id)
    if work.exists():
        raise ValueError(f"incomplete episode already exists for run_id={run_id}; save, discard, or recover it")
    (work / "cameras").mkdir(parents=True)
    episode = {
        "kind": "blacknode.episode-journal",
        "schema_version": 1,
        "run_id": run_id,
        "episode_index": episode_index,
        "task": str(manifest.get("task") or ""),
        "fps": int(manifest["fps"]),
        "started_at": _now(),
        "frames": 0,
    }
    _atomic_json(work / "episode.json", episode)
    (work / "frames.jsonl").touch()
    return work, episode_index, manifest


def append_frame(work: Path, frame: dict[str, Any], camera_images: dict[str, bytes]) -> None:
    frame_index = int(frame["frame_index"])
    camera_meta: dict[str, Any] = {}
    for name, content in camera_images.items():
        camera_dir = work / "cameras" / _slug(name)
        camera_dir.mkdir(parents=True, exist_ok=True)
        image_path = camera_dir / f"frame-{frame_index:06d}.jpg"
        image_path.write_bytes(content)
        camera_meta[name] = {
            **dict((frame.get("cameras") or {}).get(name) or {}),
            "path": str(image_path.relative_to(work)).replace("\\", "/"),
        }
    frame["cameras"] = {**dict(frame.get("cameras") or {}), **camera_meta}
    with (work / "frames.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(frame, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    episode = json.loads((work / "episode.json").read_text(encoding="utf-8"))
    episode["frames"] = frame_index + 1
    episode["updated_at"] = _now()
    _atomic_json(work / "episode.json", episode)


def read_frames(work: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (work / "frames.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _require_storage_dependencies() -> None:
    missing = []
    if pa is None or pq is None:
        missing.append("pyarrow")
    if pd is None:
        missing.append("pandas")
    if cv2 is None:
        missing.append("opencv-python")
    if missing:
        raise RuntimeError("missing dataset dependencies: " + ", ".join(missing))


def _write_frame_parquet(path: Path, frames: list[dict[str, Any]], joint_names: list[str], episode_index: int) -> None:
    assert pa is not None and pq is not None
    vector_type = pa.list_(pa.float32(), len(joint_names))
    columns: dict[str, Any] = {
        "timestamp": pa.array([float(row["timestamp"]) for row in frames], type=pa.float32()),
        "recorded_at_ns": pa.array([int(row.get("recorded_at_ns") or 0) for row in frames], type=pa.int64()),
        "frame_index": pa.array([int(row["frame_index"]) for row in frames], type=pa.int64()),
        "episode_index": pa.array([episode_index] * len(frames), type=pa.int64()),
        "observation.state": pa.array(
            [[float(row["robot"]["observation"][name]) for name in joint_names] for row in frames], type=vector_type
        ),
        "action": pa.array(
            [[float(row["robot"]["action"][name]) for name in joint_names] for row in frames], type=vector_type
        ),
        "leader.state": pa.array(
            [[float(row["robot"]["leader"][name]) for name in joint_names] for row in frames], type=vector_type
        ),
        "task": pa.array([str(row.get("task") or "") for row in frames], type=pa.string()),
        "sample_sequence": pa.array([int(row["robot"].get("sequence") or 0) for row in frames], type=pa.int64()),
        "captured_at_ns": pa.array([int(row["robot"].get("captured_at_ns") or 0) for row in frames], type=pa.int64()),
    }
    camera_names = sorted({str(name) for row in frames for name in (row.get("cameras") or {})})
    for camera in camera_names:
        columns[f"camera.{camera}.sequence"] = pa.array(
            [int(((row.get("cameras") or {}).get(camera) or {}).get("sequence") or 0) for row in frames],
            type=pa.int64(),
        )
        columns[f"camera.{camera}.captured_at_ns"] = pa.array(
            [int(((row.get("cameras") or {}).get(camera) or {}).get("captured_at_ns") or 0) for row in frames],
            type=pa.int64(),
        )
    table = pa.table(columns)
    pq.write_table(table, path, compression="snappy")


def _encode_camera(images: list[Path], output: Path, fps: int) -> dict[str, Any]:
    assert cv2 is not None
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"could not decode camera frame {images[0]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_executable()
    if ffmpeg:
        command = [
            ffmpeg, "-y", "-v", "error", "-framerate", str(int(fps)),
            "-start_number", "0", "-i", str(images[0].parent / "frame-%06d.jpg"),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and output.is_file() and output.stat().st_size > 0:
            return {
                "width": width, "height": height, "channels": 3, "frames": len(images),
                "codec": "h264", "pixel_format": "yuv420p", "has_audio": False,
            }
        output.unlink(missing_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open MP4 encoder for {output}")
    try:
        for image in images:
            frame = cv2.imread(str(image))
            if frame is None:
                raise RuntimeError(f"could not decode camera frame {image}")
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    return {
        "width": width, "height": height, "channels": 3, "frames": len(images),
        "codec": "mpeg4", "pixel_format": "yuv420p", "has_audio": False,
    }


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return str(imageio_ffmpeg.get_ffmpeg_exe() or "")
    except Exception:
        return ""


def browser_video_path(replay: dict[str, Any]) -> Path:
    """Return an H.264 playback file, transcoding old mp4v episodes into cache."""
    source = Path(str(replay.get("video_path") or "")).resolve()
    codec = str((replay.get("camera_info") or {}).get("codec") or "").lower()
    if codec in {"h264", "avc", "avc1", "vp8", "vp9", "av1"}:
        return source
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        return source
    stat = source.stat()
    cache_key = f"{source}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    import hashlib
    cache_root = Path(tempfile.gettempdir()) / "blacknode-dataset-replay"
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{hashlib.sha256(cache_key).hexdigest()[:24]}.mp4"
    if target.is_file() and target.stat().st_size > 0:
        return target
    temp = target.with_suffix(".tmp.mp4")
    command = [
        ffmpeg, "-y", "-v", "error", "-i", str(source), "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not temp.is_file() or temp.stat().st_size <= 0:
        temp.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"could not prepare browser-compatible episode video: {detail}")
    temp.replace(target)
    return target


def save_episode(path: Path, run_id: str) -> dict[str, Any]:
    _require_storage_dependencies()
    work = incomplete_episode_path(path, run_id)
    frames = read_frames(work)
    if not frames:
        raise ValueError("cannot save an episode with zero frames")
    manifest = load_manifest(path)
    episode_index = int(json.loads((work / "episode.json").read_text(encoding="utf-8"))["episode_index"])
    if episode_index != len(manifest.get("episodes") or []):
        raise ValueError("episode index changed while recording; save episodes sequentially")
    joint_names = [str(name) for name in frames[0]["robot"].get("joint_names") or []]
    if not joint_names:
        joint_names = list(frames[0]["robot"]["observation"])
    for row in frames:
        if list(row["robot"]["observation"]) != joint_names or list(row["robot"]["action"]) != joint_names:
            raise ValueError("joint names or ordering changed inside the episode")
    previous_features = manifest.get("features") or {}
    if previous_features and list(previous_features.get("joint_names") or []) != joint_names:
        raise ValueError("joint names or ordering differ from earlier episodes")

    final = path / "episodes" / f"episode-{episode_index:06d}"
    temp = path / "episodes" / f".episode-{episode_index:06d}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    _write_frame_parquet(temp / "data.parquet", frames, joint_names, episode_index)
    camera_info: dict[str, Any] = {}
    for camera_dir in sorted((work / "cameras").iterdir() if (work / "cameras").exists() else []):
        images = sorted(camera_dir.glob("frame-*.jpg"))
        if images:
            camera_info[camera_dir.name] = _encode_camera(images, temp / "cameras" / f"{camera_dir.name}.mp4", int(manifest["fps"]))
    duration = len(frames) / float(manifest["fps"])
    episode_info = {
        "kind": "blacknode.episode",
        "schema_version": 1,
        "episode_index": episode_index,
        "task": str(manifest.get("task") or ""),
        "fps": int(manifest["fps"]),
        "frames": len(frames),
        "duration_seconds": duration,
        "joint_names": joint_names,
        "units": str(frames[0]["robot"].get("units") or "radians"),
        "robot": {
            key: frames[0]["robot"].get(key)
            for key in ("leader_hardware_id", "follower_hardware_id", "leader_calibration_path", "follower_calibration_path")
        },
        "cameras": camera_info,
        "saved_at": _now(),
    }
    _atomic_json(temp / "episode.json", episode_info)
    temp.replace(final)
    shutil.rmtree(work)
    manifest["features"] = {
        "joint_names": joint_names,
        "units": episode_info["units"],
        "cameras": camera_info,
    }
    manifest.setdefault("episodes", []).append({
        "episode_index": episode_index,
        "path": str(final.relative_to(path)).replace("\\", "/"),
        "frames": len(frames),
        "duration_seconds": duration,
        "task": episode_info["task"],
        "saved_at": episode_info["saved_at"],
    })
    manifest["updated_at"] = _now()
    _atomic_json(path / "dataset.json", manifest)
    return episode_info


def trim_episode(path: Path, episode_index: int, keep_start: int, keep_end: int) -> dict[str, Any]:
    """Atomically keep one inclusive frame range across Parquet and every camera."""
    _require_storage_dependencies()
    path = Path(path).resolve()
    manifest = load_manifest(path)
    episodes = list(manifest.get("episodes") or [])
    index = int(episode_index)
    if index < 0 or index >= len(episodes):
        raise ValueError(f"episode_index {index} is unavailable")
    entry = dict(episodes[index])
    final = (path / str(entry.get("path") or "")).resolve()
    if path not in final.parents:
        raise ValueError("episode path escapes the dataset")
    info = json.loads((final / "episode.json").read_text(encoding="utf-8"))
    source_table = pq.read_table(final / "data.parquet")
    total = int(source_table.num_rows)
    start = int(keep_start)
    end = int(keep_end)
    if total <= 0 or start < 0 or end >= total or start > end:
        raise ValueError(f"invalid trim range {start}..{end} for {total} frames")
    if start == 0 and end == total - 1:
        raise ValueError("trim range keeps the complete episode")
    count = end - start + 1
    fps = max(1, int(info.get("fps") or manifest.get("fps") or 1))
    temp = final.parent / f".{final.name}.trim-tmp"
    backup = final.parent / f".{final.name}.trim-backup"
    for scratch in (temp, backup):
        if scratch.exists():
            shutil.rmtree(scratch)
    temp.mkdir(parents=True)
    try:
        table = source_table.slice(start, count)
        for column, values in (
            ("frame_index", list(range(count))),
            ("timestamp", [frame / float(fps) for frame in range(count)]),
        ):
            position = table.schema.get_field_index(column)
            if position >= 0:
                field = table.schema.field(position)
                table = table.set_column(position, field, pa.array(values, type=field.type))
        pq.write_table(table, temp / "data.parquet", compression="snappy")

        camera_info: dict[str, Any] = {}
        for camera, previous in sorted((info.get("cameras") or {}).items()):
            source_video = final / "cameras" / f"{camera}.mp4"
            capture = cv2.VideoCapture(str(source_video))
            frame_dir = temp / ".frames" / camera
            frame_dir.mkdir(parents=True)
            images: list[Path] = []
            source_index = 0
            try:
                while source_index <= end:
                    ok, image = capture.read()
                    if not ok:
                        break
                    if source_index >= start:
                        image_path = frame_dir / f"frame-{len(images):06d}.jpg"
                        if not cv2.imwrite(str(image_path), image):
                            raise RuntimeError(f"could not write trimmed camera frame {image_path}")
                        images.append(image_path)
                    source_index += 1
            finally:
                capture.release()
            if len(images) != count:
                raise RuntimeError(f"camera {camera} decoded {len(images)} trimmed frames; expected {count}")
            camera_info[camera] = {
                **dict(previous or {}),
                **_encode_camera(images, temp / "cameras" / f"{camera}.mp4", fps),
            }
        shutil.rmtree(temp / ".frames", ignore_errors=True)

        trimmed_at = _now()
        updated_info = {
            **info,
            "frames": count,
            "duration_seconds": count / float(fps),
            "cameras": camera_info,
            "trimmed_at": trimmed_at,
            "trim_history": [
                *list(info.get("trim_history") or []),
                {"source_frames": total, "kept_start": start, "kept_end": end, "result_frames": count, "trimmed_at": trimmed_at},
            ],
        }
        _atomic_json(temp / "episode.json", updated_info)

        final.replace(backup)
        try:
            temp.replace(final)
            entry.update({"frames": count, "duration_seconds": count / float(fps), "trimmed_at": trimmed_at})
            episodes[index] = entry
            manifest["episodes"] = episodes
            manifest["updated_at"] = trimmed_at
            if len(episodes) == 1:
                manifest.setdefault("features", {})["cameras"] = camera_info
            _atomic_json(path / "dataset.json", manifest)
        except Exception:
            if final.exists():
                shutil.rmtree(final)
            backup.replace(final)
            raise
        shutil.rmtree(backup)
        return {
            "ok": True,
            "dataset_path": str(path),
            "episode_path": str(final),
            "episode_index": index,
            "source_frames": total,
            "removed_frames": total - count,
            "frames": count,
            "duration_seconds": count / float(fps),
            "kept_start": start,
            "kept_end": end,
        }
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def discard_episode(path: Path, run_id: str) -> bool:
    work = incomplete_episode_path(path, run_id)
    if not work.exists():
        return False
    shutil.rmtree(work)
    return True


def summarize(path: Path) -> dict[str, Any]:
    manifest = load_manifest(path)
    episodes = list(manifest.get("episodes") or [])
    return {
        **descriptor(path, manifest),
        "total_frames": sum(int(item.get("frames") or 0) for item in episodes),
        "duration_seconds": sum(float(item.get("duration_seconds") or 0.0) for item in episodes),
        "joint_names": list((manifest.get("features") or {}).get("joint_names") or []),
        "cameras": sorted(((manifest.get("features") or {}).get("cameras") or {}).keys()),
        "incomplete": sorted(item.name for item in (path / "incomplete").iterdir() if item.is_dir()),
        "episodes": episodes,
    }


def catalog(root: str = "") -> dict[str, Any]:
    """List valid Blacknode datasets beneath one user-selected storage root."""
    base = Path(root).expanduser().resolve() if str(root or "").strip() else default_home().resolve()
    datasets: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    if base.is_dir():
        for candidate in sorted((item for item in base.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
            if not (candidate / "dataset.json").is_file():
                continue
            try:
                datasets.append(summarize(candidate))
            except Exception as exc:  # noqa: BLE001 - invalid entries are reported without hiding valid datasets
                invalid.append({"path": str(candidate), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "kind": "blacknode.dataset-catalog",
        "schema_version": 1,
        "root": str(base),
        "datasets": datasets,
        "dataset_count": len(datasets),
        "invalid": invalid,
    }


def browse_dataset(root: str = "", dataset_id: str = "", episode_index: int = 0,
                   camera: str = "") -> dict[str, Any]:
    """Resolve a catalog selection into a complete read-only episode preview."""
    result = catalog(root)
    datasets = list(result["datasets"])
    requested = str(dataset_id or "").strip()
    selected = next((item for item in datasets if item.get("dataset_id") == requested), None)
    if selected is None and datasets:
        selected = datasets[0]
    result["selected_dataset"] = selected or {}
    result["selected_dataset_id"] = str((selected or {}).get("dataset_id") or "")
    result["selected_episode"] = {}
    if selected and selected.get("episodes"):
        available = len(selected["episodes"])
        index = min(max(0, int(episode_index)), available - 1)
        replay = episode_replay({"path": selected["path"]}, index, camera)
        result["selected_episode"] = replay
    return result


def episode_replay(dataset: dict[str, Any], episode_index: int, camera: str = "") -> dict[str, Any]:
    """Resolve one saved episode and camera video for read-only playback."""
    path = resolve_dataset_path(dataset)
    manifest = load_manifest(path)
    episodes = list(manifest.get("episodes") or [])
    index = int(episode_index)
    if index < 0 or index >= len(episodes):
        raise ValueError(f"episode_index {index} is unavailable; dataset has {len(episodes)} saved episode(s)")
    entry = dict(episodes[index])
    episode_path = (path / str(entry.get("path") or "")).resolve()
    if path not in episode_path.parents:
        raise ValueError("episode path escapes the dataset")
    info = json.loads((episode_path / "episode.json").read_text(encoding="utf-8"))
    cameras = sorted(str(name) for name in (info.get("cameras") or {}))
    selected = str(camera or "").strip() or (cameras[0] if cameras else "")
    if selected not in cameras:
        raise ValueError(f"camera {selected!r} is unavailable; choose one of: {', '.join(cameras) or 'none'}")
    video_path = (episode_path / "cameras" / f"{selected}.mp4").resolve()
    if not video_path.is_file() or episode_path not in video_path.parents:
        raise ValueError(f"saved camera video is missing: {video_path}")
    return {
        "kind": "blacknode.episode-replay",
        "schema_version": 1,
        "dataset_path": str(path),
        "episode_path": str(episode_path),
        "video_path": str(video_path),
        "data_path": str(episode_path / "data.parquet"),
        "episode_index": index,
        "camera": selected,
        "camera_info": dict((info.get("cameras") or {}).get(selected) or {}),
        "cameras": cameras,
        "frames": int(info.get("frames") or entry.get("frames") or 0),
        "fps": int(info.get("fps") or manifest.get("fps") or 0),
        "duration_seconds": float(info.get("duration_seconds") or entry.get("duration_seconds") or 0.0),
        "joint_names": list(info.get("joint_names") or []),
        "units": str(info.get("units") or ""),
        "robot": dict(info.get("robot") or {}),
        "task": str(info.get("task") or entry.get("task") or ""),
        "saved_at": str(info.get("saved_at") or entry.get("saved_at") or ""),
    }


def validate(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = load_manifest(path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "errors": [str(exc)], "warnings": []}
    expected_names = list((manifest.get("features") or {}).get("joint_names") or [])
    for expected_index, item in enumerate(manifest.get("episodes") or []):
        if int(item.get("episode_index", -1)) != expected_index:
            errors.append(f"episode manifest index {item.get('episode_index')} is not sequential at {expected_index}")
        episode_path = path / str(item.get("path") or "")
        for required in (episode_path / "episode.json", episode_path / "data.parquet"):
            if not required.exists():
                errors.append(f"missing {required}")
        if (episode_path / "episode.json").exists():
            info = json.loads((episode_path / "episode.json").read_text(encoding="utf-8"))
            if list(info.get("joint_names") or []) != expected_names:
                errors.append(f"episode {expected_index} joint order differs from dataset features")
            for camera in info.get("cameras") or {}:
                if not (episode_path / "cameras" / f"{camera}.mp4").exists():
                    errors.append(f"episode {expected_index} is missing camera video {camera}")
        if pq is not None and (episode_path / "data.parquet").exists():
            table = pq.read_table(episode_path / "data.parquet")
            if table.num_rows != int(item.get("frames") or 0):
                errors.append(f"episode {expected_index} frame count does not match parquet rows")
            timestamps = table.column("timestamp").to_pylist()
            if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
                errors.append(f"episode {expected_index} timestamps are not strictly increasing")
    incomplete = sorted(item.name for item in (path / "incomplete").iterdir() if item.is_dir())
    if incomplete:
        warnings.append("recoverable incomplete episodes: " + ", ".join(incomplete))
    return {"ok": not errors, "errors": errors, "warnings": warnings, "summary": summarize(path)}


def _feature_stats(values: list[list[float]]) -> dict[str, Any]:
    if not values:
        return {}
    columns = list(zip(*values, strict=True))
    return {
        "min": [min(column) for column in columns],
        "max": [max(column) for column in columns],
        "mean": [statistics.fmean(column) for column in columns],
        "std": [statistics.pstdev(column) for column in columns],
        "count": [len(column) for column in columns],
    }


def export_lerobot_v3(path: Path, output: Path, repo_id: str = "",
                      smoothing: str = "none", smoothing_strength: float = 1.0) -> dict[str, Any]:
    """Export a LeRobot v3-compatible tree without importing LeRobot."""
    _require_storage_dependencies()
    report = validate(path)
    if not report["ok"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest = load_manifest(path)
    episodes = list(manifest.get("episodes") or [])
    if not episodes:
        raise ValueError("dataset has no saved episodes")
    if output.exists():
        raise FileExistsError(f"export path already exists: {output}")
    output.mkdir(parents=True)
    joint_names = list(manifest["features"]["joint_names"])
    fps = int(manifest["fps"])
    camera_features = dict(manifest["features"].get("cameras") or {})
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [len(joint_names)], "names": joint_names},
        "action": {"dtype": "float32", "shape": [len(joint_names)], "names": joint_names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera, info in camera_features.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [int(info["height"]), int(info["width"]), int(info.get("channels") or 3)],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": int(info["height"]),
                "video.width": int(info["width"]),
                "video.codec": str(info.get("codec") or "mpeg4"),
                "video.pix_fmt": str(info.get("pixel_format") or "yuv420p"),
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": int(info.get("channels") or 3),
                "has_audio": bool(info.get("has_audio", False)),
            },
        }
    global_index = 0
    episode_meta_rows: list[dict[str, Any]] = []
    observation_values: list[list[float]] = []
    action_values: list[list[float]] = []
    for episode in episodes:
        ep_index = int(episode["episode_index"])
        source = path / episode["path"]
        table = pq.read_table(source / "data.parquet")
        observation = table.column("observation.state").to_pylist()
        action = table.column("action").to_pylist()
        timestamps = [float(value) for value in table.column("timestamp").to_pylist()]
        frame_count = len(timestamps)
        observation = _smooth_rows(observation, smoothing, smoothing_strength, fps)
        action = _smooth_rows(action, smoothing, smoothing_strength, fps)
        observation_values.extend(observation)
        action_values.extend(action)
        data_table = pa.table({
            "observation.state": pa.array(observation, type=pa.list_(pa.float32(), len(joint_names))),
            "action": pa.array(action, type=pa.list_(pa.float32(), len(joint_names))),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(list(range(frame_count)), type=pa.int64()),
            "episode_index": pa.array([ep_index] * frame_count, type=pa.int64()),
            "index": pa.array(list(range(global_index, global_index + frame_count)), type=pa.int64()),
            "task_index": pa.array([0] * frame_count, type=pa.int64()),
        })
        data_path = output / "data" / "chunk-000" / f"file-{ep_index:03d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(data_table, data_path, compression="snappy")
        meta_row: dict[str, Any] = {
            "episode_index": ep_index,
            "tasks": [str(manifest.get("task") or "")],
            "length": frame_count,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + frame_count,
            "data/chunk_index": 0,
            "data/file_index": ep_index,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for camera in camera_features:
            video_target = output / "videos" / f"observation.images.{camera}" / "chunk-000" / f"file-{ep_index:03d}.mp4"
            video_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "cameras" / f"{camera}.mp4", video_target)
            meta_row[f"videos/observation.images.{camera}/chunk_index"] = 0
            meta_row[f"videos/observation.images.{camera}/file_index"] = ep_index
            meta_row[f"videos/observation.images.{camera}/from_timestamp"] = 0.0
            meta_row[f"videos/observation.images.{camera}/to_timestamp"] = frame_count / fps
        episode_meta_rows.append(meta_row)
        global_index += frame_count
    meta_dir = output / "meta"
    meta_dir.mkdir(exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "features": features,
        "total_episodes": len(episodes),
        "total_frames": global_index,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": str(manifest.get("robot_type") or "blacknode-robot"),
        "splits": {"train": f"0:{len(episodes)}"},
    }
    _atomic_json(meta_dir / "info.json", info)
    _atomic_json(meta_dir / "stats.json", {
        "observation.state": _feature_stats(observation_values),
        "action": _feature_stats(action_values),
    })
    assert pd is not None
    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([str(manifest.get("task") or "")], name="task"))
    tasks.to_parquet(meta_dir / "tasks.parquet")
    episode_columns = {key: [row.get(key) for row in episode_meta_rows] for key in episode_meta_rows[0]}
    episodes_path = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(episode_columns), episodes_path, compression="snappy")
    (output / "README.md").write_text(
        f"---\ntags:\n- LeRobot\n- Blacknode\n---\n\n# {repo_id or manifest['dataset_id']}\n\nExported from a Blacknode episode dataset.\n",
        encoding="utf-8",
    )
    _atomic_json(output / "blacknode-export.json", {
        "kind": "blacknode.lerobot-export",
        "schema_version": 1,
        "source": str(path),
        "repo_id": repo_id,
        "exported_at": _now(),
        "lerobot_codebase_version": "v3.0",
        "smoothing": str(smoothing or "none").lower(),
        "smoothing_strength": float(smoothing_strength),
    })
    return {"ok": True, "path": str(output), "episodes": len(episodes), "frames": global_index,
            "repo_id": repo_id, "smoothing": str(smoothing or "none").lower()}


def _hdf5_column(table: Any, name: str, dtype: Any) -> Any | None:
    if name not in table.column_names:
        return None
    assert np is not None
    return np.asarray(table.column(name).to_pylist(), dtype=dtype)


def _write_hdf5_camera(
    group: Any,
    video_path: Path,
    *,
    dataset_name: str,
    frame_count: int,
    width: int,
    height: int,
    compression: str,
) -> None:
    assert cv2 is not None and np is not None
    options: dict[str, Any] = {"chunks": (1, height, width, 3)}
    if compression != "none":
        options["compression"] = compression
        if compression == "gzip":
            options["compression_opts"] = 4
    images = group.create_dataset(dataset_name, shape=(frame_count, height, width, 3), dtype=np.uint8, **options)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera video {video_path}")
    decoded = 0
    try:
        while decoded < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"camera video {video_path} changed shape: expected {width}x{height}, "
                    f"got {frame.shape[1]}x{frame.shape[0]}"
                )
            images[decoded] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded += 1
        extra, _ = capture.read()
    finally:
        capture.release()
    if decoded != frame_count or extra:
        actual = f"more than {frame_count}" if extra else str(decoded)
        raise ValueError(f"camera video {video_path} has {actual} frames; expected {frame_count}")


def export_hdf5(
    path: Path,
    output: Path,
    *,
    include_images: bool = True,
    compression: str = "gzip",
    smoothing: str = "none",
    smoothing_strength: float = 1.0,
) -> dict[str, Any]:
    """Export one ACT-style HDF5 file per saved Blacknode episode."""
    _require_storage_dependencies()
    if h5py is None or np is None:
        raise RuntimeError("missing HDF5 export dependencies: h5py, numpy")
    compression = str(compression or "gzip").lower()
    if compression not in {"gzip", "lzf", "none"}:
        raise ValueError("compression must be gzip, lzf, or none")
    report = validate(path)
    if not report["ok"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest = load_manifest(path)
    episodes = list(manifest.get("episodes") or [])
    if not episodes:
        raise ValueError("dataset has no saved episodes")
    if output.exists():
        raise FileExistsError(f"export path already exists: {output}")

    temp_output = output.with_name(f".{output.name}.tmp")
    if temp_output.exists():
        shutil.rmtree(temp_output)
    temp_output.mkdir(parents=True)
    joint_names = list((manifest.get("features") or {}).get("joint_names") or [])
    camera_features = dict((manifest.get("features") or {}).get("cameras") or {})
    string_dtype = h5py.string_dtype(encoding="utf-8")
    total_frames = 0
    try:
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            source = path / str(episode["path"])
            episode_info = json.loads((source / "episode.json").read_text(encoding="utf-8"))
            table = pq.read_table(source / "data.parquet")
            frame_count = table.num_rows
            file_path = temp_output / f"episode_{episode_index}.hdf5"
            with h5py.File(file_path, "w") as handle:
                handle.attrs["sim"] = False
                handle.attrs["compress"] = compression != "none"
                handle.attrs["blacknode_schema_version"] = 1
                handle.attrs["episode_index"] = episode_index
                handle.attrs["fps"] = int(manifest["fps"])
                handle.attrs["task"] = str(episode_info.get("task") or manifest.get("task") or "")
                handle.attrs["robot_type"] = str(manifest.get("robot_type") or "")
                handle.attrs["units"] = str(episode_info.get("units") or "radians")
                handle.attrs["image_color_space"] = "RGB"
                handle.attrs["smoothing"] = str(smoothing or "none").lower()
                handle.attrs["smoothing_strength"] = float(smoothing_strength)

                metadata = handle.create_group("metadata")
                metadata.create_dataset("joint_names", data=np.asarray(joint_names, dtype=object), dtype=string_dtype)
                metadata.attrs["source_dataset"] = str(path)
                metadata.attrs["source_episode"] = str(source)
                metadata.attrs["exported_at"] = _now()
                robot_metadata = dict(episode_info.get("robot") or {})
                for key, value in robot_metadata.items():
                    if value is not None:
                        metadata.attrs[key] = str(value)

                observations = handle.create_group("observations")
                fps = float(manifest["fps"])
                qpos = _smooth_rows(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32),
                                    smoothing, smoothing_strength, fps).astype(np.float32)
                leader = _smooth_rows(np.asarray(table.column("leader.state").to_pylist(), dtype=np.float32),
                                      smoothing, smoothing_strength, fps).astype(np.float32)
                action = _smooth_rows(np.asarray(table.column("action").to_pylist(), dtype=np.float32),
                                      smoothing, smoothing_strength, fps).astype(np.float32)
                observations.create_dataset("qpos", data=qpos)
                observations.create_dataset("leader", data=leader)
                handle.create_dataset("action", data=action)
                for source_name, target_name, dtype in (
                    ("timestamp", "timestamp", np.float64),
                    ("frame_index", "frame_index", np.int64),
                    ("sample_sequence", "sample_sequence", np.int64),
                    ("captured_at_ns", "captured_at_ns", np.int64),
                    ("recorded_at_ns", "recorded_at_ns", np.int64),
                ):
                    values = _hdf5_column(table, source_name, dtype)
                    if values is not None:
                        handle.create_dataset(target_name, data=values)

                if include_images:
                    images_group = observations.create_group("images")
                    camera_metadata = observations.create_group("camera_metadata")
                    for camera, info in camera_features.items():
                        _write_hdf5_camera(
                            images_group,
                            source / "cameras" / f"{camera}.mp4",
                            dataset_name=camera,
                            frame_count=frame_count,
                            width=int(info["width"]),
                            height=int(info["height"]),
                            compression=compression,
                        )
                        for suffix in ("sequence", "captured_at_ns"):
                            values = _hdf5_column(table, f"camera.{camera}.{suffix}", np.int64)
                            if values is not None:
                                camera_metadata.create_dataset(f"{camera}_{suffix}", data=values)
            total_frames += frame_count

        _atomic_json(temp_output / "blacknode-export.json", {
            "kind": "blacknode.hdf5-export",
            "schema_version": 1,
            "layout": "one-file-per-episode",
            "source": str(path),
            "episodes": len(episodes),
            "frames": total_frames,
            "include_images": bool(include_images),
            "compression": compression,
            "exported_at": _now(),
        })
        temp_output.replace(output)
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise
    return {"ok": True, "path": str(output), "episodes": len(episodes), "frames": total_frames}
