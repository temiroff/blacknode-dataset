"""Blacknode-native episode dataset nodes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Image, Int, List, Text, node

from . import runtime, storage

_CATEGORY = "Dataset"


def _dashboard(status: dict[str, Any]) -> str:
    return runtime.dashboard(status)


@node(name="DatasetCameraStreamList", category=_CATEGORY,
      description="Collect any number of frame-stream handles through dynamic camera_N sockets and deduplicate them by stream ID.",
      inputs={"trigger": AnyPort},
      outputs={"camera_streams": List, "camera_count": Int, "report": Text},
      variadic_input=Dict, variadic_prefix="camera")
def dataset_camera_stream_list(ctx: dict) -> dict:
    candidates = [item for item in list(ctx.get("camera_streams") or []) if isinstance(item, dict)]
    current = dict(ctx.get("camera_stream") or {})
    if current:
        candidates.append(current)
    candidates.extend(
        value for name, value in sorted(
            ((name, value) for name, value in ctx.items() if name.startswith("camera_") and name[7:].isdigit()),
            key=lambda item: int(item[0][7:]),
        )
        if isinstance(value, dict)
    )
    streams: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    invalid = 0
    for candidate in candidates:
        if candidate.get("kind") != "blacknode.frame-stream":
            invalid += 1
            continue
        stream_id = str(candidate.get("stream_id") or "").strip()
        if not stream_id or not str(candidate.get("snapshot_url") or "").strip():
            invalid += 1
            continue
        normalized = dict(candidate)
        normalized["stream_id"] = stream_id
        if stream_id in positions:
            streams[positions[stream_id]] = normalized
        else:
            positions[stream_id] = len(streams)
            streams.append(normalized)
    report = f"{len(streams)} camera stream(s) ready"
    if invalid:
        report += f"; ignored {invalid} invalid handle(s)"
    return {"camera_streams": streams, "camera_count": len(streams), "report": report}


@node(name="DatasetCreate", category=_CATEGORY,
      description="Create or reopen a Blacknode-native episode dataset.",
      inputs={"trigger": AnyPort, "dataset_id": Text(default="teleoperation-demo"), "root": Text(default=""),
              "task": Text(default="Teleoperate the robot"), "fps": Int(default=30),
              "robot_type": Text(default=""), "metadata": Dict(default={})},
      outputs={"dataset": Dict, "path": Text, "summary": Dict, "report": Text})
def dataset_create(ctx: dict) -> dict:
    try:
        dataset = storage.create_dataset(str(ctx.get("dataset_id") or ""), root=str(ctx.get("root") or ""),
                                         task=str(ctx.get("task") or ""), fps=max(1, int(ctx.get("fps") or 30)),
                                         robot_type=str(ctx.get("robot_type") or ""),
                                         metadata=dict(ctx.get("metadata") or {}))
        summary = storage.summarize(storage.resolve_dataset_path(dataset))
        return {"dataset": dataset, "path": dataset["path"], "summary": summary,
                "report": f"dataset ready: {dataset['path']}"}
    except Exception as exc:  # noqa: BLE001
        return {"dataset": {}, "path": "", "summary": {}, "report": f"dataset create FAILED: {exc}"}


@node(name="EpisodeRecorder", live=True, category=_CATEGORY,
      description="Record synchronized teleoperation samples and camera frames into a recoverable episode journal.",
      inputs={"trigger": AnyPort, "action": Enum(["status", "start", "pause", "resume", "save", "finalize", "stop", "discard"], default="status"),
              "run_id": Text(default="episode_recorder"), "dataset": Dict(default={}),
              "robot_stream": Dict(default={}), "camera_stream": Dict(default={}), "camera_streams": List(default=[]),
              "require_armed": Bool(default=True), "stale_after": Float(default=0.5),
              "request_timeout": Float(default=1.0)},
      outputs={"running": Bool, "recording": Bool, "paused": Bool, "episode_index": Int,
               "frame_count": Int, "dropped_frames": Int, "duration_seconds": Float,
               "dataset": Dict, "status": Dict, "dashboard": Image, "report": Text})
def episode_recorder(ctx: dict) -> dict:
    run_id = str(ctx.get("run_id") or "episode_recorder").strip() or "episode_recorder"
    action = str(ctx.get("action") or "status").strip().lower()
    try:
        if action == "start":
            status = runtime.start_recorder(
                run_id=run_id, dataset=dict(ctx.get("dataset") or {}),
                robot_stream=dict(ctx.get("robot_stream") or {}), camera_stream=dict(ctx.get("camera_stream") or {}),
                camera_streams=list(ctx.get("camera_streams") or []), require_armed=bool(ctx.get("require_armed", True)),
                stale_after=float(ctx.get("stale_after") or 0.5), request_timeout=float(ctx.get("request_timeout") or 1.0))
        else:
            status = runtime.control_recorder(run_id, action)
            if (not status.get("running") and status.get("last_error") == "recorder is not running"
                    and action in {"save", "finalize", "discard"} and ctx.get("dataset")):
                status = runtime.recover_episode(dict(ctx.get("dataset") or {}), run_id, action)
        dataset = dict(ctx.get("dataset") or {})
        state = "recording" if status.get("recording") else "paused" if status.get("paused") else "stopped"
        return {"running": bool(status.get("running")), "recording": bool(status.get("recording")),
                "paused": bool(status.get("paused")), "episode_index": int(status.get("episode_index") or 0),
                "frame_count": int(status.get("frame_count") or 0),
                "dropped_frames": int(status.get("dropped_frames") or 0),
                "duration_seconds": float(status.get("duration_seconds") or 0.0), "dataset": dataset,
                "status": status, "dashboard": _dashboard(status),
                "report": f"episode recorder {state}: {status.get('frame_count', 0)} frames"
                          + (f"; {status['last_error']}" if status.get("last_error") else "")}
    except Exception as exc:  # noqa: BLE001
        status = {"run_id": run_id, "running": False, "recording": False, "paused": False,
                  "frame_count": 0, "dropped_frames": 0, "last_error": str(exc)}
        return {"running": False, "recording": False, "paused": False, "episode_index": 0,
                "frame_count": 0, "dropped_frames": 0, "duration_seconds": 0.0,
                "dataset": dict(ctx.get("dataset") or {}), "status": status, "dashboard": _dashboard(status),
                "report": f"episode recorder FAILED: {exc}"}


@node(name="EpisodeDatasetSummary", category=_CATEGORY, description="Summarize saved and recoverable episodes.",
      inputs={"trigger": AnyPort, "dataset": Dict(default={})},
      outputs={"summary": Dict, "episode_count": Int, "frame_count": Int, "report": Text})
def dataset_summary(ctx: dict) -> dict:
    try:
        summary = storage.summarize(storage.resolve_dataset_path(dict(ctx.get("dataset") or {})))
        return {"summary": summary, "episode_count": int(summary["episode_count"]),
                "frame_count": int(summary["total_frames"]),
                "report": f"{summary['episode_count']} episode(s), {summary['total_frames']} frames"}
    except Exception as exc:  # noqa: BLE001
        return {"summary": {}, "episode_count": 0, "frame_count": 0, "report": f"summary FAILED: {exc}"}


@node(name="EpisodeDatasetValidate", category=_CATEGORY, description="Validate manifests, Parquet rows, videos, timestamps, and feature consistency.",
      inputs={"trigger": AnyPort, "dataset": Dict(default={})},
      outputs={"ok": Bool, "validation": Dict, "report": Text})
def dataset_validate(ctx: dict) -> dict:
    try:
        result = storage.validate(storage.resolve_dataset_path(dict(ctx.get("dataset") or {})))
        messages = list(result.get("errors") or []) + list(result.get("warnings") or [])
        return {"ok": bool(result["ok"]), "validation": result,
                "report": ("dataset valid" if result["ok"] else "dataset invalid") + (": " + "; ".join(messages) if messages else "")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "validation": {"ok": False, "errors": [str(exc)]}, "report": f"validation FAILED: {exc}"}


@node(name="LeRobotV3Export", category=_CATEGORY,
      description="Export a validated dataset to LeRobot v3 Parquet/MP4 layout without importing LeRobot.",
      inputs={"trigger": AnyPort, "dataset": Dict(default={}), "output_path": Text(default=""), "repo_id": Text(default="")},
      outputs={"ok": Bool, "export": Dict, "path": Text, "report": Text})
def lerobot_v3_export(ctx: dict) -> dict:
    try:
        source = storage.resolve_dataset_path(dict(ctx.get("dataset") or {}))
        raw_output = str(ctx.get("output_path") or "").strip()
        output = Path(raw_output).expanduser().resolve() if raw_output else source.parent / f"{source.name}-lerobot-v3"
        result = storage.export_lerobot_v3(source, output, str(ctx.get("repo_id") or "").strip())
        return {"ok": True, "export": result, "path": result["path"],
                "report": f"LeRobot v3 export ready: {result['path']}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "export": {}, "path": "", "report": f"LeRobot export FAILED: {exc}"}


@node(name="HDF5EpisodeExport", category=_CATEGORY,
      description="Check or export one ACT-style HDF5 file per saved episode, with robot state, actions, timing, and RGB cameras.",
      inputs={"trigger": AnyPort, "action": Enum(["check", "export"], default="check"),
              "dataset": Dict(default={}), "output_path": Text(default=""),
              "include_images": Bool(default=True),
              "compression": Enum(["gzip", "lzf", "none"], default="gzip")},
      outputs={"ok": Bool, "exported": Bool, "export": Dict, "path": Text, "report": Text})
def hdf5_episode_export(ctx: dict) -> dict:
    try:
        source = storage.resolve_dataset_path(dict(ctx.get("dataset") or {}))
        raw_output = str(ctx.get("output_path") or "").strip()
        output = Path(raw_output).expanduser().resolve() if raw_output else source.parent / f"{source.name}-hdf5"
        validation = storage.validate(source)
        if not validation["ok"]:
            raise ValueError("dataset validation failed: " + "; ".join(validation["errors"]))
        if not validation["summary"]["episode_count"]:
            raise ValueError("dataset has no saved episodes")
        if str(ctx.get("action") or "check").lower() == "check":
            return {"ok": True, "exported": False, "export": {}, "path": str(output),
                    "report": f"HDF5 export ready for {validation['summary']['episode_count']} episode(s); choose action=export"}
        result = storage.export_hdf5(
            source,
            output,
            include_images=bool(ctx.get("include_images", True)),
            compression=str(ctx.get("compression") or "gzip"),
        )
        return {"ok": True, "exported": True, "export": result, "path": result["path"],
                "report": f"HDF5 episode export ready: {result['path']}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exported": False, "export": {}, "path": "", "report": f"HDF5 export FAILED: {exc}"}


@node(name="HuggingFaceDatasetUpload", category=_CATEGORY,
      description="Explicitly upload a prepared LeRobot export folder to a Hugging Face dataset repository.",
      inputs={"trigger": AnyPort, "action": Enum(["check", "upload"], default="check"),
              "export_path": Text(default=""), "repo_id": Text(default=""), "private": Bool(default=False),
              "token": Text(default="")},
      outputs={"ok": Bool, "uploaded": Bool, "repo_id": Text, "url": Text, "report": Text})
def huggingface_dataset_upload(ctx: dict) -> dict:
    action = str(ctx.get("action") or "check").lower()
    path = Path(str(ctx.get("export_path") or "")).expanduser().resolve()
    repo_id = str(ctx.get("repo_id") or "").strip()
    if not path.is_dir() or not (path / "meta" / "info.json").exists():
        return {"ok": False, "uploaded": False, "repo_id": repo_id, "url": "", "report": "a valid LeRobot export_path is required"}
    if not repo_id:
        return {"ok": False, "uploaded": False, "repo_id": "", "url": "", "report": "repo_id is required"}
    if action == "check":
        return {"ok": True, "uploaded": False, "repo_id": repo_id, "url": f"https://huggingface.co/datasets/{repo_id}",
                "report": "upload inputs valid; choose action=upload to publish"}
    try:
        from huggingface_hub import HfApi

        token = str(ctx.get("token") or "").strip() or os.getenv("HF_TOKEN") or None
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=bool(ctx.get("private")), exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(path))
        url = f"https://huggingface.co/datasets/{repo_id}"
        return {"ok": True, "uploaded": True, "repo_id": repo_id, "url": url, "report": f"uploaded dataset to {url}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "uploaded": False, "repo_id": repo_id, "url": "", "report": f"Hugging Face upload FAILED: {exc}"}
