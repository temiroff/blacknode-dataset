"""Blacknode-native episode dataset nodes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Image, Int, List, Text, Video, node
from blacknode.providers.keys import api_key_for_provider

from . import runtime, storage

_CATEGORY = "Dataset"


def _dashboard(status: dict[str, Any]) -> str:
    return runtime.dashboard(status)


@node(name="DatasetCameraStreamList", component="recording", category=_CATEGORY,
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


@node(name="DatasetCreate", component="recording", category=_CATEGORY,
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


@node(name="DatasetBrowser", component="recording", category=_CATEGORY,
      description="Browse datasets and episodes, replay and trim a selected camera, and inspect synchronized robot observations, actions, leader state, timing, and artifact paths.",
      inputs={"trigger": AnyPort, "dataset": Dict(default={}), "root": Text(default=""), "dataset_id": Text(default=""),
              "episode_index": Int(default=0), "camera": Text(default=""), "refresh_key": Int(default=0)},
      outputs={"dataset": Dict, "catalog": Dict, "episode": Dict, "video": Video,
               "replay_token": Text, "stream": Dict, "episode_path": Text, "video_path": Text,
               "data_path": Text, "report": Text})
def dataset_browser(ctx: dict) -> dict:
    try:
        connected_dataset = dict(ctx.get("dataset") or {})
        connected_path = Path(str(connected_dataset.get("path") or "")).expanduser()
        root = str(ctx.get("root") or "").strip()
        if not root and connected_path.name:
            root = str(connected_path.parent)
        dataset_id = str(ctx.get("dataset_id") or "").strip() or str(connected_dataset.get("dataset_id") or "")
        catalog = storage.browse_dataset(
            root, dataset_id,
            int(ctx.get("episode_index") or 0), str(ctx.get("camera") or ""),
        )
        dataset = dict(catalog.get("selected_dataset") or {})
        episode = dict(catalog.get("selected_episode") or {})
        token = runtime.register_episode_replay(episode) if episode else ""
        video = f"/api/dataset/media/{token}" if token else ""
        catalog["replay_token"] = token
        catalog["video"] = video
        catalog["frame_url"] = f"/api/dataset/frame/{token}" if token else ""
        report = f"{catalog['dataset_count']} dataset(s) in {catalog['root']}"
        if episode:
            report += (f"; selected {dataset.get('dataset_id')} episode {episode['episode_index']} · "
                       f"{episode['camera']} · {episode['frames']} frames · {episode['duration_seconds']:.1f}s")
        stream = runtime.make_replay_stream(
            token,
            label=f"{dataset.get('dataset_id', '')} · ep{episode.get('episode_index', 0)} · {episode.get('camera', '')}",
            frames=int(episode.get("frames") or 0), fps=float(episode.get("fps") or 0),
            units=str(episode.get("units") or "radians"),
        ) if token else {}
        return {
            "dataset": dataset, "catalog": catalog, "episode": episode, "video": video,
            "replay_token": token, "stream": stream,
            "episode_path": str(episode.get("episode_path") or ""),
            "video_path": str(episode.get("video_path") or ""),
            "data_path": str(episode.get("data_path") or ""), "report": report,
        }
    except Exception as exc:  # noqa: BLE001
        return {"dataset": {}, "catalog": {}, "episode": {}, "video": "", "replay_token": "",
                "stream": {}, "episode_path": "", "video_path": "", "data_path": "",
                "report": f"dataset browser FAILED: {exc}"}


@node(name="EpisodeRecorder", component="recording", live=True, category=_CATEGORY,
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
        runtime.configure_recorder(
            run_id=run_id, dataset=dict(ctx.get("dataset") or {}),
            robot_stream=dict(ctx.get("robot_stream") or {}), camera_stream=dict(ctx.get("camera_stream") or {}),
            camera_streams=list(ctx.get("camera_streams") or []), require_armed=bool(ctx.get("require_armed", True)),
            stale_after=float(ctx.get("stale_after") or 0.5), request_timeout=float(ctx.get("request_timeout") or 1.0),
        )
        if action == "start":
            status = runtime.start_recorder(
                run_id=run_id, dataset=dict(ctx.get("dataset") or {}),
                robot_stream=dict(ctx.get("robot_stream") or {}), camera_stream=dict(ctx.get("camera_stream") or {}),
                camera_streams=list(ctx.get("camera_streams") or []), require_armed=bool(ctx.get("require_armed", True)),
                stale_after=float(ctx.get("stale_after") or 0.5), request_timeout=float(ctx.get("request_timeout") or 1.0))
        else:
            status = runtime.control_recorder(run_id, action)
            if (action == "status" and not status.get("running") and ctx.get("dataset")):
                status = runtime.recoverable_episode_status(dict(ctx.get("dataset") or {}), run_id) or status
            if (not status.get("running") and status.get("last_error") == "recorder is not running"
                    and action in {"save", "finalize", "discard"} and ctx.get("dataset")):
                status = runtime.recover_episode(dict(ctx.get("dataset") or {}), run_id, action)
        return runtime.recorder_outputs(status, dict(ctx.get("dataset") or {}))
    except Exception as exc:  # noqa: BLE001
        status = {"run_id": run_id, "running": False, "recording": False, "paused": False,
                  "frame_count": 0, "dropped_frames": 0, "last_error": str(exc)}
        return {"running": False, "recording": False, "paused": False, "episode_index": 0,
                "frame_count": 0, "dropped_frames": 0, "duration_seconds": 0.0,
                "dataset": dict(ctx.get("dataset") or {}), "status": status, "dashboard": _dashboard(status),
                "report": f"episode recorder FAILED: {exc}"}


@node(name="EpisodeDatasetSummary", component="validation", category=_CATEGORY, description="Summarize saved and recoverable episodes.",
      inputs={"trigger": AnyPort, "dataset": Dict(default={})},
      outputs={"summary": Dict, "episode_count": Int, "frame_count": Int, "report": Text})
def dataset_summary(ctx: dict) -> dict:
    try:
        summary = storage.summarize(storage.resolve_dataset_path(dict(ctx.get("dataset") or {})))
        return {"summary": summary, "episode_count": int(summary["episode_count"]),
                "frame_count": int(summary["total_frames"]),
                "report": (f"{summary['episode_count']} episode(s), {summary['total_frames']} frames · "
                           f"camera resolution(s): {summary.get('camera_shapes') or 'unknown'}")}
    except Exception as exc:  # noqa: BLE001
        return {"summary": {}, "episode_count": 0, "frame_count": 0, "report": f"summary FAILED: {exc}"}


@node(name="EpisodeReplay", component="replay", category=_CATEGORY,
      description="Replay a saved episode camera video with recorded task, timing, joint, and storage metadata. Playback never commands a robot.",
      inputs={"trigger": AnyPort, "dataset": Dict(default={}), "episode_index": Int(default=0),
              "camera": Text(default="")},
      outputs={"video": Video, "replay": Dict, "episode_path": Text, "video_path": Text,
               "data_path": Text, "report": Text})
def episode_replay(ctx: dict) -> dict:
    try:
        replay = storage.episode_replay(
            dict(ctx.get("dataset") or {}), int(ctx.get("episode_index") or 0), str(ctx.get("camera") or ""),
        )
        token = runtime.register_episode_replay(replay)
        replay["replay_token"] = token
        replay["frame_url"] = f"/api/dataset/frame/{token}"
        return {
            "video": f"/api/dataset/media/{token}",
            "replay": replay,
            "episode_path": replay["episode_path"],
            "video_path": replay["video_path"],
            "data_path": replay["data_path"],
            "report": (
                f"replay episode {replay['episode_index']} · {replay['camera']} · "
                f"{replay['frames']} frames · {replay['duration_seconds']:.1f}s · {replay['episode_path']}"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"video": "", "replay": {}, "episode_path": "", "video_path": "", "data_path": "",
                "report": f"episode replay FAILED: {exc}"}


@node(name="EpisodeDatasetValidate", component="validation", category=_CATEGORY, description="Validate manifests, Parquet rows, videos, timestamps, and feature consistency.",
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


@node(name="LeRobotV3Export", component="export", category=_CATEGORY,
      description="Export a validated dataset to LeRobot v3 Parquet/MP4 layout without importing LeRobot. Optional "
                  "before-training smoothing filters each episode's joint trajectories (zero-lag) so the policy "
                  "learns intended motion, not teleop jitter.",
      inputs={"trigger": AnyPort, "dataset": Dict(default={}), "output_path": Text(default=""), "repo_id": Text(default=""),
              "smoothing": Enum(["none", "spline", "gaussian", "savgol", "moving_average"], default="none"),
              "smoothing_strength": Float(default=1.0)},
      outputs={"ok": Bool, "export": Dict, "path": Text, "report": Text})
def lerobot_v3_export(ctx: dict) -> dict:
    try:
        source = storage.resolve_dataset_path(dict(ctx.get("dataset") or {}))
        raw_output = str(ctx.get("output_path") or "").strip()
        output = Path(raw_output).expanduser().resolve() if raw_output else source.parent / f"{source.name}-lerobot-v3"
        result = storage.export_lerobot_v3(source, output, str(ctx.get("repo_id") or "").strip(),
                                           smoothing=str(ctx.get("smoothing") or "none"),
                                           smoothing_strength=float(ctx.get("smoothing_strength") or 1.0))
        return {"ok": True, "export": result, "path": result["path"],
                "report": f"LeRobot v3 export ready: {result['path']}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "export": {}, "path": "", "report": f"LeRobot export FAILED: {exc}"}


@node(name="BlacknodeHubExport", component="publishing", category=_CATEGORY,
      description="Export a Blacknode-native Hugging Face Hub dataset with previewable Parquet frames, episode metadata, videos, and a dataset card. Existing valid exports are reused unless overwrite is enabled.",
      inputs={"trigger": AnyPort, "action": Enum(["export", "check"], default="export"),
              "dataset": Dict(default={}), "output_path": Text(default=""), "repo_id": Text(default=""),
              "include_videos": Bool(default=True), "license": Text(default=""), "overwrite": Bool(default=False)},
      outputs={"ok": Bool, "exported": Bool, "status": Text, "export": Dict, "path": Text, "report": Text},
      primary_inputs=["trigger", "action", "dataset", "output_path", "repo_id", "overwrite"],
      primary_outputs=["exported", "status", "path", "report"])
def blacknode_hub_export(ctx: dict) -> dict:
    output: Path | None = None
    try:
        source = storage.resolve_dataset_path(dict(ctx.get("dataset") or {}))
        raw_output = str(ctx.get("output_path") or "").strip()
        output = Path(raw_output).expanduser().resolve() if raw_output else source.parent / f"{source.name}-blacknode-hub"
        validation = storage.validate(source)
        if not validation["ok"]:
            raise ValueError("dataset validation failed: " + "; ".join(validation["errors"]))
        episode_count = int(validation["summary"]["episode_count"])
        if not episode_count:
            raise ValueError("dataset has no saved episodes")
        action = str(ctx.get("action") or "export").lower()
        if action == "check":
            return {
                "ok": True, "exported": False, "status": "checked_not_exported", "export": {}, "path": str(output),
                "report": (f"CHECK ONLY — no files written. {episode_count} episode(s) are ready for the Blacknode Hub format. "
                           f"Change action=export; Blacknode will create: {output}"),
            }
        if action != "export":
            raise ValueError(f"unsupported action: {action}; choose check or export")
        overwrite = bool(ctx.get("overwrite", False))
        if output.exists() and not overwrite:
            existing = _existing_export(output, "blacknode.huggingface-export")
            if existing is None:
                raise FileExistsError(f"output path exists but is not a valid Blacknode Hub export: {output}")
            return {
                "ok": True, "exported": True, "status": "exists", "export": existing, "path": str(output),
                "report": f"EXISTS — valid Blacknode Hub export left unchanged. Enable overwrite to rebuild: {output}",
            }
        result = storage.export_blacknode_hub(
            source,
            output,
            str(ctx.get("repo_id") or "").strip(),
            include_videos=bool(ctx.get("include_videos", True)),
            license_id=str(ctx.get("license") or "").strip(),
            overwrite=overwrite,
        )
        return {
            "ok": True, "exported": True, "status": "exported", "export": result, "path": result["path"],
            "report": (f"EXPORTED Blacknode Hub dataset: {result['episodes']} episode(s), {result['frames']} frame(s), "
                       f"{result['media']} video(s). Destination: {result['path']}"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False, "exported": False, "status": "failed", "export": {},
            "path": str(output) if output is not None else "",
            "report": f"Blacknode Hub export FAILED: {type(exc).__name__}: {exc}",
        }


@node(name="HDF5EpisodeExport", component="export", category=_CATEGORY,
      description="Export one ACT-style HDF5 file per saved episode and create the destination folder. Existing valid exports are reused unless overwrite is enabled.",
      inputs={"trigger": AnyPort, "action": Enum(["export", "check"], default="export"),
              "dataset": Dict(default={}), "output_path": Text(default=""),
              "include_images": Bool(default=True),
              "compression": Enum(["gzip", "lzf", "none"], default="gzip"),
              "smoothing": Enum(["none", "spline", "gaussian", "savgol", "moving_average"], default="none"),
              "smoothing_strength": Float(default=1.0), "overwrite": Bool(default=False)},
      outputs={"ok": Bool, "exported": Bool, "status": Text, "export": Dict, "path": Text, "report": Text},
      primary_inputs=["trigger", "action", "dataset", "output_path", "overwrite"], primary_outputs=["exported", "status", "path", "report"])
def hdf5_episode_export(ctx: dict) -> dict:
    output: Path | None = None
    try:
        source = storage.resolve_dataset_path(dict(ctx.get("dataset") or {}))
        raw_output = str(ctx.get("output_path") or "").strip()
        output = Path(raw_output).expanduser().resolve() if raw_output else source.parent / f"{source.name}-hdf5"
        validation = storage.validate(source)
        if not validation["ok"]:
            raise ValueError("dataset validation failed: " + "; ".join(validation["errors"]))
        if not validation["summary"]["episode_count"]:
            raise ValueError("dataset has no saved episodes")
        action = str(ctx.get("action") or "export").lower()
        if action == "check":
            return {"ok": True, "exported": False, "status": "checked_not_exported", "export": {}, "path": str(output),
                    "report": (f"CHECK ONLY — no files written. {validation['summary']['episode_count']} episode(s) are ready. "
                               f"Change action=export; Blacknode will create: {output}")}
        if action != "export":
            raise ValueError(f"unsupported action: {action}; choose check or export")
        overwrite = bool(ctx.get("overwrite", False))
        if output.exists() and not overwrite:
            existing = _existing_export(output, "blacknode.hdf5-export")
            if existing is None:
                raise FileExistsError(f"output path exists but is not a valid HDF5 episode export: {output}")
            return {
                "ok": True, "exported": True, "status": "exists", "export": existing, "path": str(output),
                "report": f"EXISTS — valid HDF5 export left unchanged. Enable overwrite to rebuild: {output}",
            }
        result = storage.export_hdf5(
            source,
            output,
            include_images=bool(ctx.get("include_images", True)),
            compression=str(ctx.get("compression") or "gzip"),
            smoothing=str(ctx.get("smoothing") or "none"),
            smoothing_strength=float(ctx.get("smoothing_strength") or 1.0),
            overwrite=overwrite,
        )
        return {"ok": True, "exported": True, "status": "exported", "export": result, "path": result["path"],
                "report": (f"EXPORTED {result['episodes']} episode file(s), {result['frames']} frame(s). "
                           f"Destination: {result['path']}")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exported": False, "status": "failed", "export": {},
                "path": str(output) if output is not None else "", "report": f"HDF5 export FAILED: {type(exc).__name__}: {exc}"}


def _existing_export(path: Path, kind: str) -> dict[str, Any] | None:
    marker = path / "blacknode-export.json"
    if not path.is_dir() or not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if str(payload.get("kind") or "") != kind:
        return None
    return {**payload, "path": str(path)}


def _huggingface_export_format(path: Path) -> str:
    if not path.is_dir() or not (path / "README.md").is_file():
        return ""
    marker = path / "blacknode-export.json"
    if marker.is_file():
        try:
            kind = str(json.loads(marker.read_text(encoding="utf-8")).get("kind") or "")
        except Exception:  # noqa: BLE001
            return ""
        if kind == "blacknode.huggingface-export" and (path / "blacknode" / "manifest.json").is_file() \
                and any((path / "data").glob("*.parquet")):
            return "blacknode-hub"
        if kind == "blacknode.lerobot-export" and (path / "meta" / "info.json").is_file():
            return "lerobot-v3"
    if (path / "meta" / "info.json").is_file():
        return "lerobot-v3"
    return ""


@node(name="HuggingFaceDatasetUpload", component="publishing", category=_CATEGORY,
      description="Check or explicitly upload a prepared Blacknode Hub or LeRobot v3 export to a Hugging Face dataset repository.",
      inputs={"trigger": AnyPort, "action": Enum(["check", "upload"], default="check"),
              "export_path": Text(default=""), "repo_id": Text(default=""), "private": Bool(default=False),
              "token": Text(default="")},
      outputs={"ok": Bool, "uploaded": Bool, "status": Text, "format": Text,
               "repo_id": Text, "url": Text, "report": Text},
      primary_inputs=["trigger", "action", "export_path", "repo_id", "private"],
      primary_outputs=["uploaded", "status", "format", "url", "report"])
def huggingface_dataset_upload(ctx: dict) -> dict:
    action = str(ctx.get("action") or "check").lower()
    path = Path(str(ctx.get("export_path") or "")).expanduser().resolve()
    repo_id = str(ctx.get("repo_id") or "").strip()
    export_format = _huggingface_export_format(path)
    if not export_format:
        return {"ok": False, "uploaded": False, "status": "failed", "format": "", "repo_id": repo_id, "url": "",
                "report": "a valid Blacknode Hub or LeRobot v3 export_path is required"}
    if not repo_id:
        return {"ok": False, "uploaded": False, "status": "failed", "format": export_format,
                "repo_id": "", "url": "", "report": "repo_id is required"}
    if action == "check":
        return {"ok": True, "uploaded": False, "status": "checked_not_uploaded", "format": export_format,
                "repo_id": repo_id, "url": f"https://huggingface.co/datasets/{repo_id}",
                "report": f"CHECK ONLY — {export_format} upload inputs are valid; choose action=upload to publish"}
    if action != "upload":
        return {"ok": False, "uploaded": False, "status": "failed", "format": export_format,
                "repo_id": repo_id, "url": "", "report": f"unsupported action: {action}; choose check or upload"}
    try:
        from huggingface_hub import HfApi

        token = api_key_for_provider(
            "Hugging Face", "HF_TOKEN", str(ctx.get("token") or "").strip(),
        ) or None
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=bool(ctx.get("private")), exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(path))
        url = f"https://huggingface.co/datasets/{repo_id}"
        return {"ok": True, "uploaded": True, "status": "uploaded", "format": export_format,
                "repo_id": repo_id, "url": url, "report": f"uploaded {export_format} dataset to {url}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "uploaded": False, "status": "failed", "format": export_format,
                "repo_id": repo_id, "url": "", "report": f"Hugging Face upload FAILED: {type(exc).__name__}: {exc}"}
