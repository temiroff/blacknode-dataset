"""Lazy, format-neutral views over Blacknode and LeRobot episode datasets."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - surfaced when an adapter is opened
    pq = None


@dataclass(frozen=True)
class RobotSpec:
    robot_type: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    state_units: str = ""
    action_units: str = ""
    identity: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    source_format: str
    source_uri: str
    source_revision: str
    task: str
    fps: float
    episode_count: int
    camera_names: tuple[str, ...]
    schema_version: int = 1


@dataclass(frozen=True)
class DatasetDescriptor:
    metadata: DatasetMetadata
    robot_spec: RobotSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "blacknode.dataset-source",
            "schema_version": 1,
            "metadata": asdict(self.metadata),
            "robot_spec": asdict(self.robot_spec),
        }


@dataclass(frozen=True)
class EpisodeDescriptor:
    episode_id: str
    episode_index: int
    task: str
    sample_count: int
    duration_seconds: float
    camera_names: tuple[str, ...]


@dataclass(frozen=True)
class CameraFrameReference:
    camera: str
    video_path: str
    timestamp: float
    captured_at_ns: int = 0
    sequence: int = 0


@dataclass(frozen=True)
class EpisodeSample:
    episode_index: int
    frame_index: int
    timestamp: float
    recorded_at_ns: int
    source_captured_at_ns: int
    source_sequence: int
    task: str
    observation: tuple[float, ...]
    action: tuple[float, ...]
    cameras: dict[str, CameraFrameReference]


class EpisodeReader(Protocol):
    descriptor: EpisodeDescriptor

    def samples(self) -> Iterator[EpisodeSample]: ...


class DatasetAdapter(Protocol):
    descriptor: DatasetDescriptor

    def episode_ids(self) -> list[str]: ...
    def open_episode(self, episode_id: str) -> EpisodeReader: ...


class BlacknodeDataset:
    """The model-independent, lazy Blacknode robotics dataset boundary."""

    def __init__(self, adapter: DatasetAdapter) -> None:
        self._adapter = adapter
        self.descriptor = adapter.descriptor
        self.robot_spec = adapter.descriptor.robot_spec
        self.metadata = adapter.descriptor.metadata

    def episode_ids(self) -> list[str]:
        return self._adapter.episode_ids()

    def open_episode(self, episode_id: str) -> EpisodeReader:
        return self._adapter.open_episode(episode_id)


def _require_parquet() -> None:
    if pq is None:
        raise RuntimeError("pyarrow is required to open an episode dataset")


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("dataset path escapes its root")
    return candidate


def _column(table: Any, name: str, default: Any) -> list[Any]:
    if name not in table.column_names:
        return [default for _ in range(table.num_rows)]
    return table.column(name).to_pylist()


class _TableEpisodeReader:
    def __init__(
        self,
        *,
        descriptor: EpisodeDescriptor,
        table: Any,
        camera_paths: dict[str, Path],
    ) -> None:
        self.descriptor = descriptor
        self._table = table
        self._camera_paths = camera_paths

    def samples(self) -> Iterator[EpisodeSample]:
        table = self._table
        count = table.num_rows
        timestamps = _column(table, "timestamp", 0.0)
        frame_indexes = _column(table, "frame_index", 0)
        observations = _column(table, "observation.state", [])
        actions = _column(table, "action", [])
        recorded = _column(table, "blacknode.recorded_at_ns", None)
        if all(value is None for value in recorded):
            recorded = _column(table, "recorded_at_ns", 0)
        captured = _column(table, "blacknode.captured_at_ns", None)
        if all(value is None for value in captured):
            captured = _column(table, "captured_at_ns", 0)
        sequences = _column(table, "blacknode.sample_sequence", None)
        if all(value is None for value in sequences):
            sequences = _column(table, "sample_sequence", 0)
        tasks = _column(table, "task", self.descriptor.task)
        camera_metadata: dict[str, tuple[list[Any], list[Any]]] = {}
        for camera in self._camera_paths:
            camera_captured = _column(
                table, f"blacknode.camera.{camera}.captured_at_ns", None
            )
            if all(value is None for value in camera_captured):
                camera_captured = _column(table, f"camera.{camera}.captured_at_ns", 0)
            camera_sequences = _column(
                table, f"blacknode.camera.{camera}.sequence", None
            )
            if all(value is None for value in camera_sequences):
                camera_sequences = _column(table, f"camera.{camera}.sequence", 0)
            camera_metadata[camera] = (camera_captured, camera_sequences)
        for position in range(count):
            timestamp = float(timestamps[position] or 0.0)
            cameras: dict[str, CameraFrameReference] = {}
            for camera, video_path in self._camera_paths.items():
                camera_captured, camera_sequences = camera_metadata[camera]
                cameras[camera] = CameraFrameReference(
                    camera=camera,
                    video_path=str(video_path),
                    timestamp=timestamp,
                    captured_at_ns=int(camera_captured[position] or 0),
                    sequence=int(camera_sequences[position] or 0),
                )
            yield EpisodeSample(
                episode_index=self.descriptor.episode_index,
                frame_index=int(frame_indexes[position] or position),
                timestamp=timestamp,
                recorded_at_ns=int(recorded[position] or 0),
                source_captured_at_ns=int(captured[position] or 0),
                source_sequence=int(sequences[position] or 0),
                task=str(tasks[position] or self.descriptor.task),
                observation=tuple(float(value) for value in observations[position]),
                action=tuple(float(value) for value in actions[position]),
                cameras=cameras,
            )


class NativeDatasetAdapter:
    def __init__(self, path: str | Path) -> None:
        _require_parquet()
        self.path = Path(path).expanduser().resolve()
        manifest_path = self.path / "dataset.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "blacknode.episode-dataset" or int(
            manifest.get("schema_version") or 0
        ) != 1:
            raise ValueError(f"unsupported Blacknode dataset manifest: {manifest_path}")
        self._manifest = manifest
        self._episodes = {
            str(item.get("episode_id") or f"episode-{int(item['episode_index']):06d}"): dict(item)
            for item in manifest.get("episodes") or []
        }
        features = dict(manifest.get("features") or {})
        names = tuple(str(name) for name in features.get("joint_names") or [])
        cameras = tuple(sorted(str(name) for name in (features.get("cameras") or {})))
        self.descriptor = DatasetDescriptor(
            metadata=DatasetMetadata(
                dataset_id=str(manifest.get("dataset_id") or self.path.name),
                source_format="blacknode-native",
                source_uri=str(self.path),
                source_revision=str(manifest.get("updated_at") or ""),
                task=str(manifest.get("task") or ""),
                fps=float(manifest.get("fps") or 0),
                episode_count=len(self._episodes),
                camera_names=cameras,
            ),
            robot_spec=RobotSpec(
                robot_type=str(manifest.get("robot_type") or ""),
                state_names=names,
                action_names=names,
                state_units=str(features.get("units") or ""),
                action_units=str(features.get("units") or ""),
                identity=dict(manifest.get("metadata") or {}),
            ),
        )

    def episode_ids(self) -> list[str]:
        return list(self._episodes)

    def open_episode(self, episode_id: str) -> EpisodeReader:
        item = self._episodes.get(str(episode_id))
        if item is None:
            raise KeyError(f"unknown episode: {episode_id}")
        episode_path = _safe_child(self.path, str(item.get("path") or ""))
        info = json.loads((episode_path / "episode.json").read_text(encoding="utf-8"))
        table = pq.read_table(episode_path / "data.parquet")
        cameras = tuple(sorted(str(name) for name in (info.get("cameras") or {})))
        return _TableEpisodeReader(
            descriptor=EpisodeDescriptor(
                episode_id=str(info.get("episode_id") or episode_id),
                episode_index=int(info.get("episode_index") or 0),
                task=str(info.get("task") or self.descriptor.metadata.task),
                sample_count=table.num_rows,
                duration_seconds=float(info.get("duration_seconds") or 0.0),
                camera_names=cameras,
            ),
            table=table,
            camera_paths={camera: episode_path / "cameras" / f"{camera}.mp4" for camera in cameras},
        )

    def open(self) -> BlacknodeDataset:
        return BlacknodeDataset(self)


class LeRobotDatasetAdapter:
    """Open the supported LeRobot v3 Parquet/MP4 layout without LeRobot imports."""

    def __init__(self, path: str | Path, *, revision: str = "") -> None:
        _require_parquet()
        self.path = Path(path).expanduser().resolve()
        info_path = self.path / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if str(info.get("codebase_version") or "") != "v3.0":
            raise ValueError(f"unsupported LeRobot dataset version in {info_path}")
        self._info = info
        self._episode_rows = self._read_episode_rows()
        features = dict(info.get("features") or {})
        state = dict(features.get("observation.state") or {})
        action = dict(features.get("action") or {})
        extension = dict(info.get("blacknode") or {})
        cameras = tuple(
            sorted(
                name.removeprefix("observation.images.")
                for name, value in features.items()
                if isinstance(value, dict) and value.get("dtype") == "video"
            )
        )
        export = {}
        export_path = self.path / "blacknode-export.json"
        if export_path.is_file():
            export = json.loads(export_path.read_text(encoding="utf-8"))
        dataset_id = str(export.get("repo_id") or self.path.name).split("/")[-1]
        task = self._task_for_index(0)
        self.descriptor = DatasetDescriptor(
            metadata=DatasetMetadata(
                dataset_id=dataset_id,
                source_format="lerobot-v3",
                source_uri=str(self.path),
                source_revision=str(revision or export.get("source_revision") or ""),
                task=task,
                fps=float(info.get("fps") or 0),
                episode_count=len(self._episode_rows),
                camera_names=cameras,
            ),
            robot_spec=RobotSpec(
                robot_type=str(info.get("robot_type") or ""),
                state_names=tuple(str(name) for name in state.get("names") or []),
                action_names=tuple(str(name) for name in action.get("names") or []),
                state_units=str(extension.get("state_units") or ""),
                action_units=str(extension.get("action_units") or ""),
                identity=dict(extension.get("robot_identity") or {}),
            ),
        )

    def _read_episode_rows(self) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        episodes_root = self.path / "meta" / "episodes"
        for parquet_path in sorted(episodes_root.rglob("*.parquet")):
            for row in pq.read_table(parquet_path).to_pylist():
                index = int(row.get("episode_index") or 0)
                rows[f"episode-{index:06d}"] = dict(row)
        if not rows and int(self._info.get("total_episodes") or 0):
            raise ValueError("LeRobot dataset has no episode metadata rows")
        return dict(sorted(rows.items(), key=lambda item: int(item[1].get("episode_index") or 0)))

    def _task_for_index(self, task_index: int) -> str:
        tasks_path = self.path / "meta" / "tasks.parquet"
        if not tasks_path.is_file():
            return ""
        table = pq.read_table(tasks_path)
        rows = table.to_pylist()
        for row in rows:
            if int(row.get("task_index") or 0) == int(task_index):
                return str(row.get("task") or "")
        # Pandas index exports may restore the task string as the first column.
        if rows:
            return str(rows[min(max(task_index, 0), len(rows) - 1)].get("task") or "")
        return ""

    def episode_ids(self) -> list[str]:
        return list(self._episode_rows)

    def open_episode(self, episode_id: str) -> EpisodeReader:
        row = self._episode_rows.get(str(episode_id))
        if row is None:
            raise KeyError(f"unknown episode: {episode_id}")
        episode_index = int(row.get("episode_index") or 0)
        data_pattern = str(self._info.get("data_path") or "")
        data_path = data_pattern.format(
            chunk_index=int(row.get("data/chunk_index") or 0),
            file_index=int(row.get("data/file_index") or 0),
        )
        table = pq.read_table(_safe_child(self.path, data_path))
        if "episode_index" in table.column_names:
            indices = table.column("episode_index").to_pylist()
            positions = [position for position, value in enumerate(indices) if int(value) == episode_index]
            if len(positions) != table.num_rows:
                table = table.take(positions)
        task_indexes = _column(table, "task_index", 0)
        task = self._task_for_index(int(task_indexes[0] or 0)) if table.num_rows else ""
        camera_paths: dict[str, Path] = {}
        video_pattern = str(self._info.get("video_path") or "")
        for camera in self.descriptor.metadata.camera_names:
            video_key = f"observation.images.{camera}"
            relative = video_pattern.format(
                video_key=video_key,
                chunk_index=int(row.get(f"videos/{video_key}/chunk_index") or 0),
                file_index=int(row.get(f"videos/{video_key}/file_index") or 0),
            )
            camera_paths[camera] = _safe_child(self.path, relative)
        return _TableEpisodeReader(
            descriptor=EpisodeDescriptor(
                episode_id=str(episode_id),
                episode_index=episode_index,
                task=task,
                sample_count=table.num_rows,
                duration_seconds=(table.num_rows / self.descriptor.metadata.fps)
                if self.descriptor.metadata.fps
                else 0.0,
                camera_names=self.descriptor.metadata.camera_names,
            ),
            table=table,
            camera_paths=camera_paths,
        )

    def open(self) -> BlacknodeDataset:
        return BlacknodeDataset(self)


def lerobot_source_descriptor(uri: str, revision: str = "") -> dict[str, Any]:
    """Create a portable local or Hugging Face LeRobot dataset descriptor."""
    raw = str(uri or "").strip()
    if not raw:
        raise ValueError("LeRobot dataset URI is required")
    local = Path(raw).expanduser()
    if local.exists():
        dataset = LeRobotDatasetAdapter(local, revision=revision).open()
        value = dataset.descriptor.to_dict()
        value.update({"uri": str(local.resolve()), "revision": str(revision or ""), "local": True})
        return value
    repo_id = raw.removeprefix("hf://")
    if repo_id.count("/") != 1 or any(part in {"", ".", ".."} for part in repo_id.split("/")):
        raise ValueError("remote LeRobot URI must be hf://owner/dataset")
    pinned = str(revision or "").strip()
    if not pinned:
        raise ValueError("remote LeRobot datasets require an immutable revision")
    return {
        "kind": "blacknode.dataset-source",
        "schema_version": 1,
        "uri": f"hf://{repo_id}",
        "revision": pinned,
        "local": False,
        "metadata": {
            "dataset_id": repo_id.split("/", 1)[1],
            "source_format": "lerobot",
            "source_uri": f"hf://{repo_id}",
            "source_revision": pinned,
        },
    }
