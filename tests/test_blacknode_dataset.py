"""blacknode-dataset storage and node contracts."""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

try:
    import h5py
except ImportError:
    h5py = None

import blacknode  # noqa: F401 - triggers package discovery
from blacknode.node import _NODE_REGISTRY
from blacknode.packages import _import_nodes_module, _tag_new_package_nodes
from blacknode.pkg.blacknode_dataset import storage
from blacknode.pkg.blacknode_dataset import runtime
from blacknode.workflow import validate_workflow

# teleoperation-episode-recording.json uses ROS2LeaderFollower, which lives in
# blacknode-skills' disabled-by-default follow-person/ros2 adapter. Tagging
# matches blacknode-skills' own test setup exactly so registration is correct
# regardless of which test file's collection imports these modules first.
_FOLLOW_PERSON_ROOT = Path(__file__).resolve().parents[2] / "blacknode-skills" / "components" / "follow-person"
_FOLLOW_PERSON_ADAPTER_NODES = _FOLLOW_PERSON_ROOT / "adapters" / "ros2" / "nodes"
_before_follow_person = dict(_NODE_REGISTRY)
_import_nodes_module("blacknode.pkg.blacknode_skills.follow_person", _FOLLOW_PERSON_ROOT / "nodes")
_import_nodes_module("blacknode.pkg.blacknode_skills.follow_person.adapters.ros2", _FOLLOW_PERSON_ADAPTER_NODES)
_tag_new_package_nodes(_before_follow_person, "blacknode-skills", _FOLLOW_PERSON_ADAPTER_NODES, "follow-person", "ros2")


EXPECTED = {
    "DatasetCameraStreamList", "DatasetCreate", "DatasetBrowser", "EpisodeRecorder", "EpisodeDatasetSummary", "EpisodeDatasetValidate",
    "EpisodeReplay", "LeRobotV3Export", "BlacknodeHubExport", "HDF5EpisodeExport", "HuggingFaceDatasetUpload",
}


def _jpeg(value: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((24, 32, 3), value, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _sample(sequence: int) -> dict:
    return {
        "kind": "blacknode.teleoperation-sample", "schema_version": 1, "sequence": sequence,
        "captured_at_ns": sequence, "joint_names": ["shoulder", "gripper"],
        "leader": {"shoulder": 0.1, "gripper": 0.2},
        "observation": {"shoulder": 0.0, "gripper": 0.1},
        "action": {"shoulder": 0.1, "gripper": 0.2}, "units": "radians",
        "armed": True, "live": True,
    }


def test_nodes_registered():
    for name in EXPECTED:
        assert name in _NODE_REGISTRY
        assert _NODE_REGISTRY[name]._bn_package == "blacknode-dataset"
        assert _NODE_REGISTRY[name]._bn_category == "Dataset"
    for name in ("HDF5EpisodeExport", "BlacknodeHubExport"):
        assert _NODE_REGISTRY[name]._bn_input_defaults["action"] == "export"
        assert _NODE_REGISTRY[name]._bn_input_choices["action"][0] == "export"


def test_episode_recorder_dashboard_is_a_renderable_svg_image():
    result = _NODE_REGISTRY["EpisodeRecorder"]({
        "action": "status",
        "run_id": "dashboard-contract-test",
    })

    prefix = "data:image/svg+xml;base64,"
    assert result["dashboard"].startswith(prefix)
    svg = base64.b64decode(result["dashboard"][len(prefix):]).decode("utf-8")
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "STOPPED" in svg
    assert "Waiting for first camera frame" in svg
    assert "SOURCE AGE" in svg


def test_episode_recorder_reports_exact_stale_robot_age(tmp_path: Path):
    recorder = runtime.EpisodeRecorder(
        run_id="stale-contract-test", dataset_path=tmp_path, robot_stream={}, cameras=[],
        require_armed=True, stale_after=0.5, request_timeout=0.5,
        work_path=tmp_path, episode_index=0, fps=30, task="test",
    )
    sample = {**_sample(1), "captured_at_ns": time.time_ns() - 2_000_000_000}

    with pytest.raises(ValueError, match=r"robot sample is stale \(2\.00s > 0\.50s\)"):
        recorder._validate_robot(sample, time.time_ns())


def test_templates_validate():
    templates = Path(__file__).resolve().parents[1] / "templates"
    for path in templates.glob("*.json"):
        result = validate_workflow(json.loads(path.read_text(encoding="utf-8")))
        assert result.ok, f"{path.name}: {result.errors}"


def test_camera_stream_list_is_chainable_and_deduplicates():
    first = {"kind": "blacknode.frame-stream", "stream_id": "front", "snapshot_url": "http://front/frame.jpg"}
    wrist = {"kind": "blacknode.frame-stream", "stream_id": "wrist", "snapshot_url": "http://wrist/frame.jpg"}
    result = _NODE_REGISTRY["DatasetCameraStreamList"]({"camera_streams": [first], "camera_stream": wrist})
    assert [item["stream_id"] for item in result["camera_streams"]] == ["front", "wrist"]
    replaced = _NODE_REGISTRY["DatasetCameraStreamList"]({
        "camera_streams": result["camera_streams"],
        "camera_stream": {**first, "snapshot_url": "http://front/new.jpg"},
    })
    assert replaced["camera_count"] == 2
    assert replaced["camera_streams"][0]["snapshot_url"] == "http://front/new.jpg"


def test_native_save_validate_and_lerobot_v3_export(tmp_path: Path):
    dataset = storage.create_dataset("pick-cube", root=str(tmp_path), task="Pick the cube", fps=10,
                                     robot_type="so_arm101")
    path = storage.resolve_dataset_path(dataset)
    work, episode_index, _ = storage.begin_episode(path, "run-one")
    assert episode_index == 0
    for index in range(3):
        storage.append_frame(work, {
            "frame_index": index, "timestamp": index / 10.0, "recorded_at_ns": index,
            "task": "Pick the cube", "robot": _sample(index),
            "cameras": {"wrist": {"sequence": index, "captured_at_ns": index}},
        }, {"wrist": _jpeg(index * 40)})
    saved = storage.save_episode(path, "run-one")
    assert saved["frames"] == 3
    assert saved["episode_id"].startswith("episode-")
    assert saved["run_id"] == "run-one"
    assert saved["started_at"]
    assert saved["completed_at"] == saved["saved_at"]
    persisted_manifest = storage.load_manifest(path)
    assert persisted_manifest["episodes"][0]["episode_id"] == saved["episode_id"]
    assert storage.validate(path)["ok"]
    replay = _NODE_REGISTRY["EpisodeReplay"]({"dataset": dataset, "episode_index": 0, "camera": "wrist"})
    assert replay["video"].startswith("/api/dataset/media/")
    assert replay["replay"]["frames"] == 3
    assert replay["episode_path"].endswith("episode-000000")
    assert Path(replay["video_path"]).is_file()
    assert Path(replay["data_path"]).is_file()
    assert runtime.replay_media_path(replay["video"].rsplit("/", 1)[-1]) == Path(replay["video_path"])
    token = replay["replay"]["replay_token"]
    frame = runtime.replay_frame(token, 2)
    assert frame["frame_index"] == 2
    assert frame["observation"] == pytest.approx({"shoulder": 0.0, "gripper": 0.1})
    assert frame["action"] == pytest.approx({"shoulder": 0.1, "gripper": 0.2})
    assert frame["leader"] == pytest.approx({"shoulder": 0.1, "gripper": 0.2})
    browser = _NODE_REGISTRY["DatasetBrowser"]({
        "root": str(tmp_path), "dataset_id": "pick-cube", "episode_index": 0, "camera": "wrist",
    })
    assert browser["catalog"]["dataset_count"] == 1
    assert browser["dataset"]["dataset_id"] == "pick-cube"
    assert browser["video"].startswith("/api/dataset/media/")
    assert browser["catalog"]["frame_url"].startswith("/api/dataset/frame/")
    connected_browser = _NODE_REGISTRY["DatasetBrowser"]({
        "dataset": dataset, "root": "", "dataset_id": "", "episode_index": 0, "camera": "wrist",
    })
    assert connected_browser["catalog"]["root"] == str(tmp_path.resolve())
    assert connected_browser["dataset"]["dataset_id"] == "pick-cube"
    assert pq.read_table(path / "episodes/episode-000000/data.parquet").num_rows == 3
    native_table = pq.read_table(path / "episodes/episode-000000/data.parquet")
    assert native_table.column("recorded_at_ns").to_pylist() == [0, 1, 2]
    assert native_table.column("camera.wrist.sequence").to_pylist() == [0, 1, 2]
    assert native_table.column("camera.wrist.captured_at_ns").to_pylist() == [0, 1, 2]

    output = tmp_path / "lerobot"
    exported = storage.export_lerobot_v3(path, output, "owner/pick-cube")
    assert exported["frames"] == 3
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["features"]["observation.images.wrist"]["dtype"] == "video"
    tasks = pd.read_parquet(output / "meta/tasks.parquet")
    assert tasks.index.name == "task"
    assert tasks.loc["Pick the cube", "task_index"] == 0
    assert pq.read_table(output / "data/chunk-000/file-000.parquet").num_rows == 3
    assert (output / "videos/observation.images.wrist/chunk-000/file-000.mp4").exists()

    hub_output = tmp_path / "blacknode-hub"
    hub_exported = storage.export_blacknode_hub(
        path, hub_output, "owner/pick-cube-blacknode", include_videos=True, license_id="apache-2.0",
    )
    assert hub_exported["format"] == "blacknode-hub"
    assert hub_exported["frames"] == 3
    hub_frames = pq.read_table(hub_output / "data/train-000000.parquet")
    assert hub_frames.column("episode_id").to_pylist() == [saved["episode_id"]] * 3
    assert hub_frames.column("dataset_id").to_pylist() == ["pick-cube"] * 3
    hub_episodes = pq.read_table(hub_output / "episodes.parquet")
    assert hub_episodes.column("wrist_file_name").to_pylist() == ["media/videos/wrist/episode-000000.mp4"]
    assert (hub_output / "media/videos/wrist/episode-000000.mp4").is_file()
    assert (hub_output / "media/metadata.parquet").is_file()
    media_schema_metadata = pq.read_schema(hub_output / "media/metadata.parquet").metadata or {}
    assert b'"_type":"Video"' in media_schema_metadata[b"huggingface"]
    assert (hub_output / "blacknode/manifest.json").is_file()
    hub_card = (hub_output / "README.md").read_text(encoding="utf-8")
    assert "config_name: frames" in hub_card
    assert "config_name: episodes" in hub_card
    assert "config_name: media" in hub_card
    assert "license: \"apache-2.0\"" in hub_card

    if h5py is not None:
        hdf5_output = tmp_path / "hdf5"
        hdf5_exported = storage.export_hdf5(path, hdf5_output)
        assert hdf5_exported["frames"] == 3
        with h5py.File(hdf5_output / "episode_0.hdf5", "r") as episode:
            assert episode["observations/qpos"].shape == (3, 2)
            assert episode["observations/leader"].shape == (3, 2)
            assert episode["action"].shape == (3, 2)
            assert episode["observations/images/wrist"].shape == (3, 24, 32, 3)
            assert episode["observations/camera_metadata/wrist_sequence"][:].tolist() == [0, 1, 2]
            assert episode["recorded_at_ns"][:].tolist() == [0, 1, 2]
            assert episode["metadata/joint_names"].asstr()[:].tolist() == ["shoulder", "gripper"]
            assert episode.attrs["image_color_space"] == "RGB"


def test_trim_replay_keeps_selected_frame_and_synchronizes_every_artifact(tmp_path: Path):
    dataset = storage.create_dataset("trim-review", root=str(tmp_path), task="Trim review", fps=10)
    path = storage.resolve_dataset_path(dataset)
    work, _, _ = storage.begin_episode(path, "trim-run")
    for index in range(5):
        storage.append_frame(work, {
            "frame_index": index, "timestamp": index / 10.0, "recorded_at_ns": index,
            "task": "Trim review", "robot": _sample(index),
            "cameras": {"wrist": {"sequence": index, "captured_at_ns": index}},
        }, {"wrist": _jpeg(index * 35)})
    storage.save_episode(path, "trim-run")

    replay = storage.episode_replay(dataset, 0, "wrist")
    token = runtime.register_episode_replay(replay)
    trimmed = runtime.trim_replay_episode(token, 2, "before")

    assert trimmed["source_frames"] == 5
    assert trimmed["removed_frames"] == 2
    assert trimmed["frames"] == 3
    assert runtime.replay_frame(token, 0) is None  # destructive edits expire stale replay sessions

    episode_path = path / "episodes/episode-000000"
    table = pq.read_table(episode_path / "data.parquet")
    assert table.column("frame_index").to_pylist() == [0, 1, 2]
    assert table.column("timestamp").to_pylist() == pytest.approx([0.0, 0.1, 0.2])
    assert table.column("sample_sequence").to_pylist() == [2, 3, 4]
    assert table.column("camera.wrist.sequence").to_pylist() == [2, 3, 4]

    episode_info = json.loads((episode_path / "episode.json").read_text(encoding="utf-8"))
    assert episode_info["frames"] == 3
    assert episode_info["cameras"]["wrist"]["frames"] == 3
    assert episode_info["trim_history"][-1]["kept_start"] == 2
    manifest = json.loads((path / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["episodes"][0]["frames"] == 3
    assert storage.summarize(path)["total_frames"] == 3
    assert storage.validate(path)["ok"]

    capture = cv2.VideoCapture(str(episode_path / "cameras/wrist.mp4"))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        capture.release()

    refreshed = storage.episode_replay(dataset, 0, "wrist")
    refreshed_token = runtime.register_episode_replay(refreshed)
    assert refreshed_token != token
    cut_after = runtime.trim_replay_episode(refreshed_token, 1, "after")
    assert cut_after["removed_frames"] == 1
    assert cut_after["frames"] == 2
    final_table = pq.read_table(episode_path / "data.parquet")
    assert final_table.column("sample_sequence").to_pylist() == [2, 3]
    assert final_table.column("frame_index").to_pylist() == [0, 1]
    assert storage.validate(path)["ok"]


def test_hdf5_node_check_is_non_mutating(tmp_path: Path):
    dataset = storage.create_dataset("empty", root=str(tmp_path), task="Empty", fps=10)
    result = _NODE_REGISTRY["HDF5EpisodeExport"]({"action": "check", "dataset": dataset})
    assert not result["ok"]
    assert not result["exported"]
    assert result["status"] == "failed"


def test_blacknode_hub_node_check_and_export(tmp_path: Path):
    dataset = storage.create_dataset("hub-node", root=str(tmp_path), task="Hub", fps=10)
    path = storage.resolve_dataset_path(dataset)
    work, _, _ = storage.begin_episode(path, "hub-node-run")
    storage.append_frame(work, {
        "frame_index": 0, "timestamp": 0.0, "recorded_at_ns": 1,
        "task": "Hub", "robot": _sample(0), "cameras": {},
    }, {})
    storage.save_episode(path, "hub-node-run")
    output = tmp_path / "nested" / "hub-export"

    checked = _NODE_REGISTRY["BlacknodeHubExport"]({
        "action": "check", "dataset": dataset, "output_path": str(output), "repo_id": "owner/hub-node",
    })
    assert checked["ok"] and not checked["exported"]
    assert checked["status"] == "checked_not_exported"
    assert not output.exists()

    exported = _NODE_REGISTRY["BlacknodeHubExport"]({
        "dataset": dataset, "output_path": str(output),
        "repo_id": "owner/hub-node", "include_videos": False,
    })
    assert exported["ok"] and exported["exported"]
    assert exported["status"] == "exported"
    assert (output / "data/train-000000.parquet").is_file()
    assert (output / "episodes.parquet").is_file()
    assert not (output / "media").exists()
    existing = _NODE_REGISTRY["BlacknodeHubExport"]({
        "dataset": dataset, "output_path": str(output), "include_videos": False,
    })
    assert existing["ok"] and existing["exported"]
    assert existing["status"] == "exists"
    assert "left unchanged" in existing["report"]
    rebuilt = _NODE_REGISTRY["BlacknodeHubExport"]({
        "dataset": dataset, "output_path": str(output), "include_videos": False, "overwrite": True,
    })
    assert rebuilt["ok"] and rebuilt["status"] == "exported"


@pytest.mark.skipif(h5py is None, reason="h5py is installed by Blacknode package setup")
def test_hdf5_node_export_creates_nested_destination_and_reports_written_files(tmp_path: Path):
    dataset = storage.create_dataset("node-export", root=str(tmp_path), task="Export", fps=10)
    path = storage.resolve_dataset_path(dataset)
    work, _, _ = storage.begin_episode(path, "node-export-run")
    storage.append_frame(work, {
        "frame_index": 0, "timestamp": 0.0, "recorded_at_ns": 1,
        "task": "Export", "robot": _sample(0), "cameras": {},
    }, {})
    storage.save_episode(path, "node-export-run")
    output = tmp_path / "new" / "nested" / "hdf5-export"

    checked = _NODE_REGISTRY["HDF5EpisodeExport"]({
        "action": "check", "dataset": dataset, "output_path": str(output), "include_images": False,
    })
    assert checked["ok"] and not checked["exported"]
    assert checked["status"] == "checked_not_exported"
    assert "CHECK ONLY" in checked["report"]
    assert not output.exists()

    result = _NODE_REGISTRY["HDF5EpisodeExport"]({
        "dataset": dataset, "output_path": str(output), "include_images": False,
    })
    assert result["ok"] and result["exported"]
    assert result["status"] == "exported"
    assert result["path"] == str(output.resolve())
    assert (output / "episode_0.hdf5").is_file()
    assert (output / "blacknode-export.json").is_file()
    assert "EXPORTED 1 episode file" in result["report"]
    existing = _NODE_REGISTRY["HDF5EpisodeExport"]({
        "dataset": dataset, "output_path": str(output), "include_images": False,
    })
    assert existing["ok"] and existing["exported"]
    assert existing["status"] == "exists"
    assert "left unchanged" in existing["report"]
    rebuilt = _NODE_REGISTRY["HDF5EpisodeExport"]({
        "dataset": dataset, "output_path": str(output), "include_images": False, "overwrite": True,
    })
    assert rebuilt["ok"] and rebuilt["status"] == "exported"


@pytest.mark.skipif(h5py is None, reason="h5py is installed by Blacknode package setup")
def test_hdf5_dependency_is_available_after_package_setup():
    assert h5py is not None


def test_stop_preserves_incomplete_and_discard_removes_it(tmp_path: Path):
    dataset = storage.create_dataset("recover", root=str(tmp_path), task="Recover", fps=5)
    path = storage.resolve_dataset_path(dataset)
    storage.begin_episode(path, "interrupted")
    assert storage.summarize(path)["incomplete"] == ["interrupted"]
    recorder_status = _NODE_REGISTRY["EpisodeRecorder"]({
        "action": "status", "run_id": "interrupted", "dataset": dataset,
    })
    assert recorder_status["status"]["recoverable"] is True
    assert recorder_status["frame_count"] == 0
    assert "discard it before recording again" in recorder_status["report"]
    assert storage.discard_episode(path, "interrupted")
    assert storage.summarize(path)["incomplete"] == []


def test_upload_check_has_no_network_side_effect(tmp_path: Path):
    export = tmp_path / "export"
    (export / "meta").mkdir(parents=True)
    (export / "meta/info.json").write_text("{}", encoding="utf-8")
    (export / "README.md").write_text("# LeRobot", encoding="utf-8")
    result = _NODE_REGISTRY["HuggingFaceDatasetUpload"]({
        "action": "check", "export_path": str(export), "repo_id": "owner/dataset",
    })
    assert result["ok"]
    assert not result["uploaded"]
    assert result["status"] == "checked_not_uploaded"
    assert result["format"] == "lerobot-v3"

    blacknode_export = tmp_path / "blacknode-export"
    (blacknode_export / "data").mkdir(parents=True)
    (blacknode_export / "blacknode").mkdir()
    (blacknode_export / "README.md").write_text("# Blacknode", encoding="utf-8")
    (blacknode_export / "data/train-000000.parquet").write_bytes(b"test")
    (blacknode_export / "blacknode/manifest.json").write_text("{}", encoding="utf-8")
    (blacknode_export / "blacknode-export.json").write_text(
        json.dumps({"kind": "blacknode.huggingface-export"}), encoding="utf-8",
    )
    blacknode_result = _NODE_REGISTRY["HuggingFaceDatasetUpload"]({
        "action": "check", "export_path": str(blacknode_export), "repo_id": "owner/blacknode-dataset",
    })
    assert blacknode_result["ok"] and not blacknode_result["uploaded"]
    assert blacknode_result["format"] == "blacknode-hub"


def test_upload_uses_shared_hugging_face_credential_without_exposing_it(tmp_path: Path, monkeypatch):
    export = tmp_path / "export"
    (export / "meta").mkdir(parents=True)
    (export / "meta/info.json").write_text("{}", encoding="utf-8")
    (export / "README.md").write_text("# LeRobot", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeApi:
        def __init__(self, token=None):
            captured["token"] = token

        def create_repo(self, **kwargs):
            captured["create"] = kwargs

        def upload_folder(self, **kwargs):
            captured["upload"] = kwargs

    upload = _NODE_REGISTRY["HuggingFaceDatasetUpload"]
    monkeypatch.setitem(
        upload.__globals__, "api_key_for_provider",
        lambda provider, env_var, explicit: "terminal-or-shared-token",
    )
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    result = upload({
        "action": "upload", "export_path": str(export), "repo_id": "owner/dataset",
    })
    assert result["ok"] and result["uploaded"]
    assert captured["token"] == "terminal-or-shared-token"
    assert "terminal-or-shared-token" not in json.dumps(result)


def test_recorder_captures_streams_and_saves(tmp_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):  # noqa: N802
            now = time.time_ns()
            if self.path == "/sample":
                body = json.dumps({**_sample(1), "captured_at_ns": now}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                body = _jpeg(80)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("X-Blacknode-Frame-Sequence", "1")
                self.send_header("X-Blacknode-Captured-At-Ns", str(now))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dataset = storage.create_dataset("runtime", root=str(tmp_path), task="Runtime", fps=20)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        runtime.start_recorder(
            run_id="live", dataset=dataset,
            robot_stream={"kind": "blacknode.sample-stream", "url": base + "/sample"},
            camera_stream={"kind": "blacknode.frame-stream", "stream_id": "wrist", "snapshot_url": base + "/frame"},
            camera_streams=[], require_armed=True, stale_after=0.5, request_timeout=0.5,
        )
        deadline = time.monotonic() + 3
        while runtime.control_recorder("live", "status").get("frame_count", 0) < 2 and time.monotonic() < deadline:
            time.sleep(0.03)
        live_status = runtime.runtime_status()
        assert live_status["node_outputs"][0]["node_type"] == "EpisodeRecorder"
        assert live_status["node_outputs"][0]["outputs"]["recording"] is True
        dashboard_image = live_status["node_outputs"][0]["outputs"]["dashboard"]
        assert dashboard_image.startswith("data:image/svg+xml;base64,")
        dashboard_svg = base64.b64decode(dashboard_image.split(",", 1)[1]).decode("utf-8")
        assert "data:image/jpeg;base64," in dashboard_svg
        assert "wrist" in dashboard_svg
        assert "frames" in dashboard_svg
        assert live_status["managed_runs"][0]["capture_rate_hz"] > 0
        runtime.configure_recorder(
            run_id="live", dataset=dataset,
            robot_stream={"kind": "blacknode.sample-stream", "url": base + "/sample"},
            camera_stream={"kind": "blacknode.frame-stream", "stream_id": "wrist", "snapshot_url": base + "/frame"},
            camera_streams=[], require_armed=True, stale_after=0.5, request_timeout=0.5,
        )
        result = runtime.control_configured_recorder("live", "save")
        assert result["status"]["saved"]
        assert result["status"]["episode"]["frames"] >= 2
        saved_count = result["frame_count"]
        time.sleep(0.1)
        assert runtime.control_recorder("live", "status")["frame_count"] == 0
        assert result["status"]["saved_path"].endswith("episode-000000")
        assert saved_count >= 2
        assert storage.validate(storage.resolve_dataset_path(dataset))["ok"]
    finally:
        runtime.stop_runtime_services()
        server.shutdown()
        server.server_close()
