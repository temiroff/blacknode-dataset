"""Before-training smoothing in the dataset exporters.

Builds a small dataset whose action/observation trajectories are deliberately
shaky, then exports with and without smoothing and checks that smoothing calms
the exported trajectories (lower jerk) and records provenance, while the default
(``none``) leaves values unchanged. Uses the LeRobot Parquet export (pyarrow) so
the core checks run without h5py; the HDF5 path is checked separately when h5py
is present.
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

import blacknode  # noqa: F401 - triggers package discovery
from blacknode.pkg.blacknode_dataset import filters, storage

_JOINTS = ["shoulder", "gripper"]


def _jpeg(value: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((24, 32, 3), value % 255, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _shaky_sample(i: int) -> dict:
    clean = math.sin(i * 0.15)
    jitter = 0.3 if i % 2 else -0.3  # high-frequency shake
    value = clean + jitter
    return {
        "kind": "blacknode.teleoperation-sample", "schema_version": 1, "sequence": i,
        "captured_at_ns": i, "joint_names": _JOINTS,
        "leader": {"shoulder": value, "gripper": value * 0.5},
        "observation": {"shoulder": value, "gripper": value * 0.5},
        "action": {"shoulder": value, "gripper": value * 0.5},
        "units": "radians", "armed": True, "live": True,
    }


def _build_dataset(tmp_path, frames: int = 40):
    dataset = storage.create_dataset("shaky", root=str(tmp_path), task="wiggle", fps=20,
                                     robot_type="so_arm101")
    path = storage.resolve_dataset_path(dataset)
    work, _, _ = storage.begin_episode(path, "run")
    for i in range(frames):
        storage.append_frame(work, {
            "frame_index": i, "timestamp": i / 20.0, "recorded_at_ns": i,
            "task": "wiggle", "robot": _shaky_sample(i),
            "cameras": {"wrist": {"sequence": i, "captured_at_ns": i}},
        }, {"wrist": _jpeg(i)})
    storage.save_episode(path, "run")
    return path


def _exported_action(output):
    table = storage.pq.read_table(output / "data" / "chunk-000" / "file-000.parquet")
    return np.asarray(table.column("action").to_pylist(), dtype=np.float32)


def test_export_smoothing_reduces_jerk_and_records_provenance(tmp_path):
    path = _build_dataset(tmp_path)

    raw_out = tmp_path / "lerobot-raw"
    storage.export_lerobot_v3(path, raw_out, smoothing="none")
    raw_action = _exported_action(raw_out)

    smooth_out = tmp_path / "lerobot-smooth"
    result = storage.export_lerobot_v3(path, smooth_out, smoothing="gaussian", smoothing_strength=1.5)
    smooth_action = _exported_action(smooth_out)

    assert result["smoothing"] == "gaussian"
    assert result["frames"] == 40
    provenance = json.loads((smooth_out / "blacknode-export.json").read_text(encoding="utf-8"))
    assert provenance["smoothing"] == "gaussian"

    # smoothing must actually calm the exported trajectory, without reshaping it
    assert filters.jerk_rms(smooth_action) < filters.jerk_rms(raw_action)
    assert smooth_action.shape == raw_action.shape


def test_export_smoothing_default_is_unchanged(tmp_path):
    path = _build_dataset(tmp_path, frames=12)
    out = tmp_path / "lerobot-default"
    result = storage.export_lerobot_v3(path, out)
    assert result["smoothing"] == "none"
    exported = _exported_action(out)
    native = storage.pq.read_table(path / "episodes/episode-000000/data.parquet")
    recorded = np.asarray(native.column("action").to_pylist(), dtype=np.float32)
    assert np.allclose(exported, recorded)


def test_hdf5_export_smoothing_when_available(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = _build_dataset(tmp_path, frames=30)
    out = tmp_path / "hdf5-smooth"
    storage.export_hdf5(path, out, include_images=False, smoothing="spline", smoothing_strength=1.0)
    with h5py.File(out / "episode_0.hdf5", "r") as episode:
        action = np.asarray(episode["action"][:])
        assert str(episode.attrs["smoothing"]) == "spline"
    raw_out = tmp_path / "hdf5-raw"
    storage.export_hdf5(path, raw_out, include_images=False)
    with h5py.File(raw_out / "episode_0.hdf5", "r") as episode:
        raw_action = np.asarray(episode["action"][:])
    assert filters.jerk_rms(action) < filters.jerk_rms(raw_action)
