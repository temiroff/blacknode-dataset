"""EpisodeEvaluator: self-calibrating analysis and caller-supplied success.

All fixtures are synthesised numpy signals + a synthetic ``data.parquet`` — no
robot, no cameras, no ffmpeg. Verifies that nothing task-specific is assumed:
the gripper joint, bands, engaged state, and segments are all discovered.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import blacknode  # noqa: F401 - triggers package discovery
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_dataset import analysis, evaluate


def _episode(*, slip: bool) -> dict:
    """A 3-joint pick: two smooth arm joints + one bimodal gripper.

    The gripper is *commanded* open (0-19), closed (20-58), open (59+). In the
    clean run the achieved aperture follows the command. In the slip run the
    command still says "hold closed" through frame 58, but the achieved gripper
    springs open at frame 40 — a hold failure the ``action`` channel exposes
    without any guess about which aperture means "closed".
    """
    t = np.arange(60, dtype=float)
    arm0 = np.concatenate([np.linspace(0, 1.0, 20), np.linspace(1.0, 0.2, 40)])  # reach then carry
    arm1 = np.concatenate([np.linspace(0, 0.5, 20), np.linspace(0.5, 1.2, 40)])
    gripper_cmd = np.where(t < 20, 0.9, 0.05)  # open ~0.9, closed ~0.05
    gripper_cmd[t >= 59] = 0.9                   # commanded release
    gripper_obs = gripper_cmd.copy()
    if slip:
        gripper_obs[(t >= 40) & (t < 59)] = 0.9  # achieved gripper lost the hold
    obs = np.stack([arm0, arm1, gripper_obs], axis=1)
    action = np.stack([arm0, arm1, gripper_cmd], axis=1)
    return {"timestamp": t, "observation": obs, "action": action, "leader": obs.copy(),
            "joint_count": 3, "task": "pick the cube"}


def test_gripper_is_discovered_not_positional() -> None:
    m = _episode(slip=False)
    g = analysis.discover_gripper_joint(m["observation"])
    assert g["index"] == 2                      # the bimodal joint, found by shape
    assert g["confidence"] > 0.5
    # A purely smooth arm-only episode should NOT confidently claim a gripper.
    smooth = np.stack([np.linspace(0, 1, 60), np.linspace(0, 2, 60)], axis=1)
    assert analysis.discover_gripper_joint(smooth)["confidence"] < 0.5


def test_clean_carry_has_no_slip_but_slip_variant_does() -> None:
    clean = analysis.derive_signals(_episode(slip=False),
                                    analysis.discover_gripper_joint(_episode(slip=False)["observation"]))
    slipped = analysis.derive_signals(_episode(slip=True),
                                      analysis.discover_gripper_joint(_episode(slip=True)["observation"]))
    assert clean["grip_slip"] is False
    assert slipped["grip_slip"] is True
    assert any(s["tag"] == "move" for s in clean["segments"])
    assert any(s["tag"] == "grip-change" for s in slipped["segments"])


def test_success_requires_a_supplied_criterion() -> None:
    m = _episode(slip=False)
    signals = analysis.derive_signals(m, analysis.discover_gripper_joint(m["observation"]))
    # No criterion -> refuses to decide.
    ctx_none = {"episode": {}, "dataset": {}}
    verdict = evaluate._pick_success_strategy(ctx_none, signals)
    assert verdict["success"] is None
    # A rule expression the caller owns.
    good = analysis.judge_by_expression(signals, "not grip_slip and 'move' in tags")
    assert good["success"] is True
    slip_signals = analysis.derive_signals(_episode(slip=True),
                                            analysis.discover_gripper_joint(_episode(slip=True)["observation"]))
    bad = analysis.judge_by_expression(slip_signals, "not grip_slip and 'move' in tags")
    assert bad["success"] is False


def test_reference_flags_the_outlier() -> None:
    good_rows = [analysis._reference_features(
        analysis.derive_signals(_episode(slip=False),
                                analysis.discover_gripper_joint(_episode(slip=False)["observation"])))
        for _ in range(5)]
    ref = analysis.build_reference(good_rows, z_threshold=3.0)
    slip_signals = analysis.derive_signals(_episode(slip=True),
                                           analysis.discover_gripper_joint(_episode(slip=True)["observation"]))
    verdict = analysis.judge_by_reference(slip_signals, ref)
    assert verdict["success"] is False  # slip is an outlier vs good demos


def _write_parquet(path: Path, m: dict) -> None:
    j = m["observation"].shape[1]
    vec = pa.list_(pa.float32(), j)
    pq.write_table(pa.table({
        "timestamp": pa.array(m["timestamp"], type=pa.float32()),
        "observation.state": pa.array(m["observation"].tolist(), type=vec),
        "action": pa.array(m["action"].tolist(), type=vec),
        "leader.state": pa.array(m["leader"].tolist(), type=vec),
        "task": pa.array([m["task"]] * len(m["timestamp"]), type=pa.string()),
    }), path)


def test_node_end_to_end_and_writeback(tmp_path, monkeypatch) -> None:
    m = _episode(slip=True)
    ep_dir = tmp_path / "episodes" / "episode-000000"
    ep_dir.mkdir(parents=True)
    _write_parquet(ep_dir / "data.parquet", m)
    (ep_dir / "episode.json").write_text(json.dumps({"episode_index": 0, "frames": 60}), encoding="utf-8")
    (tmp_path / "dataset.json").write_text(json.dumps({
        "kind": "blacknode.episode-dataset", "schema_version": 1, "dataset_id": "t", "fps": 30,
        "episodes": [{"episode_index": 0, "path": "episodes/episode-000000", "frames": 60}],
    }), encoding="utf-8")

    monkeypatch.setattr(evaluate.storage, "episode_replay", lambda dataset, index, camera="": {
        "episode_index": 0, "data_path": str(ep_dir / "data.parquet"),
        "episode_path": str(ep_dir), "joint_names": ["shoulder", "elbow", "gripper"], "task": m["task"],
    })

    out = _NODE_REGISTRY["EpisodeEvaluator"]({"episode": {"episode_index": 0}, "dataset": {"path": str(tmp_path)},
                                              "success_rule": "not grip_slip", "save_label": True})

    assert out["success"] is False
    assert out["verdict"]["gripper_joint"] == "gripper"   # named, discovered by shape
    assert out["failed_stage"]
    assert 0.0 <= out["confidence"] <= 1.0
    written = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
    assert written["evaluation"]["success"] is False       # remembered on the episode

    sout = _NODE_REGISTRY["EpisodeStats"]({"dataset": {"path": str(tmp_path)}})
    assert sout["evaluated"] == 1 and sout["success_rate"] == 0.0
