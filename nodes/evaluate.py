"""Episode evaluation nodes — the robot-episode counterpart to ``RateOutput``.

``EpisodeEvaluator`` judges a recorded episode into
``{success, score, failed_stage, confidence}`` plus a discovered subtask timeline
and evidence signals, and writes the verdict back into the episode so a
re-browsed episode already remembers how it did. Nothing about the task is
baked in: the analysis (in :mod:`.analysis`) self-calibrates from the episode's
own statistics, and the *definition of success* is always supplied by the caller
— a rule expression, a reference distribution, or (phase 2) a vision verdict.
Give it no criterion and it reports ``success=None`` and says why.

``EpisodeStats`` aggregates the verdicts already written to a dataset's episodes
into a success rate and a ranked failure breakdown — no new sensing or models.

See ``docs/episode-evaluator.md`` for the full design.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Float, Int, List, Text, node

from . import analysis, storage

_CATEGORY = "Dataset"


def _resolve_replay(ctx: dict) -> dict[str, Any]:
    """Resolve the target saved episode via the same path DatasetBrowser uses."""
    episode = dict(ctx.get("episode") or {})
    dataset = dict(ctx.get("dataset") or {})
    if not dataset:
        dataset_id = str(ctx.get("dataset_id") or "").strip()
        root = str(ctx.get("root") or "").strip()
        if dataset_id:
            dataset = {"path": str((Path(root).expanduser() / dataset_id) if root
                                   else storage.default_home() / dataset_id)}
    index = int(episode.get("episode_index", ctx.get("episode_index") or 0))
    return storage.episode_replay(dataset, index)


def _pick_success_strategy(ctx: dict, signals: dict) -> dict[str, Any]:
    """Judge success only from what the caller supplied; never invent a definition."""
    expression = str(ctx.get("success_rule") or "").strip()
    reference = dict(ctx.get("reference") or {})
    if expression:
        return analysis.judge_by_expression(signals, expression)
    if reference.get("features"):
        return analysis.judge_by_reference(signals, reference)
    return {
        "success": None, "confidence": 0.0, "rater": "none",
        "reason": ("no success criterion supplied — pass a `success_rule` expression over signals, "
                   "a `reference` built from known-good demos, or (later) a vision_model; "
                   "signals and the subtask timeline are still computed"),
    }


def _failed_stage(signals: dict, verdict: dict) -> str:
    """Name the first discovered segment that carries failure evidence, else ""."""
    if verdict.get("success") in (True, None):
        return ""
    for seg in signals.get("segments") or []:
        start, end = int(seg.get("start_frame", 0)), int(seg.get("end_frame", 0))
        if any(start <= f < end for f in signals.get("slip_frames") or []):
            return f"{seg.get('tag')}@{start}-{end} (slip)"
        if any(start <= f < end for f in signals.get("stall_frames") or []):
            return f"{seg.get('tag')}@{start}-{end} (stall)"
    if signals.get("grip_slip"):
        return "grip_slip"
    if signals.get("stall_frames"):
        return "stall"
    return "unknown"


def _score(signals: dict, verdict: dict) -> float:
    if verdict.get("success") is True:
        return 1.0
    if verdict.get("success") is None:
        return 0.0
    penalty = 0.4 * float(bool(signals.get("grip_slip")))
    frames = max(1, int(signals.get("frames") or 1))
    penalty += 0.4 * min(1.0, len(signals.get("stall_frames") or []) / frames)
    return float(max(0.0, 1.0 - penalty))


@node(
    name="EpisodeEvaluator", category=_CATEGORY,
    description="Judge a recorded episode into success/score/failed_stage/confidence with a "
                "self-calibrating subtask timeline and evidence signals. No task assumptions: "
                "the analysis derives everything from the episode; success needs a caller-supplied "
                "rule expression or reference distribution.",
    inputs={
        "trigger": AnyPort,
        "episode": Dict(default={}),
        "dataset": Dict(default={}),
        "dataset_id": Text(default=""),
        "root": Text(default=""),
        "episode_index": Int(default=0),
        "success_rule": Text(default=""),   # e.g. "not grip_slip and 'carry' in tags"
        "reference": Dict(default={}),        # from analysis.build_reference over good demos
        "save_label": Bool(default=True),
    },
    outputs={
        "success": Bool, "score": Float, "failed_stage": Text, "confidence": Float,
        "subtasks": List, "signals": Dict, "verdict": Dict, "episode": Dict, "report": Text,
    },
    primary_inputs=["trigger", "episode"],
    primary_outputs=["success", "failed_stage", "report"],
)
def episode_evaluator(ctx: dict) -> dict:
    try:
        replay = _resolve_replay(ctx)
        matrices = analysis.load_episode_matrices(replay["data_path"])
        gripper = analysis.discover_gripper_joint(matrices["observation"])
        signals = analysis.derive_signals(matrices, gripper)
        signals["gripper"] = gripper
        signals["task"] = matrices.get("task") or replay.get("task") or ""

        judged = _pick_success_strategy(ctx, signals)
        success = judged.get("success")
        score = _score(signals, judged)
        failed_stage = _failed_stage(signals, judged)
        confidence = float(min(judged.get("confidence", 0.0), gripper.get("confidence", 0.0))) \
            if success is not None else 0.0

        joint_names = replay.get("joint_names") or []
        gi = gripper.get("index", -1)
        verdict = {
            "success": success, "score": score, "failed_stage": failed_stage,
            "confidence": confidence, "rater": judged.get("rater"),
            "reason": judged.get("reason", ""),
            "gripper_joint": joint_names[gi] if 0 <= gi < len(joint_names) else gi,
            "gripper_confidence": gripper.get("confidence", 0.0),
            "subtasks": signals.get("segments") or [],
            "task": signals.get("task", ""),
            "ts": time.time(),
        }

        if _bool(ctx.get("save_label", True)):
            _write_back(replay.get("episode_path", ""), verdict)

        episode_out = {**dict(ctx.get("episode") or {}),
                       "episode_index": replay.get("episode_index"), "evaluation": verdict}
        state = "success" if success else "FAIL" if success is False else "no-criterion"
        report = (f"episode {replay.get('episode_index')} · {state} · score {score:.2f} · "
                  f"conf {confidence:.2f} · gripper={verdict['gripper_joint']}"
                  f"{' · ' + failed_stage if failed_stage else ''}")
        return {"success": bool(success), "score": score, "failed_stage": failed_stage,
                "confidence": confidence, "subtasks": verdict["subtasks"], "signals": signals,
                "verdict": verdict, "episode": episode_out, "report": report}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "score": 0.0, "failed_stage": "", "confidence": 0.0,
                "subtasks": [], "signals": {}, "verdict": {}, "episode": dict(ctx.get("episode") or {}),
                "report": f"evaluate FAILED: {exc}"}


def _write_back(episode_path: str, verdict: dict) -> None:
    if not episode_path:
        return
    info_path = Path(episode_path) / "episode.json"
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["evaluation"] = verdict
    tmp = info_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
    tmp.replace(info_path)


def _bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() not in ("false", "0", "no", "off", "")


@node(
    name="EpisodeStats", category=_CATEGORY,
    description="Aggregate the verdicts already written to a dataset's episodes into a success "
                "rate and a ranked failure breakdown. Pure aggregation — no sensing, no models.",
    inputs={"trigger": AnyPort, "dataset": Dict(default={}), "root": Text(default=""),
            "dataset_id": Text(default="")},
    outputs={"success_rate": Float, "attempts": Int, "evaluated": Int,
             "by_stage": Dict, "common_failures": List, "report": Text},
    primary_inputs=["trigger"], primary_outputs=["success_rate", "report"],
)
def episode_stats(ctx: dict) -> dict:
    try:
        dataset = dict(ctx.get("dataset") or {})
        if not dataset:
            dataset_id = str(ctx.get("dataset_id") or "").strip()
            root = str(ctx.get("root") or "").strip()
            dataset = {"path": str((Path(root).expanduser() / dataset_id) if root
                                   else storage.default_home() / dataset_id)}
        path = storage.resolve_dataset_path(dataset)
        manifest = storage.load_manifest(path)
        episodes = list(manifest.get("episodes") or [])
        attempts, evaluated, successes = len(episodes), 0, 0
        by_stage: dict[str, int] = {}
        for entry in episodes:
            info_path = path / str(entry.get("path") or "") / "episode.json"
            if not info_path.is_file():
                continue
            verdict = json.loads(info_path.read_text(encoding="utf-8")).get("evaluation")
            if not verdict:
                continue
            evaluated += 1
            if verdict.get("success") is True:
                successes += 1
            elif verdict.get("failed_stage"):
                by_stage[verdict["failed_stage"]] = by_stage.get(verdict["failed_stage"], 0) + 1
        rate = successes / evaluated if evaluated else 0.0
        common = sorted(by_stage.items(), key=lambda kv: kv[1], reverse=True)
        report = (f"{successes}/{evaluated} success ({rate:.0%}) over {attempts} episode(s)"
                  + (f"; top failure: {common[0][0]} ×{common[0][1]}" if common else ""))
        return {"success_rate": rate, "attempts": attempts, "evaluated": evaluated,
                "by_stage": by_stage, "common_failures": [{"stage": s, "count": c} for s, c in common],
                "report": report}
    except Exception as exc:  # noqa: BLE001
        return {"success_rate": 0.0, "attempts": 0, "evaluated": 0, "by_stage": {},
                "common_failures": [], "report": f"stats FAILED: {exc}"}
