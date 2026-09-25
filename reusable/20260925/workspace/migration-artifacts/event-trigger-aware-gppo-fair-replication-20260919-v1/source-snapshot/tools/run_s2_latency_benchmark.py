"""Measure the frozen three-type GPPO decision path with nested timings.

This tool is an instrumented benchmark, not a training or acceptance runner.
It uses the fixed S1-R2 tapes and checkpoint, keeps execution-layer checks,
and emits raw per-decision samples plus percentile summaries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tools.run_s1_r2_acceptance import SCENARIOS, MAX_DECISIONS, import_baseline, make_tape, json_safe, sha256
except ImportError:  # server run places both runners in one independent directory
    from run_s1_r2_acceptance import SCENARIOS, MAX_DECISIONS, import_baseline, make_tape, json_safe, sha256


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int) -> float:
    return (now_ns() - start_ns) / 1_000_000.0


def synchronize(device) -> None:
    if getattr(device, "type", str(device)) == "cuda":
        import torch
        torch.cuda.synchronize(device)


def metadata_summary(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {"type": type(metadata).__name__}
    summary: dict[str, Any] = {}
    for key in ("baseline_commit", "accepted_decision_steps", "variant", "seed"):
        if key in metadata:
            summary[key] = json_safe(metadata[key])
    history = metadata.get("history")
    if isinstance(history, list):
        summary["history_records"] = len(history)
    return summary


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * fraction)


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean_ms": None if not values else float(statistics.fmean(values)),
        "p50_ms": quantile(values, 0.50),
        "p95_ms": quantile(values, 0.95),
        "p99_ms": quantile(values, 0.99),
        "max_ms": None if not values else float(max(values)),
        "p99_estimate_unstable": len(values) < 100,
    }


def model_step(model, graph, torch, device, mode: str, path: str):
    synchronize(device)
    started = now_ns()
    if path == "source_act":
        action, _, value, _ = model.act(graph, deterministic=True)
        synchronize(device)
        return elapsed_ms(started), value, int(action)
    context = torch.inference_mode() if mode == "inference_mode" else torch.no_grad()
    with context:
        logits, value, _ = model(graph)
    synchronize(device)
    return elapsed_ms(started), value, logits


def run_episode(imports, model, torch, device, scenario: str, index: int, mode: str, path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ActionSubmission, Env, EventTape, RandomEvent, RandomEventType, decode_edge_action, _ = imports
    tape = make_tape(EventTape, RandomEvent, RandomEventType, scenario, index)
    env = Env(initial_seed=tape.initial_seed, event_seed=tape.event_seed, mode=tape.mode, event_tape=tape, max_decisions=MAX_DECISIONS)
    episode_started = now_ns()
    env.reset()
    segments: defaultdict[str, list[float]] = defaultdict(list)
    samples: list[dict[str, Any]] = []
    original_step = env._step_current

    def timed_step(*args, **kwargs):
        started = now_ns()
        try:
            return original_step(*args, **kwargs)
        finally:
            segments["environment_progress"].append(elapsed_ms(started))

    env._step_current = timed_step
    terminated = truncated = False
    for step in range(MAX_DECISIONS):
        decision_started = now_ns()
        started = now_ns()
        ctx = env.begin_decision()
        context_ms = elapsed_ms(started)
        segments["graph_decision_context"].append(context_ms)

        started = now_ns()
        policy_graph = ctx.graph.to(device) if str(device) != "cpu" else ctx.graph
        feature_transfer_ms = elapsed_ms(started)
        segments["feature_tensor_prepare"].append(feature_transfer_ms)
        forward_ms, value, policy_output = model_step(model, policy_graph, torch, device, mode, path)
        segments["policy_forward"].append(forward_ms)

        started = now_ns()
        if path == "source_act":
            action = int(policy_output)
        else:
            action = int(torch.argmax(policy_output).item())
            if not (0 <= action < int(ctx.graph.num_actions) and bool(ctx.graph.action_mask[action].item())):
                action = int(ctx.graph.noop_action)
        selection_ms = elapsed_ms(started)
        segments["action_selection_legality"].append(selection_ms)

        advance_ms = 0.0
        if scenario in {"communication_interrupt", "composite_three_factor"} and step == 0:
            started = now_ns()
            advanced = env.advance_time(3.0)
            advance_ms = elapsed_ms(started)
            segments["in_flight_advance"].append(advance_ms)
        else:
            advanced = []

        started = now_ns()
        result = env.submit_action(ActionSubmission.from_decision(action, ctx))
        submit_ms = elapsed_ms(started)
        segments["submit_execution_total"].append(submit_ms)
        info = dict(result[-1])
        retry = False
        retry_timings: dict[str, float] = {}
        if bool(info.get("stale_decision", False)):
            retry = True
            started = now_ns()
            retry_ctx = env.begin_decision()
            retry_timings["graph_decision_context"] = elapsed_ms(started)
            started = now_ns()
            retry_graph = retry_ctx.graph.to(device) if str(device) != "cpu" else retry_ctx.graph
            retry_timings["feature_tensor_prepare"] = elapsed_ms(started)
            retry_forward, retry_value, retry_output = model_step(model, retry_graph, torch, device, mode, path)
            retry_timings["policy_forward"] = retry_forward
            started = now_ns()
            if path == "source_act":
                retry_action = int(retry_output)
            else:
                retry_action = int(torch.argmax(retry_output).item())
                if not (0 <= retry_action < int(retry_ctx.graph.num_actions) and bool(retry_ctx.graph.action_mask[retry_action].item())):
                    retry_action = int(retry_ctx.graph.noop_action)
            retry_timings["action_selection_legality"] = elapsed_ms(started)
            started = now_ns()
            retry_result = env.submit_action(ActionSubmission.from_decision(retry_action, retry_ctx))
            retry_timings["submit_execution_total"] = elapsed_ms(started)
            result = retry_result
            info = dict(result[-1])
            segments["retry_graph_decision_context"].append(retry_timings["graph_decision_context"])
            segments["retry_feature_tensor_prepare"].append(retry_timings["feature_tensor_prepare"])
            segments["retry_policy_forward"].append(retry_timings["policy_forward"])
            segments["retry_action_selection_legality"].append(retry_timings["action_selection_legality"])
            segments["retry_submit_execution_total"].append(retry_timings["submit_execution_total"])

        started = now_ns()
        log_row = {"scenario": scenario, "index": index, "step": step, "action": action, "retry": retry, "advanced": [str(x) for x in advanced], "info": json_safe(info), "value": json_safe(value)}
        json.dumps(log_row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        log_ms = elapsed_ms(started)
        segments["logging_serialization"].append(log_ms)
        wall_ms = elapsed_ms(decision_started)
        segments["decision_wall_clock"].append(wall_ms)
        samples.append({
            "scenario": scenario,
            "tape_index": index,
            "step": step,
            "decision_wall_clock_ms": wall_ms,
            "graph_decision_context_ms": context_ms,
            "feature_tensor_prepare_ms": feature_transfer_ms,
            "policy_forward_ms": forward_ms,
            "action_selection_legality_ms": selection_ms,
            "in_flight_advance_ms": advance_ms,
            "submit_execution_total_ms": submit_ms,
            "logging_serialization_ms": log_ms,
            "retry": retry,
            "graph_version": int(getattr(ctx, "graph_version", -1)),
            "action_version": int(getattr(ctx, "action_version", -1)),
            "action": action,
            "result_action": info.get("repaired_action"),
            "stale_decision": bool(info.get("stale_decision", False)),
            "terminated": bool(result[2]),
            "truncated": bool(result[3]),
        })
        terminated, truncated = bool(result[2]), bool(result[3])
        if terminated or truncated:
            break

    episode_ms = elapsed_ms(episode_started)
    segments["episode_wall_clock"].append(episode_ms)
    return samples, {
        "scenario": scenario,
        "tape_index": index,
        "decisions": len(samples),
        "episode_wall_clock_ms": episode_ms,
        "terminated": terminated,
        "truncated": truncated,
        "segment_values": {key: list(value) for key, value in segments.items()},
    }


def collect(args) -> int:
    imports = import_baseline(args.baseline_root.resolve())
    GraphActorCritic = imports[-1]
    import torch
    device = torch.device(args.device)
    model, metadata = GraphActorCritic.load(args.checkpoint.resolve(), map_location=device)
    model.to(device)
    model.eval()
    all_samples: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    for warmup in range(args.warmup):
        for scenario in SCENARIOS:
            for index in range(args.tapes_per_scenario):
                run_episode(imports, model, torch, device, scenario, index, args.mode, args.path)
    for repeat in range(args.repeat):
        for scenario in SCENARIOS:
            for index in range(args.tapes_per_scenario):
                samples, episode = run_episode(imports, model, torch, device, scenario, index, args.mode, args.path)
                for row in samples:
                    row["repeat"] = repeat
                episode["repeat"] = repeat
                all_samples.extend(samples)
                all_episodes.append(episode)

    segment_values: defaultdict[str, list[float]] = defaultdict(list)
    for row in all_samples:
        for key in ("decision_wall_clock_ms", "graph_decision_context_ms", "feature_tensor_prepare_ms", "policy_forward_ms", "action_selection_legality_ms", "in_flight_advance_ms", "submit_execution_total_ms", "logging_serialization_ms"):
            if key != "in_flight_advance_ms" or float(row[key]) > 0.0:
                segment_values[key].append(float(row[key]))
    for episode in all_episodes:
        segment_values["episode_wall_clock_ms"].append(float(episode["episode_wall_clock_ms"]))
        for key, values in episode["segment_values"].items():
            if key != "episode_wall_clock":
                segment_values[f"{key}_ms"].extend(float(value) for value in values)

    scenario_stats: dict[str, Any] = {}
    for scenario in SCENARIOS:
        rows = [row for row in all_samples if row["scenario"] == scenario]
        scenario_stats[scenario] = {key: stats([float(row[key]) for row in rows]) for key in ("decision_wall_clock_ms", "graph_decision_context_ms", "feature_tensor_prepare_ms", "policy_forward_ms", "submit_execution_total_ms", "logging_serialization_ms")}

    report = {
        "format": "m09-s2-latency-measurement/1.0.0",
        "mode": args.mode,
        "path": args.path,
        "device": str(device),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "tapes_per_scenario": args.tapes_per_scenario,
        "independent_tapes": len(SCENARIOS) * args.tapes_per_scenario,
        "total_decisions": len(all_samples),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "model_metadata": metadata_summary(metadata),
        "segments": {key: stats(values) for key, values in sorted(segment_values.items())},
        "scenario_segments": scenario_stats,
        "episodes": all_episodes,
        "raw_samples": all_samples,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "measurement.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"mode": args.mode, "device": str(device), "independent_tapes": report["independent_tapes"], "total_decisions": report["total_decisions"], "segments": report["segments"]}, indent=2, sort_keys=True))
    return 0


def cold_start(args) -> int:
    imports = import_baseline(args.baseline_root.resolve())
    GraphActorCritic = imports[-1]
    import torch
    started = now_ns()
    model, metadata = GraphActorCritic.load(args.checkpoint.resolve(), map_location=args.device)
    model.eval()
    load_ms = elapsed_ms(started)
    payload = {"format": "m09-s2-cold-start/1.0.0", "device": args.device, "checkpoint_sha256": sha256(args.checkpoint.resolve()), "model_load_ms": load_ms, "python": platform.python_version(), "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "model_metadata": metadata_summary(metadata)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cold-start.json").write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("baseline", "inference_mode"), default="baseline")
    parser.add_argument("--path", choices=("direct", "source_act"), default="direct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tapes-per-scenario", type=int, default=10)
    parser.add_argument("--cold-start", action="store_true")
    args = parser.parse_args()
    return cold_start(args) if args.cold_start else collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
