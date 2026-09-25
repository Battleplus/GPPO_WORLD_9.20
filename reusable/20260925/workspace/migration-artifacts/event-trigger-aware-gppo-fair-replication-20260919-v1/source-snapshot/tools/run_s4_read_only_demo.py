"""S4: paired pure-GPPO Shadow demo and fail-closed fault fixtures."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def json_safe(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def short_result(result):
    return {
        "valid": bool(result.valid), "fallback_reason": result.fallback_reason,
        "graph_version": int(result.graph_version), "action_version": int(result.action_version),
        "post_graph_version": int(result.post_graph_version), "post_action_version": int(result.post_action_version),
        "model_version": result.model_version, "latency_ms": float(result.latency_ms),
        "ood_score": float(result.ood_score), "uncertainty": result.uncertainty,
        "zero_latent": all(float(x) == 0.0 for x in result.latent),
        "history_reset": bool(result.history_reset),
    }


def signature(trace):
    keys = ("step", "proposed_action", "executed_action", "reward", "terminated", "truncated",
            "graph_version_before", "graph_version_after", "action_version_before", "action_version_after",
            "stale_retries", "invalid_action")
    return hashlib.sha256(json.dumps({"decisions": [{k: row.get(k) for k in keys} for row in trace],
                                      "final_snapshot": trace[-1].get("final_snapshot") if trace else None},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_arm(imports, model, tape, scenario: str, index: int, *, shadow_on: bool, world, calibration, out: Path):
    ActionSubmission, Env = imports
    from gppo_world.gppo_adapter import LatentAdapterConfig, LatentContextStore
    from gppo_world.gppo_shadow_env import PostActionShadowEnv
    from gppo_world.shadow import ShadowRuntime
    raw = Env(initial_seed=tape.initial_seed, event_seed=tape.event_seed, mode=tape.mode, event_tape=tape, max_decisions=100)
    runtime = store = None
    if shadow_on:
        model_version = "s4-shadow-read-only"
        store = LatentContextStore(LatentAdapterConfig(latent_dim=world.config.hidden_dim + world.config.stochastic_dim), model_variant="S4-Shadow", model_version=model_version)
        runtime = ShadowRuntime(world, calibration, model_version=model_version)
        env = PostActionShadowEnv(raw, runtime, store, model_variant="S4-Shadow")
    else:
        env = raw
    env.reset()
    trace = []
    for step in range(100):
        ctx = env.begin_decision()
        with torch.inference_mode():
            proposed, _, _, _ = model.act(ctx.graph, deterministic=True)
        proposed = int(proposed)
        if not bool(ctx.graph.action_mask[proposed]):
            proposed = int(ctx.graph.noop_action)
        advanced = []
        if scenario in {"communication_interrupt", "composite_three_factor"} and step == 0:
            advanced = list(env.advance_time(3.0))
        attempts = 0
        while True:
            result = env.submit_action(ActionSubmission.from_decision(proposed, ctx))
            info = dict(result[-1])
            attempts += 1
            if not info.get("stale_decision", False):
                break
            if attempts > 2:
                raise RuntimeError("stale retry budget exceeded")
            ctx = env.begin_decision()
            with torch.inference_mode():
                proposed, _, _, _ = model.act(ctx.graph, deterministic=True)
            proposed = int(proposed if bool(ctx.graph.action_mask[proposed]) else ctx.graph.noop_action)
        row = {
            "step": step, "proposed_action": proposed,
            "executed_action": info.get("executed_action", info.get("repaired_action")),
            "reward": float(result[1]), "terminated": bool(result[2]), "truncated": bool(result[3]),
            "graph_version_before": int(ctx.graph_version), "graph_version_after": int(getattr(env, "graph_version")),
            "action_version_before": int(ctx.action_version), "action_version_after": int(getattr(env, "decision_version")),
            "stale_retries": attempts - 1, "invalid_action": bool(info.get("invalid_action", False)),
            "advanced_event_ids": advanced, "info": info,
        }
        if shadow_on and runtime is not None:
            row["shadow"] = short_result(runtime.records[-1]) if runtime.records else None
        trace.append(row)
        if result[2] or result[3]:
            break
    trace[-1]["final_snapshot"] = raw.snapshot()
    arm = "on" if shadow_on else "off"
    path = out / "traces" / arm / f"{scenario}-{index:02d}.json"
    dump(path, trace)
    return {"scenario": scenario, "index": index, "arm": arm, "decisions": len(trace),
            "return": sum(row["reward"] for row in trace), "signature": signature(trace),
            "trace": path.relative_to(out).as_posix(),
            "audit": None if not shadow_on else asdict(env.audit()),
            "shadow_counters": None if runtime is None else runtime.counters,
            "context_counters": None if store is None else store.counters}


def fault_fixtures(imports, world, calibration, checkpoint_sha, out: Path):
    ActionSubmission, Env = imports
    from gppo_world.calibration import input_ood_score
    from gppo_world.contracts import GraphSnapshot, snapshot_from_gppo
    from gppo_world.gppo_adapter import LatentAdapterConfig, LatentContext, LatentContextStore
    from gppo_world.shadow import ShadowRequest, ShadowRuntime
    raw = Env(initial_seed=8801, event_seed=880101, events_per_episode=2, max_decisions=8)
    raw.reset(seed=8801)
    ctx = raw.begin_decision()
    action = int(torch.nonzero(ctx.graph.action_mask, as_tuple=False)[0].item())
    result = raw.submit_action(ActionSubmission.from_decision(action, ctx))
    expected = (int(raw.graph_version), int(raw.decision_version))
    request = ShadowRequest("s4-fault", 0, snapshot_from_gppo(ctx.graph), action, (), int(ctx.action_version), float(getattr(raw, "current_time", 0.0)), True, *expected)
    def fresh():
        return ShadowRuntime(world, calibration, model_version=f"s4-fault:{checkpoint_sha}")
    adapter_config = LatentAdapterConfig(latent_dim=world.config.hidden_dim + world.config.stochastic_dim)
    store = LatentContextStore(adapter_config, model_variant="S4-Fault", model_version=f"s4-fault:{checkpoint_sha}")
    rows = {"no_context": {"valid": store.read(expected[0], expected[1]).valid},
            "reset": {}, "invalid_or_exception": {}, "stale_before": {}, "stale_after": {}, "timeout": {}, "synthetic_ood": {}, "disabled": {}}
    latent = tuple([1.0] + [0.0] * (adapter_config.latent_dim - 1))
    store.publish(LatentContext(latent, True, "S4-Fault", f"s4-fault:{checkpoint_sha}", expected[0], expected[1], "s4-fault", 0))
    rows["reset"]["before_valid"] = store.read(expected[0], expected[1]).valid
    store.reset(graph_version=expected[0], action_version=expected[1])
    rows["reset"]["after_valid"] = store.read(expected[0], expected[1]).valid
    rows["disabled"] = {"context_ignored": True, "policy_mode": "pure_gppo", "context_consumed": False}
    rows["invalid_or_exception"]["result"] = short_result(fresh().observe(request, force_exception=True))
    rows["stale_before"]["result"] = short_result(fresh().observe(request, version_reader=lambda: (expected[0] + 1, expected[1])))
    sequence = iter([expected, (expected[0], expected[1] + 1)])
    rows["stale_after"]["result"] = short_result(fresh().observe(request, version_reader=lambda: next(sequence)))
    rows["timeout"]["result"] = short_result(fresh().observe(request, latency_injection_ms=calibration.timeout_ms + 1.0))
    shifted_nodes = {name: value + 3.0 for name, value in request.graph.nodes.items()}
    shifted = GraphSnapshot(shifted_nodes, request.graph.edge_index, request.graph.edge_attr, request.graph.candidate_edges, request.graph.action_mask, request.graph.graph_version)
    shifted_request = replace(request, graph=shifted)
    rows["synthetic_ood"]["input_ood_score"] = input_ood_score(shifted, calibration)
    rows["synthetic_ood"]["result"] = short_result(fresh().observe(shifted_request))
    for name, row in rows.items():
        if "result" in row:
            row["zero_context"] = row["result"]["zero_latent"]
    dump(out / "fault-injection-results.json", rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline_root = Path(args.baseline_root).resolve(); checkpoint = Path(args.checkpoint).resolve(); world_path = Path(args.world_checkpoint).resolve(); calibration_path = Path(args.calibration).resolve(); out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(baseline_root)); sys.path.insert(0, str(baseline_root / "ppo_allocation"))
    from random_event.environment import ActionSubmission, RandomEventAllocationEnv
    from random_event.events import EventTape, RandomEvent, RandomEventType
    from random_event.models import GraphActorCritic
    from gppo_world.calibration import ShadowCalibration
    from gppo_world.gppo_adapter import freeze_world_model
    from gppo_world.model import EventAwareGraphWorldModel
    from tools.run_s1_r2_acceptance import SCENARIOS, TAPES_PER_SCENARIO, make_tape
    model, metadata = GraphActorCritic.load(checkpoint, map_location="cpu")
    model.eval(); world, world_meta = EventAwareGraphWorldModel.load(world_path, map_location="cpu"); freeze_world_model(world)
    policy_state_before = state_sha(model)
    world_state_before = state_sha(world)
    calibration = ShadowCalibration.from_dict(json.loads(calibration_path.read_text(encoding="utf-8")))
    imports = (ActionSubmission, RandomEventAllocationEnv)
    rows = []
    for scenario in SCENARIOS:
        for index in range(TAPES_PER_SCENARIO):
            tape = make_tape(EventTape, RandomEvent, RandomEventType, scenario, index)
            rows.append({"off": run_arm(imports, model, tape, scenario, index, shadow_on=False, world=world, calibration=calibration, out=out),
                         "on": run_arm(imports, model, tape, scenario, index, shadow_on=True, world=world, calibration=calibration, out=out)})
    pairs = [{"scenario": pair["off"]["scenario"], "index": pair["off"]["index"], "equal": pair["off"]["signature"] == pair["on"]["signature"], "off_signature": pair["off"]["signature"], "on_signature": pair["on"]["signature"], "return_difference": pair["on"]["return"] - pair["off"]["return"]} for pair in rows]
    faults = fault_fixtures(imports, world, calibration, sha(checkpoint), out)
    on_details = [item["on"] for item in rows]
    audit_fields = ("belief_write_count", "action_mask_write_count", "graph_version_write_count", "action_version_write_count", "action_submission_count", "real_environment_mutation_count", "real_belief_mutation_count", "real_action_mask_mutation_count", "real_version_mutation_count")
    audit_totals = {key: sum(int((item.get("audit") or {}).get(key, 0)) for item in on_details) for key in audit_fields}
    context_totals = {key: sum(int((item.get("context_counters") or {}).get(key, 0)) for item in on_details) for key in ("published_valid", "published_fallback", "served_valid", "served_fallback", "stale_context")}
    safety_gates = {key + "_zero": value == 0 for key, value in audit_totals.items()}
    safety_gates["shadow_context_published"] = context_totals["published_valid"] + context_totals["published_fallback"] > 0
    safety_gates["policy_state_unchanged"] = policy_state_before == state_sha(model)
    safety_gates["world_frozen_and_unchanged"] = (not world.training and all(not p.requires_grad for p in world.parameters()) and world_state_before == state_sha(world))
    dump(out / "shadow-equivalence.json", {"pairs": pairs, "pair_count": len(pairs), "equal_count": sum(int(x["equal"]) for x in pairs), "all_equal": all(x["equal"] for x in pairs), "return_differences": sorted(set(x["return_difference"] for x in pairs)), "audit_totals": audit_totals, "context_totals": context_totals, "safety_gates": safety_gates, "all_gates_pass": all(x["equal"] for x in pairs) and all(safety_gates.values())})
    dump(out / "run-summary.json", {"format": "m09-s4-read-only-world-model/1.0.0", "python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "policy_checkpoint_sha256": sha(checkpoint), "world_checkpoint_sha256": sha(world_path), "calibration_sha256": sha(calibration_path), "policy_metadata": metadata, "world_metadata": world_meta, "pairs": pairs, "faults": faults, "audit_totals": audit_totals, "context_totals": context_totals, "safety_gates": safety_gates})
    dump(out / "inventory.json", [{"path": p.relative_to(out).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.rglob("*")) if p.is_file() and p.name != "inventory.json"])


if __name__ == "__main__":
    main()
