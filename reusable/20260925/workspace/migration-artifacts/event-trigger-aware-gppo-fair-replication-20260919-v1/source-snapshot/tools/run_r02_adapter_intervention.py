"""R02/D04A: actual adapter intervention on the fixed D02 development bank."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
from statistics import mean
import subprocess
import sys
import tarfile
import traceback
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARCHIVE_HASH = "a26f68fed5e3005a9d1808d62ab01a3492a0964dc2811f7bf23d3f2768697a7b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        out.write("\n")


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def trajectory_signature(trace):
    # Wall-clock latency is deliberately excluded; it is not a behavioral state.
    keys = ("raw_action", "repaired_action", "reward", "graph_version_before",
            "graph_version_after", "pending_regions_after", "active_events_before",
            "simulation_time_before", "simulation_time_after", "invalid_action")
    value = {"decisions": [{k: d[k] for k in keys} for d in trace["decisions"]],
             "final_snapshot": trace["final_snapshot"],
             "terminated": trace["terminated"], "truncated": trace["truncated"]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def execute(args, output):
    started = time.monotonic()
    config_path = ROOT / "nodes/D-02/protocol.json"
    r02_path = ROOT / "nodes/R-02/protocol.json"
    r02 = json.loads(r02_path.read_text(encoding="utf-8"))
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "nodes/T-05/server-training-config.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "nodes/T-05/evidence/server-run-manifest.json").read_text(encoding="utf-8"))
    baseline = Path(args.baseline_root).resolve()
    if git(ROOT, "status", "--porcelain") or git(baseline, "status", "--porcelain"):
        raise ValueError("Both source worktrees must be clean before the run")
    if git(baseline, "rev-parse", "HEAD") != protocol["baseline_commit"]:
        raise ValueError("Baseline commit mismatch")
    if sha(args.run_archive) != ARCHIVE_HASH:
        raise ValueError("Run archive checksum mismatch")
    sys.path.insert(0, str(baseline))
    import ppo_allocation.random_event.experiment as experiment
    from ppo_allocation.random_event.baselines import GraphPolicyAdapter
    from ppo_allocation.random_event.models import GraphActorCritic, GraphModelConfig
    from gppo_world.adapter_probe import AdapterProbe
    from gppo_world.gppo_adapter import LatentAugmentedActorCritic, LatentContextStore, freeze_world_model
    from gppo_world.gppo_shadow_env import PostActionShadowEnv
    from gppo_world.model import EventAwareGraphWorldModel
    from gppo_world.calibration import ShadowCalibration
    from gppo_world.shadow import ShadowRuntime

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    write(output / "protocol.json", r02)
    write(output / "provenance.json", {
        "source_commit": git(ROOT, "rev-parse", "HEAD"),
        "baseline_commit": protocol["baseline_commit"], "protocol_sha256": sha(r02_path),
        "development_generation_protocol_sha256": sha(config_path),
        "adapter_intervention": True, "policy_updates": 0,
        "run_archive_sha256": ARCHIVE_HASH, "python": sys.version,
        "torch": torch.__version__, "platform": platform.platform(), "device": "cpu",
        "formal_gpu_result_reproduction": False,
    })
    old_test = json.loads((ROOT / "nodes/T-05/evidence/server-test-bank-manifest.json").read_text(encoding="utf-8"))["entries"]
    forbidden = {r[k] for r in old_test for k in ("initial_seed", "event_seed")}
    forbidden_hash = {r["canonical_tape_sha256"] for r in old_test}
    spec = protocol["development_bank"]
    rng = random.Random(spec["master_seed"])
    tapes, entries = [], []
    for mode in spec["modes"]:
        for index in range(spec["tapes_per_mode"]):
            initial, event = rng.getrandbits(31), rng.getrandbits(63)
            if {initial, event} & forbidden:
                raise ValueError("Development/Test seed overlap")
            tape = experiment._make_tape(initial, event, mode, spec["events_per_tape"])
            digest = hashlib.sha256(tape.to_bytes()).hexdigest()
            if digest in forbidden_hash:
                raise ValueError("Development/Test canonical tape overlap")
            tape_id = f"dev-{mode}-{index:04d}-{digest[:12]}"
            write(output / "tapes" / f"{tape_id}.json", json.loads(tape.to_json()))
            entries.append({"tape_id": tape_id, "mode": mode, "initial_seed": initial,
                            "event_seed": event, "canonical_tape_sha256": digest})
            tapes.append((tape_id, tape))
    write(output / "development-bank.json", {"split": "development_not_test", "entries": entries,
          "original_test_seed_overlap": 0, "original_test_tape_overlap": 0})

    frozen_bank = json.loads((ROOT / "nodes/D-02/evidence/development-bank.json").read_text(encoding="utf-8"))
    if entries != frozen_bank["entries"]:
        raise ValueError("D02 development bank mismatch")

    chosen = [r for r in manifest["runs"] if r["group"] in protocol["groups"]]
    if {(r["group"], r["seed"]) for r in chosen} != {
        (g, s) for g in protocol["groups"] for s in protocol["seeds"]
    } or len(chosen) != 9:
        raise ValueError("Incomplete checkpoint matrix")
    files, hashes = {}, {}
    with tarfile.open(args.run_archive) as tar:
        for run in chosen:
            checkpoint = next(c for c in run["checkpoints"] if c["step"] == 50000)
            member = f"{run['run_directory']}/{checkpoint['path']}"
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != checkpoint["sha256"]:
                raise ValueError("Policy checksum mismatch")
            path = output / "checkpoints" / run["group"] / Path(member).name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as out:
                out.write(data)
            files[run["group"], run["seed"]] = path
            hashes[str(path)] = checkpoint["sha256"]

    def base_factory(spec):
        return GraphActorCritic(spec["node_dims"], GraphModelConfig(**spec["config"]),
                               edge_dims={tuple(k.split("__")): v for k, v in spec["edge_dims"].items()})

    original_env = experiment.RandomEventAllocationEnv
    summaries, all_rows, pairs, episode_rows = [], [], [], []
    for run in chosen:
        group, seed = run["group"], run["seed"]
        world_seed = frozen["world_seed_by_training_seed"][str(seed)]
        world_name = frozen["groups"][group]["world_model_pattern"].format(world_seed=world_seed)
        world_path = Path(args.world_checkpoint_dir) / world_name
        world_sha = frozen["world_checkpoint_sha256"][world_name]
        if sha(world_path) != world_sha:
            raise ValueError("World model checksum mismatch")
        hashes[str(world_path)] = world_sha
        side_signatures = {}
        side_returns = {}
        for observed in (False, True):
            torch.manual_seed(seed)
            world, _ = EventAwareGraphWorldModel.load(world_path, map_location="cpu")
            freeze_world_model(world)
            policy_model, metadata = LatentAugmentedActorCritic.load(
                files[group, seed], base_factory=base_factory, map_location="cpu")
            if (metadata["t05_group"] != group or metadata["training_seed"] != seed
                    or metadata["accepted_decision_steps"] != 50000
                    or metadata["world_checkpoint_sha256"] != world_sha):
                raise ValueError("Checkpoint metadata mismatch")
            policy_model.enabled = observed
            policy_model.eval()
            store = LatentContextStore(policy_model.adapter_config, model_variant=group,
                                       model_version=policy_model.model_version)
            policy_model.context_store = store
            calibration = ShadowCalibration.from_dict(json.loads(
                (ROOT / frozen["shadow_calibration"]["path"]).read_text(encoding="utf-8")))
            runtime = ShadowRuntime(world, calibration, model_version=policy_model.model_version)
            envs = []

            def env_factory(*a, **kw):
                env = PostActionShadowEnv(original_env(*a, **kw), runtime, store, model_variant=group)
                envs.append(env)
                return env

            experiment.RandomEventAllocationEnv = env_factory
            probe = AdapterProbe(policy_model)
            policy = GraphPolicyAdapter(model=probe or policy_model, name=group)
            state_before = {k: v.clone() for k, v in policy_model.state_dict().items()}
            try:
                for tape_id, tape in tapes:
                    if time.monotonic() - started > r02["adapter_intervention"]["max_wall_seconds"]:
                        raise TimeoutError("Fixed intervention budget exceeded")
                    start = len(probe.records)
                    episode, trace = experiment.run_episode(policy, tape_id=tape_id, tape=tape,
                        algorithm=group, max_decisions=protocol["execution"]["max_decisions_per_episode"])
                    signature = trajectory_signature(trace)
                    side = "on" if observed else "off"
                    write(output / "traces" / group / str(seed) / side / f"{tape_id}.json", trace)
                    total_return = sum(d["reward"] for d in trace["decisions"])
                    episode_rows.append({"group": group, "seed": seed, "tape_id": tape_id, "arm": side,
                        "all_decision_return": total_return, "original_event_attributed_return": episode.to_dict()["episode_return"],
                        "episode": episode.to_dict(), "terminated": trace["terminated"], "truncated": trace["truncated"]})
                    for row in probe.records[start:]:
                        all_rows.append({"group": group, "seed": seed, "tape_id": tape_id, "arm": side, **row})
                    if not observed:
                        side_signatures[tape_id] = signature
                        side_returns[tape_id] = total_return
                    else:
                        equal = side_signatures[tape_id] == signature
                        pairs.append({"group": group, "seed": seed, "tape_id": tape_id,
                                      "signature_on": signature, "signature_off": side_signatures[tape_id], "equal": equal,
                                      "return_on": total_return, "return_off": side_returns[tape_id],
                                      "return_difference_on_minus_off": total_return - side_returns[tape_id]})
            finally:
                experiment.RandomEventAllocationEnv = original_env
            audits = [asdict(env.audit()) for env in envs]
            write(output / "audits" / group / f"{seed}-{side}.json", {"environments": audits,
                  "shadow_counters": runtime.counters, "context_counters": store.counters})
            safety_keys = ("belief_write_count", "action_mask_write_count", "graph_version_write_count",
                "action_version_write_count", "action_submission_count", "real_environment_mutation_count",
                "real_belief_mutation_count", "real_action_mask_mutation_count", "real_version_mutation_count")
            if any(a[k] for a in audits for k in safety_keys):
                raise RuntimeError("Nonzero safety mutation count")
            if any(not torch.equal(v, state_before[k]) for k, v in policy_model.state_dict().items()):
                raise RuntimeError("Probe/evaluation changed policy state")
        rows = [r for r in all_rows if r["group"] == group and r["seed"] == seed and r["arm"] == "on"]
        used = [r for r in rows if r["reason"] == "used"]
        summaries.append({"group": group, "seed": seed, "decisions": len(rows),
            "reason_counts": dict(Counter(r["reason"] for r in rows)),
            "argmax_disagreements": sum(r["argmax_disagreement"] for r in rows),
            "used_count": len(used), "used_argmax_disagreements": sum(r["argmax_disagreement"] for r in used),
            "used_mean_probability_tv": mean(r["legal_probability_total_variation"] for r in used) if used else None,
            "used_mean_actor_residual_l2": mean(r["legal_actor_residual_l2"] for r in used) if used else None,
            "used_mean_critic_residual": mean(r["critic_residual"] for r in used) if used else None,
            "used_mean_latent_l2": mean(r["latent_l2"] for r in used) if used else None})
        paired = [p for p in pairs if p["group"] == group and p["seed"] == seed]
        summaries[-1].update({"mean_paired_return_difference": mean(p["return_difference_on_minus_off"] for p in paired),
            "changed_trajectories": sum(not p["equal"] for p in paired)})
        print(json.dumps(summaries[-1]), flush=True)
        write(output / "run-summaries" / f"{group}-{seed}.json", summaries[-1])
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise RuntimeError("Original checkpoint hash changed")
    if len(pairs) != protocol["execution"]["diagnosed_episodes"]:
        raise RuntimeError("Wrong episode count")
    write(output / "decision-probes.json", all_rows)
    disabled = [r for r in all_rows if r["arm"] == "off"]
    if any(r["reason"] != "disabled" or r["legal_actor_residual_l2"] != 0 for r in disabled):
        raise RuntimeError("Adapter off did not return exact base logits")
    if len(episode_rows) != 216:
        raise RuntimeError("Incomplete intervention matrix")
    write(output / "paired-interventions.json", pairs)
    write(output / "episodes.json", episode_rows)
    write(output / "summary.json", {"status": "PASS", "paired_episodes": len(pairs),
          "executed_episodes": 2 * len(pairs), "probe_decisions": len(all_rows),
          "checkpoint_hashes_unchanged": True, "all_safety_mutations_zero": True,
          "changed_trajectory_pairs": sum(not p["equal"] for p in pairs), "adapter_off_exact_base_logits": True,
          "elapsed_seconds": time.monotonic() - started,
          "interpretation": "Development intervention only; not an independently trained GPPO comparison", "runs": summaries})
    write(output / "inventory.json", [{"path": p.relative_to(output).as_posix(), "sha256": sha(p),
          "bytes": p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--run-archive", required=True)
    parser.add_argument("--world-checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        execute(args, output)
    except Exception as error:
        write(output / "failure.json", {"status": "FAILED", "exception": str(error),
                                       "traceback": traceback.format_exc(), "overwritten": False})
        raise


if __name__ == "__main__":
    main()
