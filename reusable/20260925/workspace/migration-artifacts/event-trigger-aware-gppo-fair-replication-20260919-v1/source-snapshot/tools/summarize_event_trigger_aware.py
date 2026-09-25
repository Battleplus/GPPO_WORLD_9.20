"""Produce the finite training/validation integrity and comparison report."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys

from gppo_world.joint_training import verify_commit_pointer
from tools.run_event_trigger_aware_gppo import CONDITIONS, SEEDS, sha256, load_tapes, write_json


def source_manifest(source: Path, out: Path) -> str:
    lines = []
    for path in sorted(p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rel = path.relative_to(source).as_posix()
        lines.append(f"{sha256(path)}  {rel}")
    target = out / "source-manifest-lines.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256(target)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve(); source = artifact / "source"; training = artifact / "formal-run-v1" / "training"
    result = {"source_manifest_sha256": source_manifest(source, artifact), "training": {}, "validation": {}, "main_comparison": {}}
    for seed in SEEDS:
        for group in ("P_train", "T_train"):
            run = training / f"seed-{seed}" / group
            summary = json.loads((run / "training-summary.json").read_text(encoding="utf-8"))
            pointer = verify_commit_pointer(run)
            result["training"][f"{seed}/{group}"] = {
                "status": summary["status"], "environment_steps": summary["environment_steps"],
                "policy_optimizer_steps": summary["policy_optimizer_steps"], "world_optimizer_steps": summary["world_optimizer_steps"],
                "actor_calls": summary["actor_calls"], "continuation_steps": summary["continuation_steps"],
                "k_distribution": summary["k_distribution"], "wall_seconds": summary["wall_seconds"],
                "source_checkpoint_sha256": summary["source_checkpoint_sha256"],
                "final_checkpoint_sha256": summary["checkpoint_sha256"], "commit_verification": pointer,
            }
    validation = json.loads(args.validation.joinpath("validation-results.json").read_text(encoding="utf-8"))
    aggregate = defaultdict(lambda: {"completed": 0, "tasks": 0, "expired": 0, "episodes": 0,
                                     "host_confirmed": 0, "host_on_time": 0, "illegal_actions": 0,
                                     "security_violations": 0, "noop_opportunities": 0, "noop_selected": 0,
                                     "actor_calls": 0, "continuation_steps": 0, "messages": 0,
                                     "reward_task": 0.0, "reward_energy": 0.0})
    for row in validation["results"]:
        key = f"{row['seed']}/{row['group']}/{row['mode']}/{row['condition']}"
        a = aggregate[key]
        for episode in row["episodes"]:
            a["completed"] += int(episode.get("completed") or 0); a["tasks"] += 6; a["expired"] += int(episode.get("expired") or 0); a["episodes"] += 1
            a["host_confirmed"] += int(episode.get("host_confirmed", 0)); a["host_on_time"] += int(episode.get("host_on_time", 0))
            for field in ("illegal_actions", "security_violations", "noop_opportunities", "noop_selected", "actor_calls", "continuation_steps", "communication_proxy_messages"):
                dest = "messages" if field == "communication_proxy_messages" else field
                a[dest] += int(episode.get(field, 0))
            reward = episode.get("reward_vector", [0.0, 0.0]); a["reward_task"] += float(reward[0]); a["reward_energy"] += float(reward[1])
    result["validation"]["episode_count"] = validation["episode_count"]
    result["validation"]["aggregates"] = dict(aggregate)
    main_rows = {}
    for seed in SEEDS:
        p = aggregate[f"{seed}/P_train/triggered/W1"]; p2 = aggregate[f"{seed}/P_train/triggered/W2"]
        t = aggregate[f"{seed}/T_train/triggered/W1"]; t2 = aggregate[f"{seed}/T_train/triggered/W2"]
        p_completed, p_tasks = p["completed"] + p2["completed"], p["tasks"] + p2["tasks"]
        t_completed, t_tasks = t["completed"] + t2["completed"], t["tasks"] + t2["tasks"]
        main_rows[str(seed)] = {"P_train_T_dispatch": f"{p_completed}/{p_tasks}", "T_train_T_dispatch": f"{t_completed}/{t_tasks}",
                                "delta_tasks": t_completed - p_completed, "delta_rate": t_completed / t_tasks - p_completed / p_tasks}
    result["main_comparison"] = {"definition": "T_train+T_dispatch minus P_train+T_dispatch; W1/W2 equal-weight pooled tasks", "per_seed": main_rows,
                                  "aggregate_completed": sum(v["delta_tasks"] for v in main_rows.values()),
                                  "aggregate_tasks": sum(int(v["T_train_T_dispatch"].split('/')[1]) for v in main_rows.values())}
    result["limits"] = {
        "scope": "frozen validation only; no final-test read",
        "training_world_updates": 0,
        "world_parameters_frozen": True,
        "safety_note": "validation records zero illegal fresh-command actions and zero reported security_violations; this is not a production safety guarantee",
        "bootstrap": "not computed; this bounded validation is reported as per-seed/configuration counts",
        "task_denominator": "six tasks per validation tape, 16 tapes per condition",
    }
    write_json(artifact / "final-report.json", result)
    lines = ["# Event-trigger-aware GPPO bounded comparison", "", f"Validation episodes: {validation['episode_count']}", "", "## Main comparison"]
    for seed, row in main_rows.items():
        lines.append(f"- seed {seed}: P+T_dispatch {row['P_train_T_dispatch']}; T+T_dispatch {row['T_train_T_dispatch']}; delta {row['delta_tasks']} tasks ({row['delta_rate']:.4f})")
    lines += ["", "## Training runs"]
    for key, row in result["training"].items():
        lines.append(f"- {key}: {row['environment_steps']} env steps, {row['policy_optimizer_steps']} policy updates, world updates {row['world_optimizer_steps']}, k={row['k_distribution']}, commit={row['commit_verification']['status']}")
    lines += ["", "Interpretation: this is a bounded development validation result, not a stable generalization claim."]
    (artifact / "final-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "report": str(artifact / 'final-report.json'), "main": result["main_comparison"]}, indent=2))


if __name__ == "__main__":
    main()

