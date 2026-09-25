"""Read-only audit of event-trigger training-window coverage and validation identity.

This script never creates environments, runs model inference, or updates parameters.
It reconstructs window membership from the persisted per-step ledger and compares
closed windows with the persisted PPO update summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ledger_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line"] = line_no
            rows.append(row)
    return rows


def audit_run(run_dir: Path):
    rows = ledger_rows(run_dir / "training-ledger.jsonl")
    rows.sort(key=lambda r: (int(r["step"]), int(r["_line"])))
    steps = [int(r["step"]) for r in rows]
    dup_steps = sorted(k for k, v in Counter(steps).items() if v > 1)
    expected = list(range(min(steps), max(steps) + 1)) if steps else []
    missing_steps = sorted(set(expected) - set(steps))
    summary = load_json(run_dir / "training-summary.json")
    ppo_updates = summary.get("ppo_updates", [])

    windows = []
    current = None
    for row in rows:
        is_actor = bool(row.get("actor_decision", False))
        if is_actor:
            if current is not None:
                windows.append(current)
            current = {
                "start_step": int(row["step"]),
                "rows": [row],
                "actor_policy_version": row.get("policy_version"),
                "actor_action": row.get("action"),
            }
        elif current is None:
            # This should never happen for the current contract, but keep it visible.
            windows.append({"start_step": None, "rows": [row], "orphan": True})
        else:
            current["rows"].append(row)
    open_window = current
    if open_window is not None:
        # The final actor window is retained in runtime and is not closed by a later
        # actor decision. It is therefore not part of the completed training set.
        open_window = dict(open_window)
        open_window["open"] = True
    closed = windows

    def enrich(w):
        rs = w["rows"]
        versions = [r.get("policy_version") for r in rs]
        return {
            "start_step": w.get("start_step"),
            "end_step": int(rs[-1]["step"]),
            "k": len(rs),
            "actor_policy_version": w.get("actor_policy_version"),
            "policy_versions": sorted(set(versions)),
            "crosses_policy_version": len(set(versions)) > 1,
            "actor_action": w.get("actor_action"),
        }

    closed_info = [enrich(w) for w in closed]
    open_info = enrich(open_window) if open_window is not None else None
    closed_k = sum(w["k"] for w in closed_info)
    open_k = int(open_info["k"]) if open_info else 0
    decision_samples = sum(int(x.get("decision_samples", 0)) for x in ppo_updates)
    crossing = [w for w in closed_info if w["crosses_policy_version"]]

    commit = load_json(run_dir / "ledger-commit.json")
    status = load_json(run_dir / "run-status.json")
    runtime_pending = None
    checkpoint = run_dir / "last-recovery.pt"
    # Avoid importing torch unless the checkpoint exists; audit remains usable on
    # machines without the training environment.
    if checkpoint.exists():
        try:
            import torch

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            runtime = payload.get("runtime", {}) if isinstance(payload, dict) else {}
            counters = payload.get("counters", {}) if isinstance(payload, dict) else {}
            pending = runtime.get("pending")
            if pending is not None:
                runtime_pending = {
                    "present": True,
                    "k": len(pending.get("rewards", [])),
                    "keys": sorted(pending.keys()),
                    "has_sample": pending.get("sample") is not None,
                    "sample_keys": sorted((pending.get("sample") or {}).keys()),
                    "has_bootstrap": pending.get("bootstrap") is not None,
                    "persistence": payload.get("persistence"),
                    "counters": {k: counters.get(k) for k in ("environment_steps", "policy_optimizer_steps", "world_optimizer_steps", "pending_window_k", "last_committed_step")},
                }
            else:
                runtime_pending = {"present": False, "persistence": payload.get("persistence"), "counters": counters}
        except Exception as exc:  # preserve audit facts without hiding failure
            runtime_pending = {"checkpoint_read_error": repr(exc)}

    return {
        "run": str(run_dir),
        "status": status,
        "commit": commit,
        "ledger_sha256": sha256(run_dir / "training-ledger.jsonl"),
        "checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else None,
        "env_steps": len(rows),
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "duplicate_steps": dup_steps,
        "missing_steps": missing_steps,
        "actor_rows": sum(bool(r.get("actor_decision", False)) for r in rows),
        "continuation_rows": sum(not bool(r.get("actor_decision", False)) for r in rows),
        "windows_closed": len(closed_info),
        "windows_open": 1 if open_info else 0,
        "closed_window_k_sum": closed_k,
        "open_window_k": open_k,
        "closed_plus_open_steps": closed_k + open_k,
        "ppo_updates": len(ppo_updates),
        "ppo_decision_samples_sum": decision_samples,
        "closed_windows_minus_trained_samples": len(closed_info) - decision_samples,
        "closed_steps_minus_trained_steps": closed_k - decision_samples,
        "orphan_rows": sum(bool(w.get("orphan")) for w in closed),
        "cross_version_closed_windows": len(crossing),
        "cross_version_examples": crossing[:20],
        "open_window": open_info,
        "runtime_pending": runtime_pending,
        "has_explicit_window_id": all("window_id" in r for r in rows),
        "has_per_step_old_logprob": all("old_log_prob" in r for r in rows),
        "has_per_step_hidden_cache": all("hidden_state" in r and "candidate_cache" in r for r in rows),
        "has_per_window_bootstrap": all("bootstrap" in r for r in rows),
    }


def audit_validation(validation_dir: Path):
    plan = load_json(validation_dir / "evaluation-plan.json")
    raw = load_json(validation_dir / "validation-results.json")
    plan_items = plan.get("plan", plan if isinstance(plan, list) else [])
    result_items = raw.get("results", raw if isinstance(raw, list) else [])

    def key(x, condition_override=None):
        # The plan is one episode per item. The result file is one aggregate row
        # per seed/group/mode/condition, with the tape identity inside episodes.
        return (x.get("seed"), x.get("group"), x.get("mode"), condition_override if condition_override is not None else x.get("condition"), x.get("tape_id"))

    plan_keys = [key(x) for x in plan_items]
    result_keys = []
    flattened_results = []
    for row in result_items:
        for ep in row.get("episodes", []):
            flat = dict(row)
            flat.update(ep)
            flat["condition"] = row.get("condition")
            flattened_results.append(flat)
            result_keys.append(key(flat))
    plan_count = Counter(plan_keys)
    result_count = Counter(result_keys)
    counts = defaultdict(int)
    for x in flattened_results:
        counts[(x.get("seed"), x.get("group"), x.get("mode"), x.get("condition"))] += 1
    return {
        "plan_items": len(plan_items),
        "plan_unique_keys": len(plan_count),
        "plan_duplicate_keys": [list(k) for k, v in plan_count.items() if v > 1],
        "result_rows": len(result_items),
        "result_items": len(flattened_results),
        "result_unique_keys": len(result_count),
        "result_duplicate_keys": [list(k) for k, v in result_count.items() if v > 1],
        "result_count_by_seed_group_mode_condition": {"|".join(map(str, k)): v for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))},
        "episodes_with_steps": sum(1 for x in flattened_results if "steps" in x),
        "environment_steps": sum(int(x.get("steps", 0)) for x in flattened_results),
    }


def audit_validation_attempts(root: Path):
    out = {}
    for name in ("validation-v1", "validation-v2", "validation-v3", "validation-v4"):
        d = root / name
        if (d / "validation-results.json").exists():
            out[name] = audit_validation(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    training = args.root / "formal-run-v1" / "training"
    runs = []
    for seed_dir in sorted(training.glob("seed-*")):
        for group_dir in sorted(seed_dir.iterdir()):
            if group_dir.is_dir() and (group_dir / "training-ledger.jsonl").exists():
                runs.append(audit_run(group_dir))
    result = {
        "audit_version": "window-audit-v1",
        "read_only": True,
        "new_environment_steps": 0,
        "new_optimizer_updates": 0,
        "runs": runs,
        "validation_attempts": audit_validation_attempts(args.root),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
