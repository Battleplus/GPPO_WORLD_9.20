"""Build the final report and artifact hash index without running the environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(r"E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1")
SOURCE = Path(r"E:\Z博士\migration-artifacts\wd-event-trigger-aware-gppo-20260918\window-budget-fix-v1\source")
REPAIR = Path(r"E:\Z博士\migration-artifacts\wd-event-trigger-aware-gppo-20260918\window-budget-fix-v1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    analysis = json.loads((ROOT / "analysis" / "fair-replication-analysis.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    evidence = ROOT / "evidence"; evidence.mkdir(parents=True, exist_ok=True)
    for name in ("failure-prefix-contract-v1.json", "replay-prefix-verification-v4.json", "final-history-replay-repair-report-20260918.md"):
        shutil.copy2(REPAIR / name, evidence / name)
    source_files = [SOURCE / "tools/run_event_trigger_aware_gppo.py", SOURCE / "gppo_world/budget_executor.py", SOURCE / "tests/test_event_trigger_history_replay.py", SOURCE / "tools/verify_t_replay_prefix.py"]
    checkpoints = [ROOT / "training" / f"seed-{seed}" / group / "last-recovery.pt" for seed in (1101, 2203, 3307) for group in ("P_train", "T_train")]
    artifacts = [ROOT / "protocol.json", ROOT / "source-manifest.json", ROOT / "tapes/tape-manifest.json", ROOT / "tapes/training-tapes.json", ROOT / "tapes/evaluation-tapes.json", ROOT / "budget.sqlite3", ROOT / "evaluation/episodes.jsonl", ROOT / "evaluation/evaluation-summary.json", ROOT / "analysis/fair-replication-analysis.json"] + checkpoints + source_files + [evidence / name for name in ("failure-prefix-contract-v1.json", "replay-prefix-verification-v4.json", "final-history-replay-repair-report-20260918.md")]
    artifact_index = {str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts if path.is_file()}
    (ROOT / "artifact-manifest.json").write_text(json.dumps({"schema": "fair-replication-artifact-manifest/1.0.0", "artifacts": artifact_index}, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    rate = analysis["physical_arrival_rate"]
    cb = analysis["bootstrap"]["C_minus_B"]; ca = analysis["bootstrap"]["C_minus_A"]
    report = f"""# Event-trigger-aware GPPO fair replication

Experiment ID: `event-trigger-aware-gppo-fair-replication-20260919-v1`  
Status: complete. The pre-registered acceptance gate is **not passed**; this is a retained negative result.

## Outcome

The primary comparison is C - B under W1/W2 with equal condition weight and equal training-seed weight. Physical on-time arrival rates were A `{rate['A']:.4f}`, B `{rate['B']:.4f}`, and C `{rate['C']:.4f}`. C - B was `{cb['point'] * 100:.3f}` percentage points, with the paired-parent 95% bootstrap interval `[{cb['ci95_lower'] * 100:.3f}, {cb['ci95_upper'] * 100:.3f}]` percentage points. The interval lower bound is not above zero and the required 1 percentage-point improvement was not reached. C - A was `{ca['point'] * 100:.3f}` percentage points, interval `[{ca['ci95_lower'] * 100:.3f}, {ca['ci95_upper'] * 100:.3f}]` percentage points.

Two of three seeds had a strictly positive C - B difference, but that criterion alone does not pass the gate. Maximum recorded security violations and illegal actions were both zero. The supported conclusion is therefore only that this protocol and these measured seeds did not establish an improvement from event-trigger training.

## Per-seed W1/W2 rates

| seed | A | B | C | C - B (pp) | C - A (pp) |
|---:|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {seed} | {analysis['per_seed'][str(seed)]['A']['W1']:.4f}/{analysis['per_seed'][str(seed)]['A']['W2']:.4f} | {analysis['per_seed'][str(seed)]['B']['W1']:.4f}/{analysis['per_seed'][str(seed)]['B']['W2']:.4f} | {analysis['per_seed'][str(seed)]['C']['W1']:.4f}/{analysis['per_seed'][str(seed)]['C']['W2']:.4f} | {analysis['per_seed'][str(seed)]['C_minus_B'] * 100:.3f} | {analysis['per_seed'][str(seed)]['C_minus_A'] * 100:.3f} |" for seed in (1101, 2203, 3307)) + f"""

## Training and evaluation budget

All six groups completed 8192 verified environment steps. Policy optimizer calls were 63 per group (378 total, below the 64-call maximum per group); world optimizer updates were zero. Evaluation completed 1728 unique episodes using 24437 verified environment steps and zero optimizer calls. Evaluation cost was:

| config | env steps | actor calls | continuation steps | world forwards |
|---|---:|---:|---:|---:|
| A | {analysis['evaluation_cost']['A']['environment_steps']} | {analysis['evaluation_cost']['A']['actor_calls']} | {analysis['evaluation_cost']['A']['continuation_steps']} | {analysis['evaluation_cost']['A']['world_forward_calls']} |
| B | {analysis['evaluation_cost']['B']['environment_steps']} | {analysis['evaluation_cost']['B']['actor_calls']} | {analysis['evaluation_cost']['B']['continuation_steps']} | {analysis['evaluation_cost']['B']['world_forward_calls']} |
| C | {analysis['evaluation_cost']['C']['environment_steps']} | {analysis['evaluation_cost']['C']['actor_calls']} | {analysis['evaluation_cost']['C']['continuation_steps']} | {analysis['evaluation_cost']['C']['world_forward_calls']} |

The actor-call reduction in B/C is not reported as a reduction in total compute cost. Decision latency includes observation encoding, trigger decision, actor/world inference, hidden/cache action transition and action selection; environment stepping and persistence I/O are excluded. Pooled P95/P99 were `{analysis['latency']['pooled_p95_ns'] / 1e6:.3f}`/`{analysis['latency']['pooled_p99_ns'] / 1e6:.3f}` ms over `{analysis['latency']['pooled_count']}` decisions.

## Replay repair evidence

The preserved real failure prefix identifies episode 8, scenario `train-mixed-seed-851003-W2`, with the first divergence at step 3: action 1, `submit_command=false`, actor decision false, public new-command mask false, but `continuation_actions=[1]` and valid continuation feedback `reuse_existing`. Step 12 repeats the same class for action 22. The repair was therefore exactly the suspected semantic error: replay used the new-command mask to reject a legal continuation. It was not inferred from a masked action alone; the paired ledger explicitly restored decision type.

The repaired contract keeps current public command-mask checks for new actor actions, checks continuations only against the saved public continuation set and `by_action`, and uses the same hidden transition function online and during replay. It does not delete safety checks, turn continuation into a new submission, use NOOP fallback, call `env.step`, resample, read future truth, clear hidden state, or call optimizer during replay. The real prefix replay had 15 steps, zero environment steps, zero optimizer calls, exact discrete fields, and maximum stored-vs-rebuilt policy/world hidden difference 0.0. Targeted fixtures passed 14 tests in the launch check, including legal continuation with a false new mask, rejection of illegal new assignment and invalid continuation, and episode-boundary decision-type preservation.

## Provenance and publication

The three original WD checkpoint hashes, source snapshot manifest, new training/evaluation tape manifests, SQLite budget database, checkpoints, raw episode ledger, analysis and evidence hashes are in `artifact-manifest.json`. The old development validation, historical failure runs and old budget database remain excluded from the independent result. No tuning, seed replacement, tape replacement or repeated evaluation was performed.

Git publication is a separate step from local archival; signature retry and upload claims are reported separately in the final task response.
"""
    (ROOT / "final-report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(ROOT / "final-report.md"), "artifact_manifest": str(ROOT / "artifact-manifest.json"), "artifact_count": len(artifact_index)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
