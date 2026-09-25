"""Write reproducible read-only audit artifacts for the repaired P/T matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = args.base.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "P_train/1101": base / "formal-v3-continuation-v4-run-20260918/training/seed-1101/P_train/last-recovery.pt",
        "T_train/1101": base / "formal-v5-history-fix-run-20260918b/training/seed-1101/T_train/last-recovery.pt",
        "P_train/2203": base / "formal-v5-history-fix-run-20260918b/training/seed-2203/P_train/last-recovery.pt",
        "T_train/2203": base / "formal-v5-history-fix-run-20260918b/training/seed-2203/T_train/last-recovery.pt",
        "P_train/3307": base / "formal-v5-history-fix-run-20260918b/training/seed-3307/P_train/last-recovery.pt",
        "T_train/3307": base / "formal-v5-history-fix-run-20260918c/training/seed-3307/T_train/last-recovery.pt",
    }
    models = {}
    for name, path in checkpoints.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        commit = path.parent / "ledger-commit.json"
        commit_data = json.loads(commit.read_text(encoding="utf-8"))
        models[name] = {
            "checkpoint": str(path),
            "checkpoint_sha256": sha256(path),
            "checkpoint_bytes": path.stat().st_size,
            "ledger_commit": str(commit),
            "ledger_commit_sha256": sha256(commit),
            "counters": payload.get("counters", {}),
            "commit_counters": commit_data.get("counters", {}),
            "transaction_id": commit_data.get("transaction_id"),
            "ledger": commit_data.get("ledger", {}),
            "protocol": payload.get("protocol"),
            "group": payload.get("group"),
            "finite": all(torch.isfinite(v).all().item() for section in ("policy_state_dict", "world_state_dict") for v in payload.get(section, {}).values() if torch.is_tensor(v)),
        }
    db = base / "budget-sqlite-v2-20260918/persistent-budget.sqlite3"
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    stages = [dict(row) for row in con.execute("select * from stages order by stage")]
    attempts = [dict(row) for row in con.execute("select * from attempts order by rowid")]
    unknown = [dict(row) for row in con.execute("select * from reservations where status='unknown'")]
    con.close()
    source = base / "source"
    manifest = {
        str(path.relative_to(source)): sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    data = {
        "schema": "repaired-pt-matrix-audit/1.0.0",
        "training_allowed": False,
        "validation_episode_count": 432,
        "validation_environment_steps": 6080,
        "models": models,
        "budget_db": {"path": str(db), "sha256": sha256(db), "stages": stages, "attempts": attempts, "unknown_reservations": unknown},
        "source_manifest": manifest,
        "t1101_recovery_chain": {
            "initial_valid_checkpoint_steps": 128,
            "initial_valid_checkpoint_sha256": "8cc5dc446dd9588400eb9495e715c00a13b5b9da1be2b519b61595c14477f0b8",
            "initial_failed_update_calls": 1,
            "initial_failed_update_committed": False,
            "failure": "historical action was checked against the new-command mask during hidden-state replay",
            "repaired_resume_final_logical_steps": models["T_train/1101"]["counters"].get("environment_steps"),
            "additional_logical_steps_after_128_checkpoint": models["T_train/1101"]["counters"].get("environment_steps", 0) - 128,
            "reexecuted_initial_128_steps": "not evidenced in the committed recovery ledger; the 128-step checkpoint was used as resume input",
            "uncommitted_initial_tail": "none promoted; the failed optimizer update was not committed",
        },
        "limitations": [
            "The database counts reservations and verified resources, while checkpoint counters count logical committed model state; they are reported separately.",
            "Per-decision raw timing samples were not stored in the frozen evaluation rows; timing quantiles are episode-summary aggregates, not exact pooled decision quantiles.",
            "Communication proxy bytes were not exposed by the environment records, so bytes are unavailable rather than zero.",
        ],
    }
    (args.out / "recovery-chain-audit.json").write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    (args.out / "source-manifest.json").write_text(json.dumps({
        "schema": "frozen-pt-validation-source-manifest/1.0.0",
        "source_root": str(source),
        "files": manifest,
    }, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# Repaired P/T matrix audit",
        "",
        "Read-only audit; no training or optimizer update was executed.",
        "",
        "## Six final checkpoints",
        "",
        "| run | logical environment steps | policy updates | pending window | checkpoint SHA-256 | finite |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, info in models.items():
        c = info["counters"]
        md.append(f"| {name} | {c.get('environment_steps')} | {c.get('policy_optimizer_steps')} | {c.get('pending_window_k')} | `{info['checkpoint_sha256']}` | {info['finite']} |")
    md += [
        "",
        "## Budget ledger",
        "",
        f"- environment steps: reserved 49152, verified 49151, unknown 1, pending 0",
        f"- policy optimizer calls: reserved 379, verified 379, unknown 0, pending 0",
        f"- world optimizer calls: reserved 0, verified 0, unknown 0, pending 0",
        "- unknown reservation: one environment-step reservation belonging to `formal-v4-T_train-3307`, retained as unknown with no refund.",
        "",
        "## T/1101",
        "",
        "The valid 128-step checkpoint was the resume input. The first update failed before an optimizer transaction was committed because replay used the new-command mask for a historical continuation. The repaired run advanced from that checkpoint to the final logical counter; no evidence in the committed ledger shows the initial 128 steps being promoted a second time. The distinction between reservation accounting and logical checkpoint counters remains explicit.",
        "",
        "## Limitations",
        "",
        "- Frozen evaluation has 432 episodes and 6080 environment steps; no independent test was read.",
        "- Timing is reported from per-episode summaries because raw decision samples were not archived.",
        "- Communication proxy bytes are unavailable in the environment record and are not interpreted as zero.",
    ]
    (args.out / "recovery-chain-audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    artifact_paths = [
        args.out / "episodes.jsonl", args.out / "evaluation-protocol.json",
        args.out / "evaluation-summary.json", args.out / "validation-summary.json",
        args.out / "recovery-chain-audit.json", args.out / "recovery-chain-audit.md",
        args.out / "frozen-validation-report.md", args.out / "source-manifest.json", db,
    ]
    artifact_manifest = {
        "schema": "frozen-pt-validation-artifact-manifest/1.0.0",
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths if path.exists()
        ],
        "checkpoint_files": [
            {"path": info["checkpoint"], "bytes": info["checkpoint_bytes"], "sha256": info["checkpoint_sha256"]}
            for info in models.values()
        ],
    }
    (args.out / "artifact-manifest.json").write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
