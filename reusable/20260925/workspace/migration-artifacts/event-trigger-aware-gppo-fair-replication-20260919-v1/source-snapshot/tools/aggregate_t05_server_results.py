"""Aggregate the 12 fixed T-05 held-out results without selecting a winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

GROUPS = ("GPPO", "WM-GPPO", "EA-noGES-GPPO", "EAWM-GPPO")
SEEDS = (1101, 2202, 3303)
METRICS = (
    "event_success_rate",
    "recovery_delay",
    "cumulative_uncovered_time",
    "legal_coverage_rate",
    "final_infeasible_rate",
    "episode_return",
    "fixed_j",
    "repair_count",
    "inference_latency_ms",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    parser.add_argument("output")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--expected-target-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    args = parser.parse_args(argv)
    baseline_root = Path(args.baseline_root).resolve()
    sys.path.insert(0, str(baseline_root))
    from ppo_allocation.random_event.metrics import (  # noqa: PLC0415
        descriptive_statistics,
        paired_metric_report,
    )

    root = Path(args.results_root).resolve()
    paths = list(root.rglob("evaluation.json"))
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        key = (str(value["group"]), int(value["seed"]))
        if key in records:
            raise SystemExit(f"duplicate evaluation for {key}")
        records[key] = value
    expected = {(group, seed) for group in GROUPS for seed in SEEDS}
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise SystemExit(f"expected exactly 12 evaluations; missing={missing}, extra={extra}")
    manifest_hashes = {item["test_manifest_sha256"] for item in records.values()}
    if len(manifest_hashes) != 1 or any(item["tape_count"] != 100 for item in records.values()):
        raise SystemExit("all evaluations must use the same frozen 100-tape Test bank")
    common_provenance = {}
    for key in ("target_commit", "baseline_commit", "config_sha256", "calibration_sha256"):
        values = {item.get(key) for item in records.values()}
        if len(values) != 1 or None in values:
            raise SystemExit(f"all evaluations must share one non-null {key}")
        common_provenance[key] = next(iter(values))
    if common_provenance["target_commit"] != args.expected_target_commit:
        raise SystemExit("aggregate target commit differs from the campaign lock")
    if common_provenance["config_sha256"] != args.expected_config_sha256:
        raise SystemExit("aggregate config SHA-256 differs from the campaign lock")
    for (group, seed), item in records.items():
        metadata = item.get("checkpoint_metadata", {})
        expected = {
            "t05_group": group,
            "training_seed": seed,
            "accepted_decision_steps": 50_000,
            "target_commit": common_provenance["target_commit"],
            "baseline_commit": common_provenance["baseline_commit"],
            "t05_config_sha256": common_provenance["config_sha256"],
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise SystemExit(f"checkpoint provenance mismatch for {group} seed {seed}")
        if metadata.get("world_checkpoint_sha256") != item.get("world_checkpoint_sha256"):
            raise SystemExit(f"world checkpoint provenance mismatch for {group} seed {seed}")
        if group == "GPPO" and item.get("world_checkpoint_sha256") is not None:
            raise SystemExit("GPPO control must not bind a world checkpoint")
        if group != "GPPO" and not item.get("world_checkpoint_sha256"):
            raise SystemExit(f"{group} must bind a world checkpoint")
    for seed in SEEDS:
        reference_ids = [row["tape_id"] for row in records[("GPPO", seed)]["episode_records"]]
        for group in GROUPS:
            ids = [row["tape_id"] for row in records[(group, seed)]["episode_records"]]
            if ids != reference_ids:
                raise SystemExit(f"paired tape order mismatch for {group} seed {seed}")
    for group in GROUPS[1:]:
        for seed in SEEDS:
            latency = records[(group, seed)].get("shadow_latency", {})
            required_latency = (
                "count",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "max_ms",
                "fallback_count",
                "timeout_count",
                "p95_budget_ms",
                "p99_budget_ms",
                "p95_within_budget",
                "p99_within_budget",
            )
            if any(key not in latency for key in required_latency):
                raise SystemExit(f"missing Shadow latency schema for {group} seed {seed}")
            if int(latency["count"]) <= 0 or any(
                not isinstance(latency[key], (int, float))
                for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms")
            ):
                raise SystemExit(f"invalid Shadow latency values for {group} seed {seed}")

    per_seed_effects: dict[str, Any] = {}
    for group in GROUPS[1:]:
        per_seed_effects[group] = {}
        for seed in SEEDS:
            candidate = records[(group, seed)]["episode_records"]
            control = records[("GPPO", seed)]["episode_records"]
            per_seed_effects[group][str(seed)] = {
                metric: paired_metric_report(
                    candidate,
                    control,
                    metric,
                    n_resamples=2000,
                    seed=20260902 + seed,
                )
                for metric in METRICS
            }

    seed_stability: dict[str, Any] = {}
    for group in GROUPS:
        seed_stability[group] = {}
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                summary = records[(group, seed)]["summary"]["metrics"][metric]
                if summary["mean"] is not None:
                    values.append(float(summary["mean"]))
            seed_stability[group][metric] = descriptive_statistics(values)
    effect_stability: dict[str, Any] = {}
    for group in GROUPS[1:]:
        effect_stability[group] = {
            metric: descriptive_statistics(
                [
                    per_seed_effects[group][str(seed)][metric]["mean_difference"]
                    for seed in SEEDS
                    if per_seed_effects[group][str(seed)][metric]["mean_difference"] is not None
                ]
            )
            for metric in METRICS
        }

    safety = {
        "all_shadow_writes_zero": all(
            not any(
                item["safety"][key]
                for key in (
                    "environment_mutations",
                    "belief_mutations",
                    "action_mask_mutations",
                    "version_mutations",
                    "action_submissions_by_shadow",
                )
            )
            for item in records.values()
        ),
        "per_run": {
            f"{group}/seed{seed}": records[(group, seed)]["safety"]
            for group in GROUPS
            for seed in SEEDS
        },
    }
    shadow_latency_gates = {
        f"{group}/seed{seed}": (
            records[(group, seed)].get("shadow_latency", {}).get("p95_within_budget") is True
            and records[(group, seed)].get("shadow_latency", {}).get("p99_within_budget") is True
        )
        for group in GROUPS[1:]
        for seed in SEEDS
    }
    result = {
        "format": "gppo-t05-four-group-ablation/0.1.0",
        "status": "evaluated_no_checkpoint_selection",
        "groups": list(GROUPS),
        "seeds": list(SEEDS),
        "test_manifest_sha256": next(iter(manifest_hashes)),
        "protocol_provenance": common_provenance,
        "per_run_summaries": {
            f"{group}/seed{seed}": records[(group, seed)]["summary"]
            for group in GROUPS
            for seed in SEEDS
        },
        "paired_effects_candidate_minus_gppo": per_seed_effects,
        "seed_stability": seed_stability,
        "paired_effect_seed_stability": effect_stability,
        "resource_and_latency": {
            f"{group}/seed{seed}": {
                "parameter_counts": records[(group, seed)].get("parameter_counts"),
                "shadow_latency": records[(group, seed)].get("shadow_latency"),
            }
            for group in GROUPS
            for seed in SEEDS
        },
        "safety": safety,
        "shadow_latency_gates": {
            "per_run": shadow_latency_gates,
            "all_pass": all(shadow_latency_gates.values()),
        },
        "claim_rule": "A group-level benefit must be visible across independent seeds; a best-seed result is insufficient.",
    }
    if not safety["all_shadow_writes_zero"]:
        raise SystemExit("cannot aggregate: at least one safety counter is non-zero")
    write_json(Path(args.output).resolve(), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
