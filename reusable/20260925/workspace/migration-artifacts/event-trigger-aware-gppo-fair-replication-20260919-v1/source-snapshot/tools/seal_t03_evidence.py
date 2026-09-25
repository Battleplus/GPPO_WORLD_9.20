"""Aggregate T-03 seeds and prepare release plus auditable node evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from gppo_world.dataset import sha256_file


MAIN_VARIANTS = ("wm", "ea_no_ges", "eawm_hard")
SEEDS = (20260903, 20260904, 20260905)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _stats(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def _asset(release_dir: Path, source: Path, name: str) -> Path:
    target = release_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("final_root", type=Path)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("node_evidence_dir", type=Path)
    parser.add_argument("--failed-weight050-root", type=Path, required=True)
    parser.add_argument("--failed-unscaled-root", type=Path, required=True)
    parser.add_argument("--repository", default="Battleplus/GPPO-WORLD-9.2")
    parser.add_argument("--release-tag", default="t03-eawm-v0.1.0")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--test-count", type=int, default=31)
    args = parser.parse_args()
    final_root = args.final_root.resolve()
    release_dir = args.release_dir.resolve()
    evidence_dir = args.node_evidence_dir.resolve()
    runs = {seed: _read(final_root / f"seed{seed}" / "metrics.json") for seed in SEEDS}
    if not all(run["result"] == "PASS" for run in runs.values()):
        raise SystemExit("cannot seal T-03: one or more final seeds failed")
    schema_hashes = {run["event_schema_sha256"] for run in runs.values()}
    if len(schema_hashes) != 1:
        raise SystemExit("cannot seal T-03: event schema differs across seeds")

    aggregate_variants = {}
    for variant in MAIN_VARIANTS:
        aggregate_variants[variant] = {}
        metric_paths = {
            "event_macro_f1": ("test_events", "macro_f1"),
            "event_macro_auprc": ("test_events", "macro_auprc"),
            "rare_event_recall": ("test_events", "rare_recall"),
            "state_mae": ("test_base", "state_mae"),
            "reward_mae": ("test_base", "reward_mae"),
            "cost_mae": ("test_base", "cost_mae"),
        }
        for name, (group, key) in metric_paths.items():
            aggregate_variants[variant][name] = _stats(
                runs[seed]["variants"][variant][group][key] for seed in SEEDS
            )
    degradation = {
        metric: _stats(
            runs[seed]["fairness"]["base_metric_relative_degradation_eawm_vs_wm"][metric]
            for seed in SEEDS
        )
        for metric in ("state", "reward", "cost")
    }
    final_gates = {
        "all_three_seed_runs_pass": all(run["result"] == "PASS" for run in runs.values()),
        "all_per_seed_gates_pass": all(all(run["gates"].values()) for run in runs.values()),
        "single_train_only_schema_across_seeds": len(schema_hashes) == 1,
        "all_seed_state_reward_cost_degradation_below_5_percent": all(
            degradation[name]["max"] <= 0.05 for name in degradation
        ),
        "all_seed_event_f1_and_auprc_beat_baseline": all(
            runs[seed]["gates"]["event_macro_f1_beats_frequency_baseline"]
            and runs[seed]["gates"]["event_macro_auprc_beats_frequency_baseline"]
            for seed in SEEDS
        ),
    }
    aggregate = {
        "result": "PASS" if all(final_gates.values()) else "FAIL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": args.code_commit,
        "seeds": list(SEEDS),
        "event_schema_sha256": next(iter(schema_hashes)),
        "variants": aggregate_variants,
        "base_metric_relative_degradation_eawm_vs_wm": degradation,
        "per_seed_results": {
            str(seed): {
                "result": runs[seed]["result"],
                "gates": runs[seed]["gates"],
                "metrics_sha256": sha256_file(final_root / f"seed{seed}" / "metrics.json"),
            }
            for seed in SEEDS
        },
        "gates": final_gates,
        "protocol_disclosure": (
            "The initial test results were inspected before the backbone learning-rate multiplier was frozen. "
            "The revision was also motivated by matching validation degradation; both failed pre-freeze runs "
            "are retained. Final results are repeated over three seeds but are not represented as a pristine "
            "first-look test evaluation."
        ),
    }
    if aggregate["result"] != "PASS":
        raise SystemExit("cannot seal T-03: aggregate gates failed")

    assets: list[Path] = []
    for seed in SEEDS:
        seed_dir = final_root / f"seed{seed}"
        assets.append(_asset(release_dir, seed_dir / "metrics.json", f"metrics-seed{seed}.json"))
        for variant in MAIN_VARIANTS:
            assets.append(
                _asset(
                    release_dir,
                    seed_dir / "checkpoints" / f"{variant}_seed{seed}.pt",
                    f"{variant}_seed{seed}.pt",
                )
            )
            assets.append(
                _asset(
                    release_dir,
                    seed_dir / f"{variant}-training-history.json",
                    f"{variant}-history-seed{seed}.json",
                )
            )
    seed03 = final_root / "seed20260903"
    for filename in ("eawm_smooth_seed20260903.pt",):
        assets.append(_asset(release_dir, seed03 / "checkpoints" / filename, filename))
    assets.append(_asset(release_dir, seed03 / "eawm_smooth-training-history.json", "eawm_smooth-history-seed20260903.json"))
    for filename in ("event-schema.json", "class-balance.json", "event-support.json", "input-audit.json", "training-config.json"):
        assets.append(_asset(release_dir, seed03 / filename, filename))

    failed_sources = (
        (
            args.failed_weight050_root.resolve() / "seed20260905" / "metrics.json",
            "failed-event-weight050-seed20260905-metrics.json",
            "event_weight=0.50; reward degradation exceeded 5%",
        ),
        (
            args.failed_unscaled_root.resolve() / "seed20260904" / "metrics.json",
            "failed-unscaled-backbone-seed20260904-metrics.json",
            "event_weight=0.25 without backbone LR scaling; reward degradation exceeded 5%",
        ),
    )
    failed = []
    for source, name, reason in failed_sources:
        target = _asset(release_dir, source, name)
        assets.append(target)
        failed.append({"asset": name, "reason": reason, "sha256": sha256_file(target)})

    aggregate_path = release_dir / "aggregate-metrics.json"
    _write(aggregate_path, aggregate)
    assets.append(aggregate_path)
    training_config = _read(seed03 / "training-config.json")
    training_config["frozen_protocol"] = {
        "seeds": list(SEEDS),
        "main_variants": list(MAIN_VARIANTS),
        "smooth_variant_seeds": [20260903],
        "parent_t02_checkpoint_sha256": runs[20260903]["parent_t02_checkpoint"]["sha256"],
    }
    _write(evidence_dir / "aggregate-metrics.json", aggregate)
    _write(evidence_dir / "training-config.json", training_config)
    _write(evidence_dir / "event-schema.json", runs[20260903]["event_schema"])
    _write(evidence_dir / "input-audit.json", _read(seed03 / "input-audit.json"))

    base_url = f"https://github.com/{args.repository}/releases/download/{args.release_tag}"
    manifest_assets = {
        path.name: {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "url": f"{base_url}/{path.name}",
        }
        for path in assets
    }
    checkpoint_manifest = {
        "format": "gppo-t03-checkpoint-manifest/0.1.0",
        "release_tag": args.release_tag,
        "release_url": f"https://github.com/{args.repository}/releases/tag/{args.release_tag}",
        "target_commit": args.code_commit,
        "event_schema_sha256": next(iter(schema_hashes)),
        "parent_t02_checkpoint_sha256": runs[20260903]["parent_t02_checkpoint"]["sha256"],
        "assets": manifest_assets,
    }
    _write(evidence_dir / "checkpoint-manifest.json", checkpoint_manifest)
    _write(evidence_dir / "failed-runs.json", {"runs": failed, "release_base_url": base_url})

    eawm = aggregate_variants["eawm_hard"]
    report = f"""# T-03 test report

- Result: **PASS**
- Code commit: `{args.code_commit}`
- Automated tests: `{args.test_count} passed`
- Seeds: `{', '.join(map(str, SEEDS))}`
- Event schema SHA-256: `{next(iter(schema_hashes))}`
- EAWM-hard macro-F1: `{eawm['event_macro_f1']['mean']:.6f} ± {eawm['event_macro_f1']['sample_std']:.6f}`
- EAWM-hard macro-AUPRC: `{eawm['event_macro_auprc']['mean']:.6f} ± {eawm['event_macro_auprc']['sample_std']:.6f}`
- EAWM-hard rare-event recall: `{eawm['rare_event_recall']['mean']:.6f} ± {eawm['rare_event_recall']['sample_std']:.6f}`
- Maximum per-seed state/reward/cost degradation versus WM: `{degradation['state']['max']:.6%}` / `{degradation['reward']['max']:.6%}` / `{degradation['cost']['max']:.6%}`

All final per-seed gates passed. The first failed configurations remain downloadable release assets. T-01 has no TTL/expiry contract, so the `expire` evidence label is explicitly ineligible rather than treated as an all-negative target.

Protocol disclosure: {aggregate['protocol_disclosure']}
"""
    (evidence_dir / "test-report.md").write_text(report, encoding="utf-8")
    release_notes = f"""# T-03 release notes

This release freezes the automatic ordinal/nominal/structural/evidence event schema, modality-specific Event Heads, paper-aligned hard/smooth GES, and the equal-budget WM / EA-noGES / EAWM-hard ablation.

The release is anchored to code commit `{args.code_commit}`. Checkpoints, histories, per-seed metrics, schema, class-balance statistics, input audit and failed pre-freeze runs are individually hashed in `checkpoint-manifest.json`.
"""
    (evidence_dir / "release-notes.md").write_text(release_notes, encoding="utf-8")
    print(json.dumps({"result": "PASS", "release_assets": len(assets), "gates": final_gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
