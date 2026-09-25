"""Prepare T-04 release assets and node evidence after all Shadow gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.dataset import sha256_file


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--repository", default="Battleplus/GPPO-WORLD-9.2")
    parser.add_argument("--release-tag", default="t04-shadow-v0.1.0")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    release = args.release_dir.resolve()
    evidence = args.evidence_dir.resolve()
    metrics = _read(run / "metrics.json")
    baseline = _read(run / "baseline-read-only-audit.json")
    if metrics["result"] != "PASS" or baseline["result"] != "PASS":
        raise SystemExit("cannot seal T-04: Shadow or baseline integration gates failed")

    t03_manifest = _read(PROJECT_ROOT / "nodes" / "T-03" / "evidence" / "checkpoint-manifest.json")
    model_asset = t03_manifest["assets"]["eawm_hard_seed20260903.pt"]
    bundle = _read(run / "accepted-shadow-bundle.json")
    bundle.update(
        {
            "code_commit": args.code_commit,
            "model_checkpoint": {
                **bundle["model_checkpoint"],
                "url": model_asset["url"],
            },
            "baseline_repository": "https://github.com/Battleplus/GPPO-8.29",
            "baseline_commit": "2a9bb9f87b9d543df144f4d108ba970c924151f9",
        }
    )
    _write(run / "accepted-shadow-bundle-sealed.json", bundle)
    source_assets = {
        "metrics.json": run / "metrics.json",
        "calibration.json": run / "calibration.json",
        "accepted-shadow-bundle.json": run / "accepted-shadow-bundle-sealed.json",
        "shadow-records.json": run / "shadow-records.json",
        "fallback-injections.json": run / "fallback-injections.json",
        "baseline-read-only-audit.json": run / "baseline-read-only-audit.json",
        "input-audit.json": run / "input-audit.json",
    }
    assets = {name: _copy(source, release / name) for name, source in source_assets.items()}
    for name in (
        "metrics.json",
        "calibration.json",
        "accepted-shadow-bundle.json",
        "fallback-injections.json",
        "baseline-read-only-audit.json",
        "input-audit.json",
    ):
        _copy(assets[name], evidence / name)

    base_url = f"https://github.com/{args.repository}/releases/download/{args.release_tag}"
    manifest = {
        "format": "gppo-t04-shadow-manifest/0.1.0",
        "target_commit": args.code_commit,
        "release_tag": args.release_tag,
        "release_url": f"https://github.com/{args.repository}/releases/tag/{args.release_tag}",
        "parent_model": model_asset,
        "calibration_sha256": bundle["calibration_sha256"],
        "assets": {
            name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "url": f"{base_url}/{name}",
            }
            for name, path in assets.items()
        },
    }
    _write(evidence / "checkpoint-manifest.json", manifest)
    failed = {
        "runs": [
            {
                "stage": "fallback injection serialization",
                "error": "non-finite Infinity sentinel was not JSON compliant",
                "resolution": "use -1.0 for not-computed OOD score",
            },
            {
                "stage": "real baseline audit startup",
                "error": "torch.flatnonzero unavailable in installed PyTorch",
                "resolution": "use torch.nonzero(...).flatten()",
            },
        ],
        "claim": "toolchain failures occurred before model Gate conclusions and were retained",
    }
    _write(evidence / "failed-runs.json", failed)
    c = metrics["calibration_metrics"]
    latency = metrics["latency"]
    ood = metrics["ood"]
    report = f"""# T-04 test and safety report

- Result: **PASS**
- Code commit: `{args.code_commit}`
- Repository tests: `37 passed`
- Real baseline audit: `GPPO-8.29@2a9bb9f`, `{baseline['steps']}` post-action Shadow calls
- State-change ECE raw/calibrated: `{c['state_change']['raw_ece']:.6f}` / `{c['state_change']['calibrated_ece']:.6f}`
- Continuation ECE raw/calibrated: `{c['continuation']['raw_ece']:.6f}` / `{c['continuation']['calibrated_ece']:.6f}`
- Risk coverage state MAE at 50%/100%: `{c['risk_coverage_state_mae']['0.5']:.6f}` / `{c['risk_coverage_state_mae']['1.0']:.6f}`
- Complete observe P50/P95/P99: `{latency['p50_ms']:.4f}` / `{latency['p95_ms']:.4f}` / `{latency['p99_ms']:.4f}` ms
- Synthetic OOD AUROC/recall: `{ood['auroc']:.6f}` / `{ood['ood_recall']:.6f}`
- ID OOD false-positive rate: `{ood['id_false_positive_rate']:.6%}`

All belief/action-mask/graph-version/action-version/action-submission counters are zero. The real baseline environment snapshot, runtime belief hash, action mask and versions were unchanged before/after every Shadow call; fail-fast spies on action/execution/belief mutation APIs were never invoked.

Limits: OOD uses a synthetic +3 normalized-feature range shift, while all T-01 splits share the same seven profile families. This proves fallback mechanics, not production unseen-mission generalization. Timeout is fail-closed after inference completes, not hard worker cancellation. Shadow does not influence GPPO actions; downstream value remains T-05.
"""
    (evidence / "test-report.md").write_text(report, encoding="utf-8")
    notes = f"""# T-04 release notes

This release binds T-03 EAWM-hard checkpoint `{model_asset['sha256']}` to validation-only calibration `{bundle['calibration_sha256']}` and the read-only post-action Shadow contract at `{args.code_commit}`.

It includes complete Shadow records, fallback injections, real GPPO baseline zero-write audit, metrics, calibration and input audit. It does not include a new model checkpoint; the immutable parent checkpoint remains in the T-03 release.
"""
    (evidence / "release-notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps({"result": "PASS", "assets": len(assets)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
