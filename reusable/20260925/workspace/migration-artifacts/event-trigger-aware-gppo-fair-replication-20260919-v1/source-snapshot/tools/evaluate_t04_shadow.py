"""Calibrate and evaluate the read-only T-04 Shadow runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from gppo_world.calibration import (
    ShadowCalibration,
    evaluate_shadow_calibration,
    fit_shadow_calibration,
    input_ood_score,
)
from gppo_world.contracts import GraphSnapshot
from gppo_world.data import audit_training_inputs, group_episodes, load_jsonl
from gppo_world.dataset import sha256_file
from gppo_world.model import EventAwareGraphWorldModel
from gppo_world.shadow import ShadowRequest, ShadowRuntime, graph_snapshot_sha256


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _synthetic_ood(graph: GraphSnapshot) -> GraphSnapshot:
    return GraphSnapshot(
        nodes={name: value + 3.0 for name, value in graph.nodes.items()},
        edge_index=graph.edge_index,
        edge_attr={relation: value + 3.0 for relation, value in graph.edge_attr.items()},
        candidate_edges=graph.candidate_edges,
        action_mask=graph.action_mask,
        graph_version=graph.graph_version,
    )


def _auroc(negative_scores, positive_scores) -> float:
    negative = np.asarray(negative_scores, dtype=np.float64)
    positive = np.asarray(positive_scores, dtype=np.float64)
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def _request(item) -> ShadowRequest:
    executed_action = item.action if item.execution_accepted else None
    return ShadowRequest(
        episode_id=item.episode_id,
        step=item.step,
        graph=item.graph,
        executed_action=executed_action,
        evidence=item.evidence,
        action_version=item.action_version,
        decision_time=item.decision_time,
        execution_accepted=item.execution_accepted,
        expected_post_graph_version=item.next_graph.graph_version,
        expected_post_action_version=item.action_version + 1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-version", default="t03-eawm-v0.1.0/eawm_hard_seed20260903")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    dataset_dir = args.dataset_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    manifest_path = (args.manifest or dataset_dir.parent / "dataset-manifest.json").resolve()
    audit = audit_training_inputs(manifest_path, dataset_dir)
    _write(output / "input-audit.json", audit)
    if not audit["passed"]:
        return 2
    episodes = {
        split: group_episodes(load_jsonl(dataset_dir / f"{split}.jsonl"))
        for split in ("train", "validation", "test")
    }
    model, checkpoint_metadata = EventAwareGraphWorldModel.load(checkpoint)
    calibration = fit_shadow_calibration(model, episodes["train"], episodes["validation"])
    _write(output / "calibration.json", calibration.to_dict())
    calibration_metrics = evaluate_shadow_calibration(model, episodes["test"], calibration)

    runtime = ShadowRuntime(model, calibration, model_version=args.model_version)
    records = []
    input_mutations = 0
    for episode in episodes["test"]:
        runtime.reset()
        for item in episode:
            before = graph_snapshot_sha256(item.graph)
            result = runtime.observe(_request(item))
            after = graph_snapshot_sha256(item.graph)
            input_mutations += int(before != after)
            records.append(asdict(result))
    counters = runtime.counters
    latency = np.asarray([record["latency_ms"] for record in records], dtype=np.float64)
    latency_metrics = {
        "count": len(latency),
        "mean_ms": float(latency.mean()),
        "p50_ms": float(np.quantile(latency, 0.50)),
        "p95_ms": float(np.quantile(latency, 0.95)),
        "p99_ms": float(np.quantile(latency, 0.99)),
        "max_ms": float(latency.max()),
        "p95_budget_ms": calibration.latency_p95_budget_ms,
        "p99_budget_ms": calibration.latency_p99_budget_ms,
    }

    flat_test = [item for episode in episodes["test"] for item in episode]
    id_scores = [input_ood_score(item.graph, calibration) for item in flat_test]
    ood_graphs = [_synthetic_ood(item.graph) for item in flat_test]
    ood_scores = [input_ood_score(graph, calibration) for graph in ood_graphs]
    ood_metrics = {
        "method": "train moments + validation q99.5 threshold; synthetic +3 normalized-feature stress",
        "id_count": len(id_scores),
        "ood_count": len(ood_scores),
        "id_mean": float(np.mean(id_scores)),
        "ood_mean": float(np.mean(ood_scores)),
        "threshold": calibration.ood_score_threshold,
        "id_false_positive_rate": float(np.mean(np.asarray(id_scores) > calibration.ood_score_threshold)),
        "ood_recall": float(np.mean(np.asarray(ood_scores) > calibration.ood_score_threshold)),
        "auroc": _auroc(id_scores, ood_scores),
        "scope_limit": "synthetic feature-range OOD only; not evidence of real unseen mission generalization",
    }

    sample = flat_test[0]
    request = _request(sample)
    stale_before = ShadowRuntime(model, calibration, model_version=args.model_version).observe(
        request,
        version_reader=lambda: (
            request.expected_post_graph_version + 1,
            request.expected_post_action_version,
        ),
    )
    versions = iter(
        (
            (request.expected_post_graph_version, request.expected_post_action_version),
            (request.expected_post_graph_version + 1, request.expected_post_action_version),
        )
    )
    stale_after = ShadowRuntime(model, calibration, model_version=args.model_version).observe(
        request, version_reader=lambda: next(versions)
    )
    timeout = ShadowRuntime(model, calibration, model_version=args.model_version).observe(
        request, latency_injection_ms=calibration.timeout_ms + 1.0
    )
    exception = ShadowRuntime(model, calibration, model_version=args.model_version).observe(
        request, force_exception=True
    )
    ood_request = replace(request, graph=ood_graphs[0])
    ood = ShadowRuntime(model, calibration, model_version=args.model_version).observe(ood_request)
    low_uncertainty_calibration = ShadowCalibration.from_dict(
        {**calibration.to_dict(), "uncertainty_threshold": -1.0}
    )
    high_uncertainty = ShadowRuntime(
        model, low_uncertainty_calibration, model_version=args.model_version
    ).observe(request)
    injected = {
        "stale_before": asdict(stale_before),
        "stale_after": asdict(stale_after),
        "timeout": asdict(timeout),
        "exception": asdict(exception),
        "ood": asdict(ood),
        "high_uncertainty": asdict(high_uncertainty),
    }
    injection_expected = {
        "stale_before": "stale_before",
        "stale_after": "stale_after",
        "timeout": "timeout",
        "exception": "exception",
        "ood": "ood",
        "high_uncertainty": "high_uncertainty",
    }
    fallback_injections_pass = all(
        not injected[name]["valid"]
        and injected[name]["fallback_reason"] == expected
        and not any(injected[name]["latent"])
        for name, expected in injection_expected.items()
    )
    write_names = (
        "belief_write_count",
        "action_mask_write_count",
        "graph_version_write_count",
        "action_version_write_count",
        "action_submission_count",
    )
    gates = {
        "strict_input_audit_passed": audit["passed"],
        "calibration_uses_no_test_statistics": "test" not in calibration.source_split,
        "state_change_ece_not_worse_by_more_than_0_01": (
            calibration_metrics["state_change"]["calibrated_ece"]
            <= calibration_metrics["state_change"]["raw_ece"] + 0.01
        ),
        "state_change_brier_not_worse_by_more_than_0_005": (
            calibration_metrics["state_change"]["calibrated_brier"]
            <= calibration_metrics["state_change"]["raw_brier"] + 0.005
        ),
        "continuation_ece_not_worse_by_more_than_0_01": (
            calibration_metrics["continuation"]["calibrated_ece"]
            <= calibration_metrics["continuation"]["raw_ece"] + 0.01
        ),
        "low_risk_half_has_lower_state_mae": calibration_metrics["low_risk_half_beats_full"],
        "synthetic_ood_auroc_at_least_0_95": ood_metrics["auroc"] >= 0.95,
        "synthetic_ood_recall_at_least_0_95": ood_metrics["ood_recall"] >= 0.95,
        "fallback_injections_all_zero_context": fallback_injections_pass,
        "belief_mask_version_and_action_writes_zero": all(counters[name] == 0 for name in write_names),
        "input_snapshot_mutations_zero": input_mutations == 0 and counters["input_mutation_count"] == 0,
        "latency_p95_within_budget": latency_metrics["p95_ms"] <= calibration.latency_p95_budget_ms,
        "latency_p99_within_budget": latency_metrics["p99_ms"] <= calibration.latency_p99_budget_ms,
        "required_pressure_profiles_reported": {
            "weak_comm", "long_gap", "burst"
        }.issubset(calibration_metrics["profile_state_mae"]),
    }
    bundle = {
        "format": "gppo-shadow-bundle/0.1.0",
        "model_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "model_version": args.model_version,
        "event_schema_sha256": model.event_schema.sha256(),
        "calibration": calibration.to_dict(),
        "calibration_sha256": calibration.sha256(),
        "fallback_policy": {
            "reasons": list(injection_expected.values()),
            "output": "88-dimensional zero context",
            "commit_hidden_on_fallback": False,
        },
    }
    _write(output / "accepted-shadow-bundle.json", bundle)
    _write(output / "shadow-records.json", records)
    _write(output / "fallback-injections.json", injected)
    result = {
        "result": "PASS" if all(gates.values()) else "FAIL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "checkpoint_metadata_keys": sorted(checkpoint_metadata),
        "calibration_sha256": calibration.sha256(),
        "calibration_metrics": calibration_metrics,
        "ood": ood_metrics,
        "latency": latency_metrics,
        "shadow_counters": counters,
        "input_mutation_count_external_audit": input_mutations,
        "fallback_injections": {
            name: {
                "valid": value["valid"],
                "fallback_reason": value["fallback_reason"],
                "zero_context": not any(value["latent"]),
            }
            for name, value in injected.items()
        },
        "gates": gates,
        "limitations": [
            "OOD Gate is validated on synthetic feature-range shift, not a truly unseen mission distribution.",
            "T-01 train/validation/test all contain the same seven scenario profile families.",
            "Shadow produces no action and no claimed GPPO return improvement; that remains T-05.",
        ],
    }
    _write(output / "metrics.json", result)
    print(json.dumps({"result": result["result"], "gates": gates, "latency": latency_metrics, "ood": ood_metrics}, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
