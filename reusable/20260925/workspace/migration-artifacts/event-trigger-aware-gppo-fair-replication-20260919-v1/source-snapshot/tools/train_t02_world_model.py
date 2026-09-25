"""Train Graph-WM and flat-GRU baseline, then evaluate T-02 gates once."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import inspect
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from gppo_world.data import STATE_DIM, RELATION_ORDER, audit_training_inputs, group_episodes, load_jsonl
from gppo_world.dataset import sha256_file
from gppo_world.model import FlatGRUWorldModel, GraphWorldModel, WorldModelConfig
from gppo_world.registry import FEATURE_REGISTRY, SCHEMA_VERSION
from gppo_world.training import (
    TrainingConfig,
    baseline_metrics,
    calibrate_change_threshold,
    config_dict,
    evaluate_model,
    paired_bootstrap_action_degradation,
    paired_bootstrap_no_action_degradation,
    parameter_count,
    rollout_errors,
    seed_everything,
    train_model,
    uncertainty_diagnostics,
)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _finite(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _budget_matched_flat_config(base: WorldModelConfig, target_parameters: int) -> WorldModelConfig:
    candidates = [replace(base, hidden_dim=hidden) for hidden in range(32, 193)]
    return min(
        candidates,
        key=lambda candidate: abs(parameter_count(FlatGRUWorldModel(candidate)) - target_parameters),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        parser.error("the frozen protocol permits training only on Python 3.10/3.11")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    dataset_dir = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    manifest_path = (args.manifest or dataset_dir.parent / "dataset-manifest.json").resolve()
    input_audit = audit_training_inputs(manifest_path, dataset_dir)
    if not input_audit["passed"]:
        _write(output / "input-audit.json", input_audit)
        print(json.dumps(input_audit, indent=2, sort_keys=True))
        return 2
    _write(output / "input-audit.json", input_audit)
    paths = {split: dataset_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    episodes = {split: group_episodes(load_jsonl(path)) for split, path in paths.items()}
    training_config = TrainingConfig(seed=args.seed, epochs=args.epochs, patience=args.patience)
    model_config = WorldModelConfig()

    seed_everything(args.seed)
    graph_candidate = GraphWorldModel(model_config)
    started = time.perf_counter()
    graph_model, graph_history, graph_selection = train_model(
        graph_candidate, episodes["train"], episodes["validation"], training_config
    )
    graph_seconds = time.perf_counter() - started
    graph_calibration = calibrate_change_threshold(graph_model, episodes["validation"])
    graph_selection["change_threshold_calibration"] = graph_calibration
    flat_config = _budget_matched_flat_config(model_config, parameter_count(graph_model))
    seed_everything(args.seed)
    flat_candidate = FlatGRUWorldModel(flat_config)
    started = time.perf_counter()
    flat_model, flat_history, flat_selection = train_model(
        flat_candidate, episodes["train"], episodes["validation"], training_config
    )
    flat_seconds = time.perf_counter() - started
    flat_calibration = calibrate_change_threshold(flat_model, episodes["validation"])
    flat_selection["change_threshold_calibration"] = flat_calibration

    graph_test = evaluate_model(graph_model, episodes["test"])
    flat_test = evaluate_model(flat_model, episodes["test"])
    last_value = baseline_metrics(episodes["train"], episodes["test"])
    action_shuffle = paired_bootstrap_action_degradation(graph_model, episodes["test"], seed=args.seed)
    no_action = paired_bootstrap_no_action_degradation(graph_model, episodes["test"], seed=args.seed)
    uncertainty = uncertainty_diagnostics(graph_model, episodes["test"])
    multistep = rollout_errors(graph_model, episodes["test"])
    graph_path = output / "checkpoints" / f"graph_wm_seed{args.seed}.pt"
    flat_path = output / "checkpoints" / f"flat_gru_seed{args.seed}.pt"
    graph_model.save(
        graph_path,
        extra={
            "training_config": config_dict(training_config),
            "selection": graph_selection,
            "test_metrics": graph_test,
            "source_baseline_commit": "2a9bb9f87b9d543df144f4d108ba970c924151f9",
        },
    )
    flat_model.save(
        flat_path,
        extra={
            "training_config": config_dict(training_config),
            "selection": flat_selection,
            "test_metrics": flat_test,
            "source_baseline_commit": "2a9bb9f87b9d543df144f4d108ba970c924151f9",
        },
    )
    restored, metadata = GraphWorldModel.load(graph_path)
    restored.eval()
    sample = episodes["test"][0][0]
    expected, _ = graph_model.step(sample.graph, sample.action, sample=False)
    actual, _ = restored.step(sample.graph, sample.action, sample=False)
    roundtrip_max_abs = float((expected["state_delta"] - actual["state_delta"]).abs().max())
    flat_restored, flat_metadata = FlatGRUWorldModel.load(flat_path)
    flat_restored.eval()
    flat_expected, _ = flat_model.step(sample.graph, sample.action, sample=False)
    flat_actual, _ = flat_restored.step(sample.graph, sample.action, sample=False)
    flat_roundtrip_max_abs = float((flat_expected["state_delta"] - flat_actual["state_delta"]).abs().max())
    graph_parameters = parameter_count(graph_model)
    flat_parameters = parameter_count(flat_model)
    parameter_gap_ratio = abs(graph_parameters - flat_parameters) / graph_parameters

    gates = {
        "strict_split_manifest_revalidated": input_audit["passed"],
        "future_interval_input_absent": "delta_time" not in inspect.signature(GraphWorldModel.step).parameters,
        "all_registered_relations_predicted": set(RELATION_ORDER) == set(FEATURE_REGISTRY.edges),
        "graph_beats_last_value_state_mae": graph_test["state_mae"] < last_value["state_mae"],
        "legal_action_counterfactual_degrades_state_reward_cost": all(
            action_shuffle["metrics"][name]["bootstrap_ci95_low"] > 0.0
            for name in ("state", "reward", "cost")
        ) and action_shuffle["illegal_alternative_count"] == 0,
        "no_action_ablation_reported": no_action["evaluated_transitions"] == graph_test["transition_count"],
        "uncertainty_ranks_error_risk": (
            uncertainty["spearman_rank_correlation"] > 0.0
            and uncertainty["high_vs_low_risk_ratio"] > 1.0
        ),
        "checkpoint_roundtrip_exact": roundtrip_max_abs == 0.0,
        "flat_checkpoint_roundtrip_exact": flat_roundtrip_max_abs == 0.0,
        "flat_parameter_budget_within_5_percent": parameter_gap_ratio <= 0.05,
        "one_three_five_step_reported": set(multistep) == {"1", "3", "5"},
    }
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
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": FEATURE_REGISTRY.sha256(),
        "dataset": {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
                "episodes": len(episodes[split]),
                "transitions": sum(len(episode) for episode in episodes[split]),
            }
            for split, path in paths.items()
        },
        "input_audit": input_audit,
        "training_config": config_dict(training_config),
        "model_config": {"graph_wm": asdict(model_config), "flat_gru": asdict(flat_config)},
        "state_dimension": STATE_DIM,
        "predicted_relations": ["/".join(relation) for relation in RELATION_ORDER],
        "parameters": {
            "graph_wm": graph_parameters,
            "flat_gru": flat_parameters,
            "gap_ratio": parameter_gap_ratio,
        },
        "training_seconds": {"graph_wm": graph_seconds, "flat_gru": flat_seconds},
        "selection": {"graph_wm": graph_selection, "flat_gru": flat_selection},
        "test": {"graph_wm": graph_test, "flat_gru": flat_test, "last_value": last_value},
        "action_shuffle": action_shuffle,
        "no_action": no_action,
        "uncertainty": uncertainty,
        "multi_step": multistep,
        "checkpoint_roundtrip_max_abs": roundtrip_max_abs,
        "flat_checkpoint_roundtrip_max_abs": flat_roundtrip_max_abs,
        "checkpoints": {
            "graph_wm": {"path": str(graph_path), "sha256": sha256_file(graph_path)},
            "flat_gru": {"path": str(flat_path), "sha256": sha256_file(flat_path)},
        },
        "gates": gates,
        "checkpoint_metadata_keys": sorted(metadata),
        "flat_checkpoint_metadata_keys": sorted(flat_metadata),
    }
    _write(output / "metrics.json", _finite(result))
    _write(output / "graph-training-history.json", _finite(graph_history))
    _write(output / "flat-training-history.json", _finite(flat_history))
    print(json.dumps(_finite(result), indent=2, sort_keys=True))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
