"""Evaluate frozen candidate-consequence predictions against simple public baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.consequence_data import ConsequenceExample, load_consequence_jsonl  # noqa: E402
from gppo_world.consequence_model import ConsequenceModelConfig, Graph5ActionConsequenceWorldModel, graph_to_device  # noqa: E402
from gppo_world.dataset import sha256_file  # noqa: E402


HEADS = ("travel_time", "service_progress", "energy_delta", "deadline_risk")
LOWER_IS_BETTER = {"travel_time": True, "service_progress": False, "energy_delta": True, "deadline_risk": True}


def grouped(examples: list[ConsequenceExample]) -> list[list[ConsequenceExample]]:
    values: dict[tuple[str, str], list[ConsequenceExample]] = defaultdict(list)
    for example in examples:
        values[(example.parent_episode_id, example.prefix_id)].append(example)
    return list(values.values())


def training_constants(examples: list[ConsequenceExample]) -> dict[str, float]:
    result = {}
    for head in HEADS:
        values = [float(getattr(item.target, head)) for item in examples if bool(item.masks[head])]
        if not values:
            raise ValueError(f"training split has no valid labels for {head}")
        result[head] = float(np.mean(values))
    return result


def public_physics(example: ConsequenceExample, horizon: float = 6.0) -> dict[str, float]:
    graph, action = example.graph, int(example.target.action)
    uav_count, task_capacity = 4, 6
    idle_power, service_power, travel_power = 0.05, 1.0, 0.35
    energy = 0.0
    for uav in range(uav_count):
        idle_value = float(graph.nodes["uav"][uav, 20])
        idle_valid = float(graph.nodes["uav"][uav, 22]) > 0.5
        energy += horizon * (idle_power if idle_valid and idle_value > 0.5 else service_power)
    if action == 24:
        return {"travel_time": 0.0, "service_progress": 0.0, "energy_delta": energy, "deadline_risk": 0.5}
    uav, task = divmod(action, task_capacity)
    distance = max(0.0, float(graph.candidate_features[action, 0])) * 10.0
    remaining_service = max(0.0, float(graph.nodes["task"][task, 12]))
    service = min(remaining_service, max(0.0, horizon - distance))
    energy -= horizon * idle_power
    energy += min(horizon, distance) * travel_power + service * service_power + max(0.0, horizon - distance - service) * idle_power
    now = float(graph.global_features[0]) * 18.0
    deadline = float(graph.nodes["task"][task, 8])
    risk = float(now + distance + remaining_service > deadline)
    return {"travel_time": distance, "service_progress": service, "energy_delta": energy, "deadline_risk": risk}


def model_predictions(model: Graph5ActionConsequenceWorldModel, examples: list[ConsequenceExample], device: torch.device) -> dict[tuple[str, str, int], dict[str, float]]:
    result = {}
    model.eval()
    with torch.no_grad():
        for members in grouped(examples):
            actions = [item.target.action for item in members]
            graph = graph_to_device(members[0].graph, device)
            history = members[0].history.to(device) if members[0].history is not None else None
            prediction = model.predict_candidates(graph, actions, history=history).as_dict()
            for index, item in enumerate(members):
                key = (item.parent_episode_id, item.prefix_id, item.target.action)
                result[key] = {
                    "travel_time": float(prediction["travel_time"][index].cpu()),
                    "service_progress": float(prediction["service_progress"][index].cpu()),
                    "energy_delta": float(prediction["energy_delta"][index].cpu()),
                    "deadline_risk": float(prediction["deadline_risk"][index].cpu()),
                }
    return result


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = predicted - actual
    return {
        "n": int(actual.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def probability_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    probability = np.clip(predicted, 1e-7, 1.0 - 1e-7)
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        chosen = bins == index
        if np.any(chosen):
            ece += float(np.mean(chosen)) * abs(float(np.mean(probability[chosen])) - float(np.mean(actual[chosen])))
    return {
        "n": int(actual.size),
        "positive": int(np.sum(actual > 0.5)),
        "negative": int(np.sum(actual <= 0.5)),
        "brier": float(np.mean(np.square(probability - actual))),
        "log_loss": float(-np.mean(actual * np.log(probability) + (1.0 - actual) * np.log(1.0 - probability))),
        "ece_10bin": ece,
    }


def ranking_metrics(
    examples: list[ConsequenceExample],
    predictions: dict[tuple[str, str, int], dict[str, float]],
    head: str,
) -> dict[str, float | int]:
    pair_scores, regrets = [], []
    comparable_groups = 0
    for members in grouped(examples):
        valid = [item for item in members if bool(item.masks[head])]
        if len(valid) < 2:
            continue
        comparable_groups += 1
        actual = np.asarray([float(getattr(item.target, head)) for item in valid])
        predicted = np.asarray([predictions[(item.parent_episode_id, item.prefix_id, item.target.action)][head] for item in valid])
        for left in range(len(valid)):
            for right in range(left + 1, len(valid)):
                delta_actual = actual[left] - actual[right]
                if abs(delta_actual) <= 1e-12:
                    continue
                delta_predicted = predicted[left] - predicted[right]
                pair_scores.append(0.5 if abs(delta_predicted) <= 1e-12 else float(np.sign(delta_actual) == np.sign(delta_predicted)))
        selected = int(np.argmin(predicted) if LOWER_IS_BETTER[head] else np.argmax(predicted))
        best = float(np.min(actual) if LOWER_IS_BETTER[head] else np.max(actual))
        regrets.append(float(actual[selected] - best if LOWER_IS_BETTER[head] else best - actual[selected]))
    return {
        "comparable_prefixes": comparable_groups,
        "comparable_pairs": len(pair_scores),
        "pairwise_accuracy": float(np.mean(pair_scores)) if pair_scores else None,
        "mean_selection_regret": float(np.mean(regrets)) if regrets else None,
    }


def evaluate_split(
    examples: list[ConsequenceExample],
    predictors: dict[str, dict[tuple[str, str, int], dict[str, float]]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "records": len(examples),
        "parent_episodes": len({item.parent_episode_id for item in examples}),
        "prefixes": len({(item.parent_episode_id, item.prefix_id) for item in examples}),
        "heads": {},
    }
    for head in HEADS:
        valid = [item for item in examples if bool(item.masks[head])]
        actual = np.asarray([float(getattr(item.target, head)) for item in valid], dtype=np.float64)
        head_report: dict[str, Any] = {"valid": len(valid), "censored_or_not_applicable": len(examples) - len(valid), "methods": {}}
        for name, values in predictors.items():
            predicted = np.asarray([values[(item.parent_episode_id, item.prefix_id, item.target.action)][head] for item in valid], dtype=np.float64)
            metrics = probability_metrics(actual, predicted) if head == "deadline_risk" else point_metrics(actual, predicted)
            metrics["ranking"] = ranking_metrics(examples, values, head)
            head_report["methods"][name] = metrics
        report["heads"][head] = head_report
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("validation", "test", "ood"), default=("validation",))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if manifest["protocol"] != protocol["protocol"]:
        raise ValueError("manifest/protocol mismatch")
    horizon = int(protocol["prediction_horizon_steps"])
    train = load_consequence_jsonl(args.data / manifest["files"]["train"]["path"], expected_horizon_steps=horizon, strict_identity=True)
    constants = training_constants(train)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Graph5ActionConsequenceWorldModel(ConsequenceModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    report: dict[str, Any] = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_sha256": sha256_file(args.manifest),
        "protocol_sha256": sha256_file(args.protocol),
        "device": str(device),
        "selection_rule": protocol["prediction_gate"],
        "train_only_constants": constants,
        "splits": {},
    }
    for split in args.splits:
        examples = load_consequence_jsonl(args.data / manifest["files"][split]["path"], expected_horizon_steps=horizon, strict_identity=True)
        model_values = model_predictions(model, examples, device)
        mean_values, physics_values = {}, {}
        for item in examples:
            key = (item.parent_episode_id, item.prefix_id, item.target.action)
            mean_values[key] = dict(constants)
            physics_values[key] = public_physics(item, float(horizon))
        report["splits"][split] = evaluate_split(examples, {"model": model_values, "train_mean_or_rate": mean_values, "public_physics": physics_values})
    if "validation" in report["splits"]:
        validation = report["splits"]["validation"]["heads"]
        continuous_wins = sum(
            validation[head]["methods"]["model"]["mae"]
            < min(validation[head]["methods"]["train_mean_or_rate"]["mae"], validation[head]["methods"]["public_physics"]["mae"])
            for head in ("travel_time", "service_progress", "energy_delta")
        )
        deadline_win = validation["deadline_risk"]["methods"]["model"]["brier"] < validation["deadline_risk"]["methods"]["train_mean_or_rate"]["brier"]
        ranking_heads = sum(
            (validation[head]["methods"]["model"]["ranking"]["pairwise_accuracy"] or 0.0)
            >= float(protocol["prediction_gate"]["pairwise_accuracy_threshold"])
            for head in HEADS
        )
        gate = {
            "continuous_heads_beating_best_simple_baseline": continuous_wins,
            "deadline_brier_beats_train_event_rate": deadline_win,
            "ranking_heads_at_or_above_threshold": ranking_heads,
        }
        gate["passed"] = bool(
            continuous_wins >= int(protocol["prediction_gate"]["continuous_heads_beating_best_simple_baseline_min"])
            and deadline_win
            and ranking_heads >= int(protocol["prediction_gate"]["ranking_heads_with_pairwise_accuracy_at_least"])
        )
        report["validation_gate"] = gate
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "validation_gate": report.get("validation_gate")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
