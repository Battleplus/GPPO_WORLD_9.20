"""One-shot test/OOD evaluation for the frozen arrival consequence model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.arrival_consequence_data import file_sha256, load_arrival_jsonl  # noqa: E402
from gppo_world.arrival_consequence_model import ArrivalModelConfig, Graph5ArrivalConsequenceModel  # noqa: E402


def brier(values: list[tuple[float, float]]) -> float | None:
    return float(np.mean([(p - y) ** 2 for p, y in values])) if values else None


def calibration(values: list[tuple[float, float]], bins: int = 5) -> list[dict[str, float | int]]:
    result = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        group = [(p, y) for p, y in values if (low <= p < high) or (index == bins - 1 and p <= high)]
        if group:
            result.append({"bin": index, "count": len(group), "mean_predicted": float(np.mean([p for p, _ in group])), "mean_observed": float(np.mean([y for _, y in group]))})
    return result


def target_value(item: Any, name: str) -> float | None:
    value = item.target.get(name)
    return None if value is None else float(value)


def public_baselines(item: Any, train_rates: dict[str, float]) -> dict[str, float]:
    action = item.action
    if action >= 24:
        return {"arrival_time": 0.0, "deadline": train_rates["deadline"], "failure": train_rates["failure"]}
    relation = float(item.graph.candidate_features[action, 0]) * 10.0
    task_index = action % 6
    task_node = item.graph.nodes["task"][task_index]
    uav_node = item.graph.nodes["uav"][action // 6]
    current_time = float(item.graph.global_features[0]) * 18.0
    deadline = float(task_node[8])
    energy = float(uav_node[8])
    deadline_public = float(current_time + relation <= deadline) if float(task_node[10]) > 0.5 else train_rates["deadline"]
    failure_public = float(energy < relation * 0.35) if float(uav_node[10]) > 0.5 else train_rates["failure"]
    return {"arrival_time": relation, "deadline": deadline_public, "failure": failure_public}


@torch.no_grad()
def evaluate_split(model: Graph5ArrivalConsequenceModel, items: list[Any], train_rates: dict[str, float], device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions: dict[tuple[str, str, int], dict[str, float]] = {}
    arrival_errors: dict[str, list[float]] = {"model": [], "distance_speed": []}
    risks: dict[str, list[tuple[float, float]]] = {"model_deadline": [], "rate_deadline": [], "public_deadline": [], "model_failure": [], "rate_failure": [], "public_failure": []}
    t0 = time.perf_counter()
    for item in items:
        prediction = model.predict_candidates(item.graph.to(device), [item.action])
        pred = {"arrival_time": float(prediction.arrival_mean[0].cpu()), "deadline": float(torch.sigmoid(prediction.deadline_logits[0]).cpu()), "failure": float(torch.sigmoid(prediction.failure_logits[0]).cpu())}
        key = (item.parent_episode_id, item.prefix_id, item.action)
        predictions[key] = pred
        public = public_baselines(item, train_rates)
        arrival = target_value(item, "arrival_time")
        if item.masks["arrival_time"] and arrival is not None and item.action != 24:
            arrival_errors["model"].append(abs(pred["arrival_time"] - arrival))
            arrival_errors["distance_speed"].append(abs(public["arrival_time"] - arrival))
        if item.masks["deadline"]:
            y = float(bool(item.target["arrival_before_deadline_physical"]))
            risks["model_deadline"].append((pred["deadline"], y)); risks["rate_deadline"].append((train_rates["deadline"], y)); risks["public_deadline"].append((public["deadline"], y))
        if item.masks["failure"]:
            y = float(bool(item.target["execution_or_energy_failure"]))
            risks["model_failure"].append((pred["failure"], y)); risks["rate_failure"].append((train_rates["failure"], y)); risks["public_failure"].append((public["failure"], y))
    elapsed = time.perf_counter() - t0
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for item in items:
        if item.action != 24:
            groups[(item.parent_episode_id, item.prefix_id)].append(item)
    ranking = {"groups_total": len(groups), "arrival_groups_comparable": 0, "deadline_groups_comparable": 0, "arrival": {}, "deadline": {}}
    model_selected_deadline: list[float] = []; rule_selected_deadline: list[float] = []
    model_regret: list[float] = []; rule_regret: list[float] = []
    for group in groups.values():
        valid_arrival = [x for x in group if x.masks["arrival_time"] and target_value(x, "arrival_time") is not None]
        if len(valid_arrival) >= 2:
            ranking["arrival_groups_comparable"] += 1
            model_pick = min(valid_arrival, key=lambda x: predictions[(x.parent_episode_id, x.prefix_id, x.action)]["arrival_time"])
            rule_pick = min(valid_arrival, key=lambda x: public_baselines(x, train_rates)["arrival_time"])
            best = min(float(x.target["arrival_time"]) for x in valid_arrival)
            model_regret.append(float(model_pick.target["arrival_time"]) - best); rule_regret.append(float(rule_pick.target["arrival_time"]) - best)
        valid_deadline = [x for x in group if x.masks["deadline"]]
        if len(valid_deadline) >= 2:
            ranking["deadline_groups_comparable"] += 1
            model_pick = max(valid_deadline, key=lambda x: predictions[(x.parent_episode_id, x.prefix_id, x.action)]["deadline"])
            rule_pick = max(valid_deadline, key=lambda x: public_baselines(x, train_rates)["deadline"])
            model_selected_deadline.append(float(bool(model_pick.target["arrival_before_deadline_physical"])))
            rule_selected_deadline.append(float(bool(rule_pick.target["arrival_before_deadline_physical"])))
    ranking["arrival"] = {"model_mean_selection_regret": float(np.mean(model_regret)) if model_regret else None, "public_distance_mean_selection_regret": float(np.mean(rule_regret)) if rule_regret else None, "model_zero_regret_rate": float(np.mean(np.isclose(model_regret, 0.0))) if model_regret else None, "public_distance_zero_regret_rate": float(np.mean(np.isclose(rule_regret, 0.0))) if rule_regret else None}
    ranking["deadline"] = {"model_selected_on_time_rate": float(np.mean(model_selected_deadline)) if model_selected_deadline else None, "public_selected_on_time_rate": float(np.mean(rule_selected_deadline)) if rule_selected_deadline else None}
    metrics = {"arrival_time": {"valid_n": len(arrival_errors["model"]), "model_mae": float(np.mean(arrival_errors["model"])) if arrival_errors["model"] else None, "distance_speed_mae": float(np.mean(arrival_errors["distance_speed"])) if arrival_errors["distance_speed"] else None}, "deadline": {"valid_n": len(risks["model_deadline"]), "model_brier": brier(risks["model_deadline"]), "occurrence_rate_brier": brier(risks["rate_deadline"]), "public_physics_brier": brier(risks["public_deadline"]), "model_calibration": calibration(risks["model_deadline"])}, "failure": {"valid_n": len(risks["model_failure"]), "model_brier": brier(risks["model_failure"]), "occurrence_rate_brier": brier(risks["rate_failure"]), "public_energy_brier": brier(risks["public_failure"]), "model_calibration": calibration(risks["model_failure"])}, "candidate_ranking": ranking, "inference": {"candidate_records": len(items), "wall_seconds": elapsed, "seconds_per_record": elapsed / max(1, len(items)), "device": str(device), "warmup": "none; one-shot offline evaluation"}}
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    horizon = int(manifest["prediction_horizon_steps"])
    train = load_arrival_jsonl(args.data / manifest["files"]["train"]["path"], horizon)
    train_rates = {"deadline": float(np.mean([bool(x.target["arrival_before_deadline_physical"]) for x in train if x.masks["deadline"]])), "failure": float(np.mean([bool(x.target["execution_or_energy_failure"]) for x in train if x.masks["failure"]]))}
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Graph5ArrivalConsequenceModel(ArrivalModelConfig(hidden_dim=64, horizon_steps=horizon))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    result: dict[str, Any] = {"checkpoint_sha256": file_sha256(args.checkpoint), "checkpoint_format": payload.get("format"), "selection_rule": "best validation total; test/OOD were not used for selection", "train_occurrence_rates": train_rates, "splits": {}}
    for split in ("test", "ood"):
        items = load_arrival_jsonl(args.data / manifest["files"][split]["path"], horizon)
        result["splits"][split] = evaluate_split(model, items, train_rates, torch.device("cpu"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "splits": list(result["splits"]), "checkpoint_sha256": result["checkpoint_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
