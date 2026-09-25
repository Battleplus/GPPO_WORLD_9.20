"""Frozen-budget offline JEPA experiment. Train/Val selection precedes Test loading."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from gppo_world.data import STATE_DIM, load_jsonl, audit_training_inputs
from gppo_world.jepa import GraphJEPA, JEPAConfig, batch_graphs, select_batch, representation_loss
from gppo_world.model import evidence_features
from gppo_world.training import seed_everything
from tools.run_d02_adapter_diagnostics import write, sha, git


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(rows):
    return {"graph": batch_graphs([t.graph for t in rows]), "next": batch_graphs([t.next_graph for t in rows]),
            "actions": torch.tensor([t.action for t in rows]),
            "evidence": torch.stack([evidence_features(t.evidence) for t in rows]),
            "y": torch.stack([torch.cat([t.target_delta, torch.tensor([t.reward]), t.costs, torch.tensor([t.continuation])]) for t in rows])}


def batches(count, size, generator=None):
    indices = torch.randperm(count, generator=generator) if generator is not None else torch.arange(count)
    pieces = list(indices.split(size))
    if len(pieces) > 1 and len(pieces[-1]) == 1:
        last = pieces.pop()
        pieces[-1] = torch.cat([pieces[-1], last])
    return pieces


@torch.no_grad()
def representations(model, data):
    online, prediction, target, persistence = [], [], [], []
    model.eval()
    for indices in batches(len(data["actions"]), 64):
        graph = select_batch(data["graph"], indices)
        current, future = model(graph, data["actions"][indices], data["evidence"][indices])
        online.append(current)
        prediction.append(future)
        target.append(model.target(select_batch(data["next"], indices)))
        persistence.append(model.target(graph))
    return {k: torch.cat(v).double() for k, v in
            {"online": online, "prediction": prediction, "target": target, "persistence": persistence}.items()}


def task_errors(prediction, truth, scale, active):
    squared = ((prediction - truth) / scale).square()
    state = squared[:, :STATE_DIM][:, active].mean(1)
    reward = squared[:, STATE_DIM]
    costs = squared[:, STATE_DIM + 1:STATE_DIM + 8].mean(1)
    continuation = squared[:, -1]
    tasks = torch.stack([state, reward, costs, continuation], 1)
    return tasks, tasks.mean(1)


def fit_probe(train_x, train_y, val_x, val_y, scale, active, grid):
    # All centering/scaling is from Train. Intercept is unpenalized.
    mean, std = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(0.05)
    x, v = (train_x - mean) / std, (val_x - mean) / std
    y_mean = train_y.mean(0)
    y = (train_y - y_mean) / scale
    results = []
    for ridge in grid:
        weight = torch.linalg.solve(x.T @ x + ridge * torch.eye(x.shape[1], dtype=x.dtype), x.T @ y)
        prediction = (v @ weight) * scale + y_mean
        score = task_errors(prediction, val_y, scale, active)[1].mean().item()
        results.append((score, ridge, weight))
    score, ridge, weight = min(results, key=lambda r: (r[0], r[1]))
    return {"mean": mean, "std": std, "y_mean": y_mean, "scale": scale, "weight": weight,
            "ridge": ridge, "validation_score": score, "grid_scores": [{"ridge": r, "score": s} for s, r, _ in results]}


def predict_probe(probe, x):
    return ((x - probe["mean"]) / probe["std"] @ probe["weight"]) * probe["scale"] + probe["y_mean"]


def collapse_stats(x):
    centered = x - x.mean(0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / max(len(x) - 1, 1)).clamp_min(0)
    if eigenvalues.sum() < 1e-15:
        rank = 0.0
    else:
        p = eigenvalues / eigenvalues.sum()
        rank = float((-(p * p.clamp_min(1e-30).log()).sum()).exp())
    return {"mean_std": x.std(0, unbiased=False).mean().item(), "effective_rank_covariance_entropy": rank,
            "min_std": x.std(0, unbiased=False).min().item(), "finite": bool(torch.isfinite(x).all())}


def cluster_interval(values, tape_ids, samples=2000, seed=94010404):
    ids = sorted(set(tape_ids))
    groups = np.array([np.mean([v for v, t in zip(values, tape_ids) if t == key]) for key in ids])
    if not len(groups):
        return {"mean": None, "ci95": [None, None], "tapes": 0}
    rng = np.random.default_rng(seed)
    boot = groups[rng.integers(0, len(groups), (samples, len(groups)))].mean(1)
    return {"mean": float(groups.mean()), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])], "tapes": len(groups)}


def train_all(protocol, data_dir, output):
    train = prepare(load_jsonl(data_dir / "dataset/train.jsonl"))
    validation = prepare(load_jsonl(data_dir / "dataset/validation.jsonl"))
    scale = train["y"].double().std(0, unbiased=False).clamp_min(protocol["probe"]["target_scale_floor"])
    active = train["y"][:, :STATE_DIM].std(0, unbiased=False) > 1e-6
    if not active.any():
        raise ValueError("no changing state dimensions")
    write(output / "target-normalization.json", {"scale": scale.tolist(), "active_state_dimensions": active.nonzero().flatten().tolist()})
    config = protocol["training"]
    selected = []
    for seed in protocol["seeds"]:
        for group in [*protocol["groups"], "random_untrained"]:
            seed_everything(seed)
            model = GraphJEPA(JEPAConfig(**protocol["model"]), "action_jepa" if group == "random_untrained" else group)
            model_dir = output / f"{group}-{seed}"
            model_dir.mkdir()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=config["learning_rate"], weight_decay=config["weight_decay"])
            generator = torch.Generator().manual_seed(seed)
            epochs = 0 if group == "random_untrained" else config["epochs"]
            updates, order_hashes = 0, []
            import hashlib
            started = time.monotonic()
            with (model_dir / "history.jsonl").open("x", encoding="utf-8") as history:
                for epoch in range(epochs):
                    model.train()
                    losses = []
                    order = batches(len(train["actions"]), config["batch_size"], generator)
                    order_hashes.append(hashlib.sha256(torch.cat(order).numpy().tobytes()).hexdigest())
                    for indices in order:
                        optimizer.zero_grad(set_to_none=True)
                        current, future = model(select_batch(train["graph"], indices), train["actions"][indices], train["evidence"][indices])
                        if group == "supervised_graph":
                            target = (train["y"][indices] / scale.float())
                            decoded = model.decoder(future)
                            # Equal task weighting, rather than letting state dimension count dominate.
                            parts, per_row = task_errors(decoded, target, torch.ones_like(scale).float(), active)
                            loss = per_row.mean()
                        else:
                            target = model.target(select_batch(train["next"], indices))
                            loss, _ = representation_loss(current, future, target, std_target=config["std_target"],
                                variance_weight=config["variance_weight"], covariance_weight=config["covariance_weight"])
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite training loss")
                        loss.backward()
                        gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["grad_clip"])
                        if not torch.isfinite(gradient):
                            raise ValueError("nonfinite gradient")
                        optimizer.step()
                        if group != "supervised_graph":
                            model.update_target()
                        updates += 1
                        losses.append(float(loss.detach()))
                    history.write(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses)), "updates": updates}) + "\n")
                    history.flush()
                    if (epoch + 1) % 10 == 0:
                        print(json.dumps({"group": group, "seed": seed, "epoch": epoch + 1, "loss": float(np.mean(losses)), "seconds": time.monotonic() - started}), flush=True)
            checkpoint = model_dir / "terminal.pt"
            metadata = {"group": group, "seed": seed, "epoch": epochs, "optimizer_updates": updates,
                        "protocol_sha256": sha(ROOT / "nodes/J-01/protocol.json"), "seconds": time.monotonic() - started,
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "total_parameters": sum(p.numel() for p in model.parameters()), "order_hashes": order_hashes}
            model.save(checkpoint, metadata)
            # No Test loaded yet. Common readout fitting uses Train and Validation only.
            probe = fit_probe(representations(model, train)["prediction"], train["y"].double(),
                              representations(model, validation)["prediction"], validation["y"].double(),
                              scale, active, protocol["probe"]["ridge_grid"])
            probe_path = model_dir / "probe.pt"
            torch.save(probe, probe_path)
            entry = {**metadata, "checkpoint": str(checkpoint.relative_to(output)), "checkpoint_sha256": sha(checkpoint),
                     "probe": str(probe_path.relative_to(output)), "probe_sha256": sha(probe_path),
                     "ridge": probe["ridge"], "validation_score": probe["validation_score"], "grid_scores": probe["grid_scores"]}
            write(model_dir / "selection.json", entry)
            selected.append(entry)
    for seed in protocol["seeds"]:
        entries = [x for x in selected if x["seed"] == seed and x["group"] != "random_untrained"]
        if len({json.dumps(x["order_hashes"]) for x in entries}) != 1 or len({x["optimizer_updates"] for x in entries}) != 1:
            raise ValueError("training budgets/orders differ")
    write(output / "selection-frozen.json", {"frozen_at": now(), "test_model_evaluation_started": False, "models": selected})


def evaluate(protocol, data_dir, output):
    frozen = json.loads((output / "selection-frozen.json").read_text(encoding="utf-8"))
    normalization = json.loads((output / "target-normalization.json").read_text(encoding="utf-8"))
    scale = torch.tensor(normalization["scale"], dtype=torch.float64)
    active = torch.zeros(STATE_DIM, dtype=torch.bool)
    active[normalization["active_state_dimensions"]] = True
    # Final Test is opened for model evaluation only after every model/probe is sealed.
    write(output / "test-start.json", {"started_at": now(), "selection_sha256": sha(output / "selection-frozen.json")})
    rows = load_jsonl(data_dir / "dataset/test.jsonl")
    data = prepare(rows)
    tape_ids = [r.episode_id.split("/", 2)[2] for r in rows]
    alternate, eligible = [], []
    rng = np.random.default_rng(protocol["gates"]["bootstrap_seed"])
    for row in rows:
        legal = [i for i, ok in enumerate(row.graph.action_mask.tolist()) if ok and i != row.action]
        alternate.append(int(rng.choice(legal)) if legal else row.action)
        eligible.append(bool(legal))
    alternatives = torch.tensor(alternate)
    eligible_tensor = torch.tensor(eligible)
    results = []
    arrays = {}
    for entry in frozen["models"]:
        checkpoint, probe_path = output / entry["checkpoint"], output / entry["probe"]
        if sha(checkpoint) != entry["checkpoint_sha256"] or sha(probe_path) != entry["probe_sha256"]:
            raise ValueError("sealed artifact changed")
        model, _ = GraphJEPA.load(checkpoint)
        probe = torch.load(probe_path, weights_only=True)
        rep = representations(model, data)
        predicted = predict_probe(probe, rep["prediction"])
        tasks, error = task_errors(predicted, data["y"].double(), scale, active)
        raw_mae = (predicted - data["y"].double()).abs().mean(0)
        key = f"{entry['group']}-{entry['seed']}"
        arrays[key] = error.tolist()
        result = {"group": entry["group"], "seed": entry["seed"], "probe_macro_normalized_mse": error.mean().item(),
                  "task_normalized_mse": dict(zip(protocol["probe"]["tasks"], tasks.mean(0).tolist())),
                  "raw_mae": {"state_delta_active": raw_mae[:STATE_DIM][active].mean().item(), "state_delta_all": raw_mae[:STATE_DIM].mean().item(),
                              "reward": raw_mae[STATE_DIM].item(), "costs": raw_mae[STATE_DIM + 1:STATE_DIM + 8].tolist(), "continuation": raw_mae[-1].item()},
                  "representation": {name: collapse_stats(value) for name, value in rep.items() if name != "persistence"}}
        per_row = {"tape_ids": tape_ids, "episode_ids": [r.episode_id for r in rows], "steps": [r.step for r in rows],
                   "probe_macro_error": error.tolist(), "probe_task_errors": tasks.tolist(), "probe_prediction": predicted.tolist()}
        if entry["group"] in {"action_jepa", "no_action_jepa", "random_untrained"}:
            actual_loss = (rep["prediction"] - rep["target"]).square().mean(1)
            persist_loss = (rep["persistence"] - rep["target"]).square().mean(1)
            alternate_data = {**data, "actions": alternatives}
            changed = representations(model, alternate_data)["prediction"]
            alternate_loss = (changed - rep["target"]).square().mean(1)
            delta = (alternate_loss - actual_loss)[eligible_tensor].tolist()
            result["latent_prediction_mse"] = actual_loss.mean().item()
            result["latent_persistence_mse"] = persist_loss.mean().item()
            result["latent_persistence_improvement"] = 1 - actual_loss.mean().item() / max(persist_loss.mean().item(), 1e-12)
            result["action_degradation"] = cluster_interval(delta, [t for t, ok in zip(tape_ids, eligible) if ok],
                protocol["gates"]["bootstrap_samples"], protocol["gates"]["bootstrap_seed"])
            result["action_eligible_transitions"] = sum(eligible)
            per_row.update({"actual_latent_mse": actual_loss.tolist(), "persistence_latent_mse": persist_loss.tolist(),
                            "alternative_latent_mse": alternate_loss.tolist(), "alternative_actions": alternate, "action_eligible": eligible})
        write(output / key / "test-per-transition.json", per_row)
        results.append(result)
    by_key = {(r["group"], r["seed"]): r for r in results}
    improvements, ratios, gate_rows = [], [], []
    for seed in protocol["seeds"]:
        action = by_key["action_jepa", seed]
        no_action = by_key["no_action_jepa", seed]
        supervised = by_key["supervised_graph", seed]
        improvements.append(1 - action["probe_macro_normalized_mse"] / no_action["probe_macro_normalized_mse"])
        ratios.append(action["probe_macro_normalized_mse"] / supervised["probe_macro_normalized_mse"])
        stats = action["representation"]["target"]
        gates = protocol["gates"]
        gate_rows.append({"seed": seed, "latent_std": stats["mean_std"] >= gates["every_seed_latent_mean_std_min"],
            "latent_rank": stats["effective_rank_covariance_entropy"] >= gates["every_seed_effective_rank_min"],
            "beats_persistence": action["latent_persistence_improvement"] >= gates["every_seed_latent_persistence_improvement_min"],
            "action_sensitive": action["action_degradation"]["ci95"][0] is not None and action["action_degradation"]["ci95"][0] > 0})
    comparison = {"probe_improvement_vs_no_action_by_seed": improvements, "mean_probe_improvement_vs_no_action": float(np.mean(improvements)),
                  "probe_ratio_to_supervised_by_seed": ratios, "mean_probe_ratio_to_supervised": float(np.mean(ratios))}
    passed = all(all(v for k, v in row.items() if k != "seed") for row in gate_rows) and np.mean(improvements) >= gates["mean_probe_improvement_vs_no_action_min"] and sum(x > 0 for x in improvements) >= gates["probe_better_than_no_action_seed_count_min"] and np.mean(ratios) <= gates["mean_probe_ratio_to_supervised_max"]
    write(output / "results.json", {"finished_at": now(), "models": results, "comparison": comparison, "per_seed_gates": gate_rows,
        "all_gates_passed": bool(passed), "next_action": "design_separately_calibrated_integration" if passed else "stop_before_gppo_training_preserve_negative_results",
        "test_transitions": len(rows), "test_tapes": len(set(tape_ids)), "active_state_dimensions": int(active.sum()),
        "no_policy_training": True, "no_gppo_benefit_claim": True})
    print(json.dumps({"finished": True, "all_gates_passed": bool(passed), **comparison}), flush=True)


def main(args):
    protocol_path = ROOT / "nodes/J-01/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if git(ROOT, "status", "--porcelain"):
        raise ValueError("commit before training")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(protocol["training"]["threads"])
    torch.use_deterministic_algorithms(True)
    write(output / "protocol.json", protocol)
    write(output / "provenance.json", {"source_commit": git(ROOT, "rev-parse", "HEAD"), "protocol_sha256": sha(protocol_path),
          "started_at": now(), "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(),
          "device": "cpu", "formal_t05_reproduction": False, "data_manifest_sha256": sha(args.data / "dataset-manifest.json")})
    audit = audit_training_inputs(args.data / "dataset-manifest.json", args.data / "dataset")
    write(output / "data-audit.json", audit)
    if not audit["passed"]:
        raise ValueError(audit["errors"])
    train_all(protocol, args.data, output)
    evaluate(protocol, args.data, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        if args.output.is_dir():
            write(args.output / "failure.json", {"at": now(), "traceback": traceback.format_exc()})
        raise
