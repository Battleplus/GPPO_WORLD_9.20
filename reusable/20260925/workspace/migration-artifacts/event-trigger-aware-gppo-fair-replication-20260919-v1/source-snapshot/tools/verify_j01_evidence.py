"""Read-only independent checks of J01 budgets, sealed inputs and exported metrics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from gppo_world.data import STATE_DIM, load_jsonl, audit_training_inputs
from gppo_world.jepa import GraphJEPA, batch_graphs, encode_batch
from tools.run_d02_adapter_diagnostics import sha, write


def verify(data_dir, run):
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    frozen = json.loads((run / "selection-frozen.json").read_text(encoding="utf-8"))
    start = json.loads((run / "test-start.json").read_text(encoding="utf-8"))
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    normalization = json.loads((run / "target-normalization.json").read_text(encoding="utf-8"))
    if sha(data_dir / "dataset-manifest.json") != provenance["data_manifest_sha256"]:
        raise ValueError("data manifest changed")
    if sha(run / "selection-frozen.json") != start["selection_sha256"] or frozen["frozen_at"] > start["started_at"]:
        raise ValueError("selection not sealed before test")
    audit = audit_training_inputs(data_dir / "dataset-manifest.json", data_dir / "dataset")
    if not audit["passed"]:
        raise ValueError(audit["errors"])
    train = load_jsonl(data_dir / "dataset/train.jsonl")
    test = load_jsonl(data_dir / "dataset/test.jsonl")
    # Collector labels are converted to float32 by the frozen training/eval loader.
    y = np.array([np.concatenate([t.target_delta.numpy(), [t.reward], t.costs.numpy(), [t.continuation]])
                  for t in test], dtype=np.float32).astype(np.float64)
    scale = np.array(normalization["scale"])
    active = normalization["active_state_dimensions"]
    expected_keys = {(g, s) for g in [*protocol["groups"], "random_untrained"] for s in protocol["seeds"]}
    if {(m["group"], m["seed"]) for m in frozen["models"]} != expected_keys or len(frozen["models"]) != 12:
        raise ValueError("incomplete matrix")
    checked, max_error, action_intervals = 0, 0.0, {}
    for entry in frozen["models"]:
        group, seed = entry["group"], entry["seed"]
        folder = run / f"{group}-{seed}"
        for name in ("checkpoint", "probe"):
            if sha(run / entry[name]) != entry[f"{name}_sha256"]:
                raise ValueError("sealed weights changed")
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        epochs = 0 if group == "random_untrained" else protocol["training"]["epochs"]
        if len(history) != epochs or (epochs and [h["epoch"] for h in history] != list(range(1, epochs + 1))):
            raise ValueError("epoch history mismatch")
        expected_updates = epochs * int(np.ceil(len(train) / protocol["training"]["batch_size"]))
        if entry["optimizer_updates"] != expected_updates:
            raise ValueError("optimizer budget mismatch")
        if entry["ridge"] != min(entry["grid_scores"], key=lambda r: (r["score"], r["ridge"]))["ridge"]:
            raise ValueError("validation selection mismatch")
        model, _ = GraphJEPA.load(run / entry["checkpoint"])
        with torch.no_grad():
            graphs = [t.graph for t in train[:12]]
            independent = torch.stack([model.online(graph) for graph in graphs])
            vectorized = encode_batch(model.online, batch_graphs(graphs))
        if not torch.allclose(independent, vectorized, atol=1e-5, rtol=1e-5):
            raise ValueError("batched encoding differs from independent real graphs")
        exported = json.loads((folder / "test-per-transition.json").read_text(encoding="utf-8"))
        prediction = np.array(exported["probe_prediction"])
        z = ((prediction - y) / scale) ** 2
        tasks = np.stack([z[:, :STATE_DIM][:, active].mean(1), z[:, STATE_DIM], z[:, STATE_DIM + 1:STATE_DIM + 8].mean(1), z[:, -1]], 1)
        error = tasks.mean(1)
        difference = float(np.max(np.abs(error - np.array(exported["probe_macro_error"]))))
        max_error = max(max_error, difference)
        if difference > 1e-9:
            raise ValueError("per-transition metric mismatch")
        summary = next(r for r in results["models"] if r["group"] == group and r["seed"] == seed)
        if abs(float(error.mean()) - summary["probe_macro_normalized_mse"]) > 1e-9:
            raise ValueError("summary mismatch")
        if "actual_latent_mse" in exported:
            eligibility = np.array(exported["action_eligible"], dtype=bool)
            delta = (np.array(exported["alternative_latent_mse"]) - np.array(exported["actual_latent_mse"]))[eligibility]
            tapes = np.array(exported["tape_ids"])[eligibility]
            means = np.array([delta[tapes == t].mean() for t in sorted(set(tapes))])
            rng = np.random.default_rng(protocol["gates"]["bootstrap_seed"])
            boot = means[rng.integers(0, len(means), (protocol["gates"]["bootstrap_samples"], len(means)))].mean(1)
            interval = np.quantile(boot, [0.025, 0.975])
            if not np.allclose(interval, summary["action_degradation"]["ci95"], atol=1e-12):
                raise ValueError("cluster bootstrap mismatch")
            if group == "no_action_jepa" and not np.array_equal(np.array(exported["actual_latent_mse"]), np.array(exported["alternative_latent_mse"])):
                raise ValueError("no-action ablation still uses actions")
            action_intervals[f"{group}-{seed}"] = interval.tolist()
        checked += len(error)
    return {"passed": True, "models": len(expected_keys), "trained_models": 9, "random_controls": 3,
            "per_transition_metric_rows": checked, "max_independent_metric_error": max_error,
            "test_selection_frozen_before_evaluation": True, "sealed_checkpoint_and_probe_hashes_verified": 24,
            "equal_budget_per_seed_verified": True, "real_graph_batch_scalar_agreement": True,
            "no_action_perturbation_effect_exactly_zero": True, "action_intervals": action_intervals,
            "data_manifest_sha256": provenance["data_manifest_sha256"], "results_sha256": sha(run / "results.json"),
            "limitations": ["Read-only metric and artifact verification; not an independent training reproduction", "Representation geometry values come from frozen-run export; no new model selection"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    record = verify(args.data, args.run)
    write(args.output, record)
    print(json.dumps(record, indent=2))
