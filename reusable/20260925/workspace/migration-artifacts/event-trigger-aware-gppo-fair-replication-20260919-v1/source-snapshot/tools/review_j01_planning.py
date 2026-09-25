"""Read-only, posthoc planning audit of the sealed J01 data and metric exports.

No model loading, probe fitting, training, environment actions or gate changes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def audit(data, run):
    inventory = read(ROOT / "nodes/J-01/evidence/release-inventory.json")
    verified = 0
    for archive in inventory["archives"]:
        folder = data if archive["name"] == "j01-dataset.tar.gz" else run
        for member in archive["members"]:
            path = folder / member["path"]
            if not path.is_file() or path.stat().st_size != member["bytes"] or digest(path) != member["sha256"]:
                raise ValueError(f"sealed input changed: {member['path']}")
            verified += 1
    manifest = read(data / "dataset-manifest.json")
    results = read(run / "results.json")
    profiles, rows_by_split = {}, {}
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (data / f"dataset/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows_by_split[split] = rows
        lengths = Counter(r["episode_id"] for r in rows)
        gaps = [r["next_decision_time"] - r["decision_time"] for r in rows]
        legal_counts = Counter(sum(r["graph_t"]["action_mask"]) for r in rows)
        identities = [(r["episode_id"], r["step"]) for r in rows]
        profiles[split] = {
            "transitions": len(rows), "episodes": len(lengths), "tapes": len({r["tape_id"] for r in rows}),
            "duplicate_episode_step_keys": len(identities) - len(set(identities)),
            "episode_length_mean": mean(list(lengths.values())), "episode_length_max": max(lengths.values()),
            "episodes_with_at_least_4_decisions": sum(n >= 4 for n in lengths.values()),
            "transitions_with_4_observed_decision_contexts": sum(r["step"] >= 3 for r in rows),
            "legal_action_count_distribution": dict(sorted(legal_counts.items())),
            "one_legal_action_transitions": legal_counts[1],
            "multi_legal_action_transitions": sum(n for k, n in legal_counts.items() if k > 1),
            "executed_action_counts": dict(sorted(Counter(r["execution"]["executed_action"] for r in rows).items())),
            "decision_gap": {"min": min(gaps), "median": statistics.median(gaps), "mean": mean(gaps), "max": max(gaps),
                             "greater_than_1_with_1e_minus_9_tolerance": sum(g > 1 + 1e-9 for g in gaps)},
            "zero_reward_transitions": sum(abs(r["reward"]) < 1e-12 for r in rows),
            "continuation_counts": dict(Counter(str(r["continuation"]) for r in rows)),
            "constraint_violation_positive_transitions": sum(r["costs"]["constraint_violation"] > 0 for r in rows),
            "by_profile": {p: {"rows": len(selected), "mean_gap": mean([r["next_decision_time"] - r["decision_time"] for r in selected]),
                               "multi_action_rows": sum(sum(r["graph_t"]["action_mask"]) > 1 for r in selected)}
                           for p in manifest["profiles"] if (selected := [r for r in rows if r["scenario_id"] == p])},
        }
    test_rows = rows_by_split["test"]
    subgroup = {}
    seed_rows = []
    for entry in results["models"]:
        key = f"{entry['group']}-{entry['seed']}"
        exported = read(run / key / "test-per-transition.json")
        if list(zip(exported["episode_ids"], exported["steps"])) != [(r["episode_id"], r["step"]) for r in test_rows]:
            raise ValueError("evaluation/data row ordering mismatch")
        errors = exported["probe_macro_error"]
        if abs(mean(errors) - entry["probe_macro_normalized_mse"]) > 1e-12:
            raise ValueError("published metric differs from export")
        subgroup[key] = {
            "published_transition_weighted_error": mean(errors),
            "tape_equal_weight_error": mean([mean([e for r, e in zip(test_rows, errors) if r["tape_id"] == t])
                                            for t in sorted({r["tape_id"] for r in test_rows})]),
            "multi_action_error": mean([e for r, e in zip(test_rows, errors) if sum(r["graph_t"]["action_mask"]) > 1]),
            "one_action_error": mean([e for r, e in zip(test_rows, errors) if sum(r["graph_t"]["action_mask"]) == 1]),
            "by_profile": {p: mean([e for r, e in zip(test_rows, errors) if r["scenario_id"] == p]) for p in manifest["profiles"]},
        }
        if entry["group"] == "action_jepa":
            seed = entry["seed"]
            no_action = next(x for x in results["models"] if x["group"] == "no_action_jepa" and x["seed"] == seed)
            random = next(x for x in results["models"] if x["group"] == "random_untrained" and x["seed"] == seed)
            supervised = next(x for x in results["models"] if x["group"] == "supervised_graph" and x["seed"] == seed)
            seed_rows.append({"seed": seed, "improvement_vs_no_action": 1 - entry["probe_macro_normalized_mse"] / no_action["probe_macro_normalized_mse"],
                             "error_ratio_to_random": entry["probe_macro_normalized_mse"] / random["probe_macro_normalized_mse"],
                             "error_ratio_to_supervised": entry["probe_macro_normalized_mse"] / supervised["probe_macro_normalized_mse"],
                             "persistence_improvement": entry["latent_persistence_improvement"]})
    return {"as_of": "2026-09-05", "kind": "posthoc_readonly_planning_audit", "assessment": "share_with_caveats",
            "sealed_member_hashes_verified": verified, "data_profiles": profiles, "seed_comparisons": seed_rows,
            "recomputed_mean_improvement_vs_no_action": mean([r["improvement_vs_no_action"] for r in seed_rows]),
            "mean_error_ratio_to_random": mean([r["error_ratio_to_random"] for r in seed_rows]),
            "posthoc_subgroup_errors": subgroup,
            "inputs": {"release_inventory_sha256": digest(ROOT / "nodes/J-01/evidence/release-inventory.json"),
                       "dataset_manifest_sha256": digest(data / "dataset-manifest.json"), "results_sha256": digest(run / "results.json")},
            "limitations": ["Historical Test inspected for planning; not a new held-out experiment or changed acceptance gate",
                            "Subgroup and tape-equal means are descriptive posthoc values without confirmatory intervals",
                            "Hash checks do not validate physical realism or causal correctness",
                            "Next-graph changes alone cannot attribute action versus externally arriving event effects"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data.resolve(), args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"posthoc_subgroup_errors"}}, ensure_ascii=False, indent=2))
