"""Read-only, descriptive post-hoc analysis of the sealed T-05 evaluation archive.

Does not load checkpoints, alter training, select models, or claim causality.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import tarfile

GROUPS = ("GPPO", "WM-GPPO", "EA-noGES-GPPO", "EAWM-GPPO")
SEEDS = (1101, 2202, 3303)
SCENARIOS = ("single", "sequential", "overlap", "burst", "unseen")
COMPONENTS = ("uncovered", "distance", "load_gap", "switches", "recovery_delay")
METRICS = ("episode_return", "all_decision_return", "fixed_j", "recovery_delay", "decision_count",
           "value_error", "pre_mask_invalid_probability")
ARCHIVE_SHA256 = "1c457d9c15d0e7b98cc50030c0f132a5171530ba6d5c4c7fd50281d8a3ab650e"


def scenario(tape_id: str) -> str:
    parts = tape_id.split("-")
    if len(parts) != 4 or parts[0] != "test" or parts[1] not in SCENARIOS:
        raise ValueError(f"Unknown frozen tape ID: {tape_id}")
    return parts[1]


def paired_differences(candidate: list[dict], baseline: list[dict]) -> dict:
    """Pair by identity, never by array position; reject missing/duplicate tapes."""
    a = {r["tape_id"]: r for r in candidate}
    b = {r["tape_id"]: r for r in baseline}
    if not a or len(a) != len(candidate) or len(b) != len(baseline) or a.keys() != b.keys():
        raise ValueError("Duplicate, empty, or unmatched tape IDs")
    return {
        "n": len(a),
        "candidate_minus_gppo": {
            metric: mean(float(a[k][metric]) - float(b[k][metric]) for k in sorted(a))
            for metric in METRICS
        },
    }


def reward_components(trace: dict) -> dict:
    values = {key: 0.0 for key in COMPONENTS}
    rewards = []
    for decision in trace["decisions"]:
        components = decision["reward_trace"]["reward_components"]
        if set(components) != set(COMPONENTS):
            raise ValueError("Unexpected reward component schema")
        reward = float(decision["reward"])
        if not math.isclose(sum(components.values()), reward, abs_tol=1e-7):
            raise ValueError("Reward component sum mismatch")
        rewards.append(reward)
        # Frozen baseline attributes reward once to the first active event;
        # decisions with no active event are absent from episode_return.
        if decision["active_events_before"]:
            for key in COMPONENTS:
                values[key] += float(components[key])
    if not math.isclose(sum(rewards), trace["episode_return_check"], abs_tol=1e-7):
        raise ValueError("Episode return mismatch")
    if not math.isclose(sum(values.values()), trace["episode"]["episode_return"], abs_tol=1e-7):
        raise ValueError("Event-attributed return mismatch")
    return values


def analyze(archive: Path) -> dict:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Archive is not the frozen T-05 Release asset")
    records, components, adapter = {}, {}, {}
    with tarfile.open(archive, "r:gz") as tar:
        # Read selected members in memory; never extract or execute archive files.
        for group in GROUPS:
            for seed in SEEDS:
                prefix = f"evaluation/{group}/seed{seed}"
                evaluation = json.load(tar.extractfile(f"{prefix}/evaluation.json"))
                if evaluation["group"] != group or evaluation["seed"] != seed:
                    raise ValueError("Run identity mismatch")
                rows = evaluation["episode_records"]
                if len(rows) != 100 or len({r["tape_id"] for r in rows}) != 100:
                    raise ValueError("Expected 100 unique tapes")
                records[group, seed] = rows
                totals = defaultdict(float)
                for row in rows:
                    tape = row["tape_id"]
                    scenario(tape)
                    trace = json.load(tar.extractfile(f"{prefix}/traces/{tape}.json"))
                    if trace["tape_id"] != tape or not math.isclose(
                        trace["episode"]["episode_return"], row["episode_return"], abs_tol=1e-7
                    ):
                        raise ValueError(f"Trace/evaluation mismatch: {group}/{seed}/{tape}: "
                                         f"{trace['episode_return_check']} vs {row['episode_return']}")
                    components[group, seed, tape] = reward_components(trace)
                    row["all_decision_return"] = float(trace["episode_return_check"])
                    for decision in trace["decisions"]:
                        totals["decisions"] += 1
                        used = False if group == "GPPO" else decision["diagnostics"]["latent_adapter_used"]
                        totals["adapter_used"] += bool(used)
                        totals["decisions_without_active_events"] += not bool(decision["active_events_before"])
                adapter[f"{group}/seed{seed}"] = dict(totals)
    paired, contributions = {}, {}
    for group in GROUPS[1:]:
        paired[group] = {}
        contributions[group] = {}
        for seed in SEEDS:
            base, candidate = records["GPPO", seed], records[group, seed]
            paired_differences(candidate, base)
            paired[group][str(seed)] = {}
            for category in SCENARIOS:
                a = [r for r in candidate if scenario(r["tape_id"]) == category]
                b = [r for r in base if scenario(r["tape_id"]) == category]
                if len(a) != 20 or len(b) != 20:
                    raise ValueError("Expected 20 tapes per frozen scenario")
                paired[group][str(seed)][category] = paired_differences(a, b)
            contributions[group][str(seed)] = {
                key: mean(components[group, seed, r["tape_id"]][key]
                          - components["GPPO", seed, r["tape_id"]][key] for r in base)
                for key in COMPONENTS
            }
    return {
        "format": "t05-posthoc-diagnostics/0.1.0",
        "analysis_type": "descriptive_posthoc_not_confirmatory_not_causal",
        "source_archive_sha256": digest,
        "trace_count": 1200,
        "metric_directions": {"episode_return": "higher", "all_decision_return": "higher", "fixed_j": "lower"},
        "return_semantics": "episode_return is active-event-attributed; all_decision_return sums every accepted trace decision",
        "paired_by_scenario": paired,
        "reward_component_mean_difference": contributions,
        "adapter_usage_counts": adapter,
        "warning": "Do not select checkpoints or tune on these already viewed Test tapes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
