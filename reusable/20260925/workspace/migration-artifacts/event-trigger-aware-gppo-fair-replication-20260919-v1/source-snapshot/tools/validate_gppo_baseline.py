"""Validate the frozen GPPO graph/action contract against a local checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    ppo_root = source_root / "ppo_allocation"
    if not (ppo_root / "random_event" / "environment.py").is_file():
        parser.error(f"not a GPPO-8.29 checkout: {source_root}")

    # The upstream project intentionally uses script-style imports such as
    # ``from config import ...``. Put ppo_allocation first and run from there.
    os.chdir(ppo_root)
    sys.path.insert(0, str(ppo_root))

    from random_event.environment import RandomEventAllocationEnv  # noqa: PLC0415
    from random_event.models import GraphActorCritic  # noqa: PLC0415

    target_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(target_root))
    from gppo_world.contracts import snapshot_from_gppo  # noqa: PLC0415
    from gppo_world.registry import FEATURE_REGISTRY  # noqa: PLC0415

    env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001)
    graph, _ = env.reset()
    snapshot = snapshot_from_gppo(graph)
    model = GraphActorCritic.from_graph(graph)
    action, _, value, _ = model.act(graph, deterministic=True)
    report = {
        "node_shapes": {name: list(value.shape) for name, value in snapshot.nodes.items()},
        "candidate_edges": int(snapshot.candidate_edges.shape[0]),
        "actions": snapshot.num_actions,
        "noop_action": snapshot.noop_action,
        "legal_action_count": int(snapshot.action_mask.sum().item()),
        "deterministic_action": int(action),
        "deterministic_action_is_legal": bool(snapshot.action_mask[action].item()),
        "critic_value_is_finite": bool(float("-inf") < value < float("inf")),
        "registry_sha256": FEATURE_REGISTRY.sha256(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["node_shapes"] != {"uav": [4, 12], "region": [4, 12], "target": [3, 16]}:
        return 1
    if report["candidate_edges"] != 16 or report["actions"] != 17 or report["noop_action"] != 16:
        return 1
    if not report["deterministic_action_is_legal"] or not report["critic_value_is_finite"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
