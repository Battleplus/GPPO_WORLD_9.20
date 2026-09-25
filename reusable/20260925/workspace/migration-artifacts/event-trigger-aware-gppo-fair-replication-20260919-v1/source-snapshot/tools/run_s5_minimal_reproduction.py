"""S5: minimal independent reproduction from the frozen delivery inputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_safe(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    world_path = Path(args.world_checkpoint).resolve()
    calibration_path = Path(args.calibration).resolve()
    source_archive = Path(args.source_archive).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(baseline_root / "ppo_allocation"))
    from random_event.environment import ActionSubmission, RandomEventAllocationEnv
    from random_event.events import EventTape, RandomEvent, RandomEventType
    from random_event.models import GraphActorCritic
    from gppo_world.calibration import ShadowCalibration
    from gppo_world.gppo_adapter import freeze_world_model
    from gppo_world.model import EventAwareGraphWorldModel
    from tools.run_s1_r2_acceptance import make_tape
    from tools.run_s4_read_only_demo import fault_fixtures, run_arm

    model, metadata = GraphActorCritic.load(checkpoint, map_location="cpu")
    model.eval()
    world, world_metadata = EventAwareGraphWorldModel.load(world_path, map_location="cpu")
    freeze_world_model(world)
    calibration = ShadowCalibration.from_dict(json.loads(calibration_path.read_text(encoding="utf-8")))
    imports = (ActionSubmission, RandomEventAllocationEnv)
    policy_before = state_sha(model)
    world_before = state_sha(world)
    selected = [
        "normal",
        "energy_insufficient",
        "uav_damage",
        "communication_interrupt",
        "composite_three_factor",
    ]
    rows = []
    for scenario in selected:
        tape = make_tape(EventTape, RandomEvent, RandomEventType, scenario, 0)
        off = run_arm(imports, model, tape, scenario, 0, shadow_on=False, world=world, calibration=calibration, out=out)
        on = run_arm(imports, model, tape, scenario, 0, shadow_on=True, world=world, calibration=calibration, out=out)
        rows.append({
            "scenario": scenario,
            "off": {key: off[key] for key in ("decisions", "return", "signature", "trace")},
            "on": {key: on[key] for key in ("decisions", "return", "signature", "trace")},
            "signature_equal": off["signature"] == on["signature"],
            "return_difference": on["return"] - off["return"],
            "shadow_audit": on["audit"],
            "shadow_counters": on["shadow_counters"],
            "context_counters": on["context_counters"],
        })
    faults = fault_fixtures(imports, world, calibration, sha(checkpoint), out)
    summary = {
        "format": "m09-s5-minimal-reproduction/1.0.0",
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "source_archive_sha256": sha(source_archive),
        "policy_checkpoint_sha256": sha(checkpoint),
        "world_checkpoint_sha256": sha(world_path),
        "calibration_sha256": sha(calibration_path),
        "selected_scenarios": selected,
        "paired_scenario_count": len(rows),
        "paired_equal_count": sum(int(row["signature_equal"]) for row in rows),
        "all_selected_pairs_equal": all(row["signature_equal"] for row in rows),
        "pairs": rows,
        "faults": faults,
        "policy_state_unchanged": policy_before == state_sha(model),
        "world_frozen_and_unchanged": (
            not world.training
            and all(not parameter.requires_grad for parameter in world.parameters())
            and world_before == state_sha(world)
        ),
        "metadata": metadata,
        "world_metadata": world_metadata,
    }
    write_json(out / "reproduction-summary.json", summary)
    write_json(out / "inventory.json", [
        {"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "inventory.json"
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
