"""Collect fresh tape-disjoint J01 data. Immutable inputs; never retrain the sampler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tools.collect_t01_dataset import _collect_episode, _make_tape
from tools.run_d02_adapter_diagnostics import write, sha, git
from gppo_world.data import audit_training_inputs
from gppo_world.dataset import audit_manifest
from gppo_world.recorder import TransitionRecorder
from gppo_world.registry import FEATURE_REGISTRY, SCHEMA_VERSION


def collect(args):
    protocol = json.loads((ROOT / "nodes/J-01/protocol.json").read_text(encoding="utf-8"))
    baseline = args.baseline_root.resolve()
    if git(baseline, "rev-parse", "HEAD") != protocol["baseline_commit"] or git(baseline, "status", "--porcelain"):
        raise ValueError("baseline must be clean and pinned")
    if git(ROOT, "status", "--porcelain"):
        raise ValueError("commit implementation before collecting")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(baseline))
    from ppo_allocation.random_event.environment import RandomEventAllocationEnv, ActionSubmission
    from ppo_allocation.random_event.baselines import GreedyCostPolicy, MaskedRandomPolicy
    from ppo_allocation.random_event.events import EventTape
    from ppo_allocation.random_event.runtime_bridge import DetectorConfig
    from ppo_allocation.random_event.trainer import PPOTrainer

    class CheckedEnv(RandomEventAllocationEnv):
        def submit_action(self, submission):
            old_rejections = self.stale_rejection_count
            result = super().submit_action(submission)
            info = result[-1]
            if info.get("stale_decision") or info.get("invalid_action") or self.stale_rejection_count != old_rejections or info.get("raw_action") != info.get("repaired_action"):
                raise RuntimeError("collection aborted: execution changed/rejected, cannot mark accepted")
            return result

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    old_manifest = json.loads(args.old_manifest.read_text(encoding="utf-8"))
    old_checkpoint = Path(old_manifest["gppo_behavior_checkpoint"]["path"])
    if sha(old_checkpoint) != old_manifest["gppo_behavior_checkpoint"]["sha256"]:
        raise ValueError("behavior checkpoint changed")
    checkpoint = output / "behavior.pt"
    shutil.copy2(old_checkpoint, checkpoint)
    env = RandomEventAllocationEnv(initial_seed=1101, event_seed=1101 * 1009)
    trainer, _ = PPOTrainer.load(checkpoint, env=env, device="cpu")
    if trainer.total_steps != protocol["data"]["behavior_gppo_steps"]:
        raise ValueError("behavior budget mismatch")
    policy = trainer.model.eval()
    env.close()

    forbidden_seeds, forbidden_hashes = set(), set()
    def scan(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if "seed" in key and isinstance(child, int):
                    forbidden_seeds.add(child)
                if "sha256" in key and isinstance(child, str):
                    forbidden_hashes.add(child)
                scan(child)
        elif isinstance(value, list):
            for child in value:
                scan(child)
    references = [args.old_manifest, ROOT / "nodes/T-05/evidence/server-test-bank-manifest.json", args.old_development]
    for path in references:
        scan(json.loads(path.read_text(encoding="utf-8")))
    # Conservative upper bounds cover previous T05 training episode reset seeds.
    def historical_training_seed(value):
        return any(s * multiplier <= value <= s * multiplier + 50000
                   for s in (1101, 2202, 3303) for multiplier in (1000003, 10000019))

    episodes, tapes, files = [], [], {}
    for split, master in protocol["data"]["master_seeds"].items():
        recorder = TransitionRecorder()
        rng = random.Random(master)
        for profile in protocol["data"]["profiles"]:
            for index in range(protocol["data"]["tapes_per_profile"][split]):
                initial, event = rng.getrandbits(31), rng.getrandbits(63)
                if {initial, event} & forbidden_seeds or historical_training_seed(initial) or historical_training_seed(event):
                    raise ValueError("new/historical seed overlap; freeze a revised protocol before retry")
                forbidden_seeds.update((initial, event))
                tape = _make_tape(CheckedEnv, EventTape, initial, event, profile)
                import hashlib
                digest = hashlib.sha256(tape.to_bytes()).hexdigest()
                if digest in forbidden_hashes:
                    raise ValueError("canonical tape overlap")
                forbidden_hashes.add(digest)
                tape_id = f"j01-{split}-{profile}-{index:02d}-{digest[:12]}"
                write(output / "tapes" / f"{tape_id}.json", json.loads(tape.to_json()))
                tapes.append({"split": split, "profile": profile, "tape_id": tape_id,
                              "initial_seed": initial, "event_seed": event, "sha256": digest})
                policies = {"random_legal": MaskedRandomPolicy(seed=initial ^ 0xA5A5), "greedy": GreedyCostPolicy(), "gppo": policy}
                for name, sampler in policies.items():
                    with torch.no_grad():
                        episodes.append(_collect_episode(RandomEventAllocationEnv=CheckedEnv, ActionSubmission=ActionSubmission,
                            DetectorConfig=DetectorConfig, policy=sampler, behavior_policy=name, split=split,
                            profile=profile, tape_id=tape_id, tape=tape, seed=initial, recorder=recorder))
        path = recorder.write_jsonl(output / "dataset" / f"{split}.jsonl")
        files[split] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "transitions": len(recorder.items)}
        print(json.dumps({"split": split, **files[split]}), flush=True)
    manifest = {"manifest_version": "gppo-world-dataset/0.1.0", "schema_version": SCHEMA_VERSION,
        "registry_sha256": FEATURE_REGISTRY.sha256(), "source_commit": protocol["baseline_commit"],
        "collector_commit": git(ROOT, "rev-parse", "HEAD"), "protocol_sha256": sha(ROOT / "nodes/J-01/protocol.json"),
        "profiles": protocol["data"]["profiles"], "behavior_policies": protocol["data"]["behavior_policies"],
        "files": files, "episodes": episodes, "tapes": tapes,
        "gppo_behavior_checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint), "decision_steps": trainer.total_steps},
        "overlap_checks": {"new_split_seed_and_tape_overlap": 0, "referenced_historical_seed_and_tape_overlap": 0,
            "historical_training_seed_range_overlap": 0, "references": [{"path": str(p), "sha256": sha(p)} for p in references]},
        "execution_rejections_or_repairs": 0, "truth_only_online_field_count": 0}
    manifest["audit"] = audit_manifest(manifest)
    write(output / "dataset-manifest.json", manifest)
    audit = audit_training_inputs(output / "dataset-manifest.json", output / "dataset")
    write(output / "data-audit.json", audit)
    if not audit["passed"]:
        raise ValueError(audit["errors"])
    print(json.dumps({"passed": True, "episodes": len(episodes), "tapes": len(tapes)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--old-development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        collect(arguments)
    except Exception:
        import traceback
        if arguments.output.is_dir():
            write(arguments.output / "collection-failure.json", {"traceback": traceback.format_exc()})
        raise
