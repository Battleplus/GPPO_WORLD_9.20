"""Train the fair T-03 WM / EA-noGES / EAWM event-aware ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from gppo_world.data import audit_training_inputs, group_episodes, load_jsonl
from gppo_world.dataset import sha256_file
from gppo_world.event_training import (
    EventTrainingConfig,
    VARIANTS,
    config_dict,
    evaluate_events,
    freeze_class_balance,
    ges_diagnostics,
    train_event_model,
)
from gppo_world.events import (
    EVIDENCE_EVENTS,
    MODALITIES,
    event_support_report,
    freeze_event_schema,
    label_digest,
    label_episodes,
)
from gppo_world.model import EventAwareGraphWorldModel
from gppo_world.registry import FEATURE_REGISTRY, SCHEMA_VERSION
from gppo_world.training import evaluate_model, parameter_count, seed_everything


MAIN_VARIANTS = ("wm", "ea_no_ges", "eawm_hard")


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_finite(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _relative_degradation(candidate: float, reference: float) -> float:
    return (candidate - reference) / max(abs(reference), 1e-12)


def _roundtrip_check(model, path: Path, sample) -> tuple[float, list[str]]:
    restored, metadata = EventAwareGraphWorldModel.load(path)
    model.eval()
    restored.eval()
    expected, _ = model.step(
        sample.graph, sample.action, sample=False, evidence=sample.evidence
    )
    actual, _ = restored.step(
        sample.graph, sample.action, sample=False, evidence=sample.evidence
    )
    event_names = (
        "ordinal_event_logits",
        "nominal_event_logits",
        "structural_event_logits",
        "evidence_event_logits",
    )
    maximum = max(float((expected[name] - actual[name]).abs().max()) for name in event_names)
    return maximum, sorted(metadata)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("base_checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--variants",
        default=",".join((*MAIN_VARIANTS, "eawm_smooth")),
        help="comma separated: wm,ea_no_ges,eawm_hard,eawm_smooth",
    )
    args = parser.parse_args()
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        parser.error("the frozen protocol permits training only on Python 3.10/3.11")
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    if not variants or any(item not in VARIANTS for item in variants):
        parser.error(f"variants must be selected from {sorted(VARIANTS)}")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    dataset_dir = args.dataset_dir.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    output = args.output_dir.resolve()
    manifest_path = (args.manifest or dataset_dir.parent / "dataset-manifest.json").resolve()

    input_audit = audit_training_inputs(manifest_path, dataset_dir)
    _write(output / "input-audit.json", input_audit)
    if not input_audit["passed"]:
        print(json.dumps(input_audit, indent=2, sort_keys=True))
        return 2
    paths = {split: dataset_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    episodes = {split: group_episodes(load_jsonl(path)) for split, path in paths.items()}

    schema = freeze_event_schema(episodes["train"])
    labeled = {split: label_episodes(value, schema) for split, value in episodes.items()}
    repeated_train_digest = label_digest(label_episodes(episodes["train"], schema))
    label_digests = {split: label_digest(value) for split, value in labeled.items()}
    deterministic_labels = label_digests["train"] == repeated_train_digest
    balance = freeze_class_balance(labeled["train"])
    support = {split: event_support_report(value) for split, value in labeled.items()}
    _write(output / "event-schema.json", schema.to_dict())
    _write(output / "class-balance.json", balance)
    _write(output / "event-support.json", support)

    base_sha = sha256_file(base_checkpoint)
    config = EventTrainingConfig(seed=args.seed, epochs=args.epochs)
    results = {}
    initial_hashes = {}
    parameter_counts = {}
    for variant in variants:
        seed_everything(args.seed)
        candidate, base_metadata = EventAwareGraphWorldModel.from_base_checkpoint(
            base_checkpoint, schema
        )
        initial_hashes[variant] = _state_dict_sha256(candidate)
        parameter_counts[variant] = parameter_count(candidate)
        started = time.perf_counter()
        trained, history, selection = train_event_model(
            candidate,
            labeled["train"],
            labeled["validation"],
            schema,
            balance,
            config,
            variant=variant,
        )
        elapsed = time.perf_counter() - started
        test_base = evaluate_model(trained, episodes["test"])
        test_events = evaluate_events(
            trained,
            labeled["test"],
            balance,
            rare_threshold=config.rare_prevalence_threshold,
        )
        checkpoint_path = output / "checkpoints" / f"{variant}_seed{args.seed}.pt"
        trained.save(
            checkpoint_path,
            extra={
                "variant": variant,
                "training_config": config_dict(config),
                "selection": selection,
                "test_base": test_base,
                "test_events": test_events,
                "parent_t02_checkpoint_sha256": base_sha,
                "source_baseline_commit": "2a9bb9f87b9d543df144f4d108ba970c924151f9",
            },
        )
        roundtrip_max_abs, metadata_keys = _roundtrip_check(
            trained, checkpoint_path, episodes["test"][0][0]
        )
        _write(output / f"{variant}-training-history.json", history)
        results[variant] = {
            "variant_definition": VARIANTS[variant],
            "training_seconds": elapsed,
            "selection": selection,
            "test_base": test_base,
            "test_events": test_events,
            "ges_train": ges_diagnostics(labeled["train"], schema, VARIANTS[variant]["ges_mode"]),
            "ges_test": ges_diagnostics(labeled["test"], schema, VARIANTS[variant]["ges_mode"]),
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "checkpoint_roundtrip_max_abs": roundtrip_max_abs,
            "checkpoint_metadata_keys": metadata_keys,
            "base_checkpoint_metadata_keys": sorted(base_metadata),
        }

    main_present = all(name in results for name in MAIN_VARIANTS)
    eawm = results.get("eawm_hard", {})
    wm = results.get("wm", {})
    eawm_events = eawm.get("test_events", {})
    base_degradation = {
        name: _relative_degradation(
            eawm["test_base"][f"{name}_mae"], wm["test_base"][f"{name}_mae"]
        )
        for name in ("state", "reward", "cost")
    } if main_present else {}
    equal_steps = len({results[name]["selection"]["executed_epochs"] for name in variants}) == 1
    gates = {
        "strict_input_audit_passed": input_audit["passed"],
        "automatic_labels_byte_identical": deterministic_labels,
        "thresholds_and_balance_train_only": schema.source_split == "train" and balance["source_split"] == "train",
        "future_targets_absent_from_model_signature": not any(
            name in inspect.signature(EventAwareGraphWorldModel.step).parameters
            for name in ("next_graph", "reward", "costs", "continuation", "next_evidence")
        ),
        "main_ablation_complete": main_present,
        "identical_initialization": len(set(initial_hashes.values())) == 1,
        "equal_parameter_count": len(set(parameter_counts.values())) == 1,
        "equal_fixed_epoch_budget": equal_steps,
        "event_macro_f1_beats_frequency_baseline": bool(main_present) and (
            eawm_events["macro_f1"] > eawm_events["baseline_macro_f1"]
        ),
        "event_macro_auprc_beats_frequency_baseline": bool(main_present) and (
            eawm_events["macro_auprc"] > eawm_events["baseline_macro_auprc"]
        ),
        "rare_event_recall_beats_frequency_baseline": bool(main_present) and (
            eawm_events["rare_recall"] > eawm_events["baseline_rare_recall"]
        ),
        "base_state_reward_cost_within_5_percent": bool(main_present) and all(
            value <= config.max_base_metric_degradation for value in base_degradation.values()
        ),
        "checkpoints_roundtrip_exact": all(
            value["checkpoint_roundtrip_max_abs"] == 0.0 for value in results.values()
        ),
        "expiry_without_contract_marked_ineligible": (
            balance["modalities"]["evidence"]["slot_valid"][EVIDENCE_EVENTS.index("expire")] == 0
        ),
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
        "event_schema_sha256": schema.sha256(),
        "event_schema": schema.to_dict(),
        "label_digests": label_digests,
        "class_balance": balance,
        "event_support": support,
        "dataset": {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
                "episodes": len(episodes[split]),
                "transitions": sum(len(episode) for episode in episodes[split]),
            }
            for split, path in paths.items()
        },
        "parent_t02_checkpoint": {"path": str(base_checkpoint), "sha256": base_sha},
        "training_config": config_dict(config),
        "variants": results,
        "fairness": {
            "initial_state_dict_sha256": initial_hashes,
            "parameter_count": parameter_counts,
            "base_metric_relative_degradation_eawm_vs_wm": base_degradation,
            "same_dataset_seed_epochs_optimizer_and_episode_order": True,
            "test_metrics_computed_on_complete_ungated_targets": True,
        },
        "evidence_limitations": {
            "new": "supported: adjacent cumulative evidence introduces a new episode-scoped event_id",
            "duplicate": "supported as within-record multiplicity increase; no delivery_id is available",
            "conflict": "detectable by semantic payload hash but train/test positive support may be zero",
            "confirm": "detectable only for an existing non-confirmed to confirmed transition",
            "expire": "ineligible because T-01 has no expires_at/ttl contract",
        },
        "gates": gates,
    }
    _write(output / "metrics.json", result)
    _write(output / "training-config.json", {"config": asdict(config), "variants": variants})
    summary = {
        "result": result["result"],
        "seed": args.seed,
        "epochs": args.epochs,
        "event_schema_sha256": schema.sha256(),
        "gates": gates,
        "base_metric_relative_degradation_eawm_vs_wm": base_degradation,
        "variants": {
            name: {
                "event_macro_f1": value["test_events"]["macro_f1"],
                "event_macro_auprc": value["test_events"]["macro_auprc"],
                "rare_event_recall": value["test_events"]["rare_recall"],
                "state_mae": value["test_base"]["state_mae"],
                "reward_mae": value["test_base"]["reward_mae"],
                "cost_mae": value["test_base"]["cost_mae"],
            }
            for name, value in results.items()
        },
    }
    print(json.dumps(_finite(summary), indent=2, sort_keys=True))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
