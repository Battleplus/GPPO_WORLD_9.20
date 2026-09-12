"""Strict loader and split audit for action-consequence supervision."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .consequence_model import ConsequenceTarget
from .data import graph_from_dict
from .dataset import sha256_file


CONSEQUENCE_SPLITS = ("train", "validation", "test", "ood")
FORBIDDEN_ONLINE_FIELDS = frozenset(
    {
        "graph_tp1",
        "next_graph",
        "future_graph",
        "hidden_state",
        "simulator_state",
        "future_events",
        "full_environment_state",
    }
)


@dataclass(frozen=True)
class ConsequenceExample:
    graph: Any
    target: ConsequenceTarget
    history: torch.Tensor | None = None


def example_from_dict(record: Mapping[str, Any], *, expected_horizon_steps: int | None = None) -> ConsequenceExample:
    forbidden = sorted(FORBIDDEN_ONLINE_FIELDS.intersection(record))
    if forbidden:
        raise ValueError(f"future or hidden fields are forbidden in online input: {forbidden}")
    if "graph_t" not in record or "target" not in record:
        raise ValueError("each consequence record requires graph_t and target")
    graph = graph_from_dict(dict(record["graph_t"]))
    target = ConsequenceTarget(**dict(record["target"]))
    target.validate(expected_horizon_steps=expected_horizon_steps)
    if target.action >= graph.num_actions or not bool(graph.action_mask[target.action].item()):
        raise ValueError("counterfactual action must be legal in graph_t")
    history_value = record.get("history")
    history = None
    if history_value is not None:
        history = torch.tensor(history_value, dtype=torch.float32)
        if history.ndim != 1 or not torch.isfinite(history).all():
            raise ValueError("history must be a finite one-dimensional visible vector")
    return ConsequenceExample(graph=graph, target=target, history=history)


def load_consequence_jsonl(path: str | Path, *, expected_horizon_steps: int | None = None) -> list[ConsequenceExample]:
    examples: list[ConsequenceExample] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                examples.append(
                    example_from_dict(json.loads(line), expected_horizon_steps=expected_horizon_steps)
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid consequence record at {path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"consequence split is empty: {path}")
    return examples


def audit_consequence_manifest(manifest: Mapping[str, Any], data_dir: str | Path) -> dict[str, Any]:
    errors: list[str] = []
    episodes = list(manifest.get("episodes", ()))
    groups: dict[tuple[str, str, int], set[str]] = {}
    for item in episodes:
        split = str(item.get("split"))
        if split not in CONSEQUENCE_SPLITS:
            errors.append(f"unknown split {split!r}")
            continue
        try:
            key = (str(item["scenario_id"]), str(item["tape_id"]), int(item["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid episode grouping: {exc}")
            continue
        groups.setdefault(key, set()).add(split)
    overlaps = {
        "/".join((scenario, tape, str(seed))): sorted(splits)
        for (scenario, tape, seed), splits in groups.items()
        if len(splits) > 1
    }
    if overlaps:
        errors.append("scenario/tape/seed groups cross splits")
    files = manifest.get("files", {})
    resolved_dir = Path(data_dir)
    file_hashes: dict[str, str] = {}
    for split in CONSEQUENCE_SPLITS:
        relative = files.get(split)
        if not relative:
            errors.append(f"missing manifest file for {split}")
            continue
        path = resolved_dir / str(relative)
        if not path.is_file():
            errors.append(f"missing split file: {path}")
            continue
        file_hashes[split] = sha256_file(path)
        expected = files[split].get("sha256") if isinstance(files[split], Mapping) else None
        if expected and expected != file_hashes[split]:
            errors.append(f"sha256 mismatch for {split}")
    return {
        "passed": not errors,
        "errors": errors,
        "episode_count": len(episodes),
        "group_count": len(groups),
        "split_overlap_count": len(overlaps),
        "split_overlaps": overlaps,
        "file_sha256": file_hashes,
    }


def target_tensors(examples: Iterable[ConsequenceExample], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    targets = list(examples)
    if not targets:
        raise ValueError("at least one consequence example is required")
    return {
        name: torch.tensor([getattr(item.target, name) for item in targets], dtype=torch.float32, device=device)
        for name in ("travel_time", "service_progress", "energy_delta", "deadline_risk")
    }
