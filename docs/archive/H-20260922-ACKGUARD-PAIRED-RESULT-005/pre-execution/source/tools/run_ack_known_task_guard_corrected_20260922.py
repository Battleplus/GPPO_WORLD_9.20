"""Corrected, authorization-gated ACK-known-task guard runner.

This module is the executable entry for the corrected guard protocol.  It is
deliberately independent from the historical runner so that the two old
20-step branches remain immutable evidence.  Importing this module performs
no model loading, environment construction, SQLite writes, or execution.

The dynamic path is intentionally blocked until an explicit authorization file
pins the protocol, source identities, immutable budget identity, and the full
384-step allowance.  The phase-1 command used in this handoff only writes
read-only preflight artifacts; it never invokes ``run_matrix``.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from dataclasses import dataclass, field
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import sqlite3
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from gppo_world.ack_known_task_guard import (
    NoLegalActionError,
    PublicIdentityError,
    select_guarded_action,
)
from gppo_world.budget_executor import PersistentBudget

BASE = WORKTREE / "runs" / "finite-communication-ack-lease-fix-20260920"
OUT_DEFAULT = BASE / "ackguard-corrected-protocol-20260922"
HISTORICAL_RUN = BASE / "ack-known-task-guard-baseline-v1"
HISTORICAL_RESUME = BASE / "ack-known-task-guard-resume-20260922"
NOOP_AUDIT = BASE / "ackguard-noop-contract-audit-20260922"
PREF_TRAIN = BASE / "preference-vector-label-train-v1"
PREFLIGHT = BASE / "allocation-branch-preflight-v2"
R_CONTROL_BRIDGE_DIR = BASE / "r-control-digest-audit-20260922"
R_CONTROL_BRIDGE_SCHEMA_HELPER = WORKTREE / "tools" / "r_control_digest_schema_20260922.py"
R_CONTROL_BRIDGE_AUDIT_LOADER = WORKTREE / "tools" / "audit_r_control_digest_20260922.py"
R_CONTROL_BRIDGE_EVIDENCE = R_CONTROL_BRIDGE_DIR / "bridge-evidence.json"
R_CONTROL_BRIDGE_QUALIFIED = R_CONTROL_BRIDGE_DIR / "compatibility-bridge-qualified.json"
R_CONTROL_BRIDGE_STRICT = R_CONTROL_BRIDGE_DIR / "compatibility-strict.json"
R_CONTROL_BRIDGE_RECEIPT = Path(
    r"E:\Z博士\research-reviews\r-control-digest-audit-004\binding-final-remote-verification.json"
)
SOURCE_ROOT = Path(
    r"E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1\source-snapshot"
)
HISTORY = Path(
    r"E:\Z博士\migration-artifacts\replan-value-feature-collection-20260919-v1"
)
CHECKPOINT = Path(
    r"E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1\training\seed-1101\P_train\last-recovery.pt"
)
CONFIG = Path(
    r"E:\Z博士\migration-artifacts\preference-weighted-wm-event-cpu-20260917\final\run\training\seed-1101\WD\resolved-config.json"
)
HISTORICAL_BUDGET_PATH = HISTORICAL_RUN / "budget.sqlite3"

PROTOCOL_ID = "ackguard-corrected-protocol-20260922-v1"
PROTOCOL_SCHEMA = "ackguard-corrected-executable-protocol/1.0.0"
AUTH_SCHEMA = "ackguard-corrected-authorization/1.0.0"
BRANCH_MANIFEST_SCHEMA = "ackguard-corrected-branch-manifest/1.0.0"
COMPATIBILITY_SCHEMA = "ackguard-historical-R-compatibility/1.0.0"
RUN_ID = "ackguard-corrected-protocol-20260922"
ATTEMPT_ID = "ackguard-corrected-protocol-20260922-attempt-0001"
EXPECTED_GUARD_SHA256 = "2bc314ff9e3d4f78ec871c1a2e739a92669b1f9bd573aabce6a8be5b3e9e808a"
EXPECTED_OLD_RUNNER_SHA256 = "da768e30403c68041c01db1d32f9dd248c03c8f82d200f241506bf629b41c71b"
EXPECTED_CHECKPOINT_SHA256 = "bf10d2685a4a3e9da036689f5028b022330e86e922e09c95a7dd0a929df9bb1a"
EXPECTED_SNAPSHOT_SHA256 = "c6826f941141d1ae7594c38a3e3465584ae0954158502f19d2f4d8163a18258e"
EXPECTED_OLD_BUDGET_SHA256 = "700b0c9173c8debd5cacb8066328dd2359bb913a128f3018710c8129de5e6223"
EXPECTED_R_CONTROL_BRIDGE_SCHEMA_HELPER_SHA256 = "5d4635d05024c04b8df2fce3cdb03ce4d19b063578fe81e026c233bcebf9cc28"
EXPECTED_R_CONTROL_BRIDGE_AUDIT_LOADER_SHA256 = "62ddb2f569dacfc3a054d5cea59a97d889a361995c9915d2d1dd5a112cb1ef0a"
EXPECTED_R_CONTROL_BRIDGE_EVIDENCE_SHA256 = "892ffb6049b99ee0ae3f526e450c7f2daa91d7ec8e92a4f01c98c4dfa5666cf3"
EXPECTED_R_CONTROL_BRIDGE_QUALIFIED_SHA256 = "fef8b2efdc1178259142408ff86ba73327bb563a77dd5aaef9622c8b036716fc"
EXPECTED_R_CONTROL_BRIDGE_STRICT_SHA256 = "1fbab168f19b470ab62545a286b441a93f614a8776654b0848d30e61165ba294"
EXPECTED_R_CONTROL_BRIDGE_RECEIPT_SHA256 = "8158be2b468a3969c02cd45069bcd72be2361c3e9bb74bbd89cd793253a1f32b"
EXPECTED_R_CONTROL_BRIDGE_COMMIT = "b83fdb2c4107b0fe343bfb5e0c6a440cc18befe9"
EXPECTED_SOURCE_HASHES = {
    "m10_environment.py": "083b4ad3e47d8a44a18e7a25f429bb7c4bba4a6899249404c0842dfdc0f71132",
    "m10_communication.py": "fcaada4fea48446aae509fb97f4b6b8275d5b6b9dda3bcdef00c5c392e6011ce",
    "joint_gppo.py": "3c799cfe12050971e1b11da6b2c04fa6a7c244a36935578861c5db533396f55e",
    "joint_training.py": "808b1de65e676ad45480843c152874248be1fe91c1ce56e1a015fddf81bfd4d1",
    "task_policy_view.py": "f5877b6582a02a5bb6c9c39b469dd222e9f5727390fd8080c01e160cd7b23317",
    "task_decision_bridge.py": "dd066a4e1be16607e5005e66eed5b90b1b97b485e57d7c7236b46eb171f8b1fd",
    "budget_executor.py": "5a2a7536a0ebb0078638a1fd45491a5d13cd9a719c9f012f158b74f5050fd29b",
}
RUNTIME_DEPENDENCY_PATHS = {
    "corrected_runner": Path(__file__).resolve(),
    "guard": WORKTREE / "gppo_world" / "ack_known_task_guard.py",
    "budget_executor": WORKTREE / "gppo_world" / "budget_executor.py",
    "historical_helper": WORKTREE / "tools" / "run_replan_value_experiment.py",
    "config": CONFIG,
    "r_control_digest_schema_helper": R_CONTROL_BRIDGE_SCHEMA_HELPER,
}
R_CONTROL_BRIDGE_INPUT_PATHS = {
    "schema_helper": R_CONTROL_BRIDGE_SCHEMA_HELPER,
    "audit_loader": R_CONTROL_BRIDGE_AUDIT_LOADER,
    "bridge_evidence": R_CONTROL_BRIDGE_EVIDENCE,
    "bridge_qualified_compatibility": R_CONTROL_BRIDGE_QUALIFIED,
    "bridge_strict_compatibility": R_CONTROL_BRIDGE_STRICT,
    "remote_receipt": R_CONTROL_BRIDGE_RECEIPT,
}
HISTORICAL_AUTH_INPUT_PATHS = {
    "historical_R_labels": PREF_TRAIN / "step-vector-rewards.jsonl",
    "historical_R_reproduction": PREF_TRAIN / "historical-reproduction.json",
    "historical_R_source_checkpoint_manifest": PREF_TRAIN / "source-checkpoint-manifest.json",
    "historical_R_snapshot_identity": PREF_TRAIN / "snapshot-identity.json",
    "historical_R_snapshot_interface_check": PREFLIGHT / "snapshot-interface-check.json",
    "historical_R_branch_manifest": PREF_TRAIN / "branch-manifest.json",
    "historical_R_branch_results": PREF_TRAIN / "branch-results.jsonl",
    "historical_R_control_reuse": HISTORICAL_RUN / "control-reuse-manifest.json",
}
PARENTS = tuple(f"parent-{index:02d}" for index in range(8))
REPEATS = (0, 1, 2)
MAX_TOTAL_STEPS = 384
MAX_STEPS_PER_BRANCH = 16
GLOBAL_LIMIT_AFTER_APPROVAL = 404
GAMMA = 0.99
BEHAVIOR_PREFERENCE = (0.8, 0.2)
REWARD_SCALES = {"task": 0.5, "energy": 1.0}
TOLERANCE = 1e-6


class ProtocolError(RuntimeError):
    """A protocol or authorization gate failed."""


class AuthorizationError(ProtocolError):
    """The dynamic runner is not authorized to construct runtime objects."""


class RuntimeTechnicalStop(ProtocolError):
    """An execution error that must stop without retry or refund."""


def _jsonable(value: Any, active: set[int] | None = None) -> Any:
    """Convert common scientific values to deterministic JSON-safe values."""
    active = set() if active is None else active
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            return {"type": "mapping", "cycle": True}
        active.add(identity)
        try:
            return {str(key): _jsonable(item, active) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            return {"type": type(value).__name__, "cycle": True}
        active.add(identity)
        try:
            return [_jsonable(item, active) for item in value]
        finally:
            active.remove(identity)
    if isinstance(value, set):
        return sorted((_jsonable(item, active) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist(), active)
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item(), active)
        except Exception:
            pass
    state = getattr(value, "__dict__", None)
    if isinstance(state, Mapping):
        identity = id(value)
        if identity in active:
            return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "cycle": True}
        active.add(identity)
        try:
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": _jsonable(state, active),
            }
        finally:
            active.remove(identity)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _full_state_jsonable(value: Any, *, path: str = "$", seen: dict[int, str] | None = None) -> Any:
    """Serialize the complete supported state graph, preserving back-references.

    Environment objects contain intentional cycles (for example execution ->
    clock and bridge -> execution).  A cycle marker without its target would
    silently omit state, so references use deterministic traversal paths.  An
    unsupported value is a hard protocol error rather than an incomplete hash.
    """
    seen = {} if seen is None else seen
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    identity = id(value)
    if identity in seen:
        return {"$ref": seen[identity]}
    seen[identity] = path
    if isinstance(value, bytes):
        return {"$id": path, "$type": "bytes", "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (bytearray, memoryview)):
        raw = bytes(value)
        return {"$id": path, "$type": type(value).__name__, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, np.ndarray):
        return {"$id": path, "$type": "numpy.ndarray", "dtype": str(value.dtype), "shape": list(value.shape), "data": _full_state_jsonable(value.tolist(), path=f"{path}.data", seen=seen)}
    if isinstance(value, np.generic):
        return {"$id": path, "$type": type(value).__name__, "value": _full_state_jsonable(value.item(), path=f"{path}.value", seen=seen)}
    if isinstance(value, np.random.Generator):
        return {"$id": path, "$type": "numpy.random.Generator", "bit_generator": _full_state_jsonable(value.bit_generator.state, path=f"{path}.bit_generator", seen=seen)}
    if isinstance(value, np.random.RandomState):
        return {"$id": path, "$type": "numpy.random.RandomState", "state": _full_state_jsonable(value.get_state(), path=f"{path}.state", seen=seen)}
    if isinstance(value, Mapping):
        entries = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            entries.append({
                "key": _full_state_jsonable(key, path=f"{path}.key[{len(entries)}]", seen=seen),
                "value": _full_state_jsonable(item, path=f"{path}.value[{len(entries)}]", seen=seen),
            })
        return {"$id": path, "$type": f"{type(value).__module__}.{type(value).__qualname__}", "entries": entries}
    if isinstance(value, (list, tuple, deque)):
        return {
            "$id": path,
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            "items": [_full_state_jsonable(item, path=f"{path}[{index}]", seen=seen) for index, item in enumerate(value)],
        }
    if isinstance(value, (set, frozenset)):
        items = [_full_state_jsonable(item, path=f"{path}[{index}]", seen=seen) for index, item in enumerate(sorted(value, key=lambda item: repr(item)))]
        return {"$id": path, "$type": f"{type(value).__module__}.{type(value).__qualname__}", "items": items}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        state = {
            field.name: _full_state_jsonable(getattr(value, field.name), path=f"{path}.{field.name}", seen=seen)
            for field in dataclasses.fields(value)
        }
        return {"$id": path, "$type": f"{type(value).__module__}.{type(value).__qualname__}", "state": state}
    if type(value).__module__ == "random" and callable(getattr(value, "getstate", None)):
        return {"$id": path, "$type": f"{type(value).__module__}.{type(value).__qualname__}", "state": _full_state_jsonable(value.getstate(), path=f"{path}.state", seen=seen)}
    item = getattr(value, "item", None)
    if callable(item) and type(value).__module__.startswith("torch"):
        try:
            return {"$id": path, "$type": f"{type(value).__module__}.{type(value).__qualname__}", "value": _full_state_jsonable(item(), path=f"{path}.value", seen=seen)}
        except Exception as exc:
            raise ProtocolError(f"cannot serialize tensor-like state at {path}: {type(exc).__name__}: {exc}") from exc
    state = getattr(value, "__dict__", None)
    if isinstance(state, Mapping):
        return {
            "$id": path,
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": {
                str(key): _full_state_jsonable(item, path=f"{path}.{key}", seen=seen)
                for key, item in sorted(state.items(), key=lambda pair: str(pair[0]))
            },
        }
    raise ProtocolError(f"environment state contains unsupported value at {path}: {type(value).__module__}.{type(value).__qualname__}")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_jsonable(dict(value)), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def observation_digest(obs: Mapping[str, Any]) -> str:
    selected = {
        "flat": obs.get("flat"),
        "mask": obs.get("mask"),
        "version": obs.get("version"),
        "time": obs.get("time"),
        "public_entity_ids": obs.get("public_entity_ids"),
        "continuation_actions": list(obs.get("continuation_actions", ())),
        "trigger_flags": obs.get("trigger_flags", {}),
    }
    return canonical_hash(selected)


def public_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    return _jsonable({
        "flat": obs.get("flat"),
        "mask": obs.get("mask"),
        "version": obs.get("version"),
        "time": obs.get("time"),
        "public_entity_ids": obs.get("public_entity_ids"),
        "continuation_actions": list(obs.get("continuation_actions", ())),
        "trigger_flags": obs.get("trigger_flags", {}),
    })


def environment_digest(env: Any) -> str:
    return canonical_hash(_full_state_jsonable(env))


def hidden_digest(value: Any) -> str | None:
    return None if value is None else canonical_hash(value)


def snapshot_semantic_components(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return the separately auditable state summaries used by isolation checks."""
    if "probe" not in snapshot:
        raise RuntimeTechnicalStop("snapshot probe state is required for semantic isolation")
    return {
        "env_sha256": environment_digest(snapshot["env"]),
        "obs_sha256": observation_digest(snapshot["obs"]),
        "policy_hidden_sha256": hidden_digest(snapshot.get("policy_hidden")),
        "world_hidden_sha256": hidden_digest(snapshot.get("world_hidden")),
        "probe_sha256": canonical_hash(snapshot["probe"]),
    }


def snapshot_semantic_digest(snapshot: Mapping[str, Any]) -> str:
    return canonical_hash(snapshot_semantic_components(snapshot))


def _energy_total(env: Any) -> float:
    """Read energy from the environment resource ledger at the boundary."""
    clock = getattr(env, "clock", None)
    resources = getattr(clock, "resources", None)
    if isinstance(resources, Mapping) and resources:
        values = []
        for resource_id, resource in sorted(resources.items(), key=lambda pair: str(pair[0])):
            if not hasattr(resource, "energy"):
                raise RuntimeTechnicalStop(f"resource {resource_id!r} has no energy field")
            value = float(resource.energy)
            if not math.isfinite(value):
                raise RuntimeTechnicalStop(f"resource {resource_id!r} energy is non-finite")
            values.append(value)
        return float(sum(values))
    raise RuntimeTechnicalStop("env.clock.resources is required for initial energy")


def _task_capacity_and_fleet(env: Any) -> tuple[int, int, float]:
    config = getattr(env, "config", None)
    if config is None:
        raise RuntimeTechnicalStop("environment config is missing")
    try:
        task_capacity = int(config.task_capacity)
        uav_count = int(config.uav_count)
        initial_energy = float(config.initial_energy)
    except AttributeError as exc:
        raise RuntimeTechnicalStop(f"native energy/config contract is incomplete: {exc}") from exc
    if task_capacity <= 0 or uav_count <= 0 or not math.isfinite(initial_energy) or initial_energy <= 0:
        raise RuntimeTechnicalStop("invalid task_capacity/uav_count/initial_energy contract")
    return task_capacity, uav_count, initial_energy


def branch_key(prefix_id: str, repeat: int) -> str:
    return f"{prefix_id}|repeat-{int(repeat)}|mode-R"


def _historical_r_rows() -> list[dict[str, Any]]:
    path = PREF_TRAIN / "branch-manifest.json"
    payload = read_json(path)
    rows = payload.get("rows", payload) if isinstance(payload, Mapping) else payload
    wanted = {f"{parent}|W1|seed-1101|prefix-0" for parent in PARENTS}
    selected = [
        dict(row) for row in rows
        if row.get("mode") == "R" and row.get("prefix_id") in wanted and int(row.get("repeat", -1)) in REPEATS
    ]
    selected.sort(key=lambda row: (str(row["parent_id"]), int(row["repeat"])))
    if len(selected) != 24:
        raise ProtocolError(f"historical R manifest must contain 24 rows, found {len(selected)}")
    return selected


def build_branch_manifest() -> dict[str, Any]:
    rows = []
    for source in _historical_r_rows():
        prefix_id = str(source["prefix_id"])
        repeat = int(source["repeat"])
        branch_id = f"{prefix_id}|repeat-{repeat}|guard-on"
        rows.append({
            "branch_id": branch_id,
            "control_branch_key": f"{prefix_id}|repeat-{repeat}|mode-R",
            "prefix_id": prefix_id,
            "parent_id": str(source["parent_id"]),
            "condition": "W1",
            "model_seed": 1101,
            "prefix_index": 0,
            "repeat": repeat,
            "split": "train",
            "historical_exogenous_key": str(source.get("exogenous_key", source.get("historical_exogenous_key"))),
            "historical_step_count": int(source.get("env_steps", source.get("historical_step_count", 0))),
            "historical_end_reason": source.get("end_reason", source.get("historical_end_reason")),
            "historical_branch_row_sha256": source.get("historical_branch_row_sha256"),
            "max_steps": MAX_STEPS_PER_BRANCH,
            "rule": "ack-known-task-guard-corrected-v1",
            "outcome_blind_selection": True,
            "execution_namespace": RUN_ID,
        })
    payload = {
        "schema": BRANCH_MANIFEST_SCHEMA,
        "matrix": {"parents": list(PARENTS), "condition": "W1", "model_seed": 1101, "prefix_index": 0, "repeats": list(REPEATS)},
        "counts": {"parents": 8, "repeats": 3, "branches": 24, "new_environment_step_cap": MAX_TOTAL_STEPS},
        "old_two_guard_branches_excluded": True,
        "outcome_blind": True,
        "rows": rows,
    }
    validate_manifest(payload)
    return payload


def validate_manifest(payload: Mapping[str, Any]) -> None:
    rows = list(payload.get("rows", ()))
    if len(rows) != 24:
        raise ProtocolError(f"corrected manifest must have 24 rows, found {len(rows)}")
    ids = [str(row.get("branch_id")) for row in rows]
    if len(set(ids)) != 24:
        raise ProtocolError("corrected manifest contains duplicate branch IDs")
    expected_ids = {
        f"parent-{parent_index:02d}|W1|seed-1101|prefix-0|repeat-{repeat}|guard-on"
        for parent_index in range(8) for repeat in REPEATS
    }
    if set(ids) != expected_ids:
        raise ProtocolError("corrected manifest does not cover the fixed 8x3 identity matrix")
    historical_rows = {
        f"{row['prefix_id']}|repeat-{int(row['repeat'])}|guard-on": row
        for row in _historical_r_rows()
    }
    for row in rows:
        branch_id = str(row.get("branch_id"))
        if row.get("condition") != "W1" or int(row.get("model_seed", -1)) != 1101 or int(row.get("prefix_index", -1)) != 0:
            raise ProtocolError(f"manifest scope mismatch: {branch_id}")
        if int(row.get("repeat", -1)) not in REPEATS or int(row.get("max_steps", -1)) != MAX_STEPS_PER_BRANCH:
            raise ProtocolError(f"manifest repeat/cap mismatch: {branch_id}")
        if row.get("execution_namespace") != RUN_ID:
            raise ProtocolError(f"manifest namespace mismatch: {branch_id}")
        expected = historical_rows.get(branch_id)
        if expected is None:
            raise ProtocolError(f"manifest branch identity is not a historical R pair: {branch_id}")
        expected_control = f"{expected['prefix_id']}|repeat-{int(expected['repeat'])}|mode-R"
        expected_key = str(expected.get("exogenous_key", expected.get("historical_exogenous_key")))
        exact = {
            "control_branch_key": expected_control,
            "prefix_id": str(expected["prefix_id"]),
            "parent_id": str(expected["parent_id"]),
            "repeat": int(expected["repeat"]),
            "historical_exogenous_key": expected_key,
        }
        for key, expected_value in exact.items():
            if row.get(key) != expected_value:
                raise ProtocolError(f"manifest {key} mismatch for {branch_id}")
        if int(row.get("historical_step_count", -1)) != int(expected.get("historical_step_count", expected.get("env_steps", -1))):
            raise ProtocolError(f"manifest historical step count mismatch for {branch_id}")
        expected_row_hash = expected.get("historical_branch_row_sha256")
        if expected_row_hash is not None and row.get("historical_branch_row_sha256") != expected_row_hash:
            raise ProtocolError(f"manifest historical row hash mismatch for {branch_id}")


def _fetch_dicts(con: sqlite3.Connection, sql: str, args: Sequence[Any] = ()) -> list[dict[str, Any]]:
    cursor = con.execute(sql, tuple(args))
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def read_budget_identity(path: Path, *, run_id: str = RUN_ID) -> dict[str, Any]:
    """Read the persistent ledger through SQLite's read-only URI."""
    if not path.is_file():
        raise AuthorizationError(f"required budget database is missing: {path}")
    uri_path = str(path.resolve()).replace("\\", "/")
    con = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    try:
        integrity = str(con.execute("PRAGMA integrity_check").fetchone()[0])
        stages = _fetch_dicts(con, "SELECT stage,limit_amount,reserved,verified,unknown FROM stages ORDER BY stage")
        attempts = _fetch_dicts(con, "SELECT attempt_id,stage,reserved,verified,unknown,run_id FROM attempts ORDER BY attempt_id,stage")
        reservations = _fetch_dicts(con, "SELECT reservation_id,attempt_id,stage,amount,status,run_id FROM reservations ORDER BY reservation_id")
        metadata = {row[0]: row[1] for row in con.execute("SELECT key,value FROM metadata ORDER BY key")}
        run_totals = _fetch_dicts(con, "SELECT stage,COALESCE(SUM(amount),0) AS reserved,COALESCE(SUM(CASE WHEN status='verified' THEN amount ELSE 0 END),0) AS verified,COALESCE(SUM(CASE WHEN status='unknown' THEN amount ELSE 0 END),0) AS unknown FROM reservations WHERE run_id=? GROUP BY stage ORDER BY stage", (run_id,))
    finally:
        con.close()
    for row in stages:
        row["pending"] = int(row["reserved"]) - int(row["verified"]) - int(row["unknown"])
    for row in run_totals:
        row["pending"] = int(row["reserved"]) - int(row["verified"]) - int(row["unknown"])
    return {
        "schema": "ackguard-budget-readonly-identity/1.0.0",
        "path": str(path),
        "sha256": sha256_file(path),
        "integrity_check": integrity,
        "metadata": metadata,
        "stages": stages,
        "attempts": attempts,
        "reservations": reservations,
        "run_totals": run_totals,
        "target_run_id": run_id,
        "read_only": True,
    }


def _stage(identity: Mapping[str, Any], name: str = "environment_steps") -> Mapping[str, Any]:
    return next((row for row in identity.get("stages", ()) if row.get("stage") == name), {})


def _source_identities() -> dict[str, Any]:
    historical_files = {
        name: file_identity(SOURCE_ROOT / "gppo_world" / name)
        for name in sorted(EXPECTED_SOURCE_HASHES)
    }
    local_files = {
        "guard": file_identity(WORKTREE / "gppo_world" / "ack_known_task_guard.py"),
        "old_runner": file_identity(WORKTREE / "tools" / "run_ack_known_task_guard_baseline_v1.py"),
        "corrected_runner": file_identity(Path(__file__)),
        "budget_executor": file_identity(WORKTREE / "gppo_world" / "budget_executor.py"),
        "config": file_identity(CONFIG),
    }
    input_files = {
        "checkpoint": file_identity(CHECKPOINT),
        "prefix_snapshots": file_identity(HISTORY / "prefix-snapshots.pkl"),
        "history_branch_results": file_identity(HISTORY / "branch-results.jsonl"),
        "frozen_selection": file_identity(PREFLIGHT / "frozen-selection.json"),
        "repeat_key_manifest": file_identity(PREFLIGHT / "repeat-key-manifest.json"),
        "snapshot_interface_check": file_identity(PREFLIGHT / "snapshot-interface-check.json"),
        "historical_R_manifest": file_identity(PREF_TRAIN / "branch-manifest.json"),
        "historical_R_protocol": file_identity(PREF_TRAIN / "protocol.json"),
        "historical_R_labels": file_identity(PREF_TRAIN / "step-vector-rewards.jsonl"),
        "historical_R_reproduction": file_identity(PREF_TRAIN / "historical-reproduction.json"),
        "old_budget": file_identity(HISTORICAL_BUDGET_PATH),
    }
    return {
        "schema": "ackguard-corrected-source-input-identities/1.0.0",
        "protocol_id": PROTOCOL_ID,
        "historical_environment_contract": "native M10 25-action source snapshot; not current finite-communication and not D-02 17-action",
        "fixed_guard_sha256": EXPECTED_GUARD_SHA256,
        "fixed_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "fixed_snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
        "fixed_old_runner_sha256": EXPECTED_OLD_RUNNER_SHA256,
        "historical_source": historical_files,
        "local_inputs": local_files,
        "runtime_dependencies": runtime_dependency_identities(),
        "historical_inputs": input_files,
        "historical_input_hashes": historical_input_hashes(),
        "r_control_bridge_inputs": r_control_bridge_identities(),
        "r_control_bridge_hashes": r_control_bridge_hashes(),
        "old_guard_rule_preserved": True,
        "old_two_guard_branches_reused": False,
        "dynamic_execution_called": False,
    }


def runtime_dependency_identities() -> dict[str, dict[str, Any]]:
    """Return every local file whose import or runtime behavior is pinned."""
    return {name: file_identity(path) for name, path in sorted(RUNTIME_DEPENDENCY_PATHS.items())}


def runtime_dependency_hashes() -> dict[str, str | None]:
    return {name: identity.get("sha256") for name, identity in runtime_dependency_identities().items()}


def historical_input_hashes() -> dict[str, str | None]:
    """Pin every historical artifact used by the compatibility gate."""
    return {name: (sha256_file(path) if path.is_file() else None) for name, path in sorted(HISTORICAL_AUTH_INPUT_PATHS.items())}


def r_control_bridge_identities() -> dict[str, dict[str, Any]]:
    """Return the pinned H-004 bridge helper and provenance identities."""
    return {name: file_identity(path) for name, path in sorted(R_CONTROL_BRIDGE_INPUT_PATHS.items())}


def r_control_bridge_hashes() -> dict[str, str | None]:
    return {name: identity.get("sha256") for name, identity in r_control_bridge_identities().items()}


def _bridge_observation_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    obs = snapshot.get("obs")
    if not isinstance(obs, Mapping):
        raise ProtocolError("saved R-control bridge snapshot has no observation mapping")
    if "flat" not in obs or "mask" not in obs or "version" not in obs or "time" not in obs:
        raise ProtocolError("saved R-control bridge observation is missing flat/mask/version/time")
    summary = {
        "flat": {"dtype": str(np.asarray(obs["flat"]).dtype), "shape": list(np.asarray(obs["flat"]).shape)},
        "mask": {"dtype": str(np.asarray(obs["mask"]).dtype), "shape": list(np.asarray(obs["mask"]).shape)},
        "version": int(obs["version"]),
        "time": float(obs["time"]),
        "public_entity_ids": _jsonable(obs.get("public_entity_ids")),
        "continuation_actions": list(obs.get("continuation_actions", ())),
        "trigger_flags": _jsonable(obs.get("trigger_flags", {})),
    }
    env = snapshot.get("env")
    if env is not None:
        if hasattr(env, "_step_index"):
            summary["snapshot_step_index"] = int(env._step_index)
        if hasattr(env, "_episode_id"):
            summary["snapshot_episode_id"] = str(env._episode_id)
    return summary


def _bridge_provenance_rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    row = payload.get("row")
    if isinstance(row, Mapping) and row.get("branch_key") is not None:
        return {str(row["branch_key"]): row}
    rows = payload.get("rows", ())
    if isinstance(rows, Mapping):
        return {str(key): value for key, value in rows.items() if isinstance(value, Mapping)}
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return {
            str(value["branch_key"]): value
            for value in rows
            if isinstance(value, Mapping) and value.get("branch_key") is not None
        }
    return {}


def _validate_r_control_bridge_input_pins(supplied: Mapping[str, Any] | None) -> list[str]:
    failures: list[str] = []
    if not isinstance(supplied, Mapping):
        return ["r-control bridge source identities are missing"]
    actual = r_control_bridge_identities()
    for name, identity in actual.items():
        provided = supplied.get(name)
        if not isinstance(provided, Mapping):
            failures.append(f"missing r-control bridge source identity: {name}")
            continue
        for field in ("path", "bytes", "sha256"):
            if provided.get(field) != identity.get(field):
                failures.append(f"r-control bridge source identity mismatch: {name}.{field}")
        if identity.get("exists") is not True:
            failures.append(f"r-control bridge source is missing: {name}")
    expected_hashes = {
        "schema_helper": EXPECTED_R_CONTROL_BRIDGE_SCHEMA_HELPER_SHA256,
        "audit_loader": EXPECTED_R_CONTROL_BRIDGE_AUDIT_LOADER_SHA256,
        "bridge_evidence": EXPECTED_R_CONTROL_BRIDGE_EVIDENCE_SHA256,
        "bridge_qualified_compatibility": EXPECTED_R_CONTROL_BRIDGE_QUALIFIED_SHA256,
        "bridge_strict_compatibility": EXPECTED_R_CONTROL_BRIDGE_STRICT_SHA256,
        "remote_receipt": EXPECTED_R_CONTROL_BRIDGE_RECEIPT_SHA256,
    }
    for name, expected in expected_hashes.items():
        if actual.get(name, {}).get("sha256") != expected:
            failures.append(f"local r-control bridge input hash is not the frozen H-004 hash: {name}")
    try:
        receipt = read_json(R_CONTROL_BRIDGE_RECEIPT)
        if (
            receipt.get("verification_status") != "complete"
            or receipt.get("ref_matches") is not True
            or receipt.get("signature_verified") is not True
            or receipt.get("commit") != EXPECTED_R_CONTROL_BRIDGE_COMMIT
        ):
            failures.append("H-004 remote bridge receipt is incomplete or not verified")
    except Exception as exc:
        failures.append(f"H-004 remote bridge receipt is unreadable: {type(exc).__name__}")
    return failures


def _load_current_bridge_rows(
    manifest: Mapping[str, Any],
    provenance_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild each pinned bridge row from current saved inputs, without replay."""
    selected_keys = set(provenance_rows)
    manifest_rows = {str(row["control_branch_key"]): row for row in manifest.get("rows", ())}
    missing_manifest = sorted(selected_keys - set(manifest_rows))
    if missing_manifest:
        raise ProtocolError(f"bridge provenance contains non-manifest control keys: {missing_manifest}")
    labels = read_jsonl(PREF_TRAIN / "step-vector-rewards.jsonl")
    labels_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in labels:
        if row.get("mode") != "R":
            continue
        key = f"{row.get('prefix_id')}|repeat-{int(row.get('repeat', -1))}|mode-R"
        labels_by_key.setdefault(key, []).append(row)
    results = read_jsonl(PREF_TRAIN / "branch-results.jsonl")
    results_by_key = {
        f"{row.get('prefix_id')}|repeat-{int(row.get('repeat', -1))}|mode-R": row
        for row in results
        if row.get("mode") == "R" and row.get("split") == "train"
    }
    snapshot_ids = {str(manifest_rows[key]["prefix_id"]) for key in selected_keys}
    with (HISTORY / "prefix-snapshots.pkl").open("rb") as stream:
        snapshots = pickle.load(stream)
    snapshot_by_prefix = {
        str(row.get("prefix_id")): row
        for row in snapshots
        if isinstance(row, Mapping) and row.get("prefix_id") in snapshot_ids
    }
    if set(snapshot_by_prefix) != snapshot_ids:
        raise ProtocolError("pinned R-control bridge snapshots are incomplete")

    current_rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    from tools.r_control_digest_schema_20260922 import evaluate_bridge

    for key in sorted(selected_keys):
        item = manifest_rows[key]
        label_rows = sorted(labels_by_key.get(key, ()), key=lambda row: int(row.get("step", -1)))
        label = next((row for row in label_rows if int(row.get("step", -1)) == 1), None)
        result = results_by_key.get(key)
        snapshot = snapshot_by_prefix.get(str(item["prefix_id"]))
        if not isinstance(label, Mapping) or not isinstance(result, Mapping) or not isinstance(snapshot, Mapping):
            raise ProtocolError(f"missing current R-control bridge input for {key}")
        obs = snapshot.get("obs")
        if not isinstance(obs, Mapping):
            raise ProtocolError(f"missing saved observation for {key}")
        actual_identity = {
            "prefix_id": label.get("prefix_id"),
            "parent_id": label.get("parent_id"),
            "condition": label.get("condition"),
            "model_seed": label.get("model_seed"),
            "repeat": label.get("repeat"),
            "mode": label.get("mode"),
            "step": label.get("step"),
            "exogenous_key": result.get("exogenous_key"),
        }
        expected_identity = {
            "prefix_id": item["prefix_id"],
            "parent_id": item["parent_id"],
            "condition": item["condition"],
            "model_seed": int(item["model_seed"]),
            "repeat": int(item["repeat"]),
            "mode": "R",
            "step": 1,
            "exogenous_key": item["historical_exogenous_key"],
        }
        current = {
            "branch_key": key,
            "observation": obs,
            "label_identity": actual_identity,
            "expected_identity": expected_identity,
            "label_digest": label.get("public_observation_sha256"),
            "prefix_digest": snapshot.get("prefix_digest"),
            "prefix_time": snapshot.get("prefix_time"),
            "label_time_before": label.get("time_before"),
            "label_time_after": label.get("time_after"),
        }
        provenance = provenance_rows[key]
        for field in (
            "branch_key",
            "label_identity",
            "expected_identity",
            "label_digest",
            "prefix_digest",
            "prefix_time",
            "label_time_before",
            "label_time_after",
        ):
            if current.get(field) != provenance.get(field):
                raise ProtocolError(f"stale H-004 bridge provenance for {key}: {field}")
        if provenance.get("observation_summary") != _bridge_observation_summary(snapshot):
            raise ProtocolError(f"stale H-004 bridge provenance for {key}: observation_summary")
        evaluation = evaluate_bridge(
            current["observation"],
            label_identity=current["label_identity"],
            expected_identity=current["expected_identity"],
            label_digest=current["label_digest"],
            prefix_digest=current["prefix_digest"],
            prefix_time=current["prefix_time"],
            label_time_before=current["label_time_before"],
            label_time_after=current["label_time_after"],
        )
        if evaluation.get("schema_compatible") is not True or evaluation.get("status") != "schema_compatible":
            raise ProtocolError(f"current H-004 bridge evaluation failed for {key}: {evaluation.get('reasons')}")
        current_rows.append(current)
        evaluations.append({"branch_key": key, "status": evaluation.get("status"), "evaluation": evaluation})
    return current_rows, evaluations


def historical_compatibility_with_pinned_bridge(
    manifest: Mapping[str, Any],
    supplied_bridge_inputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recompute the H-004 bridge and then re-run the full R-control gate."""
    pin_failures = _validate_r_control_bridge_input_pins(supplied_bridge_inputs)
    if pin_failures:
        raise ProtocolError("r-control bridge input gate failed: " + "; ".join(pin_failures))
    provenance = read_json(R_CONTROL_BRIDGE_EVIDENCE)
    provenance_rows = _bridge_provenance_rows(provenance)
    if not provenance_rows:
        raise ProtocolError("H-004 bridge evidence contains no branch-bound provenance row")
    strict = historical_compatibility(manifest)
    strict_keys = {
        str(row["control_branch_key"])
        for row in strict.get("rows", ())
        if row.get("reusable_as_R_control") is False
    }
    if strict_keys - set(provenance_rows):
        raise ProtocolError(
            "H-004 bridge evidence is missing a current bridge for strict R-control failures: "
            + ", ".join(sorted(strict_keys - set(provenance_rows)))
        )
    current_rows, evaluations = _load_current_bridge_rows(manifest, provenance_rows)
    qualified = historical_compatibility(manifest, schema_bridge_evidence={"rows": current_rows})
    if qualified.get("status") != "reusable_as_R_control" or int(qualified.get("summary", {}).get("reusable_R_control_pairs", 0)) != len(manifest.get("rows", ())):
        raise ProtocolError(
            "recomputed H-004 bridge did not qualify every R control: "
            + json.dumps(qualified.get("summary", {}), sort_keys=True)
        )
    return {
        "status": qualified.get("status"),
        "strict": strict,
        "qualified": qualified,
        "bridge_evaluations": evaluations,
        "provenance_branch_keys": sorted(provenance_rows),
        "recomputed_from_current_inputs": True,
        "saved_qualified_boolean_trusted": False,
    }


def _safe_read_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, f"missing:{path}"
    try:
        return read_json(path), None
    except Exception as exc:
        return None, f"unreadable:{path}:{type(exc).__name__}:{exc}"


def _historical_control_evidence() -> dict[str, Any]:
    """Load the actual historical R evidence used by the control gate."""
    control, control_error = _safe_read_json(HISTORICAL_RUN / "control-reuse-manifest.json")
    protocol, protocol_error = _safe_read_json(PREF_TRAIN / "protocol.json")
    reproduction, reproduction_error = _safe_read_json(PREF_TRAIN / "historical-reproduction.json")
    source, source_error = _safe_read_json(PREF_TRAIN / "source-checkpoint-manifest.json")
    snapshot, snapshot_error = _safe_read_json(PREF_TRAIN / "snapshot-identity.json")
    snapshot_interface, snapshot_interface_error = _safe_read_json(PREFLIGHT / "snapshot-interface-check.json")
    labels = read_jsonl(PREF_TRAIN / "step-vector-rewards.jsonl")
    branch_manifest, branch_manifest_error = _safe_read_json(PREF_TRAIN / "branch-manifest.json")
    rows = branch_manifest.get("rows", branch_manifest) if isinstance(branch_manifest, (Mapping, list)) else []
    historical_results = read_jsonl(PREF_TRAIN / "branch-results.jsonl")
    old_guard_snapshot, old_guard_snapshot_error = _safe_read_json(HISTORICAL_RUN / "snapshot-identity.json")
    old_guard_results = read_jsonl(HISTORICAL_RUN / "branch-results.jsonl")
    return {
        "control": control,
        "control_error": control_error,
        "protocol": protocol,
        "protocol_error": protocol_error,
        "reproduction": reproduction,
        "reproduction_error": reproduction_error,
        "source": source,
        "source_error": source_error,
        "snapshot": snapshot,
        "snapshot_error": snapshot_error,
        "snapshot_interface": snapshot_interface,
        "snapshot_interface_error": snapshot_interface_error,
        "labels": labels,
        "branch_rows": rows,
        "branch_manifest_error": branch_manifest_error,
        "historical_results": historical_results,
        "old_guard_snapshot": old_guard_snapshot,
        "old_guard_snapshot_error": old_guard_snapshot_error,
        "old_guard_results": old_guard_results,
    }


def _historical_source_checkpoint_ok(evidence: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    source = evidence.get("source")
    if not isinstance(source, Mapping):
        return False, [str(evidence.get("source_error") or "source-checkpoint-manifest missing")]
    checkpoints = source.get("checkpoints", ())
    checkpoint = next((row for row in checkpoints if int(row.get("seed", -1)) == 1101 and row.get("group") == "P_train"), None)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256 or checkpoint.get("matches") is not True:
        reasons.append("seed-1101 P_train checkpoint identity mismatch")
    for path in (
        PREF_TRAIN / "protocol.json",
        PREF_TRAIN / "branch-manifest.json",
        PREF_TRAIN / "historical-reproduction.json",
        PREF_TRAIN / "step-vector-rewards.jsonl",
    ):
        if not path.is_file():
            reasons.append(f"historical input is missing: {path.name}")
    return not reasons, reasons


def _historical_label_contract(
    labels: Sequence[Mapping[str, Any]],
    item: Mapping[str, Any],
    output_result: Mapping[str, Any] | None,
    snapshot_row: Mapping[str, Any] | None,
    snapshot_interface_row: Mapping[str, Any] | None,
    schema_bridge_row: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Check the saved R labels against the immutable step contract.

    The old reuse artifact only counted labels.  This check keeps the saved
    evidence read-only while validating the step sequence, repeat key,
    discount metadata, terminal flags, reward scales and an independent
    float32 reward reconstruction for every selected pair.
    """
    expected_steps = list(range(1, int(item["historical_step_count"]) + 1))
    ordered = sorted((dict(row) for row in labels), key=lambda row: int(row.get("step", -1)))
    actual_steps = [int(row.get("step", -1)) for row in ordered]
    expected_gamma = list(range(0, len(expected_steps)))
    actual_gamma = [int(row.get("gamma_index", -1)) for row in ordered]
    actual_discount = [float(row.get("gamma", float("nan"))) for row in ordered]
    expected_key = str(item["historical_exogenous_key"])
    label_keys = sorted({str(row.get("exogenous_key", row.get("historical_exogenous_key"))) for row in ordered if row.get("exogenous_key", row.get("historical_exogenous_key")) is not None})
    result_key = output_result.get("exogenous_key") if isinstance(output_result, Mapping) else None
    observed_keys = label_keys or ([str(result_key)] if result_key is not None else [])
    key_ok = observed_keys == [expected_key] and (not label_keys or all(str(row.get("exogenous_key", row.get("historical_exogenous_key"))) == expected_key for row in ordered))
    preference_expected = list(BEHAVIOR_PREFERENCE)
    scales_expected = dict(REWARD_SCALES)
    preference_actual = [row.get("behavior_preference") for row in ordered]
    scales_actual = [row.get("reward_scales") for row in ordered]
    preference_ok = all(actual == preference_expected for actual in preference_actual)
    scales_ok = all(actual == scales_expected for actual in scales_actual)
    discount_ok = all(math.isfinite(value) and abs(value - GAMMA) <= TOLERANCE for value in actual_discount)
    identity_expected = {
        "prefix_id": str(item["prefix_id"]),
        "parent_id": str(item["parent_id"]),
        "condition": "W1",
        "model_seed": 1101,
        "repeat": int(item["repeat"]),
        "mode": "R",
    }
    identity_actual = [{field: row.get(field) for field in identity_expected} for row in ordered]
    identity_ok = all(all(actual.get(field) == expected for field, expected in identity_expected.items()) for actual in identity_actual)
    reward_failures: list[dict[str, Any]] = []
    for row in ordered:
        try:
            task_capacity = float(row["task_capacity"])
            uav_count = float(row["uav_count"])
            initial_energy = float(row["initial_energy"])
            completed_delta = float(row["completed_delta"])
            expired_delta = float(row["expired_delta"])
            used_energy = float(row["energy_used_delta"])
            expected_vector = [
                float(value)
                for value in np.asarray(
                    ((completed_delta - expired_delta) / task_capacity, -used_energy / max(1e-9, uav_count * initial_energy)),
                    dtype=np.float32,
                ).tolist()
            ]
            actual_vector = [float(value) for value in row["vector_reward"]]
            reward_ok = len(actual_vector) == 2 and all(abs(left - right) <= TOLERANCE for left, right in zip(actual_vector, expected_vector))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            expected_vector = None
            actual_vector = row.get("vector_reward")
            reward_ok = False
        if not reward_ok:
            reward_failures.append({"step": row.get("step"), "actual": actual_vector, "expected": expected_vector})
    terminal_expected = {
        "terminated": True,
        "truncated": False,
        "end_reason": item.get("historical_end_reason") or "terminated",
    }
    terminal_actual = {
        "terminated": output_result.get("terminated") if isinstance(output_result, Mapping) else None,
        "truncated": output_result.get("truncated") if isinstance(output_result, Mapping) else None,
        "end_reason": output_result.get("end_reason") if isinstance(output_result, Mapping) else None,
        "last_label_terminated": ordered[-1].get("terminated") if ordered else None,
        "last_label_truncated": ordered[-1].get("truncated") if ordered else None,
    }
    terminal_ok = (
        terminal_actual["terminated"] is True
        and terminal_actual["truncated"] is False
        and terminal_actual["end_reason"] == terminal_expected["end_reason"]
        and terminal_actual["last_label_terminated"] is True
        and terminal_actual["last_label_truncated"] is False
    )
    first_label = next((row for row in ordered if int(row.get("step", -1)) == 1), None)
    snapshot_digest = snapshot_row.get("prefix_digest") if isinstance(snapshot_row, Mapping) else None
    interface_digest = snapshot_interface_row.get("prefix_digest") if isinstance(snapshot_interface_row, Mapping) else None
    snapshot_interface_ok = bool(
        isinstance(snapshot_interface_row, Mapping)
        and snapshot_interface_row.get("prefix_id") == item["prefix_id"]
        and snapshot_interface_row.get("prefix_digest_matches_prefix_index") is True
        and isinstance(interface_digest, str)
        and interface_digest == snapshot_digest
    )
    snapshot_actual = {
        "prefix_id": item["prefix_id"],
        "step": first_label.get("step") if isinstance(first_label, Mapping) else None,
        "public_observation_sha256": first_label.get("public_observation_sha256") if isinstance(first_label, Mapping) else None,
        "label_source": first_label.get("label_source") if isinstance(first_label, Mapping) else None,
        "source": "preference-vector-label-train-v1/step-vector-rewards.jsonl",
    }
    snapshot_expected = {
        "prefix_id": item["prefix_id"],
        "prefix_digest": snapshot_digest,
        "snapshot_identity_schema": "snapshot-identity/1.0.0",
        "snapshot_interface_prefix_digest": interface_digest,
        "snapshot_interface_schema": "allocation-branch-snapshot-interface-check/2.0.0",
        "transform": "identity comparison of the R step=1 public_observation_sha256 to the immutable prefix_digest",
    }
    snapshot_ok = bool(first_label) and snapshot_interface_ok and snapshot_actual["public_observation_sha256"] == snapshot_digest
    schema_bridge_check: dict[str, Any] = {"status": "not_requested", "ok": True}
    if schema_bridge_row is not None:
        bridge_identity_fields = ("prefix_id", "parent_id", "condition", "model_seed", "repeat", "mode", "step", "exogenous_key")
        authoritative_expected_identity = {
            "prefix_id": str(item["prefix_id"]),
            "parent_id": str(item["parent_id"]),
            "condition": str(item["condition"]),
            "model_seed": int(item["model_seed"]),
            "repeat": int(item["repeat"]),
            "mode": "R",
            "step": 1,
            "exogenous_key": str(item["historical_exogenous_key"]),
        }
        authoritative_label_identity = None
        current_branch_identity = None
        if isinstance(first_label, Mapping):
            current_branch_identity = {
                field: (output_result.get(field) if isinstance(output_result, Mapping) else None)
                for field in bridge_identity_fields
                if field != "step"
            }
            authoritative_label_identity = {
                "prefix_id": first_label.get("prefix_id"),
                "parent_id": first_label.get("parent_id"),
                "condition": first_label.get("condition"),
                "model_seed": first_label.get("model_seed"),
                "repeat": first_label.get("repeat"),
                "mode": first_label.get("mode"),
                "step": first_label.get("step"),
                "exogenous_key": output_result.get("exogenous_key") if isinstance(output_result, Mapping) else None,
            }
        authoritative_prefix_time = None
        if isinstance(snapshot_interface_row, Mapping):
            authoritative_prefix_time = snapshot_interface_row.get("prefix_time")
        if authoritative_prefix_time is None and isinstance(snapshot_row, Mapping):
            authoritative_prefix_time = snapshot_row.get("prefix_time")
        authoritative_binding = {
            "label_identity": authoritative_label_identity,
            "expected_identity": authoritative_expected_identity,
            "label_digest": first_label.get("public_observation_sha256") if isinstance(first_label, Mapping) else None,
            "prefix_digest": snapshot_digest,
            "prefix_time": authoritative_prefix_time,
            "label_time_before": first_label.get("time_before") if isinstance(first_label, Mapping) else None,
            "label_time_after": first_label.get("time_after") if isinstance(first_label, Mapping) else None,
        }
        bridge_binding_mismatches: list[str] = []

        def compare_identity(name: str, actual: Any, expected: Any) -> None:
            if not isinstance(actual, Mapping):
                bridge_binding_mismatches.append(f"missing_bridge_identity:{name}")
                return
            if not isinstance(expected, Mapping):
                bridge_binding_mismatches.append(f"missing_authoritative_identity:{name}")
                return
            for field in bridge_identity_fields:
                if field not in actual:
                    bridge_binding_mismatches.append(f"missing_bridge_identity:{name}.{field}")
                elif field not in expected or actual[field] != expected[field]:
                    bridge_binding_mismatches.append(f"bridge_binding_identity:{name}.{field}")

        compare_identity("label_identity", schema_bridge_row.get("label_identity"), authoritative_label_identity)
        compare_identity("expected_identity", schema_bridge_row.get("expected_identity"), authoritative_expected_identity)
        if isinstance(first_label, Mapping) and isinstance(output_result, Mapping):
            for field in bridge_identity_fields:
                if field == "step":
                    continue
                label_value = first_label.get(field)
                branch_value = output_result.get(field)
                # Historical step labels do not carry exogenous_key; the
                # matched branch result is the authoritative source for it.
                if field == "exogenous_key" and label_value is None:
                    continue
                if label_value != branch_value:
                    bridge_binding_mismatches.append(f"current_branch_identity:{field}")
        elif not isinstance(first_label, Mapping):
            bridge_binding_mismatches.append("missing_authoritative_label")
        elif not isinstance(output_result, Mapping):
            bridge_binding_mismatches.append("missing_authoritative_branch_result")
        scalar_bindings = (
            ("label_digest", schema_bridge_row.get("label_digest"), authoritative_binding["label_digest"]),
            ("prefix_digest", schema_bridge_row.get("prefix_digest"), authoritative_binding["prefix_digest"]),
            ("prefix_time", schema_bridge_row.get("prefix_time"), authoritative_binding["prefix_time"]),
            ("label_time_before", schema_bridge_row.get("label_time_before"), authoritative_binding["label_time_before"]),
            ("label_time_after", schema_bridge_row.get("label_time_after"), authoritative_binding["label_time_after"]),
        )
        for field, actual, expected in scalar_bindings:
            if expected is None:
                bridge_binding_mismatches.append(f"missing_authoritative_{field}")
            elif actual != expected:
                bridge_binding_mismatches.append(f"bridge_binding:{field}")
        schema_bridge_check = {
            "status": "insufficient_evidence",
            "schema_compatible": False,
            "reasons": list(bridge_binding_mismatches),
            "binding_mismatches": list(bridge_binding_mismatches),
            "authoritative_binding": authoritative_binding,
            "current_branch_identity": current_branch_identity,
        }
        try:
            if not bridge_binding_mismatches:
                from tools.r_control_digest_schema_20260922 import evaluate_bridge

                schema_bridge_check = evaluate_bridge(
                    schema_bridge_row.get("observation"),
                    label_identity=authoritative_label_identity,
                    expected_identity=authoritative_expected_identity,
                    label_digest=authoritative_binding["label_digest"],
                    prefix_digest=authoritative_binding["prefix_digest"],
                    prefix_time=authoritative_binding["prefix_time"],
                    label_time_before=authoritative_binding["label_time_before"],
                    label_time_after=authoritative_binding["label_time_after"],
                )
                schema_bridge_check["binding_mismatches"] = []
                schema_bridge_check["authoritative_binding"] = authoritative_binding
                schema_bridge_check["current_branch_identity"] = current_branch_identity
        except Exception as exc:
            schema_bridge_check = {
                "status": "insufficient_evidence",
                "schema_compatible": False,
                "reasons": [f"bridge_evaluation_failed:{type(exc).__name__}:{exc}"],
                "binding_mismatches": list(bridge_binding_mismatches),
                "authoritative_binding": authoritative_binding,
                "current_branch_identity": current_branch_identity,
            }
        schema_bridge_check["ok"] = (
            schema_bridge_check.get("schema_compatible") is True
            and not schema_bridge_check.get("binding_mismatches")
        )
        if schema_bridge_check["ok"]:
            snapshot_ok = True
            snapshot_actual["schema_bridge"] = schema_bridge_check
            snapshot_expected["schema_bridge"] = "schema-compatible saved-observation bridge"
        else:
            snapshot_ok = False
    checks = {
        "step_sequence": {"actual": actual_steps, "expected": expected_steps, "ok": actual_steps == expected_steps},
        "gamma_index": {"actual": actual_gamma, "expected": expected_gamma, "ok": actual_gamma == expected_gamma},
        "identity": {"actual": identity_actual, "expected": identity_expected, "ok": identity_ok},
        "gamma": {"actual": actual_discount, "expected": GAMMA, "ok": discount_ok},
        "exogenous_key": {"actual": observed_keys, "expected": [expected_key], "ok": key_ok, "label_rows_with_key": bool(label_keys)},
        "behavior_preference": {"actual": preference_actual, "expected": [preference_expected] * len(ordered), "ok": preference_ok},
        "reward_scales": {"actual": scales_actual, "expected": [scales_expected] * len(ordered), "ok": scales_ok},
        "terminal_contract": {"actual": terminal_actual, "expected": terminal_expected, "ok": terminal_ok},
        "independent_vector_reward": {"actual_failures": reward_failures, "expected": "float32 vector from counts/energy/config", "ok": not reward_failures},
        "snapshot_schema_equivalence": {
            "actual": {
                "snapshot_identity_schema": "snapshot-identity/1.0.0",
                "snapshot_interface_schema": "allocation-branch-snapshot-interface-check/2.0.0",
                "snapshot_interface_prefix_digest": interface_digest,
                "prefix_digest_matches_prefix_index": snapshot_interface_row.get("prefix_digest_matches_prefix_index") if isinstance(snapshot_interface_row, Mapping) else None,
            },
            "expected": "validated snapshot-interface row with the same prefix_digest as snapshot-identity",
            "ok": snapshot_interface_ok,
        },
        "snapshot_initial_digest": {"actual": snapshot_actual, "expected": snapshot_expected, "ok": snapshot_ok},
        "schema_bridge": schema_bridge_check,
    }
    reasons = [f"historical label contract failed: {name}" for name, value in checks.items() if not value.get("ok")]
    return checks, reasons


def _schema_bridge_rows(evidence: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    """Index optional, externally pinned saved-observation bridge evidence."""
    if not isinstance(evidence, Mapping):
        return {}
    rows = evidence.get("rows", evidence)
    if isinstance(rows, Mapping):
        return {str(key): value for key, value in rows.items() if isinstance(value, Mapping)}
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        indexed = {}
        for row in rows:
            if isinstance(row, Mapping) and row.get("branch_key") is not None:
                indexed[str(row["branch_key"])] = row
        return indexed
    return {}


def historical_compatibility(
    manifest: Mapping[str, Any],
    *,
    schema_bridge_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Determine whether historical R rows remain reusable as original controls.

    The gate reads the saved control-reuse, protocol, reproduction, source and
    label artifacts.  It does not compare R actions with the new guard, since
    that would confuse the control contract with the treatment trajectory.
    """
    validate_manifest(manifest)
    bridge_rows = _schema_bridge_rows(schema_bridge_evidence)
    evidence = _historical_control_evidence()
    control = evidence.get("control") if isinstance(evidence.get("control"), Mapping) else {}
    control_rows = {
        str(row.get("branch_key")): row
        for row in control.get("rows", ())
        if isinstance(row, Mapping)
    }
    reproduction = evidence.get("reproduction") if isinstance(evidence.get("reproduction"), Mapping) else {}
    reproduction_rows = {
        str(row.get("branch_key")): row
        for row in reproduction.get("rows", ())
        if isinstance(row, Mapping) and str(row.get("branch_key", "")).endswith("|mode-R")
    }
    historical_result_rows = {
        f"{row.get('prefix_id')}|repeat-{int(row.get('repeat', -1))}|mode-R": row
        for row in evidence.get("historical_results", ())
        if isinstance(row, Mapping) and row.get("split") == "train" and row.get("mode") == "R"
    }
    protocol = evidence.get("protocol") if isinstance(evidence.get("protocol"), Mapping) else {}
    protocol_ok = bool(
        protocol.get("schema") == "preference-vector-label-train-v1/1.0.0"
        and protocol.get("status") == "preflight_passed"
        and protocol.get("behavior", {}).get("deterministic_frozen_policy") is True
        and protocol.get("behavior", {}).get("no_action_sampling") is True
        and float(protocol.get("behavior", {}).get("gamma", protocol.get("reward_contract", {}).get("gamma", GAMMA))) == GAMMA
        and protocol.get("reward_contract", {}).get("implementation") == "historical source gppo_world.joint_training._vector_reward"
        and protocol.get("reward_contract", {}).get("truncation") == "reported separately; no zero fill or critic bootstrap"
    )
    source_ok, source_reasons = _historical_source_checkpoint_ok(evidence)
    snapshot = evidence.get("snapshot") if isinstance(evidence.get("snapshot"), Mapping) else {}
    snapshot_payload = snapshot.get("rows", {})
    if isinstance(snapshot_payload, Mapping):
        snapshot_rows = {
            str(prefix_id): dict(row)
            for prefix_id, row in snapshot_payload.items()
            if isinstance(row, Mapping)
        }
    else:
        snapshot_rows = {
            str(row.get("prefix_id")): row
            for row in snapshot_payload
            if isinstance(row, Mapping) and row.get("prefix_id") is not None
        }
    snapshot_interface = evidence.get("snapshot_interface") if isinstance(evidence.get("snapshot_interface"), Mapping) else {}
    snapshot_interface_rows = {
        str(row.get("prefix_id")): row
        for row in snapshot_interface.get("rows", ())
        if isinstance(row, Mapping) and row.get("prefix_id") is not None
    }
    labels_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in evidence.get("labels", ()):
        if row.get("mode") == "R":
            key = f"{row.get('prefix_id')}|repeat-{int(row.get('repeat', -1))}|mode-R"
            labels_by_key.setdefault(key, []).append(row)
    rows = []
    for item in manifest["rows"]:
        key = str(item["control_branch_key"])
        control_row = control_rows.get(key)
        reproduction_row = reproduction_rows.get(key)
        historical_result = historical_result_rows.get(key)
        labels = labels_by_key.get(key, [])
        label_checks, label_reasons = _historical_label_contract(
            labels,
            item,
            historical_result,
            snapshot_rows.get(str(item["prefix_id"])),
            snapshot_interface_rows.get(str(item["prefix_id"])),
            bridge_rows.get(key),
        )
        snapshot_row = snapshot_rows.get(str(item["prefix_id"]))
        snapshot_interface_row = snapshot_interface_rows.get(str(item["prefix_id"]))
        prefix_ok = bool(
            isinstance(snapshot_row, Mapping)
            and isinstance(snapshot_row.get("prefix_digest"), str)
            and isinstance(snapshot_interface_row, Mapping)
            and snapshot_interface_row.get("prefix_digest") == snapshot_row.get("prefix_digest")
            and snapshot_interface_row.get("prefix_digest_matches_prefix_index") is True
        )
        control_ok = bool(
            control.get("status") == "matched"
            and control.get("rerun") is False
            and control.get("old_budget_modified") is False
            and isinstance(control_row, Mapping)
            and control_row.get("branch_key") == key
            and control_row.get("status") == "matched"
            and control_row.get("differences") == []
            and int(control_row.get("historical_steps", -1)) == int(item["historical_step_count"])
            and control_row.get("historical_exogenous_key") == item["historical_exogenous_key"]
            and control_row.get("historical_branch_row_sha256") == item.get("historical_branch_row_sha256")
            and len(labels) == int(item["historical_step_count"])
            and isinstance(reproduction_row, Mapping)
            and reproduction_row.get("branch_key") == key
            and reproduction_row.get("status") == "matched"
            and int(reproduction_row.get("historical_steps", -1)) == int(item["historical_step_count"])
            and isinstance(historical_result, Mapping)
            and historical_result.get("exogenous_key") == item["historical_exogenous_key"]
            and int(historical_result.get("env_steps", -1)) == int(item["historical_step_count"])
            and bool(historical_result.get("terminated")) is True
            and bool(historical_result.get("truncated")) is False
            and all(check.get("ok") for check in label_checks.values())
        )
        reasons = list(source_reasons)
        if not protocol_ok:
            reasons.append("historical R protocol does not pin deterministic eval/reward contract")
        if not prefix_ok:
            reasons.append("selected historical snapshot prefix digest/interface evidence is missing or inconsistent")
        if not control_ok:
            reasons.append("historical R control-reuse or label/reproduction row is not matched")
        reasons.extend(label_reasons)
        hidden_evidence = {"status": "insufficient_evidence", "evidence": ["historical snapshot interface records hidden presence only; no independent post-run hidden/cache comparison"]}
        rows.append({
            "branch_id": item["branch_id"],
            "control_branch_key": key,
            "parent_id": item["parent_id"],
            "repeat": int(item["repeat"]),
            "historical_step_count": int(item["historical_step_count"]),
            "historical_exogenous_key": item["historical_exogenous_key"],
            "identity": {
                "source_code_and_checkpoint": {"status": "pass" if source_ok else "insufficient_evidence", "evidence": ["source-checkpoint-manifest.json"] if source_ok else source_reasons},
                "historical_R_protocol": {"status": "pass" if protocol_ok else "insufficient_evidence", "evidence": ["protocol.json deterministic frozen policy, no sampling, gamma=0.99"] if protocol_ok else ["protocol.json contract fields do not match"]},
                "public_initial_snapshot": {
                    "status": "pass" if label_checks["snapshot_initial_digest"].get("ok") else "insufficient_evidence",
                    "evidence": ["R step=1 public_observation_sha256 directly compared with snapshot-identity prefix_digest and snapshot-interface-check"] if label_checks["snapshot_initial_digest"].get("ok") else ["R step=1 public_observation_sha256 did not match the historical snapshot prefix digest"],
                },
                "hidden_cache_non_pollution": hidden_evidence,
                "randomness_and_exogenous_repeat_key": {"status": "pass" if control_ok else "insufficient_evidence", "evidence": ["matched control-reuse row and historical exogenous key"] if control_ok else ["control/labels/reproduction key evidence incomplete"]},
                "reward_vector_and_discount": {"status": "pass" if protocol_ok else "insufficient_evidence", "evidence": ["historical protocol vector reward and gamma=0.99"] if protocol_ok else ["reward contract not pinned"]},
                "termination_contract": {"status": "pass" if control_ok else "insufficient_evidence", "evidence": ["matched R branch/reproduction terminal record"] if control_ok else ["historical R terminal reproduction not matched"]},
                "saved_label_contract": {"status": "pass" if all(check.get("ok") for check in label_checks.values()) else "insufficient_evidence", "checks": label_checks},
                "full_candidate_probabilities": {"status": "insufficient_evidence", "evidence": ["historical R decision artifacts do not contain complete all-action probability vectors"]},
                "continuation_controller_trajectory": {"status": "insufficient_evidence", "evidence": ["corrected guard trajectory is new treatment behavior and is not inferred from R"]},
            },
            "overall": "reusable_as_R_control" if control_ok and protocol_ok and source_ok and prefix_ok else "insufficient_evidence",
            "reusable_as_R_control": bool(control_ok and protocol_ok and source_ok and prefix_ok),
            "reusable_as_corrected_guard_trajectory": False,
            "historical_R_reexecuted": False,
            "actual": {
                "control_reuse": control_row,
                "reproduction": reproduction_row,
                "historical_branch_result": historical_result,
                "label_count": len(labels),
                "label_checks": label_checks,
            },
            "expected": {
                "control_branch_key": key,
                "historical_step_count": int(item["historical_step_count"]),
                "historical_exogenous_key": item["historical_exogenous_key"],
                "historical_end_reason": item.get("historical_end_reason") or "terminated",
                "snapshot_identity": snapshot_row,
                "snapshot_interface": snapshot_interface_row,
            },
            "reasons": reasons,
        })
    reusable = sum(bool(row["reusable_as_R_control"]) for row in rows)
    return {
        "schema": COMPATIBILITY_SCHEMA,
        "status": "reusable_as_R_control" if reusable == len(rows) else "insufficient_evidence",
        "historical_R_step_count": sum(int(row["historical_step_count"]) for row in rows),
        "rows": rows,
        "summary": {
            "pairs": len(rows),
            "reusable_R_control_pairs": reusable,
            "fully_reusable_pairs": sum(bool(row["reusable_as_corrected_guard_trajectory"]) for row in rows),
            "insufficient_pairs": len(rows) - reusable,
            "direct_first_label_snapshot_gate": {
                "comparison": "R step=1 public_observation_sha256 == snapshot-identity prefix_digest",
                "snapshot_identity": str(PREF_TRAIN / "snapshot-identity.json"),
                "snapshot_interface_check": str(PREFLIGHT / "snapshot-interface-check.json"),
                "old_guard_snapshot_excluded": True,
                "schema_bridge": bool(bridge_rows),
            },
            "old_two_guard_branches_excluded": True,
            "old_two_guard_evidence": {
                "control_reuse_artifact": str(HISTORICAL_RUN / "control-reuse-manifest.json"),
                "status": "not_part_of_R_control_gate",
                "reason": "old guard branches are historical treatment evidence and are not silently promoted to the corrected matrix",
            },
            "reason": "R control reuse is evaluated from saved matched rows; corrected guard trajectory remains unexecuted and cannot be inferred from R",
        },
        "evidence_files": {
            "control_reuse": file_identity(HISTORICAL_RUN / "control-reuse-manifest.json"),
            "protocol": file_identity(PREF_TRAIN / "protocol.json"),
            "historical_reproduction": file_identity(PREF_TRAIN / "historical-reproduction.json"),
            "source_checkpoint": file_identity(PREF_TRAIN / "source-checkpoint-manifest.json"),
            "snapshot_identity": file_identity(PREF_TRAIN / "snapshot-identity.json"),
            "snapshot_interface_check": file_identity(PREFLIGHT / "snapshot-interface-check.json"),
            "labels": file_identity(PREF_TRAIN / "step-vector-rewards.jsonl"),
            "old_guard_snapshot_identity_excluded": file_identity(HISTORICAL_RUN / "snapshot-identity.json"),
        },
        "no_historical_R_rerun": True,
    }


def budget_proposal(identity: Mapping[str, Any]) -> dict[str, Any]:
    stage = _stage(identity)
    current_limit = int(stage.get("limit_amount", -1)) if stage else None
    reserved = int(stage.get("reserved", 0)) if stage else None
    verified = int(stage.get("verified", 0)) if stage else None
    unknown = int(stage.get("unknown", 0)) if stage else None
    pending = int(stage.get("pending", 0)) if stage else None
    metadata = identity.get("metadata", {}) if isinstance(identity.get("metadata"), Mapping) else {}
    try:
        legacy_run_limits = json.loads(str(metadata.get("run_limits", "{}")))
    except Exception:
        legacy_run_limits = {"raw": metadata.get("run_limits")}
    migration_audit = {
        "event": "ackguard_limit_extension_proposal",
        "same_sqlite": str(HISTORICAL_BUDGET_PATH),
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "old_reserved": 20,
        "old_verified": 20,
        "old_unknown": 0,
        "old_pending": 0,
        "new_global_limit": GLOBAL_LIMIT_AFTER_APPROVAL,
        "new_run_cap": MAX_TOTAL_STEPS,
        "original_sha256": identity.get("sha256"),
        "original_metadata_run_limits_preserved": legacy_run_limits,
        "write_original_now": False,
    }
    return {
        "schema": "ackguard-corrected-budget-proposal/1.0.0",
        "status": "proposal_only_not_applied",
        "current_read_only": {
            "path": str(HISTORICAL_BUDGET_PATH),
            "sha256": identity.get("sha256"),
            "limit": current_limit,
            "reserved": reserved,
            "verified": verified,
            "unknown": unknown,
            "pending": pending,
        },
        "recommendation": {
            "same_sqlite": True,
            "global_environment_steps_limit_before": current_limit,
            "global_environment_steps_limit_after": GLOBAL_LIMIT_AFTER_APPROVAL,
            "old_consumed_steps_preserved": 20,
            "new_guard_environment_step_cap": MAX_TOTAL_STEPS,
            "total_after": 404,
            "new_run_id": RUN_ID,
            "new_attempt_id": ATTEMPT_ID,
            "old_attempts_modified": False,
            "old_results_rerun": False,
        },
        "metadata_migration": {
            "preserve_existing_run_limits_key": True,
            "existing_metadata_run_limits": legacy_run_limits,
            "new_metadata_key": "run_limits_by_run",
            "new_metadata_value": {RUN_ID: {"environment_steps": MAX_TOTAL_STEPS, "optimizer_calls": 0, "world_updates": 0, "offline_updates": 0}},
            "do_not_overwrite_existing_run_limits_with_global_limit": True,
            "reason": "metadata.run_limits is legacy executor metadata; global stage limit and this run cap are distinct contracts",
        },
        "minimum_migration_design": {
            "schema": "persistent-budget-sqlite/1.0.0-compatible",
            "required_if_schema_cannot_authoritatively_extend_limit": True,
            "transaction": [
                "BEGIN IMMEDIATE",
                "assert PRAGMA integrity_check = 'ok'",
                "assert stages.environment_steps has limit=384, reserved=20, verified=20, unknown=0, pending=0",
                "assert run_id ackguard-corrected-protocol-20260922 and attempt_id ackguard-corrected-protocol-20260922-attempt-0001 are absent",
                "UPDATE stages SET limit_amount=404 WHERE stage='environment_steps'",
                "INSERT OR REPLACE INTO metadata(key,value) VALUES ('run_limits_by_run', '<canonical JSON mapping run_id to 384-step cap>')",
                "INSERT INTO history(event,stage,amount,attempt_id,reservation_id,reservation_index,pid,reason,created_at,run_id) VALUES ('limit_extension_proposal_applied_after_approval','environment_steps',0,'ackguard-corrected-protocol-20260922-attempt-0001',NULL,NULL,<pid>,'preserve old 20; add 404 global / 384 run cap',<time>,'ackguard-corrected-protocol-20260922')",
                "COMMIT",
                "reopen read-only, recompute SHA, verify old 20 rows unchanged, verify new run/attempt totals are empty, and record audit evidence",
            ],
            "rollback": "ROLLBACK before COMMIT on any assertion failure; never rewrite or copy the formal database",
            "apply_now": False,
            "approval_required": True,
        },
        "test_only_copy_plan": {
            "copy_source": str(HISTORICAL_BUDGET_PATH),
            "copy_target": "temporary test SQLite outside the formal run directory",
            "assert_source_sha256_unchanged": identity.get("sha256"),
            "simulate_transaction": True,
            "expected_after": {"global_limit": 404, "old_reserved": 20, "old_verified": 20, "old_unknown": 0, "old_pending": 0, "new_run_reserved": 0, "new_run_verified": 0, "new_run_unknown": 0, "new_run_pending": 0},
            "formal_database_write": False,
        },
        "audit_event": migration_audit,
        "current_gate": "blocked: available=364 < required=384 and current limit=384 < proposed total=404",
    }


def validate_authorization(
    authorization_path: Path,
    *,
    manifest_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Perform every gate before runtime factories or a budget object exist."""
    if not authorization_path.is_file():
        raise AuthorizationError(f"explicit authorization file is required: {authorization_path}")
    payload = read_json(authorization_path)
    if not isinstance(payload, Mapping) or payload.get("schema") != AUTH_SCHEMA:
        raise AuthorizationError("authorization schema mismatch")
    failures: list[str] = []
    if payload.get("status") != "authorized":
        failures.append("authorization status is not authorized")
    if payload.get("protocol", {}).get("id") != PROTOCOL_ID:
        failures.append("protocol id mismatch")
    run_payload = payload.get("run", {}) if isinstance(payload.get("run"), Mapping) else {}
    if run_payload.get("run_id") != RUN_ID or run_payload.get("attempt_id") != ATTEMPT_ID:
        failures.append("run/attempt identity is not the fixed corrected identity")
    protocol = payload.get("protocol", {})
    protocol_path = Path(str(protocol.get("path", "")))
    if not protocol_path.is_file():
        failures.append("pinned protocol file is missing")
    else:
        if sha256_file(protocol_path) != str(protocol.get("sha256")):
            failures.append("pinned protocol hash mismatch")
        try:
            if read_json(protocol_path).get("protocol_id") != PROTOCOL_ID:
                failures.append("pinned protocol content id mismatch")
        except Exception as exc:
            failures.append(f"pinned protocol is not readable JSON: {type(exc).__name__}")
    runner_path = Path(str(payload.get("source", {}).get("corrected_runner_path", Path(__file__))))
    if runner_path.resolve() != Path(__file__).resolve() or not runner_path.is_file() or sha256_file(runner_path) != str(payload.get("source", {}).get("corrected_runner_sha256")):
        failures.append("corrected runner hash mismatch")
    guard_path = WORKTREE / "gppo_world" / "ack_known_task_guard.py"
    if sha256_file(guard_path) != str(payload.get("source", {}).get("guard_sha256")) or sha256_file(guard_path) != EXPECTED_GUARD_SHA256:
        failures.append("guard hash mismatch")
    checkpoint_path = Path(str(payload.get("source", {}).get("checkpoint_path", CHECKPOINT)))
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != str(payload.get("source", {}).get("checkpoint_sha256")) or sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        failures.append("checkpoint hash mismatch")
    snapshot_path = Path(str(payload.get("source", {}).get("snapshot_path", HISTORY / "prefix-snapshots.pkl")))
    if not snapshot_path.is_file() or sha256_file(snapshot_path) != str(payload.get("source", {}).get("snapshot_sha256")) or sha256_file(snapshot_path) != EXPECTED_SNAPSHOT_SHA256:
        failures.append("prefix snapshot hash mismatch")
    supplied_sources = payload.get("source", {}).get("historical_source_hashes", {})
    if dict(supplied_sources) != EXPECTED_SOURCE_HASHES:
        failures.append("historical source hash set is not the fixed native source set")
    else:
        for name, expected in EXPECTED_SOURCE_HASHES.items():
            path = SOURCE_ROOT / "gppo_world" / name
            if not path.is_file() or sha256_file(path) != expected:
                failures.append(f"historical source changed or missing: {name}")
    supplied_runtime = payload.get("source", {}).get("runtime_dependency_hashes", {})
    actual_runtime = runtime_dependency_hashes()
    if dict(supplied_runtime) != actual_runtime:
        failures.append("runtime dependency hash set mismatch")
    supplied_historical_inputs = payload.get("source", {}).get("historical_input_hashes", {})
    actual_historical_inputs = historical_input_hashes()
    if dict(supplied_historical_inputs) != actual_historical_inputs:
        failures.append("historical input hash set mismatch")
    source_payload = payload.get("source", {}) if isinstance(payload.get("source"), Mapping) else {}
    supplied_bridge_inputs = source_payload.get("r_control_bridge")
    bridge_pin_failures = _validate_r_control_bridge_input_pins(supplied_bridge_inputs)
    if bridge_pin_failures:
        failures.extend(bridge_pin_failures)
    bridge_source_ok = not bridge_pin_failures
    if not manifest_path.is_file():
        failures.append("branch manifest is missing")
        manifest = None
    else:
        manifest = read_json(manifest_path)
        try:
            validate_manifest(manifest)
        except ProtocolError as exc:
            failures.append(str(exc))
        if sha256_file(manifest_path) != str(payload.get("branch_manifest_sha256")):
            failures.append("branch manifest hash mismatch")
        if manifest is not None:
            manifest_rows = list(manifest.get("rows", ())) if isinstance(manifest, Mapping) else []
            if len(manifest_rows) != 24 or any(row.get("execution_namespace") != RUN_ID for row in manifest_rows):
                failures.append("manifest run namespace or row count mismatch")
    compatibility_gate = payload.get("historical_R_compatibility", {})
    compatibility_path = Path(str(compatibility_gate.get("control_reuse_manifest_path", HISTORICAL_RUN / "control-reuse-manifest.json")))
    if compatibility_path.resolve() != (HISTORICAL_RUN / "control-reuse-manifest.json").resolve() or not compatibility_path.is_file():
        failures.append("historical R compatibility evidence path is missing or not canonical")
    else:
        if sha256_file(compatibility_path) != str(compatibility_gate.get("control_reuse_manifest_sha256")):
            failures.append("historical R compatibility evidence hash mismatch")
        try:
            control_evidence = read_json(compatibility_path)
            if control_evidence.get("status") != "matched" or control_evidence.get("rerun") is not False or control_evidence.get("old_budget_modified") is not False or int(control_evidence.get("count", -1)) != 24 or int(control_evidence.get("historical_steps", -1)) != 299:
                failures.append("historical R control reuse is not a matched 24-branch, 299-step read-only artifact")
            if any(row.get("status") != "matched" or row.get("differences") != [] for row in control_evidence.get("rows", ())):
                failures.append("historical R control reuse contains a mismatch")
        except Exception as exc:
            failures.append(f"historical R compatibility evidence is unreadable: {type(exc).__name__}")
    bridge_compatibility: dict[str, Any] | None = None
    if manifest is not None and bridge_source_ok:
        try:
            bridge_compatibility = historical_compatibility_with_pinned_bridge(manifest, supplied_bridge_inputs)
            if bridge_compatibility.get("status") != "reusable_as_R_control" or int(bridge_compatibility.get("qualified", {}).get("summary", {}).get("reusable_R_control_pairs", 0)) != 24:
                failures.append("recomputed H-004 bridge gate did not prove 24 reusable R controls")
        except Exception as exc:
            failures.append(f"recomputed H-004 bridge gate failed: {type(exc).__name__}: {exc}")
    supplied_manifest_path = Path(str(payload.get("branch_manifest_path", "")))
    if supplied_manifest_path.resolve() != manifest_path.resolve():
        failures.append("authorization branch manifest path does not match the requested manifest")
    if out_dir.exists() and any(out_dir.iterdir()):
        failures.append("output directory is not fresh; duplicate or partial execution is refused")
    budget = payload.get("budget", {})
    budget_path = Path(str(budget.get("path", HISTORICAL_BUDGET_PATH)))
    if budget_path.resolve() != HISTORICAL_BUDGET_PATH.resolve():
        failures.append("budget path is not the authorized historical ledger")
        budget_identity = None
    else:
        try:
            budget_identity = read_budget_identity(budget_path, run_id=str(payload.get("run", {}).get("run_id", RUN_ID)))
        except Exception as exc:
            budget_identity = None
            failures.append(f"budget read-only inspection failed: {type(exc).__name__}: {exc}")
    if budget_identity is not None:
        if budget_identity["sha256"] != str(budget.get("sha256")):
            failures.append("budget SQLite hash mismatch")
        if budget_identity.get("integrity_check") != "ok":
            failures.append("budget SQLite integrity is not ok")
        stage = _stage(budget_identity)
        expected_limit = int(budget.get("global_limit", -1))
        required = int(budget.get("required_new_steps", -1))
        if expected_limit != GLOBAL_LIMIT_AFTER_APPROVAL:
            failures.append("authorization must pin global_limit=404")
        if required != MAX_TOTAL_STEPS:
            failures.append("authorization must pin required_new_steps=384")
        if int(stage.get("limit_amount", -1)) != GLOBAL_LIMIT_AFTER_APPROVAL:
            failures.append(f"budget global limit mismatch: actual={stage.get('limit_amount')} expected=404")
        if int(stage.get("reserved", 0)) != 20 or int(stage.get("verified", 0)) != 20:
            failures.append("historical reserved/verified baseline must remain exactly 20")
        if int(stage.get("reserved", 0)) + MAX_TOTAL_STEPS > GLOBAL_LIMIT_AFTER_APPROVAL:
            failures.append("budget has insufficient global available credit")
        if int(stage.get("unknown", 0)) != 0 or int(stage.get("pending", 0)) != 0 or int(stage.get("pending", 0)) != int(stage.get("reserved", 0)) - int(stage.get("verified", 0)) - int(stage.get("unknown", 0)):
            failures.append("budget contains unknown or pending reservations")
        run_id = RUN_ID
        attempt_id = ATTEMPT_ID
        if budget_identity.get("run_totals"):
            failures.append("authorized new run already has reservations")
        if any(row.get("attempt_id") == attempt_id for row in budget_identity.get("attempts", ())):
            failures.append("authorized attempt identity already exists")
    if failures:
        raise AuthorizationError("pre-execution authorization gate failed: " + "; ".join(failures))
    return {"authorization": payload, "manifest": manifest, "budget": budget_identity}


@dataclass
class Counters:
    probe_calls: int = 0
    env_step_calls: int = 0
    env_steps: int = 0
    verified_steps: int = 0
    model_forward: int = 0
    actor_forward: int = 0
    world_forward: int = 0
    optimizer_updates: int = 0
    world_updates: int = 0
    offline_updates: int = 0
    branches_completed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "probe_calls": self.probe_calls,
            "env_step_calls": self.env_step_calls,
            "env_steps": self.env_steps,
            "successful_env_steps": self.env_steps,
            "verified_steps": self.verified_steps,
            "model_forward": self.model_forward,
            "actor_forward": self.actor_forward,
            "world_forward": self.world_forward,
            "optimizer_updates": self.optimizer_updates,
            "world_updates": self.world_updates,
            "offline_updates": self.offline_updates,
            "branches_completed": self.branches_completed,
        }


@dataclass(frozen=True)
class RuntimeAdapters:
    """Injectable runtime boundary used by the real runner and protocol tests."""

    load_snapshots: Callable[[Sequence[str]], Mapping[str, Mapping[str, Any]]]
    load_runtime: Callable[[Mapping[str, Mapping[str, Any]]], Any]
    probe: Callable[[Any, Mapping[str, Any], Any, Any], Mapping[str, Any]]
    reward: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Any], Mapping[str, Any]]
    runtime_digest: Callable[[Any], str]
    task_capacity: Callable[[Any], int] = lambda env: int(getattr(getattr(env, "config", None), "task_capacity", 6))
    close: Callable[[], None] | None = None


def _validate_probabilities(obs: Mapping[str, Any], values: Sequence[Any]) -> tuple[list[float], list[bool]]:
    mask = [bool(value) for value in obs.get("mask", ())]
    probabilities = [float(value) for value in values]
    if not mask or len(mask) != len(probabilities):
        raise RuntimeTechnicalStop("model probability vector and public mask dimensions differ")
    if any(not math.isfinite(value) for value in probabilities):
        raise RuntimeTechnicalStop("model probability vector contains a non-finite value")
    if not any(mask):
        raise RuntimeTechnicalStop("public mask has no legal action")
    return probabilities, mask


def _probability_ledger(probabilities: Sequence[float], mask: Sequence[bool], original: int, selected: int) -> list[dict[str, Any]]:
    return [
        {
            "action": int(index),
            "legal": bool(mask[index]),
            "probability": float(probabilities[index]),
            "original_argmax": int(index) == int(original),
            "selected": int(index) == int(selected),
        }
        for index in range(len(probabilities))
    ]


def _numeric_vector(value: Any, *, name: str) -> list[float]:
    vector = [float(item) for item in value]
    if len(vector) != 2 or any(not math.isfinite(item) for item in vector):
        raise RuntimeTechnicalStop(f"{name} is empty or non-finite")
    return vector


def _assert_reward_recomputed(reward: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    vector = _numeric_vector(reward.get("vector", ()), name="vector_reward")
    recomputed = _numeric_vector(reward.get("recomputed_vector", ()), name="recomputed_vector_reward")
    if len(vector) != len(recomputed) or any(abs(left - right) > TOLERANCE for left, right in zip(vector, recomputed)):
        raise RuntimeTechnicalStop("vector reward cannot be reproduced from the original reward contract")
    return vector, recomputed


def _independent_vector_reward(
    info: Mapping[str, Any],
    before: Mapping[str, Any],
    reward: Mapping[str, Any],
    env: Any,
) -> list[float] | None:
    """Recompute the frozen vector reward from counts, energy and config."""
    if not isinstance(getattr(getattr(env, "clock", None), "resources", None), Mapping):
        return None
    counts = info.get("counts") if isinstance(info, Mapping) else None
    if not isinstance(counts, Mapping):
        counts = reward.get("next_counts") if isinstance(reward.get("next_counts"), Mapping) else None
    if not isinstance(counts, Mapping):
        return None
    energy_payload = info.get("energy") if isinstance(info, Mapping) else None
    if isinstance(energy_payload, Mapping):
        energy_now = float(sum(float(value) for value in energy_payload.values()))
    elif reward.get("next_energy") is not None:
        energy_now = float(reward["next_energy"])
    else:
        energy_now = _energy_total(env)
    previous_counts = before.get("counts", {})
    previous_energy = float(before.get("energy"))
    task_capacity, uav_count, initial_energy = _task_capacity_and_fleet(env)
    new_success = max(0, int(counts.get("completed", 0)) - int(previous_counts.get("completed", 0)))
    new_deadline_failure = max(0, int(counts.get("expired", 0)) - int(previous_counts.get("expired", 0)))
    used = max(0.0, previous_energy - energy_now)
    return [
        float(value)
        for value in np.asarray(
            ((new_success - new_deadline_failure) / float(task_capacity), -used / max(1e-9, uav_count * initial_energy)),
            dtype=np.float32,
        ).tolist()
    ]


def _reserve(budget: Any) -> Mapping[str, Any]:
    try:
        return budget.reserve("environment_steps", 1)
    except Exception as exc:
        raise RuntimeTechnicalStop(f"budget reservation failed before env.step: {type(exc).__name__}: {exc}") from exc


def _reservation_state(budget: Any, token: Mapping[str, Any]) -> str | None:
    """Read the actual terminal state after a best-effort finalization."""
    getter = getattr(budget, "reservation_status", None)
    if callable(getter):
        try:
            return str(getter(token))
        except Exception:
            pass
    path = getattr(budget, "path", getattr(budget, "db_path", None))
    reservation_id = token.get("reservation_id")
    if path is None or not reservation_id:
        return None
    try:
        con = sqlite3.connect(str(path), timeout=2.0)
        try:
            row = con.execute("SELECT status FROM reservations WHERE reservation_id=?", (str(reservation_id),)).fetchone()
        finally:
            con.close()
        return None if row is None else str(row[0])
    except Exception:
        return None


def _mark_unknown(budget: Any, token: Mapping[str, Any], reason: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reservation_id": token.get("reservation_id"),
        "requested_outcome": "unknown",
        "reason": reason,
    }
    try:
        budget.unknown(token, reason)
        result["unknown_call"] = "ok"
    except Exception as exc:
        result["unknown_call"] = "failed"
        result["secondary_error"] = f"{type(exc).__name__}: {exc}"
    result["actual_reservation_status"] = _reservation_state(budget, token)
    return result


def _execute_branch_impl(
    item: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    runtime: Any,
    adapters: RuntimeAdapters,
    budget: Any,
    counters: Counters,
    out_dir: Path,
) -> dict[str, Any]:
    """Execute one isolated branch with durable decision and step evidence."""
    branch_counter_start = counters.as_dict()

    def branch_count(name: str) -> int:
        return int(getattr(counters, name)) - int(branch_counter_start.get(name, 0))

    baseline_digest = snapshot_semantic_digest(source_snapshot)
    baseline_components = snapshot_semantic_components(source_snapshot)
    snapshot = copy.deepcopy(source_snapshot)
    env = snapshot["env"]
    prefix_exogenous_key = getattr(env, "_exogenous_key", None)
    branch_exogenous_key = str(item["historical_exogenous_key"])
    env._exogenous_key = branch_exogenous_key
    if getattr(env, "_exogenous_key", None) != branch_exogenous_key:
        raise RuntimeTechnicalStop(f"failed to inject historical exogenous key for {item['branch_id']}")
    obs = copy.deepcopy(snapshot["obs"])
    policy_hidden = copy.deepcopy(snapshot.get("policy_hidden"))
    world_hidden = copy.deepcopy(snapshot.get("world_hidden"))
    initial_env_digest = environment_digest(env)
    initial_obs_digest = observation_digest(obs)
    previous_counts = {
        "completed": int(getattr(env, "_last_completed", 0)),
        "expired": int(getattr(env, "_last_expired", 0)),
    }
    previous_energy = _energy_total(env)
    task_capacity, uav_count, configured_initial_energy = _task_capacity_and_fleet(env)
    initial_counts = dict(previous_counts)
    initial_energy = previous_energy
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    steps = 0
    done = False
    terminated = False
    truncated = False
    end_reason: str | None = None
    last_info: Mapping[str, Any] = {}
    last_public_digest = initial_obs_digest

    while not done:
        if steps >= int(item.get("max_steps", MAX_STEPS_PER_BRANCH)):
            truncated = True
            end_reason = "protocol_step_cap"
            break
        if counters.env_steps >= MAX_TOTAL_STEPS:
            raise RuntimeTechnicalStop(f"global step cap reached before branch step: {item['branch_id']}")
        before_env_digest = environment_digest(env)
        before_obs_digest = observation_digest(obs)
        before_context = {
            "env_state_sha256": before_env_digest,
            "observation_sha256": before_obs_digest,
            "counts": dict(previous_counts),
            "energy": previous_energy,
        }
        try:
            counters.probe_calls += 1
            probe = dict(adapters.probe(runtime, obs, policy_hidden, world_hidden))
            counters.model_forward += 1
            counters.actor_forward += 1
            counters.world_forward += 1
            probabilities, mask = _validate_probabilities(obs, probe.get("probabilities", ()))
            raw_action = int(probe.get("original_action", probe.get("raw_action", -1)))
            if raw_action < 0 or raw_action >= len(mask) or not mask[raw_action]:
                raise RuntimeTechnicalStop(f"model original action is not legal: {raw_action}")
            decision = select_guarded_action(
                obs,
                probabilities,
                task_capacity=int(adapters.task_capacity(env)),
                enabled=True,
            )
            if raw_action != int(decision.original_action):
                raise RuntimeTechnicalStop(
                    f"probe original_action disagrees with guard original_action: {raw_action}!={decision.original_action}"
                )
            action = int(decision.final_action)
            if action < 0 or action >= len(mask) or not mask[action]:
                raise RuntimeTechnicalStop(f"guard selected illegal action: {action}")
        except (PublicIdentityError, NoLegalActionError, ProtocolError) as exc:
            failure = {
                "schema": "ackguard-corrected-failure/1.0.0",
                "branch_id": item["branch_id"],
                "step": steps + 1,
                "phase": "decision",
                "reason": f"{type(exc).__name__}: {exc}",
                "probe_calls": branch_count("probe_calls"),
                "env_step_calls": branch_count("env_step_calls"),
                "model_forward_calls": branch_count("model_forward"),
            }
            append_jsonl(out_dir / "failure-ledger.jsonl", failure)
            raise RuntimeTechnicalStop(failure["reason"]) from exc
        except Exception as exc:
            failure = {
                "schema": "ackguard-corrected-failure/1.0.0",
                "branch_id": item["branch_id"],
                "step": steps + 1,
                "phase": "model_or_guard",
                "reason": f"{type(exc).__name__}: {exc}",
                "probe_calls": branch_count("probe_calls"),
                "env_step_calls": branch_count("env_step_calls"),
                "model_forward_calls": branch_count("model_forward"),
            }
            append_jsonl(out_dir / "failure-ledger.jsonl", failure)
            raise RuntimeTechnicalStop(failure["reason"]) from exc

        by_action = probe.get("by_action")
        selected_action_record = by_action.get(action) if isinstance(by_action, Mapping) else None
        if not isinstance(selected_action_record, Mapping) or "hidden" not in selected_action_record:
            failure = {
                "schema": "ackguard-corrected-failure/1.0.0",
                "branch_id": item["branch_id"],
                "step": steps + 1,
                "phase": "selected_action_hidden",
                "reason": f"missing hidden/cache transition for selected action {action}",
                "selected_action": action,
                "probe_calls": branch_count("probe_calls"),
                "env_step_calls": branch_count("env_step_calls"),
            }
            append_jsonl(out_dir / "failure-ledger.jsonl", failure)
            raise RuntimeTechnicalStop(failure["reason"])
        action_hidden = selected_action_record["hidden"]
        decision_row = {
            "schema": "ackguard-corrected-decision/1.0.0",
            "branch_id": item["branch_id"],
            "control_branch_key": item["control_branch_key"],
            "prefix_exogenous_key_before": prefix_exogenous_key,
            "branch_exogenous_key": branch_exogenous_key,
            "step": steps + 1,
            "original_action": raw_action,
            "final_action": action,
            "original_probability": probabilities[raw_action],
            "final_probability": probabilities[action],
            "probabilities": _probability_ledger(probabilities, mask, raw_action, action),
            "legal_mask": [int(value) for value in mask],
            "continuation_actions": list(obs.get("continuation_actions", ())),
            "continuation_task_ids": list(decision.continuation_task_ids),
            "excluded_actions": list(decision.excluded_actions),
            "candidate_actions": list(decision.candidate_actions),
            "action_task_ids": [[int(index), task_id] for index, task_id in decision.action_task_ids],
            "triggered": bool(decision.triggered),
            "reason": decision.reason,
            "public_observation_sha256": before_obs_digest,
            "env_state_before_sha256": before_env_digest,
            "policy_hidden_before_sha256": hidden_digest(policy_hidden),
            "world_hidden_before_sha256": hidden_digest(world_hidden),
            "selected_action_hidden_sha256": hidden_digest(action_hidden),
            "task_capacity": task_capacity,
            "uav_count": uav_count,
            "configured_initial_energy": configured_initial_energy,
            "initial_energy": initial_energy,
            "counts_before": dict(previous_counts),
            "energy_before": previous_energy,
            "budget_status": "pending_before_reserve",
            "probe_calls": branch_count("probe_calls"),
        }
        reservation: Mapping[str, Any] | None = None
        try:
            try:
                reservation = _reserve(budget)
            except RuntimeTechnicalStop as exc:
                decision_row["budget_status"] = "not_reserved"
                decision_row["failure"] = str(exc)
                append_jsonl(out_dir / "decision-ledger.jsonl", decision_row)
                append_jsonl(out_dir / "failure-ledger.jsonl", {
                    "schema": "ackguard-corrected-failure/1.0.0",
                    "branch_id": item["branch_id"],
                    "step": steps + 1,
                    "phase": "budget.reserve",
                    "reason": str(exc),
                    "budget_outcome": "not_reserved",
                })
                raise
            decision_row["budget_status"] = "reserved"
            decision_row["reservation_id"] = reservation.get("reservation_id")
            decision_row["reservation_status_before_env_step"] = _reservation_state(budget, reservation)
            counters.env_step_calls += 1
            try:
                next_obs, scalar_reward, done_return, info = env.step(action, submit_command=True)
            except Exception as exc:
                unknown_result = _mark_unknown(budget, reservation, f"env.step:{item['branch_id']}:{type(exc).__name__}:{exc}")
                decision_row["budget_status"] = "unknown"
                decision_row["failure"] = f"{type(exc).__name__}: {exc}"
                decision_row["unknown_finalization"] = unknown_result
                append_jsonl(out_dir / "decision-ledger.jsonl", decision_row)
                append_jsonl(out_dir / "failure-ledger.jsonl", {
                    "schema": "ackguard-corrected-failure/1.0.0",
                    "branch_id": item["branch_id"],
                    "step": steps + 1,
                    "phase": "env.step",
                    "reservation_id": reservation.get("reservation_id"),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "budget_outcome": "unknown",
                    "unknown_finalization": unknown_result,
                    "env_step_calls": branch_count("env_step_calls"),
                })
                raise RuntimeTechnicalStop(decision_row["failure"]) from exc
            counters.env_steps += 1
            after_env_digest = environment_digest(env)
            try:
                reward = dict(adapters.reward(info, before_context, {
                    "env_state_sha256": after_env_digest,
                    "observation_sha256": observation_digest(next_obs),
                    "done_return": bool(done_return),
                }, env))
                vector, recomputed = _assert_reward_recomputed(reward)
                independent = _independent_vector_reward(info, before_context, reward, env)
                if independent is not None:
                    if any(abs(left - right) > TOLERANCE for left, right in zip(vector, independent)) or any(abs(left - right) > TOLERANCE for left, right in zip(recomputed, independent)):
                        raise RuntimeTechnicalStop("vector reward does not match independent float32 contract")
            except Exception as exc:
                unknown_result = _mark_unknown(budget, reservation, f"reward-verify:{item['branch_id']}:{type(exc).__name__}:{exc}")
                decision_row["budget_status"] = "unknown"
                decision_row["failure"] = f"{type(exc).__name__}: {exc}"
                decision_row["unknown_finalization"] = unknown_result
                append_jsonl(out_dir / "decision-ledger.jsonl", decision_row)
                append_jsonl(out_dir / "failure-ledger.jsonl", {
                    "schema": "ackguard-corrected-failure/1.0.0",
                    "branch_id": item["branch_id"],
                    "step": steps + 1,
                    "phase": "reward_verify",
                    "reservation_id": reservation.get("reservation_id"),
                    "reason": decision_row["failure"],
                    "budget_outcome": "unknown",
                    "unknown_finalization": unknown_result,
                    "env_step_calls": branch_count("env_step_calls"),
                })
                raise RuntimeTechnicalStop(decision_row["failure"]) from exc
            next_counts = dict(reward.get("next_counts", previous_counts))
            actual_next_energy = _energy_total(env)
            next_energy = float(reward.get("next_energy", actual_next_energy))
            if abs(next_energy - actual_next_energy) > TOLERANCE:
                raise RuntimeTechnicalStop("reward energy does not match env.clock.resources")
            terminated = bool(info.get("terminated", bool(done_return) and not bool(info.get("truncated", False))))
            truncated = bool(info.get("truncated", False))
            done = bool(done_return) or terminated or truncated
            step_row = {
                "schema": "ackguard-corrected-step/1.0.0",
                "branch_id": item["branch_id"],
                "control_branch_key": item["control_branch_key"],
                "prefix_exogenous_key_before": prefix_exogenous_key,
                "branch_exogenous_key": branch_exogenous_key,
                "prefix_id": item["prefix_id"],
                "parent_id": item["parent_id"],
                "condition": item["condition"],
                "model_seed": int(item["model_seed"]),
                "repeat": int(item["repeat"]),
                "exogenous_key": item["historical_exogenous_key"],
                "step": steps + 1,
                "gamma_index": steps,
                "action": action,
                "original_action": raw_action,
                "submit_command": True,
                "public_observation_sha256": before_obs_digest,
                "public_next_observation_sha256": observation_digest(next_obs),
                "env_state_before_sha256": before_env_digest,
                "env_state_after_sha256": after_env_digest,
                "legal_mask": [int(value) for value in mask],
                "probabilities": _probability_ledger(probabilities, mask, raw_action, action),
                "time_before": obs.get("time"),
                "time_after": next_obs.get("time") if isinstance(next_obs, Mapping) else None,
                "completed_before": previous_counts["completed"],
                "expired_before": previous_counts["expired"],
                "completed_after": int(next_counts.get("completed", previous_counts["completed"])),
                "expired_after": int(next_counts.get("expired", previous_counts["expired"])),
                "completed_delta": int(next_counts.get("completed", previous_counts["completed"])) - previous_counts["completed"],
                "expired_delta": int(next_counts.get("expired", previous_counts["expired"])) - previous_counts["expired"],
                "energy_before": previous_energy,
                "energy_after": next_energy,
                "energy_used_delta": max(0.0, previous_energy - next_energy),
                "task_capacity": task_capacity,
                "uav_count": uav_count,
                "configured_initial_energy": configured_initial_energy,
                "initial_energy": initial_energy,
                "raw_scalar_reward": float(scalar_reward),
                "vector_reward": vector,
                "recomputed_vector_reward": recomputed,
                "reward_recomputed": True,
                "task_consequence": _jsonable(reward.get("consequence")),
                "gamma": GAMMA,
                "reward_scales": REWARD_SCALES,
                "behavior_preference": list(BEHAVIOR_PREFERENCE),
                "terminated": terminated,
                "truncated": truncated,
                "done_return": bool(done_return),
                "end_reason": info.get("episode_end_reason"),
                "feedback": info.get("feedback"),
                "actual_feedback": _jsonable(info),
                "policy_hidden_before_sha256": hidden_digest(policy_hidden),
                "world_hidden_before_sha256": hidden_digest(world_hidden),
                "selected_action_hidden_sha256": hidden_digest(action_hidden),
                "policy_hidden_after_sha256": hidden_digest(probe.get("next_policy_hidden")),
                "world_hidden_after_sha256": hidden_digest(action_hidden),
                "reservation_id": reservation.get("reservation_id"),
                "budget_status": "reserved_pending_finalization",
                "probe_calls": branch_count("probe_calls"),
                "env_step_calls": branch_count("env_step_calls"),
            }
            # Full immutable evidence is durable while the reservation is still
            # pending.  Finalization is recorded separately after complete().
            append_jsonl(out_dir / "decision-ledger.jsonl", decision_row)
            append_jsonl(out_dir / "step-vector-rewards.jsonl", step_row)
            try:
                budget.complete(reservation)
            except Exception as exc:
                unknown_result = _mark_unknown(budget, reservation, f"budget.complete:{item['branch_id']}:{type(exc).__name__}:{exc}")
                actual_status = unknown_result.get("actual_reservation_status")
                decision_row["budget_status"] = actual_status or "unknown"
                decision_row["failure"] = f"{type(exc).__name__}: {exc}"
                append_jsonl(out_dir / "failure-ledger.jsonl", {
                    "schema": "ackguard-corrected-failure/1.0.0",
                    "branch_id": item["branch_id"],
                    "step": steps + 1,
                    "phase": "budget.complete",
                    "reservation_id": reservation.get("reservation_id"),
                    "reason": decision_row["failure"],
                    "budget_outcome": actual_status or "unknown_or_pending",
                    "unknown_finalization": unknown_result,
                    "env_step_calls": branch_count("env_step_calls"),
                })
                if actual_status == "verified":
                    counters.verified_steps += 1
                    append_jsonl(out_dir / "budget-finalization.jsonl", {
                        "schema": "ackguard-corrected-budget-finalization/1.0.0",
                        "branch_id": item["branch_id"],
                        "step": steps + 1,
                        "reservation_id": reservation.get("reservation_id"),
                        "status": "verified",
                        "actual_reservation_status": actual_status,
                        "finalization_exception": decision_row["failure"],
                        "env_step_calls": branch_count("env_step_calls"),
                        "successful_env_steps": branch_count("env_steps"),
                        "verified_steps": branch_count("verified_steps"),
                    })
                raise RuntimeTechnicalStop(decision_row["failure"]) from exc
            decision_row["budget_status"] = "verified"
            step_row["budget_status"] = "verified"
            counters.verified_steps += 1
            append_jsonl(out_dir / "budget-finalization.jsonl", {
                "schema": "ackguard-corrected-budget-finalization/1.0.0",
                "branch_id": item["branch_id"],
                "step": steps + 1,
                "reservation_id": reservation.get("reservation_id"),
                "status": "verified",
                "actual_reservation_status": _reservation_state(budget, reservation),
                "env_step_calls": branch_count("env_step_calls"),
                "successful_env_steps": branch_count("env_steps"),
                "verified_steps": branch_count("verified_steps"),
            })
            decisions.append(dict(decision_row))
        except RuntimeTechnicalStop:
            raise
        except Exception as exc:
            if reservation is not None:
                unknown_result = _mark_unknown(budget, reservation, f"branch:{item['branch_id']}:{type(exc).__name__}:{exc}")
            else:
                unknown_result = None
            append_jsonl(out_dir / "failure-ledger.jsonl", {
                "schema": "ackguard-corrected-failure/1.0.0",
                "branch_id": item["branch_id"],
                "step": steps + 1,
                "phase": "branch",
                "reservation_id": reservation.get("reservation_id") if reservation else None,
                "reason": f"{type(exc).__name__}: {exc}",
                "budget_outcome": ((unknown_result or {}).get("actual_reservation_status") or "unknown") if reservation else "not_reserved",
                "unknown_finalization": unknown_result,
                "env_step_calls": branch_count("env_step_calls"),
            })
            raise RuntimeTechnicalStop(f"{type(exc).__name__}: {exc}") from exc

        steps += 1
        rows.append(step_row)
        previous_counts = next_counts
        previous_energy = next_energy
        last_info = info
        last_public_digest = observation_digest(next_obs)
        policy_hidden = copy.deepcopy(probe.get("next_policy_hidden"))
        world_hidden = copy.deepcopy(action_hidden)
        obs = copy.deepcopy(next_obs)
        if done:
            end_reason = info.get("episode_end_reason")

    if not done and steps >= int(item.get("max_steps", MAX_STEPS_PER_BRANCH)):
        failure = {
            "schema": "ackguard-corrected-failure/1.0.0",
            "branch_id": item["branch_id"],
            "step": steps,
            "phase": "termination",
            "reason": "protocol step cap reached before native terminated/truncated",
            "budget_outcome": "verified_steps_retained",
            "env_step_calls": branch_count("env_step_calls"),
            "successful_env_steps": branch_count("env_steps"),
            "verified_steps": branch_count("verified_steps"),
            "native_done": False,
            "protocol_truncation_is_not_success": True,
        }
        append_jsonl(out_dir / "failure-ledger.jsonl", failure)
        raise RuntimeTechnicalStop(f"{failure['reason']}: {item['branch_id']}")
    final_digest = snapshot_semantic_digest(source_snapshot)
    final_components = snapshot_semantic_components(source_snapshot)
    if final_digest != baseline_digest:
        raise RuntimeTechnicalStop(f"source snapshot mutated during isolated branch {item['branch_id']}")
    branch_counter_end = counters.as_dict()
    branch_counter_delta = {
        key: int(branch_counter_end.get(key, 0)) - int(branch_counter_start.get(key, 0))
        for key in branch_counter_end
    }
    result = {
        "schema": "ackguard-corrected-branch-result/1.0.0",
        "branch_id": item["branch_id"],
        "control_branch_key": item["control_branch_key"],
        "prefix_exogenous_key_before": prefix_exogenous_key,
        "branch_exogenous_key": branch_exogenous_key,
        "prefix_id": item["prefix_id"],
        "parent_id": item["parent_id"],
        "condition": item["condition"],
        "model_seed": int(item["model_seed"]),
        "repeat": int(item["repeat"]),
        "exogenous_key": item["historical_exogenous_key"],
        "initial_public_observation_sha256": initial_obs_digest,
        "initial_env_state_sha256": initial_env_digest,
        "final_env_state_sha256": environment_digest(env),
        "source_snapshot_digest_before": baseline_digest,
        "source_snapshot_digest_after": final_digest,
        "source_snapshot_semantics_before": baseline_components,
        "source_snapshot_semantics_after": final_components,
        "deepcopy_isolated": final_digest == baseline_digest,
        "initial_counts": initial_counts,
        "final_counts": previous_counts,
        "initial_energy": initial_energy,
        "configured_initial_energy": configured_initial_energy,
        "task_capacity": task_capacity,
        "uav_count": uav_count,
        "final_energy": previous_energy,
        "energy_used": max(0.0, initial_energy - previous_energy),
        "env_steps": steps,
        "successful_env_steps": steps,
        "verified_steps": branch_counter_delta["verified_steps"],
        "probe_calls": branch_counter_delta["probe_calls"],
        "env_step_calls": branch_counter_delta["env_step_calls"],
        "actor_forward_calls": branch_counter_delta["actor_forward"],
        "world_forward_calls": branch_counter_delta["world_forward"],
        "counter_start": branch_counter_start,
        "counter_end": branch_counter_end,
        "counter_delta": branch_counter_delta,
        "terminated": terminated,
        "truncated": truncated,
        "done": bool(done),
        "end_reason": end_reason,
        "final_public_observation_sha256": last_public_digest,
        "last_info": _jsonable(last_info),
        "guard_trigger_count": sum(bool(row.get("triggered")) for row in decisions),
        "action_change_count": sum(int(row.get("final_action")) != int(row.get("original_action")) for row in decisions),
        "noop_count": sum(int(row.get("final_action")) == 24 for row in decisions),
        "task_unavailable_count": sum(row.get("feedback") == "task_unavailable" for row in rows),
        "command_count": sum(bool(row.get("submit_command")) for row in rows),
        "rows": len(rows),
        "first_step": decisions[0] if decisions else None,
        "no_retry": True,
    }
    append_jsonl(out_dir / "branch-results.jsonl", result)
    return result


def _counter_delta(start: Mapping[str, Any], end: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: int(end.get(key, 0)) - int(start.get(key, 0))
        for key in set(start) | set(end)
        if isinstance(start.get(key, end.get(key, 0)), (int, float))
    }


def _safe_append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Keep failure reporting best-effort without masking the original error."""
    try:
        append_jsonl(path, value)
    except Exception:
        pass


def _failure_snapshot_summary(
    item: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    counter_start: Mapping[str, Any],
    counters: Counters,
    exc: BaseException,
    baseline_digest: str | None,
    baseline_components: Mapping[str, Any] | None,
) -> dict[str, Any]:
    after_components: dict[str, Any] | None = None
    after_digest: str | None = None
    try:
        after_components = snapshot_semantic_components(source_snapshot)
        after_digest = snapshot_semantic_digest(source_snapshot)
    except Exception as digest_exc:
        after_components = {"error": f"{type(digest_exc).__name__}: {digest_exc}"}
    counter_end = counters.as_dict()
    return {
        "schema": "ackguard-corrected-branch-failure-summary/1.0.0",
        "failure_summary": True,
        "branch_id": item.get("branch_id"),
        "phase": "branch_finally",
        "reason": f"{type(exc).__name__}: {exc}",
        "source_snapshot_digest_before": baseline_digest,
        "source_snapshot_digest_after": after_digest,
        "source_snapshot_semantics_before": baseline_components,
        "source_snapshot_semantics_after": after_components,
        "source_snapshot_unchanged": baseline_digest is not None and baseline_digest == after_digest,
        "counter_start": dict(counter_start),
        "counter_end": counter_end,
        "counter_delta": _counter_delta(counter_start, counter_end),
        "actual_counts": {
            "probe_calls": counter_end.get("probe_calls", 0),
            "env_step_calls": counter_end.get("env_step_calls", 0),
            "successful_env_steps": counter_end.get("successful_env_steps", 0),
            "verified_steps": counter_end.get("verified_steps", 0),
            "actor_forward": counter_end.get("actor_forward", 0),
            "world_forward": counter_end.get("world_forward", 0),
        },
    }


def execute_branch(
    item: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    runtime: Any,
    adapters: RuntimeAdapters,
    budget: Any,
    counters: Counters,
    out_dir: Path,
) -> dict[str, Any]:
    """Execute one branch and always retain failure-time source/counter evidence."""
    counter_start = counters.as_dict()
    try:
        baseline_digest = snapshot_semantic_digest(source_snapshot)
        baseline_components = snapshot_semantic_components(source_snapshot)
    except Exception:
        baseline_digest = None
        baseline_components = None
    try:
        return _execute_branch_impl(item, source_snapshot, runtime, adapters, budget, counters, out_dir)
    except Exception as exc:
        _safe_append_jsonl(
            out_dir / "failure-ledger.jsonl",
            _failure_snapshot_summary(item, source_snapshot, counter_start, counters, exc, baseline_digest, baseline_components),
        )
        raise


class _HistoricalPackageIsolation:
    """Temporarily make canonical ``gppo_world`` imports resolve to SOURCE_ROOT."""

    REQUIRED_MODULES = (
        "gppo_world.budget_executor",
        "gppo_world.joint_gppo",
        "gppo_world.joint_training",
        "gppo_world.m10_environment",
        "gppo_world.task_policy_view",
        "gppo_world.task_decision_bridge",
    )

    def __init__(self, source_root: Path):
        self.source_root = source_root.resolve()
        self.saved_modules: dict[str, Any] = {}
        self.saved_path: list[str] | None = None
        self.active = False

    def enter(self) -> None:
        if self.active:
            return
        self.saved_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "gppo_world" or name.startswith("gppo_world.")
        }
        self.saved_path = list(sys.path)
        try:
            for name in list(self.saved_modules):
                sys.modules.pop(name, None)
            sys.path[:] = [str(self.source_root)] + [path for path in sys.path if Path(path or ".").resolve() != self.source_root]
            importlib.invalidate_caches()
            for module_name in self.REQUIRED_MODULES:
                module = __import__(module_name, fromlist=["*"])
                module_path = Path(str(getattr(module, "__file__", ""))).resolve()
                source_name = module_path.name
                expected = EXPECTED_SOURCE_HASHES.get(source_name)
                if module_path.parent != self.source_root / "gppo_world" or expected is None or sha256_file(module_path) != expected:
                    raise RuntimeTechnicalStop(f"historical module identity mismatch: {module_name} -> {module_path}")
            self.active = True
        except Exception:
            self._restore()
            raise

    def close(self) -> None:
        if not self.active and self.saved_path is None and not self.saved_modules:
            return
        self._restore()

    def _restore(self) -> None:
        for name in list(sys.modules):
            if name == "gppo_world" or name.startswith("gppo_world."):
                sys.modules.pop(name, None)
        sys.modules.update(self.saved_modules)
        if self.saved_path is not None:
            sys.path[:] = self.saved_path
        self.active = False
        self.saved_modules = {}
        self.saved_path = None


def _verify_historical_classes(classes: Sequence[Any]) -> None:
    class_positions = (0, 1, 2, 6, 7)
    for index in class_positions:
        value = classes[index]
        module_name = getattr(value, "__module__", "")
        if not module_name.startswith("gppo_world."):
            raise RuntimeTechnicalStop(f"historical class has unexpected module: {value!r}")
        module = sys.modules.get(module_name)
        module_path = Path(str(getattr(module, "__file__", ""))).resolve() if module is not None else Path()
        expected = EXPECTED_SOURCE_HASHES.get(module_path.name)
        try:
            class_path = Path(inspect.getfile(value)).resolve()
        except (TypeError, OSError) as exc:
            raise RuntimeTechnicalStop(f"cannot inspect historical class {value!r}") from exc
        if class_path != module_path or class_path.parent != SOURCE_ROOT / "gppo_world" or expected is None or sha256_file(class_path) != expected:
            raise RuntimeTechnicalStop(f"historical class source identity mismatch: {value!r} -> {class_path}")


def _default_adapters() -> RuntimeAdapters:
    """Bind the frozen native source only after authorization has passed."""
    isolation = _HistoricalPackageIsolation(SOURCE_ROOT)
    helper_path = WORKTREE / "tools" / "run_replan_value_experiment.py"
    spec = importlib.util.spec_from_file_location("ackguard_historical_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeTechnicalStop(f"cannot load helper: {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    classes_holder: dict[str, Any] = {}

    def load_snapshots(prefix_ids: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        isolation.enter()
        with (HISTORY / "prefix-snapshots.pkl").open("rb") as stream:
            snapshots = pickle.load(stream)
        found = {item.get("prefix_id"): item for item in snapshots if item.get("prefix_id") in set(prefix_ids)}
        if set(found) != set(prefix_ids):
            raise RuntimeTechnicalStop("selected prefix snapshots are incomplete")
        for prefix_id, snapshot in found.items():
            env_module = type(snapshot["env"]).__module__
            env_file = Path(str(inspect.getfile(type(snapshot["env"])))).resolve()
            if env_module != "gppo_world.m10_environment" or env_file != SOURCE_ROOT / "gppo_world" / "m10_environment.py":
                raise RuntimeTechnicalStop(f"snapshot environment is not native historical class: {prefix_id}")
        return found

    def load_runtime(snapshots: Mapping[str, Mapping[str, Any]]) -> Any:
        import torch

        if not isolation.active:
            raise RuntimeTechnicalStop("historical package isolation was not established before runtime load")
        classes = helper.load_modules(SOURCE_ROOT)
        _verify_historical_classes(classes)
        classes_holder["classes"] = classes
        config = copy.deepcopy(next(iter(snapshots.values()))["env"].config)
        if int(config.action_count) != 25:
            raise RuntimeTechnicalStop("runtime action contract is not historical 25-action M10")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        device = torch.device("cpu")
        policy, world, _ = helper.load_models(1101, CHECKPOINT, config, classes, device)
        preference_fn = classes[3]
        preference = preference_fn(BEHAVIOR_PREFERENCE, device=device)
        return {"helper": helper, "classes": classes, "policy": policy, "world": world, "preference": preference, "device": device, "native_source_root": str(SOURCE_ROOT)}

    def probe(runtime: Mapping[str, Any], obs: Mapping[str, Any], policy_hidden: Any, world_hidden: Any) -> Mapping[str, Any]:
        raw = runtime["helper"].probe(
            runtime["policy"], runtime["world"], obs, policy_hidden, world_hidden,
            runtime["preference"], runtime["classes"], runtime["device"],
        )
        probabilities = raw["evaluation"]["distribution"].probs.detach().cpu().numpy().reshape(-1).tolist()
        original = int(runtime["helper"].deterministic_action(raw))
        by_action = raw.get("by_action", {})
        return {
            "probabilities": probabilities,
            "original_action": original,
            "by_action": by_action,
            "next_policy_hidden": raw.get("next_policy_hidden"),
        }

    def reward(info: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any], env: Any) -> Mapping[str, Any]:
        vector_fn = __import__("gppo_world.joint_training", fromlist=["_vector_reward"])._vector_reward
        vector, consequence, counts, energy = vector_fn(info, before["counts"], before["energy"], env.config)
        vector_list = _numeric_vector(vector, name="vector_reward")
        task_capacity, uav_count, initial_energy = _task_capacity_and_fleet(env)
        info_counts = info.get("counts")
        if not isinstance(info_counts, Mapping):
            raise RuntimeTechnicalStop("native reward info is missing counts")
        info_energy = info.get("energy")
        if not isinstance(info_energy, Mapping):
            raise RuntimeTechnicalStop("native reward info is missing resource energy")
        energy_now = float(sum(float(value) for value in info_energy.values()))
        new_success = max(0, int(info_counts["completed"]) - int(before["counts"].get("completed", 0)))
        new_deadline_failure = max(0, int(info_counts["expired"]) - int(before["counts"].get("expired", 0)))
        used = max(0.0, float(before["energy"]) - energy_now)
        recomputed = [
            float(value)
            for value in np.asarray(
                ((new_success - new_deadline_failure) / float(task_capacity), -used / max(1e-9, uav_count * initial_energy)),
                dtype=np.float32,
            ).tolist()
        ]
        return {
            "vector": vector_list,
            "recomputed_vector": recomputed,
            "consequence": consequence,
            "next_counts": counts,
            "next_energy": energy,
            "task_capacity": task_capacity,
            "uav_count": uav_count,
            "initial_energy": initial_energy,
        }

    def runtime_digest(runtime: Mapping[str, Any]) -> str:
        def model_digest(model: Any) -> str:
            digest = hashlib.sha256()
            for key, value in sorted(model.state_dict().items()):
                digest.update(str(key).encode("utf-8"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        return canonical_hash({"policy": model_digest(runtime["policy"]), "world": model_digest(runtime["world"])})

    return RuntimeAdapters(load_snapshots, load_runtime, probe, reward, runtime_digest, close=isolation.close)


def _budget_factory(path: Path, auth: Mapping[str, Any]) -> PersistentBudget:
    budget = auth["budget"]
    limits = {
        "environment_steps": int(budget.get("global_limit", GLOBAL_LIMIT_AFTER_APPROVAL)),
        "optimizer_calls": 0,
        "world_updates": 0,
        "offline_updates": 0,
    }
    run_limits = {
        "environment_steps": int(budget.get("required_new_steps", MAX_TOTAL_STEPS)),
        "optimizer_calls": 0,
        "world_updates": 0,
        "offline_updates": 0,
    }
    return PersistentBudget(
        path,
        limits=limits,
        attempt_id=str(auth["run"]["attempt_id"]),
        run_id=str(auth["run"]["run_id"]),
        run_limits=run_limits,
        require_existing=True,
    )


def run_matrix(
    authorization_path: Path,
    *,
    manifest_path: Path | None = None,
    out_dir: Path = OUT_DEFAULT,
    adapters: RuntimeAdapters | None = None,
    budget_factory: Callable[[Path, Mapping[str, Any]], Any] = _budget_factory,
) -> dict[str, Any]:
    """Run the authorized 24-branch matrix; never resume or retry a failure."""
    manifest_path = manifest_path or (out_dir / "branch-manifest.json")
    gate = validate_authorization(authorization_path, manifest_path=manifest_path, out_dir=out_dir)
    auth = gate["authorization"]
    manifest = gate["manifest"]
    out_dir.mkdir(parents=True, exist_ok=False)
    counters = Counters()
    budget: Any | None = None
    runtime_before: str | None = None
    runtime: Any = None
    try:
        write_json(out_dir / "run-status.json", {
            "schema": "ackguard-corrected-run-status/1.0.0",
            "status": "runtime_objects_allowed",
            "started_at": time.time(),
            "hard_counts": counters.as_dict(),
            "attempt_created": False,
            "no_retry": True,
        })
        budget = budget_factory(Path(auth["budget"]["path"]), auth)
        adapters = adapters or _default_adapters()
        snapshots = adapters.load_snapshots([str(row["prefix_id"]) for row in manifest["rows"]])
        if set(snapshots) != {str(row["prefix_id"]) for row in manifest["rows"]}:
            raise RuntimeTechnicalStop("runtime snapshot set does not match the fixed matrix")
        runtime = adapters.load_runtime(snapshots)
        runtime_before = adapters.runtime_digest(runtime)
        results: list[dict[str, Any]] = []
        for item in manifest["rows"]:
            result = execute_branch(item, snapshots[item["prefix_id"]], runtime, adapters, budget, counters, out_dir)
            results.append(result)
            counters.branches_completed += 1
            append_jsonl(out_dir / "progress.jsonl", {
                "schema": "ackguard-corrected-progress/1.0.0",
                "branch_id": item["branch_id"],
                "branches_completed": counters.branches_completed,
                "env_steps": counters.env_steps,
                "verified_steps": counters.verified_steps,
                "limit": MAX_TOTAL_STEPS,
            })
        runtime_after = adapters.runtime_digest(runtime)
        if runtime_after != runtime_before:
            raise RuntimeTechnicalStop("runtime model weights changed during eval-only execution")
        if len(results) != 24 or counters.env_steps > MAX_TOTAL_STEPS:
            raise RuntimeTechnicalStop(f"matrix accounting mismatch: {counters.as_dict()}")
        totals = budget.run_totals(str(auth["run"]["run_id"]))
        stage = totals.get("environment_steps", {})
        if int(stage.get("reserved", -1)) != counters.verified_steps or int(stage.get("verified", -1)) != counters.verified_steps or int(stage.get("unknown", -1)) != 0 or int(stage.get("pending", -1)) != 0:
            raise RuntimeTechnicalStop(f"budget verification mismatch: {stage}")
        payload = {
            "schema": "ackguard-corrected-run-status/1.0.0",
            "status": "completed",
            "finished_at": time.time(),
            "hard_counts": counters.as_dict(),
            "attempt_created": counters.env_steps > 0,
            "no_retry": True,
            "runtime_digest_before": runtime_before,
            "runtime_digest_after": runtime_after,
            "budget_totals": totals,
        }
        write_json(out_dir / "run-status.json", payload)
        budget.export_snapshot(out_dir / "budget-after.json")
        return payload
    except Exception as exc:
        actual_totals = {}
        if budget is not None:
            try:
                actual_totals = budget.run_totals(str(auth["run"]["run_id"]))
            except Exception as totals_exc:
                actual_totals = {"error": f"{type(totals_exc).__name__}: {totals_exc}"}
        payload = {
            "schema": "ackguard-corrected-run-status/1.0.0",
            "status": "stopped_on_technical_error",
            "finished_at": time.time(),
            "hard_counts": counters.as_dict(),
            "attempt_created": any(int(value.get("reserved", 0)) > 0 for value in actual_totals.values() if isinstance(value, Mapping)),
            "no_retry": True,
            "error": f"{type(exc).__name__}: {exc}",
            "actual_budget_totals": actual_totals,
        }
        try:
            write_json(out_dir / "run-status.json", payload)
        except Exception:
            pass
        try:
            append_jsonl(out_dir / "failure-ledger.jsonl", {
                "schema": "ackguard-corrected-failure/1.0.0",
                "phase": "runtime_setup_or_matrix",
                "reason": payload["error"],
                "hard_counts": counters.as_dict(),
                "actual_budget_totals": actual_totals,
                "attempt_status": "created" if actual_totals else "not_created_or_unavailable",
            })
        except Exception:
            pass
        try:
            if budget is not None:
                budget.export_snapshot(out_dir / "budget-after.json")
        finally:
            if adapters is not None and adapters.close is not None:
                try:
                    adapters.close()
                except Exception:
                    pass
            raise
    finally:
        if adapters is not None and adapters.close is not None:
            try:
                adapters.close()
            except Exception:
                pass


def _protocol_payload(
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    budget: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    *,
    out_dir: Path = OUT_DEFAULT,
) -> dict[str, Any]:
    manifest_path = out_dir / "branch-manifest.json"
    runtime_out_dir = out_dir / "authorized-run"
    compatibility_summary = compatibility.get("summary", {}) if isinstance(compatibility, Mapping) else {}
    return {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "phase": 1,
        "status": "rejected_intermediate_corrected_protocol_review_required",
        "dynamic_execution_called": False,
        "scope": {
            "matrix": "parent-00..07 / W1 / seed-1101 / prefix-0 / repeat-0..2",
            "new_guard_branches": 24,
            "historical_R_controls": 24,
            "historical_R_steps": 299,
            "old_two_guard_branches": "preserved in historical output and excluded from this matrix",
            "later_dynamic_matrix_and_model_comparison": "not authorized in phase 1",
        },
        "cli": {
            "preflight": f"python tools/run_ack_known_task_guard_corrected_20260922.py --mode preflight --out {OUT_DEFAULT}",
            "future_run": f"python tools/run_ack_known_task_guard_corrected_20260922.py --mode run --authorization <reviewed-authorization.json> --manifest {manifest_path} --out {runtime_out_dir}",
            "authorization_file_required": True,
        },
        "pre_execution_gates": [
            "explicit authorization status=authorized",
            "protocol id and pinned protocol SHA-256 match",
            "corrected runner, guard, native source files, checkpoint, and prefix snapshot hashes match",
            "branch manifest is exactly 24 unique identities and output directory is fresh",
            "historical SQLite is read-only inspected, global limit is 404, new-run available credit is at least 384, and unknown/pending are zero",
            "new run_id and attempt_id are absent from the ledger",
            "H-004 bridge helper/provenance hashes match and the current saved observation, label, snapshot, identity, and time are recomputed before the budget object exists",
        ],
        "runtime_contract": {
            "source": "historical native M10 25-action snapshot, isolated from current finite-communication and D-02 code",
            "policy": "frozen P_train seed-1101 checkpoint, eval-only, deterministic action, behavior preference (0.8,0.2)",
            "guard": "gppo_world.ack_known_task_guard.py at fixed corrected SHA-256",
            "branch_restore": "deep-copy env, public observation, policy hidden, world hidden, and probe per branch; inject branch exogenous key only into the deep copy",
            "step_order": ["capture full env/obs/policy_hidden/world_hidden/probe digest", "model probe", "guard selection", "select by_action[selected_action].hidden", "reserve one environment step", "env.step", "independent float32 reward recheck", "persist full decision/step evidence", "budget verify", "persist finalization"],
            "probability_log": "all action probabilities with legal mask, original argmax, and selected action",
            "failure": "unknown reservation and immediate terminal stop; no refund, retry, or zero fill",
            "termination": "native terminated/truncated flags; reaching the protocol cap before native done is a technical stop and is never promoted as a complete branch",
        },
        "logging_contract": {
            "files": ["decision-ledger.jsonl", "step-vector-rewards.jsonl", "budget-finalization.jsonl", "branch-results.jsonl", "progress.jsonl", "failure-ledger.jsonl", "run-status.json", "budget-after.json"],
            "before_after": "full env state, queue/resource/RNG graph, public observation, hidden states and probe are hashed before and after; env_state_before is captured before env.step",
            "reward": "vector reward and independently float32 recomputed_vector_reward must agree within 1e-6 and match counts/energy/config",
            "task_host_confirmation": {
                "schema": "native-m10-step-info/1.0.0",
                "per_step_source": "step-vector-rewards.jsonl actual_feedback.info.completion_records and actual_feedback.info.task_outcome when present; counts.completed/expired remain the R-compatible task outcome",
                "per_branch_source": "branch-results.jsonl last_info.info.completion_records and last_info.info.task_outcome, with final_counts and end_reason",
                "postprocessing": "last_info is retained for postprocessing; no task or host outcome is reconstructed from model output",
            },
        },
        "budget": {
            "stage": "environment_steps",
            "new_cap": 384,
            "old_consumed": 20,
            "total_after": 404,
            "optimizer_calls": 0,
            "world_updates": 0,
            "offline_updates": 0,
            "unknown_or_pending_stop": True,
        },
        "metrics": {
            "primary": "parent-macro guard-R utility at preference (0.8,0.2); mean 3 repeats within parent then equal-weight 8 parents",
            "utility": "0.8*0.5*G_task + 0.2*1.0*G_energy with gamma=0.99",
            "task_host_confirmation": "per branch, task completion and host confirmation come from native info task records/completion_records or native task_outcome, retaining the historical R definition; energy comes from env.clock.resources and final_counts/end_reason from native info",
            "secondary": ["task completion", "energy used", "task_unavailable/rejection", "NOOP count", "guard trigger/action changes", "model forward count"],
            "stop_conditions": ["any gate failure", "identity/mask/probability mismatch", "env/model/reward technical error", "unknown or pending budget", "duplicate or partial output"],
            "no_negative_result_retry": True,
        },
        "hard_counts_at_phase1": {"environment_steps": 0, "model_forward": 0, "optimizer_updates": 0, "world_updates": 0, "offline_updates": 0, "new_formal_attempt": 0},
        "source_identity_summary": {"guard_sha256": EXPECTED_GUARD_SHA256, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "snapshot_sha256": EXPECTED_SNAPSHOT_SHA256, "r_control_bridge_schema_helper_sha256": EXPECTED_R_CONTROL_BRIDGE_SCHEMA_HELPER_SHA256},
        "current_budget_identity": {"sha256": budget.get("sha256"), "limit": _stage(budget).get("limit_amount"), "reserved": _stage(budget).get("reserved"), "available": int(_stage(budget).get("limit_amount", 0)) - int(_stage(budget).get("reserved", 0))},
        "historical_R_compatibility": {
            "status": compatibility.get("status"),
            "pairs": int(compatibility_summary.get("pairs", len(manifest.get("rows", ())))),
            "reusable_R_control_pairs": int(compatibility_summary.get("reusable_R_control_pairs", 0)),
            "insufficient_pairs": int(compatibility_summary.get("insufficient_pairs", 0)),
            "direct_first_label_snapshot_gate": "R step=1 public_observation_sha256 is compared directly with the immutable snapshot prefix_digest; old guard snapshot output is excluded",
            "corrected_guard_trajectory_reuse": False,
            "bridge_recomputed_at_authorization": True,
            "bridge_saved_24_of_24_boolean_trusted": False,
        },
    }


def authorization_template(
    manifest: Mapping[str, Any],
    *,
    protocol_path: Path,
    budget_identity: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Create a complete pending template; this function can never authorize a run."""
    validate_manifest(manifest)
    stage = _stage(budget_identity)
    return {
        "schema": AUTH_SCHEMA,
        "status": "pending",
        "template_only": True,
        "cannot_authorize_here": True,
        "protocol": {
            "id": PROTOCOL_ID,
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "source": {
            "corrected_runner_path": str(Path(__file__).resolve()),
            "corrected_runner_sha256": sha256_file(Path(__file__)),
            "guard_path": str(WORKTREE / "gppo_world" / "ack_known_task_guard.py"),
            "guard_sha256": sha256_file(WORKTREE / "gppo_world" / "ack_known_task_guard.py"),
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "snapshot_path": str(HISTORY / "prefix-snapshots.pkl"),
            "snapshot_sha256": sha256_file(HISTORY / "prefix-snapshots.pkl"),
            "historical_source_hashes": dict(EXPECTED_SOURCE_HASHES),
            "runtime_dependency_hashes": runtime_dependency_hashes(),
            "historical_input_hashes": historical_input_hashes(),
            "r_control_bridge": r_control_bridge_identities(),
        },
        "branch_manifest_path": str(out_dir / "branch-manifest.json"),
        "branch_manifest_sha256": sha256_file(out_dir / "branch-manifest.json"),
        "budget": {
            "path": str(HISTORICAL_BUDGET_PATH),
            "sha256": budget_identity.get("sha256"),
            "stage": "environment_steps",
            "global_limit": GLOBAL_LIMIT_AFTER_APPROVAL,
            "required_new_steps": MAX_TOTAL_STEPS,
            "old_reserved": 20,
            "old_verified": 20,
            "old_unknown": 0,
            "old_pending": 0,
            "current_observed_limit": stage.get("limit_amount"),
            "current_observed_available": int(stage.get("limit_amount", 0)) - int(stage.get("reserved", 0)),
            "must_be_changed_before_authorization": True,
        },
        "run": {"run_id": RUN_ID, "attempt_id": ATTEMPT_ID},
        "historical_R_compatibility": {
            "required": "reusable_as_R_control",
            "control_reuse_manifest_path": str(HISTORICAL_RUN / "control-reuse-manifest.json"),
            "control_reuse_manifest_sha256": sha256_file(HISTORICAL_RUN / "control-reuse-manifest.json"),
            "expected_pairs": 24,
            "expected_steps": 299,
            "observed_status": compatibility.get("status"),
            "observed_reusable_pairs": int(compatibility.get("summary", {}).get("reusable_R_control_pairs", 0)),
            "observed_insufficient_pairs": int(compatibility.get("summary", {}).get("insufficient_pairs", 0)),
            "direct_first_label_snapshot_gate": "R step=1 public_observation_sha256 must match snapshot-identity prefix_digest; old guard snapshot output is excluded",
            "historical_R_reexecuted": False,
        },
        "gates": {
            "integrity": "ok",
            "manifest_exact_identity": True,
            "runtime_dependencies_pinned": True,
            "unknown_pending_zero": True,
            "fresh_output_required": True,
            "authorization_status_must_be_reviewed_manually": True,
        },
    }


def write_preflight_artifacts(out_dir: Path = OUT_DEFAULT, *, allow_rewrite: bool = False) -> dict[str, Any]:
    """Write phase-1 artifacts using only static files and read-only SQLite."""
    preserved = {"report-rejected-intermediate.md"}
    if not allow_rewrite and out_dir.exists() and any(path.name not in preserved for path in out_dir.iterdir()):
        raise ProtocolError(f"refusing to overwrite existing preflight output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_branch_manifest()
    write_json(out_dir / "branch-manifest.json", manifest)
    source = _source_identities()
    budget = read_budget_identity(HISTORICAL_BUDGET_PATH, run_id=RUN_ID)
    compatibility = historical_compatibility(manifest)
    proposal = budget_proposal(budget)
    protocol = _protocol_payload(manifest, source, budget, compatibility, out_dir=out_dir)
    write_json(out_dir / "protocol.json", protocol)
    authorization = authorization_template(
        manifest,
        protocol_path=out_dir / "protocol.json",
        budget_identity=budget,
        compatibility=compatibility,
        out_dir=out_dir,
    )
    executable_md = "\n".join([
        "# ACK guard corrected executable protocol",
        "",
        f"Protocol ID: `{PROTOCOL_ID}`. Phase 1 prepares and audits the entry only; dynamic execution is not called.",
        "",
        "The future run command is:",
        "",
        f"`python tools/run_ack_known_task_guard_corrected_20260922.py --mode run --authorization <reviewed-authorization.json> --manifest {out_dir / 'branch-manifest.json'} --out {out_dir / 'authorized-run'}`",
        "",
        "The authorization must pin this protocol JSON, the corrected runner, the unchanged guard, the native 25-action source snapshot, the seed-1101 checkpoint, the prefix snapshot, the same SQLite ledger, and the H-004 R-control bridge helper/provenance/receipt identities. Before the budget object or runtime exists, the runner re-reads the bridge evidence, rebuilds the strict 23/24 failure rows from the current labels, branch results, manifest, and prefix snapshots, evaluates the current saved-observation schema bridge, and re-runs the full compatibility gate; the saved 24/24 qualified boolean is provenance only and is never trusted as authorization. It must also prove a global environment-step limit of 404 with at least 384 available new-run credit, no pending/unknown reservations, and unused run/attempt identities. Every condition is checked before model or environment construction.",
        "",
        "Each of 24 branches is restored from a deep copy. A decision row records every action probability, the public legal mask, original argmax, corrected choice, continuation identities, hidden/probe digests, and the pre-step environment digest. The budget reservation occurs immediately before `env.step`; a step exception, reward mismatch, logging failure, or finalization conflict records the actual reservation state (`verified`, `unknown`, or `pending`) and stops the run. There is no refund, retry, or zero fill.",
        "",
        f"The original 299-step R record is reusable as the matched R control only for the {compatibility.get('summary', {}).get('reusable_R_control_pairs', 0)} of 24 pairs that pass the read-only compatibility gate. The gate directly compares each R step=1 `public_observation_sha256` with the immutable snapshot `prefix_digest`; the old guard snapshot output is excluded. That control statement is separate from corrected-guard trajectory reuse: the old guard treatment and a future corrected guard treatment are never inferred from R, and complete R action probabilities or post-run hidden/cache non-pollution evidence are not required for the original R control itself.",
        "",
        "Primary metric: for each parent, average the three repeats, then give the eight parents equal weight for guard minus R utility at preference `(0.8,0.2)`, with `gamma=0.99`, where utility is `0.8*0.5*G_task + 0.2*1.0*G_energy`. Per-branch task completion and host confirmation come from the native step `info` schema (`completion_records`/task records or `task_outcome`) and the retained historical R definition; energy comes from `env.clock.resources`, and terminal counts/end reason come from native `info`. Step rows retain `actual_feedback` and branch rows retain `last_info` for postprocessing. Any negative result is retained; no retry is allowed.",
        "",
        "Current preflight budget is `limit=384, reserved=20, verified=20, unknown=0, pending=0`, leaving 364 available. It cannot authorize the 384-step matrix. `budget-proposal.json` recommends an approval-only extension of the same ledger to total limit 404 while preserving the old 20; it was not applied.",
        "",
    ])
    (out_dir / "executable-protocol.md").write_text(executable_md, encoding="utf-8")
    write_json(out_dir / "historical-R-compatibility.json", compatibility)
    write_json(out_dir / "source-input-identities.json", source)
    write_json(out_dir / "budget-proposal.json", proposal)
    write_json(out_dir / "authorization-template.json", authorization)
    write_json(out_dir / "run-status.json", {
        "schema": "ackguard-corrected-phase1-status/1.0.0",
        "status": "rejected_intermediate_corrected_protocol",
        "dynamic_execution_called": False,
        "hard_counts": {"environment_steps": 0, "model_forward": 0, "optimizer_updates": 0, "world_updates": 0, "offline_updates": 0, "new_formal_attempt": 0},
        "current_budget": budget,
        "historical_R_compatibility": compatibility.get("status"),
        "next_step": "review corrected protocol, apply only the approved same-ledger migration on a controlled copy, then issue a manually reviewed authorization; do not execute in this phase",
    })
    report = "\n".join([
        "# ACK guard corrected protocol, phase 1",
        "",
        "## Result",
        "",
        "The previous preflight was rejected and is retained as report-rejected-intermediate.md. This corrected phase performed no environment step, model forward, optimizer/world/offline update, formal attempt creation, budget reservation, server action, commit, or push.",
        "",
        "The 24-branch manifest is fixed to parent-00..07, W1, seed-1101, prefix-0, repeat 0/1/2. The two old 20-step guard branches remain in their historical output and are excluded. The 24 historical R controls total 299 steps and were not rerun.",
        "",
        "## Gate status",
        "",
        "Current read-only ledger: `limit=384, reserved=20, verified=20, unknown=0, pending=0`; available credit is 364, below the required 384. A future authorization must pin the same ledger after an approved total-limit extension to 404. No extension or new ledger was created.",
        "",
        f"Historical R compatibility is `{compatibility.get('status')}` for the strict original control contract across {compatibility.get('summary', {}).get('reusable_R_control_pairs', 0)} of 24 pairs. The dynamic authorization gate separately recomputes the H-004 saved-observation bridge for the strict failure row(s) from current labels, branch results, manifest, prefix snapshots, pinned helper/provenance hashes, and the verified remote receipt; it requires qualified 24/24 at authorization time and does not trust the saved qualified JSON boolean. Each pair directly compares the R step=1 `public_observation_sha256` with the corresponding immutable snapshot `prefix_digest` from `preference-vector-label-train-v1/snapshot-identity.json`, validated by `allocation-branch-preflight-v2/snapshot-interface-check.json`; the old guard snapshot output is excluded. The new guard trajectory is never inferred from R. Complete historical action probabilities and independent hidden/cache non-pollution evidence remain insufficient for corrected-guard trajectory reuse. The reported utility is `0.8*0.5*G_task + 0.2*1.0*G_energy` at `gamma=0.99`; task/host confirmation is read from native step `info` task records/completion records or `task_outcome`, with `last_info` retained for postprocessing.",
        "",
        "## Entry behavior",
        "",
        "The runner rejects missing or unapproved authorization, any hash mismatch, a non-fresh output, duplicate/partial identities, an existing new run/attempt, an unknown/pending reservation, incompatible historical R controls, or insufficient credit before constructing models, environments, or a budget object. The pending authorization template cannot authorize a run. Once manually authorized, each step reserves before `env.step`, persists full evidence before budget completion, verifies reward and budget completion, records complete probabilities and before/after state digests, and terminally records failures while retaining the actual reservation state (`verified`, `unknown`, or `pending`) and any secondary finalization error.",
        "",
        "See `executable-protocol.md`, `protocol.json`, `authorization-template.json`, `branch-manifest.json`, `historical-R-compatibility.json`, `source-input-identities.json`, `budget-proposal.json`, and the focused test output for the reviewable contract.",
        "",
    ])
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return {"manifest": manifest, "source": source, "budget": budget, "compatibility": compatibility, "proposal": proposal, "protocol": protocol, "authorization": authorization}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        write_preflight_artifacts(args.out)
        return 0
    if args.authorization is None:
        raise SystemExit("--authorization is required for --mode run")
    run_matrix(args.authorization, manifest_path=args.manifest or (args.out / "branch-manifest.json"), out_dir=args.out)
    return 0


__all__ = [
    "ATTEMPT_ID",
    "AUTH_SCHEMA",
    "AuthorizationError",
    "BASE",
    "BEHAVIOR_PREFERENCE",
    "Counters",
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_GUARD_SHA256",
    "EXPECTED_SOURCE_HASHES",
    "GLOBAL_LIMIT_AFTER_APPROVAL",
    "HISTORICAL_BUDGET_PATH",
    "MAX_STEPS_PER_BRANCH",
    "MAX_TOTAL_STEPS",
    "OUT_DEFAULT",
    "PROTOCOL_ID",
    "RuntimeAdapters",
    "RuntimeTechnicalStop",
    "RUN_ID",
    "build_branch_manifest",
    "authorization_template",
    "canonical_hash",
    "execute_branch",
    "historical_compatibility",
    "historical_compatibility_with_pinned_bridge",
    "historical_input_hashes",
    "observation_digest",
    "read_budget_identity",
    "runtime_dependency_hashes",
    "runtime_dependency_identities",
    "r_control_bridge_hashes",
    "r_control_bridge_identities",
    "run_matrix",
    "sha256_file",
    "snapshot_semantic_digest",
    "validate_authorization",
    "validate_manifest",
    "write_preflight_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
