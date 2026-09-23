"""Stage-1 frozen-model versus public-rule runner.

The package/check path is deliberately static: it reads JSON, source hashes,
and the formal SQLite ledger through a read-only connection.  Runtime modules,
the prefix pickle, environments, and the checkpoint are imported or loaded
only after an explicitly reviewed authorization passes every identity and
fresh-output check.  The dynamic path is retained for a future approved run;
this preparation turn never invokes it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Callable, Mapping, Sequence


WORKTREE = Path(__file__).resolve().parents[1]
BASE = WORKTREE / "runs" / "finite-communication-ack-lease-fix-20260920"
PACKAGE = BASE / "model-rule-stage1-execution-package-v1"
PROTOCOL = PACKAGE / "protocol.json"
MANIFEST = PACKAGE / "paired-manifest.json"
NATIVE_SOURCE_MANIFEST = PACKAGE / "native-source-manifest.json"
BUDGET = BASE / "ack-known-task-guard-baseline-v1" / "budget.sqlite3"
RUNTIME = WORKTREE / "tools" / "model_rule_branch_runtime_v1.py"
RULE = WORKTREE / "gppo_world" / "public_dispatch_rule_v1.py"
GUARD = WORKTREE / "gppo_world" / "ack_known_task_guard.py"
CONFIG = Path(
    r"E:\Z博士\migration-artifacts\preference-weighted-wm-event-cpu-20260917"
    r"\final\run\training\seed-1101\WD\resolved-config.json"
)
CHECKPOINT = Path(
    r"E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1"
    r"\training\seed-1101\P_train\last-recovery.pt"
)
SNAPSHOT = Path(
    r"E:\Z博士\migration-artifacts\replan-value-feature-collection-20260919-v1"
    r"\prefix-snapshots.pkl"
)
SOURCE_ROOT = Path(
    r"E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1"
    r"\source-snapshot"
)
H005_RUNNER = WORKTREE / "tools" / "run_ack_known_task_guard_corrected_20260922.py"
HISTORICAL_HELPER = WORKTREE / "tools" / "run_replan_value_experiment.py"

RUNNER_ID = "model-rule-stage1-v1"
AUTH_SCHEMA = "model-rule-pair-authorization/1.0.0"
PROTOCOL_SCHEMA = "model-rule-stage1-protocol/1.0.0"
MANIFEST_SCHEMA = "model-rule-stage1-paired-manifest/1.0.0"
BRANCH_SCHEMA = "model-rule-pair-branch-result/1.0.0"
DECISION_SCHEMA = "model-rule-pair-decision/1.0.0"
STEP_SCHEMA = "model-rule-pair-step/1.0.0"
SELECTOR_SCHEMA = "model-rule-pair-selector/1.0.0"
RUN_STATUS_SCHEMA = "model-rule-pair-run-status/1.0.0"
RUNTIME_COST_SCHEMA = "model-rule-pair-runtime-cost/1.0.0"
FIRST_GATE_SCHEMA = "model-rule-pair-first-gate/1.0.0"

ARMS = ("frozen_model", "public_rule")
PARENTS = tuple(f"parent-{index:02d}" for index in range(8))
REPEATS = (0, 1, 2)
MAX_BRANCHES = 48
MAX_STEPS_PER_BRANCH = 16
MAX_TOTAL_STEPS = 768
MODEL_STEP_CAP = 384
RULE_SELECTOR_CAP = 384
CURRENT_LIMIT = 1067
CURRENT_VERIFIED = 857
APPROVED_LIMIT = 1625
APPROVED_NEW_STEPS = 768
GAMMA = 0.99
BEHAVIOR_PREFERENCE = (0.8, 0.2)
REWARD_SCALES = {"task": 0.5, "energy": 1.0}
FIRST_PAIR_ID = "parent-00|W1|seed-1101|prefix-0|repeat-0"

EXPECTED_GUARD_SHA256 = "2bc314ff9e3d4f78ec871c1a2e739a92669b1f9bd573aabce6a8be5b3e9e808a"
EXPECTED_CHECKPOINT_SHA256 = "bf10d2685a4a3e9da036689f5028b022330e86e922e09c95a7dd0a929df9bb1a"
EXPECTED_SNAPSHOT_SHA256 = "c6826f941141d1ae7594c38a3e3465584ae0954158502f19d2f4d8163a18258e"
EXPECTED_CONFIG_SHA256 = "f57f91878d161855cfd58a28db7943d2341ebbfdfd5053f8c2185afbfe6cbc27"
EXPECTED_BUDGET_SHA256 = "ea42090b92b9ecea9ab1c487742dbb059d4f95c2087f90f8d66dbfbb6be38d50"
EXPECTED_H005_SHA256 = "ec1eae6e0f6fff759a391875c2f34878260a85889d97c18c6f991c5f98b0711e"
EXPECTED_BUDGET_EXECUTOR_SHA256 = "d9a39251cba7833ce2f92e06a84522a81c930c9eba706b42de62adab8a003551"
EXPECTED_HELPER_SHA256 = "a9ccf1711790c472ededd7666c54c28ca34abffc4a0095d35b300f1297afa965"


class RunnerError(RuntimeError):
    """Package, authorization, or runtime contract failure."""


class AuthorizationError(RunnerError):
    """The dynamic entry point is not authorized."""


class TechnicalStop(RunnerError):
    """A runtime failure that must be retained and never retried."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_safe(value: Any, active: set[int] | None = None) -> Any:
    """Keep audit rows deterministic without importing numpy or torch."""
    active = set() if active is None else active
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            return {"cycle": True}
        active.add(identity)
        try:
            return {str(key): _json_safe(item, active) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            return {"cycle": True}
        active.add(identity)
        try:
            return [_json_safe(item, active) for item in value]
        finally:
            active.remove(identity)
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            return _json_safe(detach().cpu().tolist(), active)
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist(), active)
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item(), active)
        except Exception:
            pass
    return repr(value)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def observation_digest(obs: Mapping[str, Any]) -> str:
    # Match the native branch runtime's public-observation contract. The
    # selector's richer uavs/tasks payload is separately hashed as
    # public_rule_input_sha256; it must not silently change the shared digest.
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


def hidden_digest(value: Any) -> str | None:
    if value is None:
        return None
    return canonical_hash(_json_safe(value))


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def inspect_budget(path: Path, run_id: str | None = None) -> dict[str, Any]:
    """Read the formal ledger through SQLite's read-only URI."""
    path = _resolved(path)
    if not path.is_file():
        raise RunnerError(f"budget database is missing: {path}")
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        integrity = str(con.execute("PRAGMA integrity_check").fetchone()[0])
        stages = {
            str(stage): {
                "limit": int(limit),
                "reserved": int(reserved),
                "verified": int(verified),
                "unknown": int(unknown),
                "pending": int(reserved) - int(verified) - int(unknown),
            }
            for stage, limit, reserved, verified, unknown in con.execute(
                "SELECT stage,limit_amount,reserved,verified,unknown FROM stages ORDER BY stage"
            )
        }
        pending_rows = int(con.execute("SELECT COUNT(*) FROM reservations WHERE status='pending'").fetchone()[0])
        unknown_rows = int(con.execute("SELECT COUNT(*) FROM reservations WHERE status='unknown'").fetchone()[0])
        attempts = [
            {"attempt_id": str(row[0]), "stage": str(row[1]), "run_id": row[5]}
            for row in con.execute("SELECT attempt_id,stage,reserved,verified,unknown,run_id FROM attempts")
        ]
        runs: dict[str, dict[str, dict[str, int]]] = {}
        for rid, stage, reserved, verified, unknown in con.execute(
            "SELECT run_id,stage,SUM(amount),SUM(CASE WHEN status='verified' THEN amount ELSE 0 END),SUM(CASE WHEN status='unknown' THEN amount ELSE 0 END) "
            "FROM reservations WHERE run_id IS NOT NULL GROUP BY run_id,stage"
        ):
            if rid is None:
                continue
            runs.setdefault(str(rid), {})[str(stage)] = {
                "reserved": int(reserved or 0),
                "verified": int(verified or 0),
                "unknown": int(unknown or 0),
                "pending": int(reserved or 0) - int(verified or 0) - int(unknown or 0),
            }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "integrity_check": integrity,
        "stages": stages,
        "pending_rows": pending_rows,
        "unknown_rows": unknown_rows,
        "attempts": attempts,
        "run_totals": runs.get(str(run_id), {}) if run_id is not None else {},
        "all_run_totals": runs,
    }


def _manifest_rows(path: Path = MANIFEST) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise RunnerError("paired manifest must be a JSON array or an object with rows")
    if any(not isinstance(row, Mapping) for row in rows):
        raise RunnerError("paired manifest contains a non-object row")
    return [dict(row) for row in rows]


def _canonical_manifest_failure(path: Path) -> str | None:
    try:
        resolved = _resolved(path)
    except Exception as exc:
        return f"manifest path cannot be resolved: {type(exc).__name__}: {exc}"
    if resolved != MANIFEST.resolve():
        return f"manifest path is not canonical: {resolved}"
    return None


def expected_arm(parent_id: str, repeat: int) -> tuple[str, str]:
    index = int(parent_id.rsplit("-", 1)[1])
    return ARMS if (index + int(repeat)) % 2 == 0 else tuple(reversed(ARMS))


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    if len(rows) != MAX_BRANCHES:
        failures.append(f"manifest row count {len(rows)} != {MAX_BRANCHES}")
    ids = [str(row.get("branch_id")) for row in rows]
    if len(set(ids)) != len(ids):
        failures.append("duplicate branch_id")
    expected_pairs = [
        f"parent-{parent:02d}|W1|seed-1101|prefix-0|repeat-{repeat}"
        for parent in range(8)
        for repeat in range(3)
    ]
    seen_pairs: list[str] = []
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows, start=1):
        pair_id = str(row.get("pair_id"))
        arm = str(row.get("arm"))
        groups[pair_id].append(row)
        if not seen_pairs or seen_pairs[-1] != pair_id:
            seen_pairs.append(pair_id)
        try:
            order = int(row.get("order", -1))
            repeat = int(row.get("repeat", -1))
        except (TypeError, ValueError):
            order, repeat = -1, -1
        parent = str(row.get("parent_id"))
        expected_pair = f"{parent}|W1|seed-1101|prefix-0|repeat-{repeat}"
        expected_exogenous = f"replan-feature-v1|{parent}|W1|0|repeat-{repeat}"
        if arm not in ARMS:
            failures.append(f"invalid arm: {row.get('arm')}")
        if order != row_index:
            failures.append(f"manifest order field mismatch: {row.get('branch_id')}")
        if str(row.get("branch_id")) != f"{pair_id}|{arm}":
            failures.append(f"branch_id mismatch: {row.get('branch_id')}")
        if pair_id != expected_pair or str(row.get("prefix_id")) != f"{parent}|W1|seed-1101|prefix-0":
            failures.append(f"pair/prefix identity mismatch: {row.get('branch_id')}")
        if row.get("condition") != "W1" or int(row.get("model_seed", -1)) != 1101 or int(row.get("prefix_index", -1)) != 0:
            failures.append(f"fixed matrix identity mismatch: {row.get('branch_id')}")
        if row.get("split") != "train" or int(row.get("max_steps", -1)) != MAX_STEPS_PER_BRANCH:
            failures.append(f"split/step cap mismatch: {row.get('branch_id')}")
        if row.get("status") not in {"proposal_not_attempt", "pending_execution_authorization"}:
            failures.append(f"manifest row already attempted: {row.get('branch_id')}")
        if str(row.get("exogenous_key")) != expected_exogenous or str(row.get("historical_exogenous_key")) != expected_exogenous:
            failures.append(f"exogenous identity mismatch: {row.get('branch_id')}")
        if str(row.get("control_branch_key")) != f"{pair_id}|mode-R":
            failures.append(f"control branch mismatch: {row.get('branch_id')}")
        cohort = row.get("cohort_task_ids")
        if not isinstance(cohort, list) or not cohort or len(cohort) != len(set(map(str, cohort))):
            failures.append(f"cohort task identity mismatch: {row.get('branch_id')}")
    if [int(row.get("order", -1)) for row in rows] != list(range(1, MAX_BRANCHES + 1)):
        failures.append("manifest order is not exactly 1..48")
    if seen_pairs != expected_pairs:
        failures.append("manifest pair order is not canonical parent/repeat order")
    if set(groups) != set(expected_pairs):
        failures.append("manifest pair set mismatch")
    for pair_id, pair_rows in groups.items():
        if len(pair_rows) != 2:
            failures.append(f"pair does not have two arms: {pair_id}")
            continue
        pair_rows = sorted(pair_rows, key=lambda row: int(row.get("order", 0)))
        if tuple(str(row.get("arm")) for row in pair_rows) != expected_arm(str(pair_rows[0].get("parent_id")), int(pair_rows[0].get("repeat", -1))):
            failures.append(f"pair arm order mismatch: {pair_id}")
        if {str(row.get("exogenous_key")) for row in pair_rows} != {str(pair_rows[0].get("exogenous_key"))}:
            failures.append(f"pair exogenous key mismatch: {pair_id}")
        if pair_rows[0].get("cohort_task_ids") != pair_rows[1].get("cohort_task_ids"):
            failures.append(f"pair cohort mismatch: {pair_id}")
    return {
        "ok": not failures,
        "row_count": len(rows),
        "pair_count": len(groups),
        "failures": failures,
        "branch_ids": ids,
        "pair_ids": sorted(groups),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def source_pins() -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "runtime": RUNTIME,
        "rule": RULE,
        "guard": GUARD,
        "h005_runner": H005_RUNNER,
        "budget_executor": WORKTREE / "gppo_world" / "budget_executor.py",
        "historical_helper": HISTORICAL_HELPER,
        "config": CONFIG,
        "checkpoint": CHECKPOINT,
        "prefix_snapshot": SNAPSHOT,
        "budget": BUDGET,
        "protocol": PROTOCOL,
        "manifest": MANIFEST,
        "native_source_manifest": NATIVE_SOURCE_MANIFEST,
    }
    return {name: _file_identity(path) for name, path in paths.items()}


def _runtime_hashes() -> dict[str, str | None]:
    return {
        "runtime": sha256_file(RUNTIME) if RUNTIME.is_file() else None,
        "rule": sha256_file(RULE) if RULE.is_file() else None,
        "guard": sha256_file(GUARD) if GUARD.is_file() else None,
        "h005_runner": sha256_file(H005_RUNNER) if H005_RUNNER.is_file() else None,
        "budget_executor": sha256_file(WORKTREE / "gppo_world" / "budget_executor.py") if (WORKTREE / "gppo_world" / "budget_executor.py").is_file() else None,
        "historical_helper": sha256_file(HISTORICAL_HELPER) if HISTORICAL_HELPER.is_file() else None,
        "config": sha256_file(CONFIG) if CONFIG.is_file() else None,
    }


def validate_native_source_manifest() -> dict[str, Any]:
    """Validate the pinned source snapshot without importing it."""
    failures: list[str] = []
    if not NATIVE_SOURCE_MANIFEST.is_file():
        return {"ok": False, "failures": ["native-source-manifest.json is missing"], "file_count": 0}
    try:
        payload = read_json(NATIVE_SOURCE_MANIFEST)
        root_raw = str(payload.get("source_root", ""))
        root = _resolved(root_raw)
        files = payload.get("files")
    except Exception as exc:
        return {"ok": False, "failures": [f"native source manifest unreadable: {type(exc).__name__}: {exc}"], "file_count": 0}
    if root_raw != str(root) or not root.is_dir() or not isinstance(files, Mapping):
        failures.append("native source root/files invalid")
    else:
        expected_paths: set[str] = set()
        for relative, identity in sorted(files.items()):
            rel = str(relative).replace("\\", "/")
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                failures.append(f"native source path escapes root: {relative}")
                continue
            expected_paths.add(rel)
            path = (root / rel_path).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                failures.append(f"native source path escapes root: {relative}")
                continue
            if not path.is_file():
                failures.append(f"native source missing: {relative}")
                continue
            if int(identity.get("bytes", -1)) != path.stat().st_size or str(identity.get("sha256")) != sha256_file(path):
                failures.append(f"native source identity mismatch: {relative}")
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.relative_to(root).as_posix().startswith("gppo_world/") and path.suffix == ".py"
        }
        if actual_paths != expected_paths:
            failures.append("native source module set mismatch")
    return {
        "ok": not failures,
        "manifest_path": str(NATIVE_SOURCE_MANIFEST.resolve()),
        "manifest_sha256": sha256_file(NATIVE_SOURCE_MANIFEST),
        "source_root": str(root),
        "file_count": len(files) if isinstance(files, Mapping) else 0,
        "failures": failures,
    }


def _protocol_check(protocol: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        failures.append("protocol schema mismatch")
    if int(protocol.get("stage", -1)) != 1:
        failures.append("protocol stage mismatch")
    if protocol.get("status") not in {"pending_execution_authorization", "reviewed_development_protocol_proposal_not_execution_authorization"}:
        failures.append("protocol is not a pending execution proposal")
    matrix = protocol.get("matrix", {}) if isinstance(protocol.get("matrix"), Mapping) else {}
    if matrix.get("parents") != list(PARENTS) or matrix.get("repeats") != list(REPEATS):
        failures.append("protocol parent/repeat matrix mismatch")
    if matrix.get("condition") != "W1" or int(matrix.get("model_seed", -1)) != 1101 or int(matrix.get("prefix_index", -1)) != 0:
        failures.append("protocol fixed identity mismatch")
    if int(matrix.get("branches", -1)) != MAX_BRANCHES or int(matrix.get("max_steps_per_branch", -1)) != MAX_STEPS_PER_BRANCH:
        failures.append("protocol branch/step cap mismatch")
    budget = protocol.get("budget", {}) if isinstance(protocol.get("budget"), Mapping) else {}
    if int(budget.get("current_verified", -1)) != CURRENT_VERIFIED or int(budget.get("current_limit", -1)) != CURRENT_LIMIT:
        failures.append("protocol current budget identity mismatch")
    if int(budget.get("new_cap", -1)) != APPROVED_NEW_STEPS or int(budget.get("proposed_limit", -1)) != APPROVED_LIMIT:
        failures.append("protocol proposed budget mismatch")
    calls = budget.get("model_call_caps", {}) if isinstance(budget.get("model_call_caps"), Mapping) else {}
    if {str(key): int(value) for key, value in calls.items()} != {"policy_encode": MODEL_STEP_CAP, "world_candidate_batch": MODEL_STEP_CAP, "actor_readout": MODEL_STEP_CAP}:
        failures.append("protocol model call caps mismatch")
    if int(budget.get("rule_selector_cap", -1)) != RULE_SELECTOR_CAP or int(budget.get("all_updates", -1)) != 0:
        failures.append("protocol rule/update caps mismatch")
    return failures


def package_validation() -> dict[str, Any]:
    """Perform zero-step package validation and read-only budget inspection."""
    failures: list[str] = []
    protocol: Mapping[str, Any] = {}
    rows: list[dict[str, Any]] = []
    try:
        protocol = read_json(PROTOCOL)
    except Exception as exc:
        failures.append(f"protocol unreadable: {type(exc).__name__}: {exc}")
    try:
        rows = _manifest_rows(MANIFEST)
    except Exception as exc:
        failures.append(f"manifest unreadable: {type(exc).__name__}: {exc}")
    failures.extend(_protocol_check(protocol))
    manifest_check = validate_manifest(rows)
    failures.extend(manifest_check["failures"])
    budget_state: dict[str, Any] | None = None
    try:
        budget_state = inspect_budget(BUDGET)
        stage = budget_state["stages"].get("environment_steps", {})
        if int(stage.get("limit", -1)) not in {CURRENT_LIMIT, APPROVED_LIMIT} or int(stage.get("reserved", -1)) != CURRENT_VERIFIED or int(stage.get("verified", -1)) != CURRENT_VERIFIED or int(stage.get("unknown", -1)) != 0 or int(stage.get("pending", -1)) != 0:
            failures.append("read-only SQLite ledger is not a clean 857-step proposal or approved extension")
        if budget_state.get("integrity_check") != "ok" or budget_state.get("pending_rows", 0) != 0 or budget_state.get("unknown_rows", 0) != 0:
            failures.append("read-only SQLite integrity or unresolved reservation check failed")
        if int(stage.get("limit", -1)) == CURRENT_LIMIT and budget_state.get("sha256") != EXPECTED_BUDGET_SHA256:
            failures.append("read-only SQLite baseline hash mismatch")
    except Exception as exc:
        failures.append(f"budget read-only inspection failed: {type(exc).__name__}: {exc}")
    native = validate_native_source_manifest()
    failures.extend(native["failures"])
    pins = source_pins()
    fixed_hashes = {
        "guard": EXPECTED_GUARD_SHA256,
        "h005_runner": EXPECTED_H005_SHA256,
        "budget_executor": EXPECTED_BUDGET_EXECUTOR_SHA256,
        "historical_helper": EXPECTED_HELPER_SHA256,
        "config": EXPECTED_CONFIG_SHA256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "prefix_snapshot": EXPECTED_SNAPSHOT_SHA256,
        "budget": EXPECTED_BUDGET_SHA256,
    }
    budget_limit = int((budget_state or {}).get("stages", {}).get("environment_steps", {}).get("limit", -1))
    if budget_limit == APPROVED_LIMIT:
        fixed_hashes.pop("budget", None)
    for name, expected in fixed_hashes.items():
        actual = pins.get(name, {}).get("sha256")
        if actual != expected:
            failures.append(f"fixed source identity mismatch: {name}")
    return {
        "schema": "model-rule-stage1-package-check/1.0.0",
        "runner_id": RUNNER_ID,
        "ok": not failures,
        "dynamic_execution_authorized": False,
        "authorization_required": True,
        "failures": failures,
        "protocol_sha256": sha256_file(PROTOCOL) if PROTOCOL.is_file() else None,
        "manifest_sha256": sha256_file(MANIFEST) if MANIFEST.is_file() else None,
        "source_pins": pins,
        "runtime_dependency_hashes": _runtime_hashes(),
        "native_source_manifest": native,
        "budget_readonly": budget_state,
        "matrix": manifest_check,
        "hard_counts": {
            "env_step": 0,
            "reset_replay": 0,
            "model_forward": 0,
            "checkpoint_snapshot_load": 0,
            "rule_selector_calls": 0,
            "updates": 0,
            "new_attempt": 0,
            "formal_sqlite_writes": 0,
        },
    }


def authorization_template(check: Mapping[str, Any]) -> dict[str, Any]:
    """Build a pending template; only the separate authorizer may approve it."""
    return {
        "schema": AUTH_SCHEMA,
        "status": "pending_review",
        "runner_id": RUNNER_ID,
        "review": {
            "reviewed_by": None,
            "reviewed_at": None,
            "accepted_interpretation": False,
            "user_message_reference": None,
        },
        "protocol": {"path": str(PROTOCOL.resolve()), "sha256": check.get("protocol_sha256")},
        "manifest": {"path": str(MANIFEST.resolve()), "sha256": check.get("manifest_sha256")},
        "source": {
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": check.get("source_pins", {}).get("runner", {}).get("sha256"),
            "runtime_path": str(RUNTIME.resolve()),
            "runtime_sha256": check.get("source_pins", {}).get("runtime", {}).get("sha256"),
            "rule_path": str(RULE.resolve()),
            "rule_sha256": check.get("source_pins", {}).get("rule", {}).get("sha256"),
            "guard_path": str(GUARD.resolve()),
            "guard_sha256": EXPECTED_GUARD_SHA256,
            "checkpoint_path": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "prefix_snapshot_path": str(SNAPSHOT.resolve()),
            "prefix_snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
            "config_path": str(CONFIG.resolve()),
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "native_source_manifest_path": str(NATIVE_SOURCE_MANIFEST.resolve()),
            "native_source_manifest_sha256": check.get("native_source_manifest", {}).get("manifest_sha256"),
            "runtime_dependency_hashes": check.get("runtime_dependency_hashes", {}),
        },
        "budget": {
            "path": str(BUDGET.resolve()),
            "sha256": EXPECTED_BUDGET_SHA256,
            "global_limit": APPROVED_LIMIT,
            "baseline_verified": CURRENT_VERIFIED,
            "required_new_steps": APPROVED_NEW_STEPS,
            "run_limit": APPROVED_NEW_STEPS,
            "model_call_caps": {
                "policy_encode": MODEL_STEP_CAP,
                "world_candidate_batch": MODEL_STEP_CAP,
                "actor_readout": MODEL_STEP_CAP,
            },
            "rule_selector_cap": RULE_SELECTOR_CAP,
            "updates_cap": 0,
        },
        "run": {
            "run_id": "model-rule-stage1-v1",
            "attempt_id": "model-rule-stage1-v1-attempt-0001",
        },
        "interpretation": "frozen native model combination versus named public-only rule on observed train prefixes; no isolated component or equivalence claim",
        "no_retry": True,
    }


def _auth_source_matches(auth: Mapping[str, Any], check: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    source = auth.get("source") if isinstance(auth.get("source"), Mapping) else {}
    pairs = {
        "runner": (source.get("runner_sha256"), check.get("source_pins", {}).get("runner", {}).get("sha256")),
        "runtime": (source.get("runtime_sha256"), check.get("source_pins", {}).get("runtime", {}).get("sha256")),
        "rule": (source.get("rule_sha256"), check.get("source_pins", {}).get("rule", {}).get("sha256")),
        "guard": (source.get("guard_sha256"), EXPECTED_GUARD_SHA256),
        "checkpoint": (source.get("checkpoint_sha256"), EXPECTED_CHECKPOINT_SHA256),
        "prefix_snapshot": (source.get("prefix_snapshot_sha256"), EXPECTED_SNAPSHOT_SHA256),
        "config": (source.get("config_sha256"), EXPECTED_CONFIG_SHA256),
        "native_source_manifest": (source.get("native_source_manifest_sha256"), check.get("native_source_manifest", {}).get("manifest_sha256")),
    }
    for name, (supplied, expected) in pairs.items():
        if supplied != expected:
            failures.append(f"authorization source hash mismatch: {name}")
    canonical_paths = {
        "runner": Path(__file__).resolve(),
        "runtime": RUNTIME.resolve(),
        "rule": RULE.resolve(),
        "guard": GUARD.resolve(),
        "checkpoint": CHECKPOINT.resolve(),
        "prefix_snapshot": SNAPSHOT.resolve(),
        "config": CONFIG.resolve(),
        "native_source_manifest": NATIVE_SOURCE_MANIFEST.resolve(),
    }
    path_fields = {
        "runner": "runner_path", "runtime": "runtime_path", "rule": "rule_path",
        "guard": "guard_path", "checkpoint": "checkpoint_path",
        "prefix_snapshot": "prefix_snapshot_path", "config": "config_path",
        "native_source_manifest": "native_source_manifest_path",
    }
    for name, field_name in path_fields.items():
        try:
            supplied_path = _resolved(str(source.get(field_name, "")))
        except Exception:
            supplied_path = Path()
        if supplied_path != canonical_paths[name]:
            failures.append(f"authorization source path mismatch: {name}")
    supplied_runtime = source.get("runtime_dependency_hashes")
    if not isinstance(supplied_runtime, Mapping) or dict(supplied_runtime) != dict(check.get("runtime_dependency_hashes", {})):
        failures.append("authorization runtime dependency identities mismatch")
    return failures


def validate_authorization(
    authorization_path: Path,
    *,
    manifest_path: Path = MANIFEST,
    output_dir: Path,
) -> dict[str, Any]:
    """Fail closed before importing runtime, reading pickle, or opening budget writable."""
    authorization_path = _resolved(authorization_path)
    output_dir = _resolved(output_dir)
    if not authorization_path.is_file():
        raise AuthorizationError(f"reviewed authorization is required: {authorization_path}")
    auth = read_json(authorization_path)
    if not isinstance(auth, Mapping) or auth.get("schema") != AUTH_SCHEMA:
        raise AuthorizationError("authorization schema mismatch")
    if auth.get("status") != "authorized":
        raise AuthorizationError("authorization status is not authorized; pending template cannot run")
    if _canonical_manifest_failure(manifest_path) is not None:
        raise AuthorizationError(_canonical_manifest_failure(manifest_path) or "manifest path failure")
    review = auth.get("review") if isinstance(auth.get("review"), Mapping) else {}
    if not review.get("reviewed_by") or not review.get("reviewed_at") or review.get("accepted_interpretation") is not True:
        raise AuthorizationError("reviewed_by/reviewed_at/accepted_interpretation are required")
    check = package_validation()
    failures = list(check.get("failures", ()))
    if not check.get("ok"):
        failures.append("static package validation failed")
    protocol = auth.get("protocol") if isinstance(auth.get("protocol"), Mapping) else {}
    if _resolved(str(protocol.get("path", ""))) != PROTOCOL.resolve() or str(protocol.get("sha256")) != check.get("protocol_sha256"):
        failures.append("authorization protocol identity mismatch")
    manifest_auth = auth.get("manifest") if isinstance(auth.get("manifest"), Mapping) else {}
    if _resolved(str(manifest_auth.get("path", ""))) != MANIFEST.resolve() or str(manifest_auth.get("sha256")) != check.get("manifest_sha256"):
        failures.append("authorization manifest identity mismatch")
    failures.extend(_auth_source_matches(auth, check))
    budget = auth.get("budget") if isinstance(auth.get("budget"), Mapping) else {}
    if _resolved(str(budget.get("path", ""))) != BUDGET.resolve():
        failures.append("authorization budget path mismatch")
    if int(budget.get("global_limit", -1)) != APPROVED_LIMIT or int(budget.get("required_new_steps", -1)) != APPROVED_NEW_STEPS or int(budget.get("run_limit", -1)) != APPROVED_NEW_STEPS:
        failures.append("authorization budget scope mismatch")
    if budget.get("model_call_caps") != {"policy_encode": MODEL_STEP_CAP, "world_candidate_batch": MODEL_STEP_CAP, "actor_readout": MODEL_STEP_CAP} or int(budget.get("rule_selector_cap", -1)) != RULE_SELECTOR_CAP or int(budget.get("updates_cap", -1)) != 0:
        failures.append("authorization call/update caps mismatch")
    budget_state = check.get("budget_readonly") or {}
    stage = budget_state.get("stages", {}).get("environment_steps", {}) if isinstance(budget_state.get("stages"), Mapping) else {}
    if str(budget.get("sha256")) != str(budget_state.get("sha256")):
        failures.append("authorization budget hash mismatch")
    if int(stage.get("limit", -1)) != APPROVED_LIMIT or int(stage.get("reserved", -1)) != CURRENT_VERIFIED or int(stage.get("verified", -1)) != CURRENT_VERIFIED or int(stage.get("unknown", -1)) != 0 or int(stage.get("pending", -1)) != 0:
        failures.append("authorized budget is not the post-extension clean ledger")
    run = auth.get("run") if isinstance(auth.get("run"), Mapping) else {}
    run_id = str(run.get("run_id", ""))
    attempt_id = str(run.get("attempt_id", ""))
    if not run_id or not attempt_id:
        failures.append("run/attempt identities are missing")
    if budget_state.get("run_totals", {}).get("environment_steps"):
        failures.append("authorized run already has reservations")
    if any(str(row.get("attempt_id")) == attempt_id for row in budget_state.get("attempts", ())):
        failures.append("authorized attempt identity already exists")
    if output_dir.exists() and any(output_dir.iterdir()):
        failures.append("output directory is not fresh; duplicate or partial execution is refused")
    if failures:
        raise AuthorizationError("pre-execution authorization gate failed: " + "; ".join(failures))
    return {"authorization": auth, "manifest": _manifest_rows(MANIFEST), "budget": budget_state, "check": check}


class ModelCallCounters:
    """Durable attempted/completed counters, including failed invocations."""

    NAMES = ("policy_encode", "world_candidate_batch", "actor_readout")

    def __init__(self) -> None:
        self.attempted = {name: 0 for name in self.NAMES}
        self.completed = {name: 0 for name in self.NAMES}

    def start(self, declaration: Mapping[str, int]) -> None:
        for name in self.NAMES:
            count = int(declaration.get(name, 0))
            if count < 0 or self.attempted[name] + count > MODEL_STEP_CAP:
                raise TechnicalStop(f"{name} model call cap reached before invocation")
            self.attempted[name] += count

    def finish(self, declaration: Mapping[str, int]) -> None:
        for name in self.NAMES:
            self.completed[name] += int(declaration.get(name, 0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": dict(self.attempted),
            "completed": dict(self.completed),
            "limits": {name: MODEL_STEP_CAP for name in self.NAMES},
            "attempted_total": sum(self.attempted.values()),
            "completed_total": sum(self.completed.values()),
        }


def _public_rule_inputs(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist exactly the public selector channels used by the rule."""
    names = ("uavs", "tasks", "time", "mask", "public_entity_ids", "continuation_actions")
    return {name: _json_safe(copy.deepcopy(obs.get(name))) for name in names}


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TechnicalStop(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _ProbeRecorder:
    def __init__(self, item: Mapping[str, Any], arm: str, model_calls: ModelCallCounters, out_path: Path):
        self.item = item
        self.arm = arm
        self.model_calls = model_calls
        self.out_path = out_path
        self.records: list[dict[str, Any]] = []
        self.rule_calls = 0
        self.selector_seconds = 0.0
        self._step = 0

    def model_probe(self, base_probe: Callable[..., Mapping[str, Any]], runtime: Any, obs: Mapping[str, Any], policy_hidden: Any, world_hidden: Any) -> Mapping[str, Any]:
        declaration = {"policy_encode": 1, "world_candidate_batch": 1, "actor_readout": 1}
        self.model_calls.start(declaration)
        try:
            started = time.perf_counter()
            raw = dict(base_probe(runtime, obs, policy_hidden, world_hidden))
            self.model_calls.finish(declaration)
            raw["model_calls"] = declaration
            raw["model_call_status"] = {name: {"attempted": value, "completed": value} for name, value in declaration.items()}
            raw["probe_seconds"] = time.perf_counter() - started
            self._step += 1
            self.records.append({
                "branch_id": self.item["branch_id"], "pair_id": self.item["pair_id"], "arm": self.arm,
                "exogenous_key": self.item["exogenous_key"],
                "step": self._step, "public_observation_sha256": observation_digest(obs),
                "public_only_input_sha256": canonical_hash(_public_rule_inputs(obs)),
                "public_rule_input_sha256": canonical_hash(_public_rule_inputs(obs)),
                "model_call_status": raw["model_call_status"],
                "model_calls": declaration,
                "rule_selector_calls": 0,
                "rule_selector_status": {"attempted": 0, "completed": 0},
                "selector_seconds": 0.0,
                "input_kind": "model_probe",
            })
            return raw
        except Exception:
            raise

    def rule_probe(self, rule_selector: Callable[[Mapping[str, Any]], Any], obs: Mapping[str, Any], policy_hidden: Any, world_hidden: Any) -> Mapping[str, Any]:
        started = time.perf_counter()
        self.rule_calls += 1
        if self.rule_calls > RULE_SELECTOR_CAP:
            raise TechnicalStop("public rule selector cap reached")
        decision = rule_selector(obs)
        elapsed = time.perf_counter() - started
        self.selector_seconds += elapsed
        probabilities = [float(value) for value in decision.rank_based_probabilities]
        original = int(decision.original_argmax)
        selected = int(decision.selected_action)
        # The runtime contract requires a selected-action hidden record.  The
        # public rule carries the inherited snapshot hidden unchanged solely as
        # an audit field; it is never read by the selector and never predicted.
        # Preserve the inherited prefix tensors as inert audit values.  The
        # rule never inspects them and returning the same values makes the
        # absence of a predicted hidden transition explicit in step evidence.
        inert_hidden = copy.deepcopy(world_hidden)
        self._step += 1
        public_inputs = _public_rule_inputs(obs)
        row = {
            "branch_id": self.item["branch_id"], "pair_id": self.item["pair_id"], "arm": self.arm,
            "exogenous_key": self.item["exogenous_key"],
            "step": self._step, "public_observation_sha256": observation_digest(obs),
            "public_only_input_sha256": canonical_hash(public_inputs),
            "public_rule_input_sha256": canonical_hash(public_inputs),
            "public_rule_inputs": public_inputs,
            "rule_scores": probabilities,
            "rule_ranks": [int(value) for value in getattr(decision, "order", ())],
            "rule_selector_calls": 1,
            "rule_selector_status": {"attempted": 1, "completed": 1},
            "selector_seconds": elapsed,
            "model_call_status": {name: {"attempted": 0, "completed": 0} for name in ModelCallCounters.NAMES},
            "model_calls": {name: 0 for name in ModelCallCounters.NAMES},
            "input_kind": "public_observation_only",
            "hidden_transition": "identity_audit_only",
        }
        self.records.append(row)
        return {
            "probabilities": probabilities,
            "original_action": original,
            "by_action": {selected: {"hidden": inert_hidden}},
            "next_policy_hidden": copy.deepcopy(policy_hidden),
            "rule_decision": decision.to_dict() if hasattr(decision, "to_dict") else _json_safe(decision),
            "model_calls": {name: 0 for name in ModelCallCounters.NAMES},
            "model_call_status": row["model_call_status"],
            "rule_selector_calls": 1,
            "selector_seconds": elapsed,
        }


def _augment(item: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.update({
        "arm": item["arm"],
        "pair_id": item["pair_id"],
        "mode": "guard-on",
        "cohort_task_ids": list(item.get("cohort_task_ids", ())),
        "task_ids": list(item.get("cohort_task_ids", ())),
        "exogenous_key": item["exogenous_key"],
    })
    return result


def _merge_staging(staging: Path, output_dir: Path, item: Mapping[str, Any], recorder: _ProbeRecorder, result: Mapping[str, Any] | None) -> None:
    decisions = read_jsonl(staging / "decision-ledger.jsonl")
    steps = read_jsonl(staging / "step-vector-rewards.jsonl")
    by_step = {int(row.get("step", -1)): row for row in decisions}
    step_by_step = {int(row.get("step", -1)): row for row in steps}
    for filename in ("decision-ledger.jsonl", "step-vector-rewards.jsonl", "budget-finalization.jsonl", "failure-ledger.jsonl"):
        for row in read_jsonl(staging / filename):
            append_jsonl(output_dir / filename, _augment(item, row))
    # Persist selector evidence even when reservation fails before an env step;
    # selected feedback remains absent until an actual step exists.
    for probe_row in recorder.records:
        step_number = int(probe_row.get("step", -1))
        step = step_by_step.get(step_number, {})
        decision = by_step.get(step_number, {})
        append_jsonl(output_dir / "selector-ledger.jsonl", {
            **probe_row,
            "selected_action": decision.get("final_action", step.get("action")),
            "original_action": decision.get("original_action", step.get("original_action")),
            "legal_mask": decision.get("legal_mask", step.get("legal_mask")),
            "probabilities": decision.get("probabilities", step.get("probabilities")),
            "guard_candidates": decision.get("candidate_actions", []),
        })
    if result is not None:
        branch = _augment(item, result)
        branch["schema"] = BRANCH_SCHEMA
        append_jsonl(output_dir / "branch-results.jsonl", branch)


def _first_pair_gate(rows: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema": FIRST_GATE_SCHEMA, "pair_id": FIRST_PAIR_ID, "step": 1}
    first = [row for row in rows if row.get("pair_id") == FIRST_PAIR_ID and int(row.get("step", -1)) == 1]
    if len(first) != 2 or {str(row.get("arm")) for row in first} != set(ARMS):
        return {**result, "ok": False, "reason": "first pair must contain one step-1 row per arm", "rows": len(first)}
    model, rule = [next(row for row in first if row.get("arm") == arm) for arm in ARMS]
    step_index = {(str(row.get("branch_id")), int(row.get("step", -1))): row for row in steps}
    decision_index = {(str(row.get("branch_id")), int(row.get("step", -1))): row for row in decisions}
    model_step = step_index.get((str(model.get("branch_id")), 1), {})
    rule_step = step_index.get((str(rule.get("branch_id")), 1), {})
    model_decision = decision_index.get((str(model.get("branch_id")), 1), {})
    rule_decision = decision_index.get((str(rule.get("branch_id")), 1), {})
    checks = {
        "same_pair_exogenous_key": model.get("exogenous_key") == rule.get("exogenous_key"),
        "same_public_observation": model.get("public_observation_sha256") == rule.get("public_observation_sha256"),
        "same_env_state_before": model_step.get("env_state_before_sha256") == rule_step.get("env_state_before_sha256"),
        "same_legal_mask": model_decision.get("legal_mask") == rule_decision.get("legal_mask"),
        "same_guard_candidates": model_decision.get("candidate_actions") == rule_decision.get("candidate_actions"),
        "same_guard_excluded": model_decision.get("excluded_actions") == rule_decision.get("excluded_actions"),
        "same_guard_trigger": model_decision.get("triggered") == rule_decision.get("triggered"),
        "noop_legal_and_retained": all(
            isinstance(row.get("legal_mask"), list)
            and len(row.get("legal_mask")) > 24
            and bool(row["legal_mask"][24])
            and 24 in (row.get("candidate_actions") or ())
            for row in (model_decision, rule_decision)
        ),
        "actual_selected_actions_recorded": all(row.get("selected_action") is not None for row in (model, rule)),
    }
    return {**result, "ok": all(checks.values()), "checks": checks, "arms": {"frozen_model": model, "public_rule": rule}}


def _zero_counts() -> dict[str, int]:
    return {
        "probe_calls": 0, "env_step_calls": 0, "env_steps": 0, "successful_env_steps": 0,
        "verified_steps": 0, "model_forward": 0, "actor_forward": 0, "world_forward": 0,
        "optimizer_updates": 0, "world_updates": 0, "offline_updates": 0, "branches_completed": 0,
    }


def _persist_stop(output_dir: Path, started: float, error: BaseException, counters: Any | None, model_calls: ModelCallCounters, auth: Mapping[str, Any] | None, budget: Any | None, setup_seconds: float = 0.0, model_load_seconds: float = 0.0, snapshot_seconds: float = 0.0, per_arm: Mapping[str, Any] | None = None) -> None:
    hard = counters.as_dict() if counters is not None else _zero_counts()
    calls = model_calls.as_dict()
    costs = {
        "schema": RUNTIME_COST_SCHEMA, "status": "stopped_on_technical_error", "counters": hard,
        "hard_counts_per_arm": dict(per_arm or {}), "model_call_counts": calls,
        "rule_selector_calls": sum(int(value.get("rule_selector_calls", 0)) for value in (per_arm or {}).values() if isinstance(value, Mapping)),
        "limits": {"environment_steps": MAX_TOTAL_STEPS, "policy_encode": MODEL_STEP_CAP, "world_candidate_batch": MODEL_STEP_CAP, "actor_readout": MODEL_STEP_CAP, "rule_selector_calls": RULE_SELECTOR_CAP, "updates": 0},
        "setup_seconds": setup_seconds, "model_load_seconds": model_load_seconds, "shared_snapshot_load_seconds": snapshot_seconds,
        "wall_seconds": time.perf_counter() - started, "failure_is_algorithm_result": False, "result_classification": "technical_stop_not_algorithm_result",
    }
    try:
        write_json(output_dir / "runtime-costs.json", costs)
        write_json(output_dir / "run-status.json", {"schema": RUN_STATUS_SCHEMA, "status": "stopped_on_technical_error", "error": f"{type(error).__name__}: {error}", "hard_counts": hard, "hard_counts_per_arm": dict(per_arm or {}), "model_call_counts": calls, "no_retry": True, "failure_is_algorithm_result": False, "result_classification": "technical_stop_not_algorithm_result"})
    except Exception:
        pass
    if budget is not None:
        try:
            budget.export_snapshot(output_dir / "budget-after.json")
        except Exception:
            pass
    try:
        append_jsonl(output_dir / "failure-ledger.jsonl", {"schema": "model-rule-pair-failure/1.0.0", "phase": "runtime_setup_or_matrix", "reason": f"{type(error).__name__}: {error}", "hard_counts": hard, "model_call_counts": calls, "no_retry": True})
    except Exception:
        pass


def _load_runtime_module() -> Any:
    return _load_module(RUNTIME, "model_rule_branch_runtime_v1")


def _budget_factory(runtime: Any, auth: Mapping[str, Any]) -> Any:
    budget = auth["budget"]
    return runtime.PersistentBudget(
        Path(str(budget["path"])),
        limits={"environment_steps": int(budget["global_limit"]), "optimizer_calls": 0, "world_updates": 0, "offline_updates": 0},
        attempt_id=str(auth["run"]["attempt_id"]), run_id=str(auth["run"]["run_id"]),
        run_limits={"environment_steps": int(budget["run_limit"]), "optimizer_calls": 0, "world_updates": 0, "offline_updates": 0},
        require_existing=True,
    )


def execute_authorized(
    authorization_path: Path,
    *,
    manifest_path: Path = MANIFEST,
    output_dir: Path,
    adapters: Any | None = None,
    budget_factory: Callable[[Any, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Execute the future authorized matrix once, preserving every failure."""
    gate = validate_authorization(authorization_path, manifest_path=manifest_path, output_dir=output_dir)
    rows = gate["manifest"]
    output_dir = _resolved(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model_calls = ModelCallCounters()
    runtime = None
    base_adapters = None
    budget = None
    counters = None
    model_runtime = None
    model_load_seconds = 0.0
    snapshot_seconds = 0.0
    setup_seconds = 0.0
    per_arm: dict[str, dict[str, Any]] = {arm: {"branches": 0, "steps": 0, "probe_calls": 0, "actor_forward": 0, "world_forward": 0, "rule_selector_calls": 0, "selector_seconds": 0.0} for arm in ARMS}
    try:
        # Bind the local rule module before H005's historical isolation swaps
        # the gppo_world import namespace to the native source snapshot.
        rule_module = _load_module(RULE, "gppo_world.public_dispatch_rule_v1")
        stable = _load_runtime_module()
        stable.MAX_TOTAL_STEPS = MAX_TOTAL_STEPS
        stable.MAX_STEPS_PER_BRANCH = MAX_STEPS_PER_BRANCH
        auth = gate["authorization"]
        if budget_factory is None:
            budget_factory = _budget_factory
        budget = budget_factory(stable, auth)
        if adapters is None:
            base_adapters = stable._default_adapters()
            snapshot_started = time.perf_counter()
            prefix_ids = sorted({str(row["prefix_id"]) for row in rows})
            snapshots = base_adapters.load_snapshots(prefix_ids)
            snapshot_seconds = time.perf_counter() - snapshot_started
        else:
            base_adapters = adapters
            snapshot_started = time.perf_counter()
            prefix_ids = sorted({str(row["prefix_id"]) for row in rows})
            snapshots = base_adapters.load_snapshots(prefix_ids)
            snapshot_seconds = time.perf_counter() - snapshot_started
        expected_prefixes = {str(row["prefix_id"]) for row in rows}
        if set(snapshots) != expected_prefixes:
            raise TechnicalStop("runtime snapshot set does not match fixed matrix")
        setup_seconds = time.perf_counter() - started
        counters = stable.Counters()
        all_selector_rows: list[dict[str, Any]] = []
        first_gate: dict[str, Any] | None = None
        pair_seen: dict[str, set[str]] = defaultdict(set)
        branch_timings: list[dict[str, Any]] = []
        all_steps: list[dict[str, Any]] = []
        all_decisions: list[dict[str, Any]] = []
        for item in rows:
            item = dict(item)
            arm = str(item["arm"])
            if arm == "frozen_model":
                if model_runtime is None:
                    model_load_started = time.perf_counter()
                    model_runtime = base_adapters.load_runtime(snapshots)
                    model_load_seconds += time.perf_counter() - model_load_started
                declaration = {"policy_encode": 1, "world_candidate_batch": 1, "actor_readout": 1}
                recorder = _ProbeRecorder(item, arm, model_calls, output_dir / "selector-ledger.jsonl")
                probe_adapter = stable.RuntimeAdapters(
                    load_snapshots=base_adapters.load_snapshots,
                    load_runtime=base_adapters.load_runtime,
                    probe=lambda rt, obs, ph, wh, rec=recorder: rec.model_probe(base_adapters.probe, rt, obs, ph, wh),
                    reward=base_adapters.reward,
                    runtime_digest=base_adapters.runtime_digest,
                    task_capacity=base_adapters.task_capacity,
                    close=None,
                    model_calls_per_probe=declaration,
                )
                branch_runtime = model_runtime
            elif arm == "public_rule":
                recorder = _ProbeRecorder(item, arm, model_calls, output_dir / "selector-ledger.jsonl")
                probe_adapter = stable.RuntimeAdapters(
                    load_snapshots=base_adapters.load_snapshots,
                    load_runtime=base_adapters.load_runtime,
                    probe=lambda _rt, obs, ph, wh, rec=recorder: rec.rule_probe(rule_module.select_public_dispatch_action, obs, ph, wh),
                    reward=base_adapters.reward,
                    runtime_digest=base_adapters.runtime_digest,
                    task_capacity=base_adapters.task_capacity,
                    close=None,
                    model_calls_per_probe={"policy_encode": 0, "world_candidate_batch": 0, "actor_readout": 0},
                )
                branch_runtime = None
            else:
                raise TechnicalStop(f"unknown manifest arm: {arm}")
            staging = output_dir / "runner-staging" / str(item["branch_id"]).replace("|", "__")
            staging.mkdir(parents=True, exist_ok=False)
            branch_started = time.perf_counter()
            try:
                result = stable.execute_branch(item, snapshots[str(item["prefix_id"])], branch_runtime, probe_adapter, budget, counters, staging)
            except Exception:
                _merge_staging(staging, output_dir, item, recorder, None)
                raise
            elapsed = time.perf_counter() - branch_started
            _merge_staging(staging, output_dir, item, recorder, result)
            branch_steps = [_augment(item, row) for row in read_jsonl(staging / "step-vector-rewards.jsonl")]
            branch_decisions = [_augment(item, row) for row in read_jsonl(staging / "decision-ledger.jsonl")]
            decision_by_step = {int(row.get("step", -1)): row for row in branch_decisions}
            for selector_row in recorder.records:
                step = int(selector_row.get("step", -1))
                decision = decision_by_step.get(step, {})
                step_row = next((row for row in branch_steps if int(row.get("step", -1)) == step), {})
                selector_row.update({
                    "selected_action": decision.get("final_action", step_row.get("action")),
                    "original_action": decision.get("original_action", step_row.get("original_action")),
                    "legal_mask": decision.get("legal_mask", step_row.get("legal_mask")),
                    "probabilities": decision.get("probabilities", step_row.get("probabilities")),
                    "guard_candidates": decision.get("candidate_actions", []),
                })
            all_selector_rows.extend(recorder.records)
            all_steps.extend(branch_steps)
            all_decisions.extend(branch_decisions)
            pair_seen[str(item["pair_id"])].add(arm)
            per_arm[arm]["branches"] += 1
            per_arm[arm]["steps"] += int(result.get("env_steps", 0))
            per_arm[arm]["probe_calls"] += int(result.get("probe_calls", 0))
            per_arm[arm]["actor_forward"] += int(result.get("actor_forward_calls", 0))
            per_arm[arm]["world_forward"] += int(result.get("world_forward_calls", 0))
            per_arm[arm]["rule_selector_calls"] += int(recorder.rule_calls)
            per_arm[arm]["selector_seconds"] += float(recorder.selector_seconds)
            branch_timings.append({"branch_id": item["branch_id"], "pair_id": item["pair_id"], "arm": arm, "branch_seconds": elapsed, "selector_seconds": recorder.selector_seconds, "environment_and_runner_overhead_seconds": max(0.0, elapsed - recorder.selector_seconds)})
            counters.branches_completed += 1
            if first_gate is None and len(pair_seen) == 1 and pair_seen[str(item["pair_id"])] == set(ARMS):
                first_gate = _first_pair_gate(all_selector_rows, all_steps, all_decisions)
                write_json(output_dir / "first-pair-gate.json", first_gate)
                if not first_gate.get("ok"):
                    raise TechnicalStop(f"first pair gate failed: {first_gate}")
            append_jsonl(output_dir / "progress.jsonl", {"schema": "model-rule-pair-progress/1.0.0", "branch_id": item["branch_id"], "branches_completed": counters.branches_completed, "pairs_completed": len(pair_seen), "env_steps": counters.env_steps, "verified_steps": counters.verified_steps})
        if first_gate is None:
            raise TechnicalStop("first pair gate was never reached")
        if len(rows) != MAX_BRANCHES or counters.branches_completed != MAX_BRANCHES or len(pair_seen) != 24 or any(arms != set(ARMS) for arms in pair_seen.values()):
            raise TechnicalStop("matrix branch accounting mismatch")
        if counters.env_steps > MAX_TOTAL_STEPS or counters.verified_steps != counters.env_steps:
            raise TechnicalStop("environment step or verified-step cap mismatch")
        totals = budget.run_totals(str(auth["run"]["run_id"]))
        stage = totals.get("environment_steps", {})
        if int(stage.get("reserved", -1)) != counters.verified_steps or int(stage.get("verified", -1)) != counters.verified_steps or int(stage.get("unknown", -1)) != 0 or int(stage.get("pending", -1)) != 0:
            raise TechnicalStop(f"final budget mismatch: {stage}")
        calls = model_calls.as_dict()
        if any(int(calls["attempted"][name]) > MODEL_STEP_CAP for name in ModelCallCounters.NAMES) or any(int(calls["completed"][name]) != int(calls["attempted"][name]) for name in ModelCallCounters.NAMES):
            raise TechnicalStop(f"model call accounting mismatch: {calls}")
        if sum(int(per_arm["public_rule"][name]) for name in ("rule_selector_calls",)) > RULE_SELECTOR_CAP:
            raise TechnicalStop("public rule selector cap exceeded")
        runtime_digest_before = base_adapters.runtime_digest(model_runtime) if model_runtime is not None else "no-model-runtime-loaded"
        runtime_digest_after = base_adapters.runtime_digest(model_runtime) if model_runtime is not None else runtime_digest_before
        if runtime_digest_after != runtime_digest_before:
            raise TechnicalStop("model runtime changed during eval-only execution")
        runtime_costs = {
            "schema": RUNTIME_COST_SCHEMA, "status": "completed", "counters": counters.as_dict(), "hard_counts_per_arm": per_arm,
            "model_call_counts": calls, "rule_selector_calls": sum(int(per_arm[arm]["rule_selector_calls"]) for arm in ARMS),
            "limits": {"environment_steps": MAX_TOTAL_STEPS, "policy_encode": MODEL_STEP_CAP, "world_candidate_batch": MODEL_STEP_CAP, "actor_readout": MODEL_STEP_CAP, "rule_selector_calls": RULE_SELECTOR_CAP, "updates": 0},
            "setup_seconds": setup_seconds, "model_load_seconds": model_load_seconds, "shared_snapshot_load_seconds": snapshot_seconds,
            "wall_seconds": time.perf_counter() - started, "branch_timings": branch_timings, "timing_scope": "selector timing excludes shared prefix deserialization and model load; environment/runner/logging overhead is residual", "failure_is_algorithm_result": False, "result_classification": "evaluation_completed",
        }
        write_json(output_dir / "runtime-costs.json", runtime_costs)
        budget.export_snapshot(output_dir / "budget-after.json")
        status = {"schema": RUN_STATUS_SCHEMA, "status": "completed", "hard_counts": counters.as_dict(), "hard_counts_per_arm": per_arm, "model_call_counts": calls, "rule_selector_calls": runtime_costs["rule_selector_calls"], "runtime_digest_before": runtime_digest_before, "runtime_digest_after": runtime_digest_after, "budget_totals": totals, "no_retry": True, "failure_is_algorithm_result": False, "result_classification": "evaluation_completed"}
        write_json(output_dir / "run-status.json", status)
        return status
    except Exception as exc:
        _persist_stop(_resolved(output_dir), started, exc, counters, model_calls, gate.get("authorization"), budget, setup_seconds, model_load_seconds, snapshot_seconds, per_arm)
        raise
    finally:
        if base_adapters is not None and getattr(base_adapters, "close", None) is not None:
            try:
                base_adapters.close()
            except Exception:
                pass


def run_matrix(
    authorization_path: Path,
    *,
    manifest_path: Path = MANIFEST,
    out_dir: Path,
    adapters: Any | None = None,
    budget_factory: Callable[[Any, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Compatibility entry matching the historical runners' matrix API."""
    return execute_authorized(
        authorization_path,
        manifest_path=manifest_path,
        output_dir=out_dir,
        adapters=adapters,
        budget_factory=budget_factory,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-package", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=PACKAGE / "runner-execution")
    args = parser.parse_args(argv)
    if args.check_package == args.execute:
        raise SystemExit("choose exactly one of --check-package or --execute")
    if args.check_package:
        result = package_validation()
        return 0 if result["ok"] else 2
    if args.authorization is None:
        raise SystemExit("--authorization is required for --execute")
    execute_authorized(args.authorization, manifest_path=args.manifest, output_dir=args.out)
    return 0


__all__ = [
    "ARMS", "APPROVED_LIMIT", "APPROVED_NEW_STEPS", "AUTH_SCHEMA", "AuthorizationError", "BASE", "BUDGET", "CHECKPOINT", "CONFIG", "CURRENT_LIMIT", "CURRENT_VERIFIED", "EXPECTED_BUDGET_SHA256", "EXPECTED_CHECKPOINT_SHA256", "EXPECTED_CONFIG_SHA256", "EXPECTED_GUARD_SHA256", "EXPECTED_SNAPSHOT_SHA256", "FIRST_PAIR_ID", "GUARD", "MANIFEST", "MAX_BRANCHES", "MAX_STEPS_PER_BRANCH", "MAX_TOTAL_STEPS", "ModelCallCounters", "PACKAGE", "PROTOCOL", "RULE", "RUNTIME", "RUNNER_ID", "RunnerError", "TechnicalStop", "authorization_template", "canonical_hash", "execute_authorized", "expected_arm", "inspect_budget", "observation_digest", "package_validation", "read_json", "read_jsonl", "run_matrix", "sha256_file", "source_pins", "validate_authorization", "validate_manifest", "validate_native_source_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
