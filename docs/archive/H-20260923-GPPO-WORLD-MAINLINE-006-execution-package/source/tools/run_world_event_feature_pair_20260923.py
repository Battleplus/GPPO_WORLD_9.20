"""Authorization-gated paired event-feature sensitivity runner.

The static package path reads only JSON, source hashes, and the formal SQLite
ledger in read-only mode.  The dynamic path is deliberately lazy: it imports
the historical H-005 runner only after an explicit authorization has pinned
the package, source identities, expanded budget, run identity, and fresh
output directory.  H-005 owns environment stepping, reward checks, branch
isolation, reservation finalization, and failure retention.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Mapping, Sequence


WORKTREE = Path(__file__).resolve().parents[1]
BASE = WORKTREE / "runs" / "finite-communication-ack-lease-fix-20260920"
PACKAGE = BASE / "world-event-feature-execution-package-20260923"
PROTOCOL = PACKAGE / "protocol.json"
MANIFEST = PACKAGE / "paired-manifest.json"
NATIVE_SOURCE_MANIFEST = PACKAGE / "native-source-manifest.json"
ANALYZER = WORKTREE / "tools" / "analyze_world_event_feature_pair_20260923.py"
AUTHORIZER = WORKTREE / "tools" / "authorize_world_event_feature_pair_20260923.py"
ANALYZER_DEPENDENCIES = {
    "native_analysis": BASE / "ackguard-matrix-registration-20260922" / "analyze_results.py",
    "reconciliation": BASE / "ackguard-matrix-registration-20260922" / "final-analysis-v2" / "reconcile_budget_status.py",
    "public_contract": WORKTREE / "tools" / "analyze_world_increment_preflight_20260923.py",
}
CONFIG = Path(
    r"E:\Z博士\migration-artifacts\preference-weighted-wm-event-cpu-20260917\final\run\training\seed-1101\WD\resolved-config.json"
)
H005_RUNNER = WORKTREE / "tools" / "run_ack_known_task_guard_corrected_20260922.py"
GUARD = WORKTREE / "gppo_world" / "ack_known_task_guard.py"
CHECKPOINT = Path(
    r"E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1"
    r"\training\seed-1101\P_train\last-recovery.pt"
)
SNAPSHOT = Path(
    r"E:\Z博士\migration-artifacts\replan-value-feature-collection-20260919-v1"
    r"\prefix-snapshots.pkl"
)
BUDGET = BASE / "ack-known-task-guard-baseline-v1" / "budget.sqlite3"
RUNTIME_DEPENDENCIES = {
    "h005_runner": H005_RUNNER,
    "guard": GUARD,
    "budget_executor": WORKTREE / "gppo_world" / "budget_executor.py",
    "historical_helper": WORKTREE / "tools" / "run_replan_value_experiment.py",
    "config": CONFIG,
    "analyzer": ANALYZER,
    "authorizer": AUTHORIZER,
    **{f"analyzer_dependency_{name}": path for name, path in ANALYZER_DEPENDENCIES.items()},
}

RUNNER_ID = "world-event-feature-pair-20260923-v1"
AUTH_SCHEMA = "world-event-feature-authorization/1.0.0"
FEATURE_SCHEMA = "world-event-feature/1.0.0"
BRANCH_SCHEMA = "world-event-feature-branch-result/1.0.0"
MAX_BRANCHES = 48
MAX_STEPS_PER_BRANCH = 16
MAX_TOTAL_STEPS = 768
CURRENT_LIMIT = 404
APPROVED_LIMIT = 1067
HISTORICAL_VERIFIED = 299
APPROVED_NEW_STEPS = 768
PREFERENCE = (0.8, 0.2)
EVENT_START = 12
EVENT_END = 17
EXPECTED_H005_RUNNER_SHA256 = "ec1eae6e0f6fff759a391875c2f34878260a85889d97c18c6f991c5f98b0711e"
EXPECTED_GUARD_SHA256 = "2bc314ff9e3d4f78ec871c1a2e739a92669b1f9bd573aabce6a8be5b3e9e808a"
EXPECTED_CHECKPOINT_SHA256 = "bf10d2685a4a3e9da036689f5028b022330e86e922e09c95a7dd0a929df9bb1a"
EXPECTED_SNAPSHOT_SHA256 = "c6826f941141d1ae7594c38a3e3465584ae0954158502f19d2f4d8163a18258e"
EXPECTED_BUDGET_SHA256 = "9ff13f0e18ab8ce56f418f00eac786f08a47e6107a2b8d31f7184b2502c9d420"
EXPECTED_RUNTIME_DEPENDENCY_HASHES = {
    "h005_runner": EXPECTED_H005_RUNNER_SHA256,
    "guard": EXPECTED_GUARD_SHA256,
    "budget_executor": "d9a39251cba7833ce2f92e06a84522a81c930c9eba706b42de62adab8a003551",
    "historical_helper": "a9ccf1711790c472ededd7666c54c28ca34abffc4a0095d35b300f1297afa965",
    "config": "f57f91878d161855cfd58a28db7943d2341ebbfdfd5053f8c2185afbfe6cbc27",
}


class RunnerError(RuntimeError):
    """A package, authorization, or technical stop."""


class AuthorizationError(RunnerError):
    pass


class TechnicalStop(RunnerError):
    pass


class ModelCallCounters:
    """Durable sidecar counts that include calls failing after invocation."""

    LIMITS = {
        "policy_encode": MAX_TOTAL_STEPS,
        "world_candidate_batch": MAX_TOTAL_STEPS,
        "actor_readout": MAX_TOTAL_STEPS + 2,
    }

    def __init__(self) -> None:
        self.attempted = {name: 0 for name in self.LIMITS}
        self.completed = {name: 0 for name in self.LIMITS}

    def start(self, name: str) -> None:
        if name not in self.LIMITS:
            raise ValueError(f"unknown model call: {name}")
        if self.attempted[name] >= self.LIMITS[name]:
            raise TechnicalStop(f"{name} attempt cap reached before invocation")
        self.attempted[name] += 1

    def finish(self, name: str) -> None:
        if name not in self.completed:
            raise ValueError(f"unknown model call: {name}")
        self.completed[name] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": dict(self.attempted),
            "completed": dict(self.completed),
            "limits": dict(self.LIMITS),
            "attempted_total": sum(self.attempted.values()),
            "completed_total": sum(self.completed.values()),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_safe(value: Any, active: set[int] | None = None) -> Any:
    """Convert tensor/array values without importing scientific packages."""
    active = set() if active is None else active
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            return {"cycle": True}
        active.add(identity)
        try:
            return {str(key): json_safe(item, active) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        return [json_safe(item, active) for item in value]
    detach = getattr(value, "detach", None)
    if callable(detach):
        return json_safe(detach().cpu().tolist(), active)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return json_safe(tolist(), active)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item(), active)
        except Exception:
            pass
    return repr(value)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def inspect_budget(path: Path, run_id: str | None = None) -> dict[str, Any]:
    """Inspect the formal ledger through SQLite's read-only URI."""
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
        attempts = [
            {"attempt_id": str(row[0]), "stage": str(row[1]), "run_id": row[5]}
            for row in con.execute("SELECT attempt_id,stage,reserved,verified,unknown,run_id FROM attempts")
        ]
        runs: dict[str, dict[str, int]] = {}
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
        "attempts": attempts,
        "run_totals": runs.get(str(run_id), {}) if run_id is not None else {},
        "all_run_totals": runs,
    }


def _manifest_rows(manifest_path: Path = MANIFEST) -> list[dict[str, Any]]:
    payload = read_json(manifest_path)
    if isinstance(payload, Mapping):
        rows = payload.get("rows")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise RunnerError("paired manifest must be a JSON array or an object with rows")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _canonical_manifest_failure(manifest_path: Path) -> str | None:
    try:
        if _resolved(manifest_path) != MANIFEST.resolve():
            return f"manifest path is not canonical: {_resolved(manifest_path)}"
    except Exception as exc:
        return f"manifest path cannot be resolved: {type(exc).__name__}: {exc}"
    return None


def expected_arm(parent_id: str, repeat: int) -> tuple[str, str]:
    index = int(parent_id.rsplit("-", 1)[1])
    if (index + int(repeat)) % 2 == 0:
        return "normal", "event_features_off"
    return "event_features_off", "normal"


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    if len(rows) != MAX_BRANCHES:
        failures.append(f"manifest row count {len(rows)} != {MAX_BRANCHES}")
    expected_pairs = [
        f"parent-{parent:02d}|W1|seed-1101|prefix-0|repeat-{repeat}"
        for parent in range(8)
        for repeat in range(3)
    ]
    observed_pair_order: list[str] = []
    observed_orders: list[int] = []
    branch_ids = [str(row.get("branch_id")) for row in rows]
    if len(set(branch_ids)) != len(branch_ids):
        failures.append("duplicate branch_id")
    pair_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        observed_orders.append(int(row.get("order", -1)))
        pair_id_value = str(row.get("pair_id"))
        if not observed_pair_order or observed_pair_order[-1] != pair_id_value:
            observed_pair_order.append(pair_id_value)
        pair_groups[str(row.get("pair_id"))].append(row)
        arm = str(row.get("arm"))
        if arm not in {"normal", "event_features_off"}:
            failures.append(f"invalid arm: {arm}")
        if int(row.get("max_steps", -1)) != MAX_STEPS_PER_BRANCH:
            failures.append(f"invalid max_steps: {row.get('branch_id')}")
        if row.get("status") not in {"proposal_not_attempt", "pending_execution_authorization"}:
            failures.append(f"manifest row is already attempted: {row.get('branch_id')}")
        if row.get("condition") != "W1" or int(row.get("model_seed", -1)) != 1101 or int(row.get("prefix_index", -1)) != 0:
            failures.append(f"fixed matrix identity mismatch: {row.get('branch_id')}")
        if row.get("split") != "train":
            failures.append(f"manifest split mismatch: {row.get('branch_id')}")
        expected_parent = str(row.get("parent_id"))
        expected_repeat = int(row.get("repeat", -1))
        expected_pair = f"{expected_parent}|W1|seed-1101|prefix-0|repeat-{expected_repeat}"
        expected_prefix = f"{expected_parent}|W1|seed-1101|prefix-0"
        expected_exogenous = f"replan-feature-v1|{expected_parent}|W1|0|repeat-{expected_repeat}"
        if pair_id_value != expected_pair or str(row.get("prefix_id")) != expected_prefix:
            failures.append(f"canonical pair/prefix identity mismatch: {row.get('branch_id')}")
        if str(row.get("exogenous_key")) != expected_exogenous or str(row.get("historical_exogenous_key")) != expected_exogenous:
            failures.append(f"canonical exogenous identity mismatch: {row.get('branch_id')}")
        if str(row.get("branch_id", "")) != f"{pair_id_value}|{arm}":
            failures.append(f"branch_id arm mismatch: {row.get('branch_id')}")
        if not isinstance(row.get("cohort_task_ids"), list):
            failures.append(f"missing cohort_task_ids: {row.get('branch_id')}")
    for pair_id, group in pair_groups.items():
        if len(group) != 2:
            failures.append(f"pair {pair_id} does not have two arms")
            continue
        arms = {str(row.get("arm")) for row in group}
        if arms != {"normal", "event_features_off"}:
            failures.append(f"pair {pair_id} is not one normal/one event_features_off")
        keys = {str(row.get("exogenous_key")) for row in group}
        if len(keys) != 1:
            failures.append(f"pair {pair_id} exogenous key differs across arms")
        if len({str(row.get("prefix_id")) for row in group}) != 1:
            failures.append(f"pair {pair_id} prefix differs across arms")
        pair_rows = sorted(group, key=lambda row: int(row.get("order", 0)))
        parent = str(pair_rows[0].get("parent_id"))
        repeat = int(pair_rows[0].get("repeat", -1))
        expected = expected_arm(parent, repeat)
        actual = tuple(str(row.get("arm")) for row in pair_rows)
        if actual != expected:
            failures.append(f"pair {pair_id} order {actual} != {expected}")
        expected_control = f"{pair_id}|mode-R"
        if any(str(row.get("control_branch_key")) != expected_control for row in group):
            failures.append(f"pair {pair_id} control_branch_key mismatch")
        if any(row.get("cohort_task_ids") != group[0].get("cohort_task_ids") for row in group):
            failures.append(f"pair {pair_id} cohort differs across arms")
    if observed_orders != list(range(1, MAX_BRANCHES + 1)):
        failures.append("manifest order fields are not exactly 1..48 in file order")
    if observed_pair_order != expected_pairs:
        failures.append("manifest pair order is not canonical parent-00..07/repeat-0..2")
    if set(pair_groups) != set(expected_pairs):
        failures.append("manifest pair set is not exactly parent-00..07/repeat-0..2")
    for parent in range(8):
        parent_id = f"parent-{parent:02d}"
        repeats = sorted(int(row.get("repeat", -1)) for row in rows if str(row.get("parent_id")) == parent_id)
        if repeats != [0, 0, 1, 1, 2, 2]:
            failures.append(f"parent {parent_id} does not have exactly three paired repeats")
    return {
        "ok": not failures,
        "row_count": len(rows),
        "pair_count": len(pair_groups),
        "failures": failures,
        "branch_ids": branch_ids,
        "pair_ids": sorted(pair_groups),
    }


def source_pins() -> dict[str, Any]:
    paths = {
        "runner": WORKTREE / "tools" / "run_world_event_feature_pair_20260923.py",
        "h005_runner": H005_RUNNER,
        "guard": GUARD,
        "checkpoint": CHECKPOINT,
        "prefix_snapshot": SNAPSHOT,
        "budget": BUDGET,
        "protocol": PROTOCOL,
        "manifest": MANIFEST,
        "native_source_manifest": NATIVE_SOURCE_MANIFEST,
        **RUNTIME_DEPENDENCIES,
    }
    result = {}
    for name, path in paths.items():
        resolved = path.resolve()
        result[name] = {
            "path": str(resolved),
            "canonical": path.is_absolute() and resolved == path,
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return result


def validate_native_source_manifest() -> dict[str, Any]:
    """Verify the pinned transitive native source set by bytes and hashes."""
    if not NATIVE_SOURCE_MANIFEST.is_file():
        return {"ok": False, "failures": ["native-source-manifest.json is missing"], "file_count": 0}
    try:
        payload = read_json(NATIVE_SOURCE_MANIFEST)
    except Exception as exc:
        return {"ok": False, "failures": [f"native source manifest unreadable: {type(exc).__name__}: {exc}"], "file_count": 0}
    files = payload.get("files") if isinstance(payload, Mapping) else None
    source_root_raw = str(payload.get("source_root", "")) if isinstance(payload, Mapping) else ""
    source_root = _resolved(source_root_raw) if source_root_raw else Path()
    failures: list[str] = []
    if not isinstance(files, Mapping) or not source_root.is_dir():
        return {"ok": False, "failures": ["native source manifest has no valid source_root/files"], "file_count": 0}
    if source_root_raw != str(source_root):
        failures.append("native source root is not canonical")
    manifest_paths: set[str] = set()
    for relative, identity in sorted(files.items()):
        relative_text = str(relative).replace("/", "\\")
        relative_path = Path(relative_text)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            failures.append(f"native source relative path escapes root: {relative}")
            continue
        canonical_relative = relative_path.as_posix()
        if canonical_relative != str(relative).replace("\\", "/"):
            failures.append(f"native source relative path is not canonical: {relative}")
        path = (source_root / relative_path).resolve()
        try:
            path.relative_to(source_root)
        except ValueError:
            failures.append(f"native source path escapes root: {relative}")
            continue
        manifest_paths.add(canonical_relative)
        if not path.is_file():
            failures.append(f"native source missing: {relative}")
            continue
        if int(identity.get("bytes", -1)) != path.stat().st_size:
            failures.append(f"native source byte mismatch: {relative}")
        if str(identity.get("sha256")) != sha256_file(path):
            failures.append(f"native source hash mismatch: {relative}")
    actual_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if not manifest_paths.issubset(actual_paths):
        failures.append(
            f"native source manifest paths missing: manifest={len(manifest_paths)} actual={len(actual_paths)}"
        )
    return {
        "ok": not failures,
        "manifest_path": str(NATIVE_SOURCE_MANIFEST),
        "manifest_sha256": sha256_file(NATIVE_SOURCE_MANIFEST),
        "source_root": str(source_root),
        "file_count": len(files),
        "actual_file_count": len(actual_paths),
        "failures": failures,
    }


def package_validation() -> dict[str, Any]:
    failures: list[str] = []
    if not PROTOCOL.is_file() or not MANIFEST.is_file():
        failures.append("protocol.json or paired-manifest.json is missing")
        protocol: Mapping[str, Any] = {}
        rows: list[dict[str, Any]] = []
    else:
        protocol = read_json(PROTOCOL)
        try:
            rows = _manifest_rows(MANIFEST)
        except Exception as exc:
            rows = []
            failures.append(f"manifest unreadable: {type(exc).__name__}: {exc}")
    manifest_check = validate_manifest(rows)
    failures.extend(manifest_check["failures"])
    if protocol.get("schema") not in {"world-increment-preflight-protocol/1.0.0", "world-event-feature-execution-protocol/1.0.0"}:
        failures.append("protocol schema mismatch")
    if protocol.get("status") not in {"pending_execution_authorization", "reviewed_development_protocol_proposal_not_execution_authorization"}:
        failures.append("protocol is not a pending execution proposal")
    if int(protocol.get("required_new_work", {}).get("branches", -1)) != MAX_BRANCHES:
        failures.append("protocol branch count mismatch")
    if int(protocol.get("required_new_work", {}).get("new_environment_step_worst_case", -1)) != MAX_TOTAL_STEPS:
        failures.append("protocol step cap mismatch")
    budget_state: dict[str, Any] | None = None
    try:
        budget_state = inspect_budget(BUDGET)
        stage = budget_state["stages"].get("environment_steps", {})
        if int(stage.get("limit", -1)) not in {CURRENT_LIMIT, APPROVED_LIMIT}:
            failures.append("formal budget limit is neither the immutable pre-approval limit nor the approved proposal")
        if int(stage.get("reserved", -1)) != HISTORICAL_VERIFIED or int(stage.get("verified", -1)) != HISTORICAL_VERIFIED:
            failures.append("historical budget reserved/verified baseline changed")
        if int(stage.get("unknown", -1)) != 0 or int(stage.get("pending", -1)) != 0 or budget_state["pending_rows"] != 0:
            failures.append("formal budget has unknown or pending reservations")
        if budget_state["integrity_check"] != "ok":
            failures.append("formal budget integrity check failed")
    except Exception as exc:
        failures.append(f"formal budget read-only inspection failed: {type(exc).__name__}: {exc}")
    pins = source_pins()
    expected = {
        "h005_runner": EXPECTED_H005_RUNNER_SHA256,
        "guard": EXPECTED_GUARD_SHA256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "prefix_snapshot": EXPECTED_SNAPSHOT_SHA256,
        "budget": EXPECTED_BUDGET_SHA256,
    }
    for name, value in expected.items():
        if name == "budget":
            continue
        if not pins[name]["exists"] or pins[name]["sha256"] != value:
            failures.append(f"source identity mismatch: {name}")
    for name, expected_hash in EXPECTED_RUNTIME_DEPENDENCY_HASHES.items():
        identity = pins.get(name, {})
        if not identity.get("canonical") or not identity.get("exists") or identity.get("sha256") != expected_hash:
            failures.append(f"runtime dependency identity mismatch: {name}")
    for name in RUNTIME_DEPENDENCIES:
        identity = pins.get(name, {})
        if not identity.get("canonical") or not identity.get("exists"):
            failures.append(f"runtime dependency missing or noncanonical: {name}")
    native_sources = validate_native_source_manifest()
    if not native_sources["ok"]:
        failures.extend(native_sources["failures"])
    if budget_state is not None:
        current_budget_limit = int(budget_state["stages"].get("environment_steps", {}).get("limit", -1))
        if current_budget_limit == CURRENT_LIMIT and pins["budget"]["sha256"] != EXPECTED_BUDGET_SHA256:
            failures.append("pre-approval budget hash mismatch")
        if current_budget_limit == APPROVED_LIMIT and not pins["budget"]["sha256"]:
            failures.append("approved budget hash is missing")
    return {
        "schema": "world-event-feature-package-check/1.0.0",
        "runner_id": RUNNER_ID,
        "ok": not failures,
        "dynamic_execution_authorized": False,
        "authorization_required": True,
        "failures": failures,
        "manifest": manifest_check,
        "protocol_sha256": sha256_file(PROTOCOL) if PROTOCOL.is_file() else None,
        "manifest_sha256": sha256_file(MANIFEST) if MANIFEST.is_file() else None,
        "source_pins": pins,
        "native_source_manifest": native_sources,
        "budget_readonly": budget_state,
        "hard_counts": {
            "env_step": 0,
            "reset_replay": 0,
            "model_forward": 0,
            "optimizer_updates": 0,
            "world_updates": 0,
            "offline_updates": 0,
            "new_attempt": 0,
        },
    }


def authorization_template(check: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": AUTH_SCHEMA,
        "status": "pending_review",
        "runner_id": RUNNER_ID,
        "review": {"reviewed_by": None, "reviewed_at": None, "accepted_interpretation": False},
        "protocol": {"path": str(PROTOCOL), "sha256": check.get("protocol_sha256")},
        "manifest": {"path": str(MANIFEST), "sha256": check.get("manifest_sha256")},
        "source": {
            "runner_path": str(WORKTREE / "tools" / "run_world_event_feature_pair_20260923.py"),
            "runner_sha256": check.get("source_pins", {}).get("runner", {}).get("sha256"),
            "h005_runner_path": str(H005_RUNNER),
            "h005_runner_sha256": EXPECTED_H005_RUNNER_SHA256,
            "guard_path": str(GUARD),
            "guard_sha256": EXPECTED_GUARD_SHA256,
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "prefix_snapshot_path": str(SNAPSHOT),
            "prefix_snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
            "native_source_manifest_path": str(NATIVE_SOURCE_MANIFEST),
            "native_source_manifest_sha256": check.get("native_source_manifest", {}).get("manifest_sha256"),
            "runtime_dependencies": {
                name: check.get("source_pins", {}).get(name)
                for name in RUNTIME_DEPENDENCIES
            },
        },
        "budget": {
            "path": str(BUDGET),
            "sha256": EXPECTED_BUDGET_SHA256,
            "global_limit": APPROVED_LIMIT,
            "baseline_verified": HISTORICAL_VERIFIED,
            "required_new_steps": APPROVED_NEW_STEPS,
            "run_limit": APPROVED_NEW_STEPS,
            "updates_cap": 0,
        },
        "run": {
            "run_id": "world-event-feature-pair-20260923",
            "attempt_id": "world-event-feature-pair-20260923-attempt-0001",
        },
        "interpretation": "event-feature subvector sensitivity only; not a no-world-model or net world-model benefit claim",
        "no_retry": True,
    }


def _schema_document() -> dict[str, Any]:
    return {
        "schema": "world-event-feature-runner-schema/1.0.0",
        "branch_result": {
            "base": "H-005 ackguard-corrected-branch-result/1.0.0",
            "added": ["arm", "pair_id", "cohort_task_ids", "task_ids", "mode", "feature_ledger_rows"],
        },
        "decision_row": {
            "base": "H-005 ackguard-corrected-decision/1.0.0",
            "added": ["arm", "pair_id", "cohort_task_ids", "task_ids", "mode"],
        },
        "step_row": {
            "base": "H-005 step-vector-rewards/1.0.0",
            "added": ["arm", "pair_id", "cohort_task_ids", "task_ids", "mode"],
        },
        "feature_row": {
            "schema": FEATURE_SCHEMA,
            "required": [
                "branch_id", "pair_id", "arm", "step", "exogenous_key",
                "public_observation_sha256", "policy_hidden_before_sha256", "world_hidden_before_sha256",
                "candidate_features_raw_25x17", "candidate_features_actor_25x17",
                "event_features_before_25x5", "event_features_after_25x5",
                "base_logits", "preference_logits", "candidate_logits", "logits", "probabilities",
                "by_action_hidden_sha256", "next_policy_hidden_sha256", "timing", "model_calls", "model_call_status",
            ],
            "event_slice": "columns 12:17; only actor-facing candidate rows change",
        },
        "cost_limits": {
            "branches": MAX_BRANCHES,
            "environment_steps": MAX_TOTAL_STEPS,
            "policy_encodes": MAX_TOTAL_STEPS,
            "world_candidate_batches": MAX_TOTAL_STEPS,
            "actor_readouts": MAX_TOTAL_STEPS + 2,
            "updates": 0,
        },
        "model_call_accounting": {
            "fields": ["attempted", "completed"],
            "calls": ["policy_encode", "world_candidate_batch", "actor_readout"],
            "technical_stop_is_not_algorithm_result": True,
        },
    }


def check_package(*, write_outputs: bool = True) -> dict[str, Any]:
    result = package_validation()
    if write_outputs:
        template = authorization_template(result)
        write_json(PACKAGE / "runner-package-check.json", result)
        write_json(PACKAGE / "runner-authorization-template.json", template)
        write_json(PACKAGE / "runner-schema.json", _schema_document())
        readme = """# World event-feature pair runner\n\n"""
        readme += "`--check-package` is zero-step and performs only manifest, source-hash, and read-only SQLite checks.\n\n"
        readme += "The package has 48 branches (8 parents x 3 repeats x 2 arms), with at most 16 environment steps per branch. `normal` uses the frozen candidate-conditioned world features; `event_features_off` keeps the same world batch and zeroes only actor-facing columns 12:17. This is event-feature sensitivity, not a total world-model ablation.\n\n"
        readme += "`--execute` rejects the pending template. It requires a separately reviewed authorization with status `authorized`, an exact post-extension same-ledger hash and global limit 1067, a 768-step run cap, fresh run/attempt identities, and all pinned source/package hashes. The formal SQLite ledger is never replaced.\n\n"
        readme += "The dynamic path imports H-005 only after all gates pass and reuses its `execute_branch`, environment isolation, native reward checks, reservation/finalization, and no-retry failure recording. Outputs include `branch-results.jsonl`, `decision-ledger.jsonl`, `step-vector-rewards.jsonl`, `budget-finalization.jsonl`, `feature-ledger.jsonl`, `first-pair-gate.json`, `runtime-costs.json`, and `run-status.json`. `runtime-costs.json` keeps model-call `attempted` and `completed` counts separately; a technical stop is not an algorithm result.\n"
        (PACKAGE / "runner-README.md").write_text(readme, encoding="utf-8")
        hash_targets = [
            "runner-package-check.json", "runner-authorization-template.json", "runner-schema.json", "runner-README.md",
        ]
        hashes = {name: {"bytes": (PACKAGE / name).stat().st_size, "sha256": sha256_file(PACKAGE / name)} for name in hash_targets}
        runner_copy = PACKAGE / "runner-source.py"
        runner_copy.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
        hashes["runner-source.py"] = {
            "path": str(runner_copy.resolve()),
            "bytes": runner_copy.stat().st_size,
            "sha256": sha256_file(runner_copy),
            "matches_worktree_runner": sha256_file(runner_copy) == sha256_file(Path(__file__)),
        }
        write_json(PACKAGE / "runner-hashes.json", hashes)
    return result


def _auth_source_matches(auth: Mapping[str, Any], check: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    source = auth.get("source") if isinstance(auth.get("source"), Mapping) else {}
    current = check.get("source_pins", {})
    pairs = {
        "runner": (source.get("runner_sha256"), current.get("runner", {}).get("sha256")),
        "h005_runner": (source.get("h005_runner_sha256"), EXPECTED_H005_RUNNER_SHA256),
        "guard": (source.get("guard_sha256"), EXPECTED_GUARD_SHA256),
        "checkpoint": (source.get("checkpoint_sha256"), EXPECTED_CHECKPOINT_SHA256),
        "prefix_snapshot": (source.get("prefix_snapshot_sha256"), EXPECTED_SNAPSHOT_SHA256),
        "native_source_manifest": (source.get("native_source_manifest_sha256"), check.get("native_source_manifest", {}).get("manifest_sha256")),
    }
    for name, (supplied, expected) in pairs.items():
        if supplied != expected:
            failures.append(f"authorization source hash mismatch: {name}")
    supplied_runtime = source.get("runtime_dependencies")
    actual_runtime = {
        name: check.get("source_pins", {}).get(name)
        for name in RUNTIME_DEPENDENCIES
    }
    if not isinstance(supplied_runtime, Mapping):
        failures.append("authorization runtime dependency identities are missing")
    elif dict(supplied_runtime) != actual_runtime:
        failures.append("authorization runtime dependency identities mismatch")
    return failures


def validate_authorization(auth_path: Path, *, output_dir: Path, manifest_path: Path = MANIFEST) -> dict[str, Any]:
    if not auth_path.is_file():
        raise AuthorizationError(f"explicit reviewed authorization is required: {auth_path}")
    auth = read_json(auth_path)
    if not isinstance(auth, Mapping) or auth.get("schema") != AUTH_SCHEMA:
        raise AuthorizationError("authorization schema mismatch")
    if auth.get("status") != "authorized":
        raise AuthorizationError("authorization status is not authorized; pending template cannot run")
    canonical_manifest_failure = _canonical_manifest_failure(manifest_path)
    if canonical_manifest_failure is not None:
        raise AuthorizationError(canonical_manifest_failure)
    review = auth.get("review") if isinstance(auth.get("review"), Mapping) else {}
    if not review.get("reviewed_by") or not review.get("reviewed_at") or review.get("accepted_interpretation") is not True:
        raise AuthorizationError("reviewed_by/reviewed_at/accepted_interpretation are required")
    check = package_validation()
    failures = list(check.get("failures", ()))
    if not check.get("ok"):
        failures.append("static package check failed")
    protocol = auth.get("protocol") if isinstance(auth.get("protocol"), Mapping) else {}
    manifest = auth.get("manifest") if isinstance(auth.get("manifest"), Mapping) else {}
    if _resolved(protocol.get("path", "")) != PROTOCOL or protocol.get("sha256") != sha256_file(PROTOCOL):
        failures.append("protocol identity mismatch")
    if _resolved(manifest.get("path", "")) != MANIFEST or manifest.get("sha256") != sha256_file(MANIFEST):
        failures.append("manifest identity mismatch")
    failures.extend(_auth_source_matches(auth, check))
    if _resolved((auth.get("source") or {}).get("native_source_manifest_path", "")) != NATIVE_SOURCE_MANIFEST:
        failures.append("native source manifest path mismatch")
    if not check.get("native_source_manifest", {}).get("ok"):
        failures.append("native source manifest check failed")
    budget_auth = auth.get("budget") if isinstance(auth.get("budget"), Mapping) else {}
    if _resolved(budget_auth.get("path", "")) != BUDGET:
        failures.append("budget path mismatch")
    budget_state = inspect_budget(BUDGET)
    if budget_state["sha256"] != budget_auth.get("sha256"):
        failures.append("budget SQLite hash mismatch")
    if budget_auth.get("sha256") != sha256_file(BUDGET):
        failures.append("budget SQLite changed after authorization was prepared")
    stage = budget_state["stages"].get("environment_steps", {})
    if int(budget_auth.get("global_limit", -1)) != APPROVED_LIMIT or int(stage.get("limit", -1)) != APPROVED_LIMIT:
        failures.append("approved global budget must be exactly 1067")
    if int(budget_auth.get("required_new_steps", -1)) != APPROVED_NEW_STEPS or int(budget_auth.get("run_limit", -1)) != APPROVED_NEW_STEPS:
        failures.append("approved run budget must be exactly 768")
    if int(stage.get("reserved", -1)) != HISTORICAL_VERIFIED or int(stage.get("verified", -1)) != HISTORICAL_VERIFIED or int(stage.get("unknown", -1)) != 0 or int(stage.get("pending", -1)) != 0:
        failures.append("same-ledger baseline is not exactly 299 verified with no pending/unknown")
    if budget_state["pending_rows"] != 0 or budget_state["integrity_check"] != "ok":
        failures.append("budget is not clean and integral")
    run = auth.get("run") if isinstance(auth.get("run"), Mapping) else {}
    run_id = str(run.get("run_id", ""))
    attempt_id = str(run.get("attempt_id", ""))
    if not run_id or not attempt_id or run_id == "pending" or attempt_id == "pending":
        failures.append("fresh run_id and attempt_id are required")
    if any(str(row.get("run_id")) == run_id for row in budget_state.get("attempts", ())):
        failures.append("run_id already exists in formal ledger")
    if any(str(row.get("attempt_id")) == attempt_id for row in budget_state.get("attempts", ())):
        failures.append("attempt_id already exists in formal ledger")
    if output_dir.exists() and any(output_dir.iterdir()):
        failures.append("output directory is not fresh; duplicate or partial execution is refused")
    if failures:
        raise AuthorizationError("execution authorization gate failed: " + "; ".join(failures))
    return {"authorization": dict(auth), "package_check": check, "budget": budget_state}


def _tensor_digest(value: Any) -> str:
    return canonical_hash(json_safe(value))


def _tensor_payload(value: Any) -> Any:
    return json_safe(value)


def _tensor_summary(value: Any) -> dict[str, Any]:
    payload = _tensor_payload(value)
    shape = list(getattr(value, "shape", ()))
    return {"shape": [int(item) for item in shape], "sha256": canonical_hash(payload)}


def _hook(module: Any, bucket: dict[str, Any], name: str) -> Any:
    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        bucket[name] = output.detach().clone() if hasattr(output, "detach") else copy.deepcopy(output)
    return module.register_forward_hook(capture)


def _stable_observation_digest(obs: Mapping[str, Any]) -> str:
    selected = {
        "flat": obs.get("flat"),
        "mask": obs.get("mask"),
        "version": obs.get("version"),
        "time": obs.get("time"),
        "public_entity_ids": obs.get("public_entity_ids"),
        "continuation_actions": list(obs.get("continuation_actions", ())),
        "trigger_flags": obs.get("trigger_flags", {}),
    }
    return canonical_hash(json_safe(selected))


def _stable_hidden_digest(value: Any) -> str | None:
    return None if value is None else canonical_hash(json_safe(value))


class FeatureCapture:
    def __init__(
        self,
        item: Mapping[str, Any],
        arm: str,
        call_counters: ModelCallCounters | None = None,
        *,
        observation_digest_fn: Any = _stable_observation_digest,
        hidden_digest_fn: Any = _stable_hidden_digest,
        timing_totals: dict[str, float] | None = None,
        diagnostic_first_pair: bool = False,
    ) -> None:
        self.item = dict(item)
        self.arm = arm
        self.call_counters = call_counters or ModelCallCounters()
        self.observation_digest_fn = observation_digest_fn
        self.hidden_digest_fn = hidden_digest_fn
        self.timing_totals = timing_totals if timing_totals is not None else defaultdict(float)
        self.diagnostic_first_pair = bool(diagnostic_first_pair)
        self.records: list[dict[str, Any]] = []

    def probe(self, runtime: Mapping[str, Any], obs: Mapping[str, Any], policy_hidden: Any, world_hidden: Any) -> Mapping[str, Any]:
        import torch

        classes = runtime["classes"]
        policy = runtime["policy"]
        world = runtime["world"]
        device = runtime["device"]
        mask_safe, obs_tensor = classes[4], classes[5]
        public_digest = self.observation_digest_fn(obs)
        raw_policy_hidden_digest = self.hidden_digest_fn(policy_hidden)
        raw_world_hidden_digest = self.hidden_digest_fn(world_hidden)
        p_hidden = torch.zeros((1, 1, 128), dtype=torch.float32, device=device) if policy_hidden is None else policy_hidden
        w_hidden = torch.zeros((1, 128), dtype=torch.float32, device=device) if world_hidden is None else world_hidden
        policy_input_digest = self.hidden_digest_fn(p_hidden)
        world_input_digest = self.hidden_digest_fn(w_hidden)
        step = len(self.records) + 1
        started = time.perf_counter()

        def readout(actor_candidates: Any) -> tuple[Any, dict[str, Any], float]:
            captured: dict[str, Any] = {}
            handles = [
                _hook(policy.base.pair_actor, captured, "pair"),
                _hook(policy.base.noop_actor, captured, "noop"),
                _hook(policy.preference_actor, captured, "preference"),
                _hook(policy.candidate_actor, captured, "candidate"),
            ]
            try:
                actor_start = time.perf_counter()
                self.call_counters.start("actor_readout")
                evaluation = policy.evaluate_encoded(features, pair_messages, runtime["preference"], actor_candidates, mask)
                self.call_counters.finish("actor_readout")
                actor_seconds = time.perf_counter() - actor_start
            finally:
                for handle in handles:
                    handle.remove()
            return evaluation, captured, actor_seconds

        def readout_payload(evaluation: Any, captured: Mapping[str, Any], actor_candidates: Any) -> dict[str, Any]:
            pair_logits = captured["pair"].reshape(1, -1)
            noop_logits = captured["noop"].reshape(1, -1)
            base_logits = torch.cat((pair_logits, noop_logits), dim=-1)
            preference_logits = captured["preference"]
            candidate_logits = captured["candidate"].squeeze(-1)
            logits = evaluation["logits"]
            probabilities = evaluation["distribution"].probs
            return {
                "base_logits": _tensor_payload(base_logits[0]),
                "preference_logits": _tensor_payload(preference_logits[0]),
                "candidate_logits": _tensor_payload(candidate_logits[0]),
                "logits": _tensor_payload(logits[0]),
                "probabilities": _tensor_payload(probabilities[0]),
                "candidate_features_actor_25x17": _tensor_payload(actor_candidates[0]),
                "event_features_after_25x5": _tensor_payload(actor_candidates[0, :, EVENT_START:EVENT_END]),
                "original_action": int(torch.argmax(probabilities, dim=-1).item()),
            }

        with torch.no_grad():
            encode_start = time.perf_counter()
            self.call_counters.start("policy_encode")
            features, pair_messages, next_policy_hidden = policy.encode(obs_tensor(obs, device), p_hidden)
            self.call_counters.finish("policy_encode")
            encode_seconds = time.perf_counter() - encode_start
            world_start = time.perf_counter()
            self.call_counters.start("world_candidate_batch")
            raw_candidates, by_action = world.predict_all_candidates(features, w_hidden, obs, use_events=True)
            self.call_counters.finish("world_candidate_batch")
            world_seconds = time.perf_counter() - world_start
            actor_candidates = raw_candidates.clone()
            if self.arm == "event_features_off":
                actor_candidates[:, :, EVENT_START:EVENT_END] = 0
            mask = torch.as_tensor(mask_safe(obs["mask"]), dtype=torch.bool, device=device)[None, :]
            evaluation, captured, actor_seconds = readout(actor_candidates)
            main_payload = readout_payload(evaluation, captured, actor_candidates)
            extra_payload: dict[str, Any] | None = None
            extra_actor_seconds = 0.0
            extra_arm: str | None = None
            if self.diagnostic_first_pair and step == 1:
                extra_arm = "event_features_off" if self.arm == "normal" else "normal"
                extra_candidates = raw_candidates.clone()
                if extra_arm == "event_features_off":
                    extra_candidates[:, :, EVENT_START:EVENT_END] = 0
                extra_evaluation, extra_captured, extra_actor_seconds = readout(extra_candidates)
                extra_payload = readout_payload(extra_evaluation, extra_captured, extra_candidates)
        self.timing_totals["policy_encode_seconds"] += encode_seconds
        self.timing_totals["world_candidate_batch_seconds"] += world_seconds
        self.timing_totals["actor_readout_seconds"] += actor_seconds + extra_actor_seconds
        self.timing_totals["probe_seconds"] += time.perf_counter() - started
        by_action_hidden = {
            str(action): self.hidden_digest_fn(record.get("hidden"))
            for action, record in sorted(by_action.items())
            if isinstance(record, Mapping) and "hidden" in record
        }
        by_action_summary = {
            str(action): {
                key: _tensor_summary(value)
                for key, value in sorted(record.items())
                if hasattr(value, "shape") or hasattr(value, "detach")
            }
            for action, record in sorted(by_action.items())
            if isinstance(record, Mapping)
        }
        raw_payload = _tensor_payload(raw_candidates[0])
        actor_payload = _tensor_payload(actor_candidates[0])
        diagnostic = {
            "enabled": bool(extra_payload is not None),
            "extra_arm": extra_arm,
            "immutable_inputs": {
                "public_observation_sha256": public_digest,
                "policy_hidden_before_sha256": raw_policy_hidden_digest,
                "policy_hidden_input_sha256": policy_input_digest,
                "world_hidden_before_sha256": raw_world_hidden_digest,
                "world_hidden_input_sha256": world_input_digest,
                "raw_candidate_features_sha256": _tensor_digest(raw_candidates),
                "by_action_hidden_sha256": by_action_hidden,
                "next_policy_hidden_sha256": self.hidden_digest_fn(next_policy_hidden),
            },
            "raw_candidate_features_25x17": raw_payload,
            "by_action_output_summary": by_action_summary,
            "next_policy_hidden_sha256": self.hidden_digest_fn(next_policy_hidden),
            "extra_actor_readout": extra_payload,
            "extra_actor_readout_seconds": extra_actor_seconds,
            "extra_model_call_status": {"attempted": int(extra_payload is not None), "completed": int(extra_payload is not None)},
        }
        record = {
            "schema": FEATURE_SCHEMA,
            "branch_id": self.item["branch_id"],
            "pair_id": self.item["pair_id"],
            "arm": self.arm,
            "parent_id": self.item["parent_id"],
            "repeat": int(self.item["repeat"]),
            "condition": self.item["condition"],
            "model_seed": int(self.item["model_seed"]),
            "prefix_id": self.item["prefix_id"],
            "exogenous_key": self.item["exogenous_key"],
            "step": step,
            "public_observation_sha256": public_digest,
            "policy_hidden_before_sha256": raw_policy_hidden_digest,
            "policy_hidden_input_sha256": policy_input_digest,
            "world_hidden_before_sha256": raw_world_hidden_digest,
            "world_hidden_input_sha256": world_input_digest,
            "raw_candidate_features_sha256": _tensor_digest(raw_candidates),
            "actor_candidate_features_sha256": _tensor_digest(actor_candidates),
            "candidate_features_raw_25x17": raw_payload,
            "candidate_features_actor_25x17": actor_payload,
            "event_features_before_25x5": _tensor_payload(raw_candidates[0, :, EVENT_START:EVENT_END]),
            "event_features_after_25x5": _tensor_payload(actor_candidates[0, :, EVENT_START:EVENT_END]),
            "base_logits": main_payload["base_logits"],
            "preference_logits": main_payload["preference_logits"],
            "candidate_logits": main_payload["candidate_logits"],
            "logits": main_payload["logits"],
            "probabilities": _tensor_payload(probabilities[0]),
            "legal_mask": [int(value) for value in mask[0].detach().cpu().tolist()],
            "original_action": int(torch.argmax(probabilities, dim=-1).item()),
            "by_action_hidden_sha256": by_action_hidden,
            "by_action_output_summary": by_action_summary,
            "next_policy_hidden_sha256": self.hidden_digest_fn(next_policy_hidden),
            "initial_env_state_sha256": self.item.get("initial_env_state_sha256"),
            "initial_observation_sha256": self.item.get("initial_observation_sha256"),
            "initial_policy_hidden_sha256": self.item.get("initial_policy_hidden_sha256"),
            "initial_world_hidden_sha256": self.item.get("initial_world_hidden_sha256"),
            "source_snapshot_digest_before": self.item.get("source_snapshot_digest_before"),
            "timing": {
                "policy_encode_seconds": encode_seconds,
                "world_candidate_batch_seconds": world_seconds,
                "actor_readout_seconds": actor_seconds,
                "extra_actor_readout_seconds": extra_actor_seconds,
                "probe_seconds": time.perf_counter() - started,
            },
            "model_calls": {"policy_encode": 1, "world_candidate_batch": 1, "actor_readout": 1},
            "model_call_status": {
                name: {"attempted": 1, "completed": 1}
                for name in ModelCallCounters.LIMITS
            },
            "pair_diagnostic": diagnostic,
            "intervention": {
                "api_contract": "native predict_all_candidates(use_events=False) actor projection",
                "event_slice": [EVENT_START, EVENT_END],
                "world_batch_reused": True,
                "by_action_hidden_unmodified": True,
            },
        }
        self.records.append(record)
        return {
            "probabilities": _tensor_payload(probabilities[0]),
            "original_action": record["original_action"],
            "by_action": by_action,
            "next_policy_hidden": next_policy_hidden,
        }


def compare_first_pair(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[str(record["pair_id"])][str(record["arm"])] = record
    if not grouped:
        return {"ok": False, "reason": "no feature rows"}
    pair_id = sorted(grouped)[0]
    pair = grouped[pair_id]
    if set(pair) != {"normal", "event_features_off"}:
        return {"ok": False, "pair_id": pair_id, "reason": "first pair is not two distinct arms"}
    normal = pair["normal"]
    off = pair["event_features_off"]
    checks = {
        "same_public_observation": normal.get("public_observation_sha256") == off.get("public_observation_sha256"),
        "same_policy_hidden": normal.get("policy_hidden_before_sha256") == off.get("policy_hidden_before_sha256"),
        "same_world_hidden": normal.get("world_hidden_before_sha256") == off.get("world_hidden_before_sha256"),
        "same_raw_candidate_features": normal.get("candidate_features_raw_25x17") == off.get("candidate_features_raw_25x17"),
        "same_by_action_hidden": normal.get("by_action_hidden_sha256") == off.get("by_action_hidden_sha256"),
        "same_next_policy_hidden": normal.get("next_policy_hidden_sha256") == off.get("next_policy_hidden_sha256"),
        "same_candidate_columns_0_12": [row[:EVENT_START] for row in normal.get("candidate_features_actor_25x17", ())] == [row[:EVENT_START] for row in off.get("candidate_features_actor_25x17", ())],
        "normal_event_projection": normal.get("event_features_after_25x5") == normal.get("event_features_before_25x5"),
        "off_event_zero": all(abs(float(value)) == 0.0 for row in off.get("event_features_after_25x5", ()) for value in row),
    }
    return {"schema": "world-event-feature-first-pair-gate/1.0.0", "ok": all(checks.values()), "pair_id": pair_id, "checks": checks}


def required_feature_fields(row: Mapping[str, Any]) -> list[str]:
    return [
        field for field in _schema_document()["feature_row"]["required"]
        if field not in row
    ]


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


def merge_staging(staging: Path, out_dir: Path, item: Mapping[str, Any], capture: FeatureCapture, result: Mapping[str, Any] | None) -> int:
    count = 0
    decision_rows = read_jsonl(staging / "decision-ledger.jsonl")
    by_step = {int(row.get("step", -1)): row for row in decision_rows}
    for feature in capture.records:
        decision = by_step.get(int(feature["step"]))
        if decision is not None:
            feature = dict(feature)
            feature.update({
                "original_action": decision.get("original_action"),
                "selected_action": decision.get("final_action"),
                "guard_candidates": decision.get("candidate_actions"),
                "guard_excluded_actions": decision.get("excluded_actions"),
                "selected_by_action_output": feature.get("by_action_output_summary", {}).get(str(decision.get("final_action"))),
            })
        append_jsonl(out_dir / "feature-ledger.jsonl", feature)
        count += 1
    files = ("decision-ledger.jsonl", "step-vector-rewards.jsonl", "budget-finalization.jsonl", "failure-ledger.jsonl")
    for filename in files:
        for row in read_jsonl(staging / filename):
            append_jsonl(out_dir / filename, _augment(item, row))
    if result is not None:
        branch = _augment(item, result)
        branch["schema"] = BRANCH_SCHEMA
        branch["feature_ledger_rows"] = len(capture.records)
        append_jsonl(out_dir / "branch-results.jsonl", branch)
    return count


def _zero_hard_counts() -> dict[str, int]:
    return {
        "probe_calls": 0,
        "env_step_calls": 0,
        "env_steps": 0,
        "successful_env_steps": 0,
        "verified_steps": 0,
        "model_forward": 0,
        "actor_forward": 0,
        "world_forward": 0,
        "optimizer_updates": 0,
        "world_updates": 0,
        "offline_updates": 0,
        "branches_completed": 0,
    }


def _persist_runtime_stop(
    output_dir: Path,
    *,
    started: float,
    error: BaseException,
    counters: Any | None,
    call_counters: ModelCallCounters,
    budget: Any | None,
    auth: Mapping[str, Any] | None,
    runtime_before: str | None,
) -> None:
    """Retain resource attempts and classify a technical stop as non-result."""
    hard_counts = counters.as_dict() if counters is not None else _zero_hard_counts()
    call_counts = call_counters.as_dict()
    runtime_costs = {
        "schema": "world-event-feature-runtime-cost/1.0.0",
        "status": "stopped_on_technical_error",
        "counters": hard_counts,
        "model_call_counts": call_counts,
        "limits": {
            "environment_steps": MAX_TOTAL_STEPS,
            "policy_encodes": MAX_TOTAL_STEPS,
            "world_candidate_batches": MAX_TOTAL_STEPS,
            "actor_readouts": MAX_TOTAL_STEPS + 2,
            "updates": 0,
        },
        "wall_seconds": time.perf_counter() - started,
        "failure_is_algorithm_result": False,
        "result_classification": "technical_stop_not_algorithm_result",
    }
    try:
        write_json(output_dir / "runtime-costs.json", runtime_costs)
    except Exception:
        pass
    if budget is not None:
        try:
            budget.export_snapshot(output_dir / "budget-after.json")
        except Exception:
            pass
    status: dict[str, Any] = {
        "schema": "world-event-feature-run-status/1.0.0",
        "status": "stopped_on_technical_error",
        "error": f"{type(error).__name__}: {error}",
        "hard_counts": hard_counts,
        "model_call_counts": call_counts,
        "runtime_digest_before": runtime_before,
        "no_retry": True,
        "failure_is_algorithm_result": False,
        "result_classification": "technical_stop_not_algorithm_result",
    }
    if auth is not None and budget is not None:
        try:
            status["budget_totals"] = budget.run_totals(str(auth["run"]["run_id"]))
        except Exception as exc:
            status["budget_totals_error"] = f"{type(exc).__name__}: {exc}"
    try:
        write_json(output_dir / "run-status.json", status)
    except Exception:
        pass


def _load_h005_runner_from_path() -> Any:
    """Load the pinned runner by canonical path when launched outside the repo root."""
    module_name = "world_event_feature_h005_runner_20260923"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    root_text = str(WORKTREE)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location(module_name, H005_RUNNER)
    if spec is None or spec.loader is None:
        raise AuthorizationError(f"cannot load pinned H-005 runner: {H005_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def execute_authorized(auth_path: Path, *, manifest_path: Path = MANIFEST, output_dir: Path) -> dict[str, Any]:
    gate = validate_authorization(auth_path, output_dir=output_dir, manifest_path=manifest_path)
    rows = _manifest_rows(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    call_counters = ModelCallCounters()
    write_json(output_dir / "run-status.json", {
        "schema": "world-event-feature-run-status/1.0.0",
        "status": "runtime_objects_allowed",
        "hard_counts": _zero_hard_counts(),
        "model_call_counts": call_counters.as_dict(),
        "no_retry": True,
    })
    stable = None
    base_adapters = None
    budget = None
    auth: Mapping[str, Any] | None = None
    counters: Any | None = None
    runtime_before: str | None = None
    try:
        stable = _load_h005_runner_from_path()
        stable.MAX_TOTAL_STEPS = MAX_TOTAL_STEPS
        stable.MAX_STEPS_PER_BRANCH = MAX_STEPS_PER_BRANCH
        auth = gate["authorization"]
        budget = stable.PersistentBudget(
            BUDGET,
            limits={"environment_steps": APPROVED_LIMIT, "optimizer_calls": 0, "world_updates": 0, "offline_updates": 0},
            attempt_id=str(auth["run"]["attempt_id"]), run_id=str(auth["run"]["run_id"]),
            run_limits={"environment_steps": APPROVED_NEW_STEPS, "optimizer_calls": 0, "world_updates": 0, "offline_updates": 0},
            require_existing=True,
        )
        base_adapters = stable._default_adapters()
        prefix_ids = sorted({str(row["prefix_id"]) for row in rows})
        snapshots = base_adapters.load_snapshots(prefix_ids)
        runtime = base_adapters.load_runtime(snapshots)
        runtime_before = base_adapters.runtime_digest(runtime)
        counters = stable.Counters()
        all_features: list[dict[str, Any]] = []
        pair_first_seen: dict[str, set[str]] = defaultdict(set)
        first_gate: dict[str, Any] | None = None
        completed_branches = 0
        for row in rows:
            item = dict(row)
            item["historical_exogenous_key"] = item["exogenous_key"]
            item["control_branch_key"] = item["control_branch_key"]
            capture = FeatureCapture(item, str(item["arm"]), call_counters)
            probe_adapter = stable.RuntimeAdapters(
                load_snapshots=base_adapters.load_snapshots,
                load_runtime=base_adapters.load_runtime,
                probe=capture.probe,
                reward=base_adapters.reward,
                runtime_digest=base_adapters.runtime_digest,
                task_capacity=base_adapters.task_capacity,
                close=None,
            )
            staging = output_dir / "runner-staging" / str(item["branch_id"]).replace("|", "__")
            staging.mkdir(parents=True, exist_ok=False)
            try:
                result = stable.execute_branch(item, snapshots[item["prefix_id"]], runtime, probe_adapter, budget, counters, staging)
            except Exception:
                merge_staging(staging, output_dir, item, capture, None)
                raise
            merge_staging(staging, output_dir, item, capture, result)
            all_features.extend(capture.records)
            pair_first_seen[str(item["pair_id"])].add(str(item["arm"]))
            completed_branches += 1
            counters.branches_completed = completed_branches
            if first_gate is None and len(pair_first_seen) == 1 and pair_first_seen[str(item["pair_id"])] == {"normal", "event_features_off"}:
                first_gate = compare_first_pair(all_features)
                write_json(output_dir / "first-pair-gate.json", first_gate)
                if not first_gate.get("ok"):
                    raise TechnicalStop(f"first pair same-input gate failed: {first_gate}")
            append_jsonl(output_dir / "progress.jsonl", {"schema": "world-event-feature-progress/1.0.0", "branch_id": item["branch_id"], "branches_completed": completed_branches, "pairs_completed": len(pair_first_seen), "env_steps": counters.env_steps, "verified_steps": counters.verified_steps})
        runtime_after = base_adapters.runtime_digest(runtime)
        totals = budget.run_totals(str(auth["run"]["run_id"]))
        stage = totals.get("environment_steps", {})
        if (
            len(rows) != MAX_BRANCHES
            or completed_branches != MAX_BRANCHES
            or len(pair_first_seen) != 24
            or any(arms != {"normal", "event_features_off"} for arms in pair_first_seen.values())
            or first_gate is None
            or not first_gate.get("ok")
            or counters.env_steps > MAX_TOTAL_STEPS
            or runtime_after != runtime_before
        ):
            raise TechnicalStop("matrix, cost, or frozen runtime digest mismatch")
        if int(stage.get("reserved", -1)) != counters.verified_steps or int(stage.get("verified", -1)) != counters.verified_steps or int(stage.get("unknown", -1)) != 0 or int(stage.get("pending", -1)) != 0:
            raise TechnicalStop(f"final run budget mismatch: {stage}")
        model_call_counts = call_counters.as_dict()
        if any(value > limit for value, limit in zip(model_call_counts["attempted"].values(), model_call_counts["limits"].values())) or any(model_call_counts["completed"][name] > model_call_counts["attempted"][name] for name in model_call_counts["attempted"]):
            raise TechnicalStop(f"model call accounting mismatch: {model_call_counts}")
        write_json(output_dir / "runtime-costs.json", {"schema": "world-event-feature-runtime-cost/1.0.0", "status": "completed", "counters": counters.as_dict(), "model_call_counts": model_call_counts, "limits": {"environment_steps": MAX_TOTAL_STEPS, "policy_encodes": MAX_TOTAL_STEPS, "world_candidate_batches": MAX_TOTAL_STEPS, "actor_readouts": MAX_TOTAL_STEPS + 2, "updates": 0}, "wall_seconds": time.perf_counter() - started, "failure_is_algorithm_result": False, "result_classification": "evaluation_completed"})
        budget.export_snapshot(output_dir / "budget-after.json")
        status = {"schema": "world-event-feature-run-status/1.0.0", "status": "completed", "hard_counts": counters.as_dict(), "model_call_counts": model_call_counts, "runtime_digest_before": runtime_before, "runtime_digest_after": runtime_after, "budget_totals": totals, "no_retry": True, "failure_is_algorithm_result": False, "result_classification": "evaluation_completed"}
        write_json(output_dir / "run-status.json", status)
        return status
    except Exception as exc:
        _persist_runtime_stop(output_dir, started=started, error=exc, counters=counters, call_counters=call_counters, budget=budget, auth=auth, runtime_before=runtime_before)
        raise
    finally:
        if base_adapters is not None and base_adapters.close is not None:
            base_adapters.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-package", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=PACKAGE / "runner-execution")
    args = parser.parse_args(argv)
    if args.check_package == args.execute:
        raise SystemExit("choose exactly one of --check-package or --execute")
    if args.check_package:
        result = check_package(write_outputs=True)
        return 0 if result["ok"] else 2
    if args.authorization is None:
        raise SystemExit("--authorization is required for --execute")
    execute_authorized(args.authorization, manifest_path=args.manifest, output_dir=args.out)
    return 0


__all__ = [
    "APPROVED_LIMIT", "APPROVED_NEW_STEPS", "AuthorizationError", "BUDGET", "EXPECTED_BUDGET_SHA256",
    "MAX_BRANCHES", "MAX_STEPS_PER_BRANCH", "MAX_TOTAL_STEPS", "MANIFEST", "PACKAGE", "RunnerError",
    "TechnicalStop", "canonical_hash", "check_package", "compare_first_pair", "expected_arm",
    "inspect_budget", "json_safe", "ModelCallCounters", "package_validation", "required_feature_fields",
    "sha256_file", "validate_authorization", "validate_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
