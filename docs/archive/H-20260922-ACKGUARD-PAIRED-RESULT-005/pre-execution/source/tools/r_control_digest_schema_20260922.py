"""Read-only public-observation digest schema bridge for the R-control audit.

The historical collection digest and the pilot observer digest are different
canonical JSON projections.  This module keeps that distinction explicit and
only accepts a bridge when the saved observation and branch identity are both
available.  It has no environment, model, budget, or replay dependencies.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


PILOT_SCHEMA = "pilot-observer/flat-mask-version-time/1.0.0"
HISTORICAL_SCHEMA = "replan-value-public/flat-mask-version-ids-continuation-trigger/1.0.0"
CURRENT_SCHEMA = "ackguard-observation/flat-mask-version-time-ids-continuation-trigger/1.0.0"
BRIDGE_SCHEMA = "r-control-digest-schema-bridge/1.0.0"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def canonical_json(value: Any) -> str:
    """Return the byte representation shared by the historical hash helpers."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _flat(obs: Mapping[str, Any]) -> list[Any]:
    if "flat" not in obs:
        raise KeyError("flat")
    return np.asarray(obs["flat"], dtype=np.float32).tolist()


def _mask(obs: Mapping[str, Any]) -> list[bool]:
    if "mask" not in obs:
        raise KeyError("mask")
    return np.asarray(obs["mask"], dtype=np.bool_).tolist()


def _version(obs: Mapping[str, Any]) -> int:
    if "version" not in obs:
        raise KeyError("version")
    return int(obs["version"])


def _time(obs: Mapping[str, Any]) -> float:
    if "time" not in obs:
        raise KeyError("time")
    return float(obs["time"])


def pilot_projection(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Projection used by ``pilot_vector_recorder.observer_projection``."""
    return {"flat": _flat(obs), "mask": _mask(obs), "version": _version(obs), "time": _time(obs)}


def historical_projection(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Projection used by the original collection ``public_digest``."""
    if "public_entity_ids" not in obs:
        raise KeyError("public_entity_ids")
    if "continuation_actions" not in obs:
        raise KeyError("continuation_actions")
    if "trigger_flags" not in obs:
        raise KeyError("trigger_flags")
    return {
        "flat": _flat(obs),
        "mask": _mask(obs),
        "version": _version(obs),
        "public_entity_ids": obs["public_entity_ids"],
        "continuation_actions": list(obs["continuation_actions"]),
        "trigger_flags": obs["trigger_flags"],
    }


def current_projection(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Projection used by the corrected runner's current observation digest."""
    return {**historical_projection(obs), "time": _time(obs)}


def digest_pair(obs: Mapping[str, Any]) -> dict[str, Any]:
    pilot = pilot_projection(obs)
    historical = historical_projection(obs)
    current = current_projection(obs)
    return {
        "schema": BRIDGE_SCHEMA,
        "pilot": {"schema": PILOT_SCHEMA, "projection": pilot, "sha256": canonical_digest(pilot)},
        "historical": {"schema": HISTORICAL_SCHEMA, "projection": historical, "sha256": canonical_digest(historical)},
        "current": {"schema": CURRENT_SCHEMA, "projection": current, "sha256": canonical_digest(current)},
    }


def _identity_mismatches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    required = ("prefix_id", "parent_id", "condition", "model_seed", "repeat", "mode", "step", "exogenous_key")
    mismatches = []
    for field in required:
        if field not in actual or field not in expected:
            mismatches.append(f"missing_identity:{field}")
        elif actual[field] != expected[field]:
            mismatches.append(f"identity:{field}")
    return mismatches


def evaluate_bridge(
    obs: Mapping[str, Any] | None,
    *,
    label_identity: Mapping[str, Any] | None,
    expected_identity: Mapping[str, Any] | None,
    label_digest: str | None,
    prefix_digest: str | None,
    prefix_time: float | None,
    label_time_before: float | None,
    label_time_after: float | None,
) -> dict[str, Any]:
    """Check a schema bridge without trusting digest strings alone.

    The bridge is accepted only when one saved observation recomputes both
    known digests and the branch identity/time metadata agrees.  A missing
    observation or identity is therefore ``insufficient_evidence``.
    """
    result: dict[str, Any] = {
        "schema": BRIDGE_SCHEMA,
        "status": "insufficient_evidence",
        "schema_compatible": False,
        "reasons": [],
    }
    if not isinstance(obs, Mapping):
        result["reasons"].append("missing_saved_observation")
        return result
    if not isinstance(label_identity, Mapping) or not isinstance(expected_identity, Mapping):
        result["reasons"].append("missing_identity_evidence")
        return result
    result["identity"] = {"actual": dict(label_identity), "expected": dict(expected_identity)}
    result["recomputed"] = None
    try:
        recomputed = digest_pair(obs)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reasons"].append(f"observation_projection_failed:{type(exc).__name__}:{exc}")
        return result
    result["recomputed"] = {
        "pilot_sha256": recomputed["pilot"]["sha256"],
        "historical_sha256": recomputed["historical"]["sha256"],
        "current_sha256": recomputed["current"]["sha256"],
    }
    result["observed"] = {"label_digest": label_digest, "prefix_digest": prefix_digest}
    result["identity_mismatches"] = _identity_mismatches(label_identity, expected_identity)
    if result["identity_mismatches"]:
        result["reasons"].extend(result["identity_mismatches"])
    if label_digest != recomputed["pilot"]["sha256"]:
        result["reasons"].append("label_digest_not_pilot_schema")
    if prefix_digest != recomputed["historical"]["sha256"]:
        result["reasons"].append("prefix_digest_not_historical_schema")
    try:
        observation_time = _time(obs)
    except (KeyError, TypeError, ValueError) as exc:
        result["reasons"].append(f"missing_observation_time:{type(exc).__name__}")
        return result
    result["times"] = {
        "observation": observation_time,
        "prefix_time": prefix_time,
        "label_time_before": label_time_before,
        "label_time_after": label_time_after,
    }
    for field, value in (("prefix_time", prefix_time), ("label_time_before", label_time_before)):
        if value is None:
            result["reasons"].append(f"missing_time:{field}")
        elif observation_time != float(value):
            result["reasons"].append(f"time:{field}")
    if label_time_after is None:
        result["reasons"].append("missing_time:label_time_after")
    elif float(label_time_after) <= observation_time:
        result["reasons"].append("time:label_time_after_not_after_observation")
    result["status"] = "schema_compatible" if not result["reasons"] else "insufficient_evidence"
    result["schema_compatible"] = result["status"] == "schema_compatible"
    return result


__all__ = [
    "BRIDGE_SCHEMA",
    "CURRENT_SCHEMA",
    "HISTORICAL_SCHEMA",
    "PILOT_SCHEMA",
    "canonical_digest",
    "canonical_json",
    "current_projection",
    "digest_pair",
    "evaluate_bridge",
    "historical_projection",
    "pilot_projection",
]
