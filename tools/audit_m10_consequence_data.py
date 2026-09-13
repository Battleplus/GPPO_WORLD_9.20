"""Audit generated M-10 consequence data without training or model selection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.consequence_data import audit_consequence_manifest, load_consequence_jsonl  # noqa: E402
from gppo_world.dataset import sha256_file  # noqa: E402


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit_branch_ledger(data: Path, manifest: dict, split: str) -> dict:
    spec = manifest.get("branch_ledgers", {}).get(split, {})
    errors: list[str] = []
    path = data / str(spec.get("path", ""))
    if not path.is_file():
        return {"passed": False, "errors": [f"missing branch ledger: {path}"]}
    if sha256_file(path) != spec.get("sha256"):
        errors.append("branch ledger sha256 mismatch")
    records_path = data / manifest["files"][split]["path"]
    with records_path.open("r", encoding="utf-8") as stream:
        records = {
            (row["parent_episode_id"], row["prefix_id"], int(row["target"]["action"])): row
            for row in (json.loads(line) for line in stream if line.strip())
        }
    ledgers = []
    with path.open("r", encoding="utf-8") as stream:
        ledgers = [json.loads(line) for line in stream if line.strip()]
    if len(ledgers) != int(spec.get("records", -1)) or len(ledgers) != len(records):
        errors.append("branch ledger record count mismatch")
    prefix_evidence: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    identities = set()
    for item in ledgers:
        identity = (item["parent_episode_id"], item["prefix_id"], int(item["action"]))
        if identity in identities:
            errors.append(f"duplicate ledger identity: {identity}")
            continue
        identities.add(identity)
        record = records.get(identity)
        if record is None:
            errors.append(f"ledger has no matching label record: {identity}")
            continue
        prefix_steps = int(item["prefix_steps"])
        prefix_trace = item["trace"][:prefix_steps]
        prefix_hash = digest(prefix_trace)
        branch_hash = digest(item["trace"])
        if prefix_hash != item.get("prefix_trace_sha256") or prefix_hash != record["label_provenance"].get("prefix_trace_sha256"):
            errors.append(f"prefix trace hash mismatch: {identity}")
        if branch_hash != record["label_provenance"].get("branch_trace_sha256"):
            errors.append(f"branch trace hash mismatch: {identity}")
        if item.get("exogenous_key") != record["target"].get("exogenous_key"):
            errors.append(f"exogenous key mismatch: {identity}")
        evidence = prefix_evidence[identity[:2]]
        evidence["exogenous_keys"].add(str(item.get("exogenous_key")))
        evidence["prefix_trace_hashes"].add(prefix_hash)
        evidence["prefix_actions"].add(json.dumps(item.get("prefix_actions"), separators=(",", ":")))
    inconsistent = {
        f"{parent}/{prefix}": {name: sorted(values) for name, values in evidence.items() if len(values) != 1}
        for (parent, prefix), evidence in prefix_evidence.items()
        if any(len(values) != 1 for values in evidence.values())
    }
    if inconsistent:
        errors.append("candidate branches do not share an identical realized prefix")
    return {
        "passed": not errors,
        "errors": errors,
        "records": len(ledgers),
        "prefixes": len(prefix_evidence),
        "shared_realized_prefixes": len(prefix_evidence) - len(inconsistent),
        "inconsistent_prefixes": inconsistent,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test", "ood"), default=("train", "validation", "test", "ood"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = args.data.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = audit_consequence_manifest(manifest, data)
    report = {"manifest_audit": audit, "splits": {}, "branch_ledgers": {}}
    for split in args.splits:
        path = data / manifest["files"][split]["path"]
        examples = load_consequence_jsonl(path, expected_horizon_steps=int(manifest["prediction_horizon_steps"]), strict_identity=True)
        prefixes = {(example.parent_episode_id, example.prefix_id) for example in examples}
        candidate_counts = Counter((example.parent_episode_id, example.prefix_id) for example in examples)
        values = {
            name: [float(getattr(example.target, name)) for example in examples]
            for name in ("travel_time", "service_progress", "energy_delta", "deadline_risk")
        }
        report["splits"][split] = {
            "records": len(examples),
            "parent_episode_count": len({example.parent_episode_id for example in examples}),
            "prefix_count": len(prefixes),
            "candidate_branch_count": len(examples),
            "candidate_counts_per_prefix": sorted(Counter(candidate_counts.values()).items()),
            "label_masks": {name: int(sum(bool(example.masks[name]) for example in examples)) for name in values},
            "label_ranges": {name: [min(items), max(items)] for name, items in values.items()},
            "deadline_risk_positive_negative": {"positive": sum(value > 0.5 for value in values["deadline_risk"]), "negative": sum(value <= 0.5 for value in values["deadline_risk"])},
        }
        report["branch_ledgers"][split] = audit_branch_ledger(data, manifest, split)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if audit["passed"] and all(value["passed"] for value in report["branch_ledgers"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
