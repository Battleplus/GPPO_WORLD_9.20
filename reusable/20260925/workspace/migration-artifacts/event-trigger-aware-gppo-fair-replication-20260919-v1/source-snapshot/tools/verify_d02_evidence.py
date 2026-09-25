"""Independently cross-check saved D-02 traces, probes, audits and inventories."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    failures = []
    checked = 0
    for entry in read(root / "inventory.json"):
        file = (root / entry["path"]).resolve()
        if not file.is_relative_to(root):
            raise ValueError("Inventory path escapes evidence root")
        if hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(entry["path"])
        checked += 1
    count = 0
    decisions = 0
    for on_path in sorted((root / "traces").glob("*/*/on/*.json")):
        on = read(on_path)
        off = read(on_path.parent.parent / "off" / on_path.name)
        if len(on["decisions"]) != len(off["decisions"]):
            failures.append(f"decision_count:{on_path.name}")
        for a, b in zip(on["decisions"], off["decisions"]):
            # Compare every non-diagnostic field, not just the runner signature.
            if {k:v for k,v in a.items() if k != "diagnostics"} != {
                k:v for k,v in b.items() if k != "diagnostics"
            }:
                failures.append(f"decision:{on_path.name}")
        for key in ("initial_seed", "event_seed", "tape_id", "tape_sha256",
                    "final_snapshot", "initial_seed", "terminated", "truncated",
                    "total_reward_check", "episode_return_check"):
            if on[key] != off[key]:
                failures.append(f"{key}:{on_path.name}")
        for trace in (on, off):
            rewards = sum(d["reward"] for d in trace["decisions"])
            attributed = sum(d["reward"] for d in trace["decisions"] if d["active_events_before"])
            if abs(rewards-trace["episode_return_check"]) > 1e-7 or abs(attributed-trace["episode"]["episode_return"]) > 1e-7:
                failures.append(f"return:{on_path.name}")
        count += 1
        decisions += len(on["decisions"])
    for audit in (root / "audits").glob("*/*.json"):
        value = read(audit)
        for env in value["environments"]:
            if any(v for k,v in env.items() if "mutation_count" in k or "write_count" in k or k == "action_submission_count"):
                failures.append(f"safety:{audit.name}")
    probes = read(root / "decision-probes.json")
    if len(probes) != decisions or count != 108:
        failures.append("counts")
    if not all(r["invalid_logits_unchanged"] and r["mask_and_version_unchanged"] for r in probes):
        failures.append("probe_mutation")
    result = {"status": "PASS" if not failures else "FAILED", "failures": failures,
              "inventory_files_checked": checked, "pairs_checked": count,
              "traces_checked": count*2, "probe_records_checked": len(probes),
              "return_semantics_checked": ["event_attributed", "all_decisions"],
              "checks": "full non-diagnostic decision fields plus final snapshot, reward sums, safety and file hashes"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(result))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
