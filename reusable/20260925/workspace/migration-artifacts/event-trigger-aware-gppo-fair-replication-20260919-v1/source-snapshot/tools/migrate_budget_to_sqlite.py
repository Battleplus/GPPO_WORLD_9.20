"""Migrate one frozen legacy budget JSON into an SQLite ledger once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from gppo_world.budget_executor import PersistentBudget


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    legacy = args.legacy_json.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    working_json = out_dir / "persistent-budget.json"
    sqlite_path = out_dir / "persistent-budget.sqlite3"
    shutil.copy2(legacy, working_json)
    source_hash = sha256(legacy)
    budget = PersistentBudget(sqlite_path, limits=json.loads(working_json.read_text(encoding="utf-8"))["limits"], attempt_id="migration-audit")
    snapshot = budget.snapshot()
    exported = budget.export_snapshot(out_dir / "persistent-budget-v2-export.json")
    manifest = {
        "schema": "budget-sqlite-migration/1.0.0",
        "legacy_source": str(legacy),
        "legacy_source_sha256": source_hash,
        "working_legacy_copy": str(working_json),
        "working_legacy_copy_sha256": sha256(working_json),
        "sqlite_path": str(sqlite_path),
        "sqlite_sha256": sha256(sqlite_path),
        "export_path": str(exported),
        "export_sha256": sha256(exported),
        "integrity": budget.integrity_check(),
        "stages": snapshot["stages"],
        "attempts": snapshot["attempts"],
        "history_count": len(snapshot["history"]),
        "migrated_at": time.time(),
        "json_is_snapshot_only": True,
    }
    (out_dir / "migration-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
