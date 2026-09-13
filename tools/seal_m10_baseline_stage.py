"""Seal the bounded M-10 baseline stage with member hashes and readback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing archive: {output}")
    roots = ((args.formal_run, "formal-run"), (args.pilot_run, "pilot-run"))
    files: list[tuple[Path, str]] = []
    for root, label in roots:
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"missing run directory: {root}")
        files.extend((path, f"runs/{label}/{path.relative_to(root).as_posix()}") for path in sorted(root.rglob("*")) if path.is_file())
    extras = ((args.report, "docs/m10-baseline-and-energy-residual-20260913.md"),
              (args.audit, "evidence/m10-energy-residual-no-update-audit.json"),
              (args.config, "configs/world-gppo-9.11-energy-residual-v0.1.0.json"))
    files.extend((path.resolve(), name) for path, name in extras if path.is_file())
    if len(files) != len({name for _, name in files}):
        raise ValueError("duplicate archive member")
    members = [{"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path, name in files]
    manifest = {
        "format": "m10-baseline-stage-export/1.0.0",
        "status": "complete",
        "source_commit": args.source_commit,
        "formal_budget": {"groups": ["GPPO-Graph5", "GPPO-History-Graph5"], "seeds": [1101, 2203, 3307], "steps_per_seed": 8192, "total_environment_steps": 49152},
        "contains_consequence_fusion": False,
        "contains_new_residual_training": False,
        "files": members,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr("STAGE-MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    with zipfile.ZipFile(output) as archive:
        for member in members:
            value = archive.read(member["path"])
            if len(value) != member["bytes"] or hashlib.sha256(value).hexdigest() != member["sha256"]:
                raise ValueError(f"archive readback mismatch: {member['path']}")
    result = {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256(output), "verified_members": len(members), "status": "complete"}
    output.with_suffix(output.suffix + ".sha256.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(seal(parser.parse_args()), ensure_ascii=False, sort_keys=True))
