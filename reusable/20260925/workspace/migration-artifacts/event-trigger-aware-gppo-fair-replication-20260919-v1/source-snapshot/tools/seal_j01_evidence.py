"""Seal all J01 graphs, tapes, weights, probes and results with member hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def seal(data, run, destination, compact):
    verification = json.loads((run / "verification.json").read_text(encoding="utf-8"))
    if not verification["passed"] or verification["results_sha256"] != sha(run / "results.json"):
        raise ValueError("verification absent or stale")
    if destination.resolve().is_relative_to(data.resolve()) or destination.resolve().is_relative_to(run.resolve()):
        raise ValueError("staging inside evidence")
    destination.mkdir(parents=True, exist_ok=False)
    compact.mkdir(parents=True, exist_ok=True)
    inventories = []
    for source, name in ((data, "j01-dataset.tar.gz"), (run, "j01-models-and-results.tar.gz")):
        archive = destination / name
        members = []
        with tarfile.open(archive, "x:gz") as tar:
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    raise ValueError("symlink in evidence")
                if not path.is_file():
                    continue
                label = path.relative_to(source).as_posix()
                members.append({"path": label, "bytes": path.stat().st_size, "sha256": sha(path)})
                tar.add(path, arcname=label, recursive=False)
        with tarfile.open(archive) as tar:
            if [m.name for m in tar.getmembers()] != [m["path"] for m in members]:
                raise ValueError("archive membership changed")
            for item in members:
                content = tar.extractfile(item["path"]).read()
                if hashlib.sha256(content).hexdigest() != item["sha256"]:
                    raise ValueError("archive member corruption")
        inventories.append({"name": name, "bytes": archive.stat().st_size, "sha256": sha(archive), "members": members})
    dump(destination / "j01-release-inventory.json", {"format": "j01-release-inventory/1.0.0", "archives": inventories,
         "all_archive_members_verified": True, "contains_original_graph_nodes_and_edges": True,
         "contains_12_model_checkpoints_and_12_probes": True, "contains_behavior_checkpoint": True})
    shutil.copy2(run / "results.json", destination / "j01-results.json")
    names = ("results.json", "verification.json", "provenance.json", "selection-frozen.json", "data-audit.json", "target-normalization.json")
    for name in names:
        target = compact / name
        if target.exists():
            raise ValueError("refusing to overwrite compact evidence")
        shutil.copy2(run / name, target)
    shutil.copy2(destination / "j01-release-inventory.json", compact / "release-inventory.json")
    with (destination / "SHA256SUMS.txt").open("x", encoding="utf-8", newline="\n") as stream:
        for path in sorted(destination.iterdir()):
            if path.name != "SHA256SUMS.txt":
                stream.write(f"{sha(path)}  {path.name}\n")
    print(json.dumps([{"name": a["name"], "bytes": a["bytes"], "sha256": a["sha256"], "members": len(a["members"])} for a in inventories]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("data", "run", "destination", "compact"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    seal(args.data, args.run, args.destination, args.compact)
