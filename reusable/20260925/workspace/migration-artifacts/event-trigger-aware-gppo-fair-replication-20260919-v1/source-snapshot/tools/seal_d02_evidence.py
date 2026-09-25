"""Package D-02 raw diagnostics, excluding already-released input checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("Release staging must be outside the raw evidence directory")
    destination.mkdir(parents=True, exist_ok=False)
    members = []
    archive = destination / "d02-development-diagnostics.tar.gz"
    with tarfile.open(archive, "x:gz") as tar:
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("Refusing symlink in evidence")
            if not path.is_file() or path.suffix == ".pt":
                continue
            label = path.relative_to(source).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            members.append({"path": label, "bytes": path.stat().st_size, "sha256": digest})
            tar.add(path, arcname=label, recursive=False)
    value = {"format": "d02-release-inventory/0.1.0", "members": members,
             "archive": {"name": archive.name, "bytes": archive.stat().st_size,
                         "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
             "input_checkpoints": "not duplicated; use original T-05/T-03 Releases"}
    with (destination / "d02-release-inventory.json").open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps({"members": len(members), "archive": value["archive"]}))


if __name__ == "__main__":
    main()
