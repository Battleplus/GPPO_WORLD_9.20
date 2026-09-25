import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_t05_baseline_hashes_match_clean_linux_checkout() -> None:
    """The frozen hashes are the LF Git-blob bytes used by a Linux checkout."""

    config = json.loads(
        (ROOT / "nodes" / "T-05" / "server-training-config.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["baseline"] == {
        "repository": "https://github.com/Battleplus/GPPO-8.29",
        "commit": "2a9bb9f87b9d543df144f4d108ba970c924151f9",
        "protocol_sha256": (
            "f9e476724e9cfeeb5053fcefa4ddd2d900c78bb7666e3fe62b3b8d342fcb0778"
        ),
        "seed_manifest_sha256": (
            "fc11024f06113bee25e528edc1c174edc3f59119ab3ed55d81514a9a9a1db01c"
        ),
    }

    assert config["shadow_calibration"]["sha256"] == (
        "a77f2a38ff99bc63996af343e2218ebaea84e943448fab15618454fc1265272d"
    )
