import json
import math
from pathlib import Path


def test_promoted_m1_benchmark_is_measured_and_sanitized() -> None:
    result_path = (
        Path(__file__).parents[1] / "benchmarks" / "m1" / "results" / "rtx4060_laptop_fp32.json"
    )
    document = json.loads(result_path.read_text(encoding="utf-8"))

    assert document["schema_version"] == "1.0"
    assert document["status"] == "measured"
    assert document["environment"]["gpu_name"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert document["environment"]["mmdet3d_commit"] == ("fe25f7a51d36e3702f961e198894580d83c4387b")
    assert len(document["commit_sha"]) == 40
    assert all(character in "0123456789abcdef" for character in document["commit_sha"])
    assert document["dataset"]["observed_split_size"] == 81
    assert document["warmup_iterations_per_boundary"] == 10

    for measurement in document["measurements"].values():
        statistics = measurement["statistics"]
        assert statistics["count"] == 50
        assert all(
            math.isfinite(value) and value > 0
            for key, value in statistics.items()
            if key != "count"
        )

    serialized = json.dumps(document).lower()
    for private_marker in ("/root/", "/home/", "\\users\\", "my drive", "data_root"):
        assert private_marker not in serialized
