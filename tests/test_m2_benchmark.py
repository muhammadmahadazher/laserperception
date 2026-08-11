import copy

import pytest

from laserperception.detection.m2_benchmark import build_parity_v2_benchmark_record

COMMIT = "a" * 40
ONNX_SHA256 = "b" * 64
ENGINE_SHA256 = "c" * 64
FROZEN_INDICES = list(range(20))


def _passing_parity_v2() -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "protocol_version": 2,
        "status": "pass",
        "diagnostic_only": False,
        "commit_sha": COMMIT,
        "dataset": {"sample_indices": FROZEN_INDICES},
        "artifacts": {
            "onnx": {"sha256": ONNX_SHA256},
            "engine": {"sha256": ENGINE_SHA256},
        },
        "stage_1": {"overall_pass": True, "failed_checks": []},
        "overall_pass": True,
    }


def _build(parity: dict[str, object]) -> dict[str, object]:
    return build_parity_v2_benchmark_record(
        parity,
        b'{"synthetic":"parity-v2"}\n',
        current_commit=COMMIT,
        frozen_indices=FROZEN_INDICES,
        onnx_sha256=ONNX_SHA256,
        engine_sha256=ENGINE_SHA256,
    )


def test_build_parity_v2_benchmark_record_preserves_stage_1_summary() -> None:
    record = _build(_passing_parity_v2())

    assert record["protocol_version"] == 2
    assert record["status"] == "pass"
    assert record["commit_sha"] == COMMIT
    assert record["sample_count"] == 20
    assert record["overall_pass"] is True
    assert record["stage_1"] == {"overall_pass": True, "failed_checks": []}
    assert record["result_sha256"] == (
        "53205d4175b85cc4665a6575edb570c396d659793f2ffe8785bfc7db10296904"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol_version",), 1),
        (("schema_version",), "1.0"),
        (("status",), "fail"),
        (("diagnostic_only",), True),
        (("overall_pass",), False),
        (("commit_sha",), "d" * 40),
        (("dataset", "sample_indices"), list(range(19))),
        (("stage_1", "overall_pass"), False),
        (("artifacts", "onnx", "sha256"), "d" * 64),
        (("artifacts", "engine", "sha256"), "d" * 64),
    ],
)
def test_build_parity_v2_benchmark_record_rejects_unpromotable_evidence(
    path: tuple[str, ...], value: object
) -> None:
    parity = copy.deepcopy(_passing_parity_v2())
    target = parity
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value

    with pytest.raises(ValueError):
        _build(parity)


def test_build_parity_v2_benchmark_record_rejects_v1_shape() -> None:
    parity = _passing_parity_v2()
    parity["protocol_version"] = 1
    parity["schema_version"] = "1.0"
    parity["acceptance_summary"] = parity.pop("stage_1")

    with pytest.raises(ValueError, match="protocol_version 2"):
        _build(parity)
