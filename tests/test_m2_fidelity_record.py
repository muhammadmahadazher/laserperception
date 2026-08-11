import copy

import pytest

from laserperception.detection.m2_benchmark import build_fidelity_diagnostic_record

COMMIT = "a" * 40
CHECKPOINT_SHA256 = "b" * 64
ONNX_SHA256 = "c" * 64
ENGINE_SHA256 = "d" * 64
FROZEN_INDICES = list(range(20))


def _passing_diagnostic() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "status": "diagnostic_measurement_not_canonical",
        "publication_eligible": False,
        "canonical_benchmark_run": False,
        "commit_sha": COMMIT,
        "artifacts": {
            "checkpoint": {"sha256": CHECKPOINT_SHA256},
            "onnx": {"sha256": ONNX_SHA256},
            "engine": {"sha256": ENGINE_SHA256},
        },
        "native_vs_rewritten_fidelity": {
            "materially_equivalent": True,
            "sample_indices": FROZEN_INDICES,
            "sample_count": 20,
        },
    }


def _build(diagnostic: dict[str, object]) -> dict[str, object]:
    return build_fidelity_diagnostic_record(
        diagnostic,
        b'{"synthetic":"fidelity"}\n',
        current_commit=COMMIT,
        frozen_indices=FROZEN_INDICES,
        checkpoint_sha256=CHECKPOINT_SHA256,
        onnx_sha256=ONNX_SHA256,
        engine_sha256=ENGINE_SHA256,
    )


def test_fidelity_record_preserves_exact_commit_and_full_sample_pass() -> None:
    record = _build(_passing_diagnostic())

    assert record["status"] == "pass"
    assert record["commit_sha"] == COMMIT
    assert record["sample_count"] == 20
    assert record["materially_equivalent"] is True


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "measured"),
        (("publication_eligible",), True),
        (("canonical_benchmark_run",), True),
        (("commit_sha",), "e" * 40),
        (("artifacts", "checkpoint", "sha256"), "e" * 64),
        (("artifacts", "onnx", "sha256"), "e" * 64),
        (("artifacts", "engine", "sha256"), "e" * 64),
        (("native_vs_rewritten_fidelity", "materially_equivalent"), False),
        (("native_vs_rewritten_fidelity", "sample_indices"), list(range(19))),
    ],
)
def test_fidelity_record_rejects_unpromotable_diagnostics(
    path: tuple[str, ...], value: object
) -> None:
    diagnostic = copy.deepcopy(_passing_diagnostic())
    target = diagnostic
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value

    with pytest.raises(ValueError):
        _build(diagnostic)
