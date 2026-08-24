"""Run final M6c R3 Gate 2 and Detection3DArray contract on ten frozen conditions."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from laserperception_ros.conversion import detection_frame_to_message, pointcloud2_to_model_ready
from laserperception_ros.detector_node import LaserPerceptionDetectorNode
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from laserperception.detection.m6c_contract import M6C_ENGINE_SHA256
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.parity_v2 import aggregate_acceptance_v2
from laserperception.detection.parity_validation import analyze_sample
from scripts.ros2.validate_m6c_kitti_detector import (
    _canonical_sha256,
    _capture_model_ready,
    _DetectionCapture,
    _DetectorInputPublisher,
    _frame_from_dict,
    _M6cDetectorRuntime,
)

BASE_MAIN_SHA = "ebbbc0bbc4423e3be476abcd1165f75a136fa54c"
REFERENCE_GENERATION_COMMIT = "03ce7729bea0d76028783234dee559fe32cf21db"
PROJECTED_MANIFEST_SHA256 = "c06cddc6884fef87de99d1c68ec2b5c1f1945f7f9e5ecae6fcb3e4275dd952a2"
SENTINEL_SHA256 = "e80e803fe8923a52ced1bdecd17841a4cf915352a56ea5427607746409cf3be3"
PARITY_IDENTITIES = {
    "config": (
        "configs/detection/m2_parity_v2.yaml",
        "91e7cde19076c6452d9ff8e0fefc893a6d429622ed30c2da88127d29d4418df0",
    ),
    "stage_1_evaluator": (
        "src/laserperception/detection/parity_v2.py",
        "24fd8c7bcf8ee74049682ecd7d93989f4d62736eaeb35033155c0115281c38b4",
    ),
    "sample_analyzer": (
        "src/laserperception/detection/parity_validation.py",
        "37652e464a785174170240e99d593cd9d00a8362008537e182ad0e2b0a83d7f0",
    ),
    "matcher": (
        "src/laserperception/detection/parity.py",
        "1be52b850ba5f41e1abf96e83923c1f4dbe65a5a2c592a4e6bb4185dc7e83c00",
    ),
}


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_root(), check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=_root(),
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_measurement_identity(
    *,
    protocol_commit: str,
    implementation_commit: str,
    manifest_path: Path,
    sentinels_path: Path,
) -> dict[str, str]:
    if _git("rev-parse", "HEAD") != protocol_commit:
        raise RuntimeError("M6c R3 Gate 2 must run at the exact protocol commit")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("M6c R3 Gate 2 requires a clean tracked worktree")
    for ancestor in (BASE_MAIN_SHA, REFERENCE_GENERATION_COMMIT, implementation_commit):
        if not _git_is_ancestor(ancestor, protocol_commit):
            raise RuntimeError(f"R3 protocol does not descend from required commit {ancestor}")
    protocol_relative = "docs/m6/M6C_PROTOCOL_R3.md"
    if _git("log", "-1", "--format=%H", "--", protocol_relative) != protocol_commit:
        raise RuntimeError("final R3 protocol was not frozen by the claimed protocol commit")
    if sha256_file(manifest_path) != PROJECTED_MANIFEST_SHA256:
        raise RuntimeError("projected-reference manifest SHA256 mismatch before Gate 2")
    if sha256_file(sentinels_path) != SENTINEL_SHA256:
        raise RuntimeError("frozen detector sentinel SHA256 mismatch before Gate 2")
    verified: dict[str, str] = {}
    for name, (relative, expected) in PARITY_IDENTITIES.items():
        observed = sha256_file(_root() / relative)
        if observed != expected:
            raise RuntimeError(f"unchanged parity-v2 identity mismatch: {name}")
        verified[name] = observed
    return verified


def _projected_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, object]]:
    records = manifest.get("conditions")
    if not isinstance(records, Sequence) or len(records) != 860:
        raise RuntimeError("projected-reference manifest must contain 860 conditions")
    result: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("projected-reference condition record is malformed")
        result[str(record["key"])] = record
    if len(result) != 860:
        raise RuntimeError("projected-reference condition keys are not unique")
    return result


def _compact_sample_report(report: Mapping[str, Any]) -> dict[str, object]:
    counts = report["counts"]
    unmatched = report["unmatched"]
    return {
        "sample_index": report["sample_index"],
        "sample_id": report["sample_id"],
        "counts": counts,
        "per_class_exported_counts": report["per_class_exported_counts"],
        "matched": len(report["matches"]),
        "unmatched_reference": len(unmatched["pytorch"]),
        "unmatched_candidate": len(unmatched["tensorrt"]),
        "threshold_edge_disagreement_count": len(report["threshold_edge_disagreements"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_root() / "benchmarks/m6c/preregistration/projected_reference_manifest.json",
    )
    parser.add_argument("--sentinels", type=Path, required=True)
    parser.add_argument("--progress-root", type=Path, default=_root() / ".local/m6c-r3")
    parser.add_argument("--message-timeout-sec", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    sentinel_path = args.sentinels.expanduser().resolve()
    parity_identities = _require_measurement_identity(
        protocol_commit=args.protocol_commit,
        implementation_commit=args.implementation_commit,
        manifest_path=manifest_path,
        sentinels_path=sentinel_path,
    )
    progress_root = args.progress_root.expanduser().resolve()
    input_summary = _load(progress_root / "input_gate_summary.json")
    if input_summary.get("status") != "PASS":
        raise RuntimeError("R3 Gate 1 must pass before detector startup")
    if input_summary.get("protocol_commit") != args.protocol_commit:
        raise RuntimeError("R3 Gate 1 and Gate 2 protocol identities differ")
    if input_summary.get("measurement_implementation_commit") != args.implementation_commit:
        raise RuntimeError("R3 Gate 1 and Gate 2 implementation identities differ")
    if input_summary.get("projected_manifest_sha256") != PROJECTED_MANIFEST_SHA256:
        raise RuntimeError("R3 Gate 1 and Gate 2 projected manifest identities differ")
    manifest = _load(manifest_path)
    projected = _projected_map(manifest)
    preregistration = _load(sentinel_path)
    sentinels = preregistration.get("sentinels")
    if preregistration.get("status") != "FROZEN_BEFORE_M6C_DETECTOR_EXECUTION":
        raise RuntimeError("M6c detector sentinel status changed")
    if not isinstance(sentinels, Sequence) or len(sentinels) != 10:
        raise RuntimeError("M6c detector population must contain ten frozen conditions")
    if preregistration["ros_output_contract"]["velocity_exposed"] is not False:
        raise RuntimeError("accepted ROS velocity contract changed")

    engine = args.engine.expanduser().resolve()
    runtime = _M6cDetectorRuntime(engine)
    if runtime.engine_sha256 != M6C_ENGINE_SHA256:
        raise RuntimeError("structural 40k engine identity changed")
    detector_input_topic = "/laserperception/m6c/r3/detector_input"
    detector_output_topic = "/laserperception/m6c/r3/detections"
    rclpy.init()
    detector = LaserPerceptionDetectorNode(
        runtime=runtime,
        parameter_overrides=[
            Parameter("input_topic", value=detector_input_topic),
            Parameter("output_topic", value=detector_output_topic),
            Parameter("publish_markers", value=False),
            Parameter("voxelization_mode", value="exact_fast"),
            Parameter("provenance_mode", value="full"),
            Parameter("engine_path", value=str(engine)),
        ],
    )
    publisher = _DetectorInputPublisher(detector_input_topic)
    detection_capture = _DetectionCapture(detector_output_topic)
    executor = MultiThreadedExecutor(num_threads=4)
    for node in (detector, publisher, detection_capture):
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    records: list[dict[str, object]] = []
    full_sample_reports: list[dict[str, object]] = []
    ros_contract_passes = 0
    campaign_started = time.monotonic()
    try:
        discovery_deadline = time.monotonic() + 10.0
        while (
            publisher.publisher.get_subscription_count() == 0
            or detector._detections_publisher.get_subscription_count() == 0
        ):
            if time.monotonic() >= discovery_deadline:
                raise TimeoutError("R3 detector ROS endpoints did not complete discovery")
            time.sleep(0.02)
        for case_index, sentinel in enumerate(sentinels):
            if not isinstance(sentinel, Mapping):
                raise RuntimeError("M6c detector sentinel record is malformed")
            drive = str(sentinel["drive"])
            frame_index = int(str(sentinel["frame"]))
            condition = str(sentinel["condition"])
            history_depth = int(sentinel["history_depth"])
            frame_id = str(sentinel["frame_id"])
            key = f"{drive}/{frame_index:010d}|{condition}"
            projected_record = projected[key]
            model_message, builder_counters = _capture_model_ready(
                executor,
                data_root=args.data_root.expanduser().resolve(),
                drive=drive,
                frame_index=frame_index,
                history_depth=history_depth,
                case_index=case_index,
                timeout_sec=args.message_timeout_sec,
            )
            model_cloud = pointcloud2_to_model_ready(model_message)
            if model_cloud.sha256 != projected_record["model_ready_sha256"]:
                raise RuntimeError(f"R3 detector input was not exact projected Gate 1 data: {key}")
            runtime.target_sample_id = frame_id
            runtime.last_frame = None
            runtime.last_evidence = None
            publisher.publisher.publish(model_message)
            observed_message = detection_capture.wait(args.message_timeout_sec)
            if runtime.last_frame is None or runtime.last_evidence is None:
                raise RuntimeError("R3 detector callback returned without runtime evidence")
            frame_payload = sentinel.get("detection_frame")
            if not isinstance(frame_payload, Mapping):
                raise RuntimeError("frozen M6b DetectionFrame payload is malformed")
            reference_frame = _frame_from_dict(frame_payload)
            candidate_frame = runtime.last_frame
            sample_report = analyze_sample(
                reference_frame,
                candidate_frame,
                sample_index=case_index,
                exported_threshold=0.25,
                high_confidence_threshold=0.30,
                minimum_bev_iou=0.50,
            )
            full_sample_reports.append(sample_report)
            expected_candidate_message = detection_frame_to_message(
                candidate_frame, model_message.header
            )
            conversion_checks = {
                "complete_message_exact": observed_message == expected_candidate_message,
                "timestamp_exact": observed_message.header.stamp == model_message.header.stamp,
                "frame_exact": observed_message.header.frame_id == model_message.header.frame_id,
                "detection_count_exact": len(observed_message.detections)
                == len(candidate_frame.detections),
                "class_score_geometry_orientation_exact": (
                    observed_message.detections == expected_candidate_message.detections
                ),
                "per_detection_headers_exact": all(
                    detection.header == model_message.header
                    for detection in observed_message.detections
                ),
                "velocity_exposed": False,
            }
            conversion_pass = all(
                value for name, value in conversion_checks.items() if name != "velocity_exposed"
            )
            ros_contract_passes += conversion_pass
            records.append(
                {
                    "key": key,
                    "sample_id": frame_id,
                    "condition": condition,
                    "model_ready_sha256": model_cloud.sha256,
                    "projected_model_ready_exact": True,
                    "builder_counters": builder_counters,
                    "runtime_evidence": dict(runtime.last_evidence),
                    "candidate_detection_frame_sha256": _canonical_sha256(
                        candidate_frame.to_dict()
                    ),
                    "candidate_detection_frame": candidate_frame.to_dict(),
                    "parity_sample": _compact_sample_report(sample_report),
                    "detection3darray_contract": {
                        "status": "PASS" if conversion_pass else "FAIL",
                        "checks": conversion_checks,
                    },
                }
            )
            print(
                f"R3 detector condition {case_index + 1}/10 complete: {key}; "
                f"candidate_detections={len(candidate_frame.detections)}; "
                f"ros_contract={'PASS' if conversion_pass else 'FAIL'}",
                flush=True,
            )
    finally:
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        detection_capture.destroy_node()
        publisher.destroy_node()
        detector.destroy_node()
        rclpy.shutdown()

    acceptance = aggregate_acceptance_v2(
        full_sample_reports,
        minimum_coverage=0.99,
        minimum_metric_pass_fraction=0.99,
        maximum_xy_m=0.25,
        maximum_z_m=0.25,
        maximum_dimension_relative_error=0.05,
        maximum_axis_yaw_degrees=5.0,
        maximum_score_difference=0.05,
        minimum_direction_agreement=0.99,
        maximum_aggregate_count_relative_difference=0.05,
    )
    parity_pass = bool(acceptance["overall_pass"])
    ros_contract_pass = ros_contract_passes == 10
    overall_pass = parity_pass and ros_contract_pass
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS" if overall_pass else "FAIL",
        "scientific_classification": (
            "GATE_2_AND_ROS_OUTPUT_CONTRACT_PASS"
            if overall_pass
            else "FINAL_R3_DETECTOR_OR_OUTPUT_CONTRACT_NEGATIVE"
        ),
        "protocol_commit": args.protocol_commit,
        "measurement_implementation_commit": args.implementation_commit,
        "projected_manifest_sha256": PROJECTED_MANIFEST_SHA256,
        "sentinel_preregistration_sha256": SENTINEL_SHA256,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ros_distro": "Humble",
            "rmw_implementation": "rmw_fastrtps_cpp",
        },
        "frozen_artifacts": {
            "checkpoint_sha256": runtime.assets.m1_manifest["model"]["checkpoint"]["sha256"],
            "onnx_sha256": sha256_file(runtime.assets.onnx_path),
            "engine_sha256": runtime.engine_sha256,
        },
        "parity_v2": {
            "status": "PASS" if parity_pass else "FAIL",
            "identities": parity_identities,
            "reference": "frozen accepted M6b detector results",
            "candidate": "detector results from byte-exact projected/live ROS inputs",
            "inheritance_caveat": (
                "The unchanged envelope was inherited from FP32/TensorRT deployment parity; "
                "it was not statistically derived for quaternion-projection input noise."
            ),
            "stage_1": acceptance,
            "stage_2": (
                {
                    "required": True,
                    "scope": "unchanged evaluator final-detection forensics",
                    "result_changed": False,
                    "sample_reports": full_sample_reports,
                }
                if not parity_pass
                else {"required": False}
            ),
        },
        "detection3darray_contract": {
            "required": 10,
            "passed": ros_contract_passes,
            "status": "PASS" if ros_contract_pass else "FAIL",
            "velocity_exposed": False,
        },
        "conditions": records,
        "detector_node_counters": {
            "received": detector.received_count,
            "accepted": detector.accepted_count,
            "published": detector.published_count,
            "rejected": detector.rejected_count,
        },
        "wall_clock_progress_seconds": time.monotonic() - campaign_started,
        "wall_clock_note": "orchestration metadata only; not performance evidence",
        "performance_campaign": False,
    }
    output = progress_root / "detector_gate_summary.json"
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if len(encoded.encode("utf-8")) >= 5_000_000:
        raise RuntimeError("final R3 detector evidence reached the 5 MB hard stop")
    _atomic_write(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "parity_v2": result["parity_v2"]["status"],  # type: ignore[index]
                "detection3darray": result["detection3darray_contract"]["status"],  # type: ignore[index]
            }
        )
    )
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
