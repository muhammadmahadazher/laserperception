"""CPU inspection and evaluation of saved semantic results."""

import argparse
import json

from .evaluation import evaluate_semantics
from .serialization import load_semantic_frame


def configure_semantic(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="action", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("result")
    inspect.add_argument("--json", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("prediction")
    evaluate.add_argument("ground_truth")
    evaluate.add_argument("--json", action="store_true")


def run_semantic(args: argparse.Namespace) -> int:
    if args.action == "evaluate":
        result = evaluate_semantics(
            load_semantic_frame(args.prediction), load_semantic_frame(args.ground_truth)
        )
        if args.json:
            print(result.to_json(), end="")
        else:
            print(
                f"points={result.evaluated_points} accuracy={result.overall_accuracy} "
                f"mIoU={result.mean_iou} excluded={result.excluded_mean_iou_class_ids}"
            )
    else:
        frame = load_semantic_frame(args.result)
        summary = {
            "sample_id": frame.sample_id,
            "frame_id": frame.frame_id,
            "source_point_count": frame.source_point_count,
            "result_rows": len(frame.class_ids),
            "taxonomy": frame.taxonomy.to_dict(),
            "coordinates": frame.coordinates.to_dict(),
            "provenance": frame.provenance.to_dict(),
            "confidence_available": frame.confidence is not None,
            "source_row_semantics": "strictly-increasing-source-row-indices",
        }
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0
