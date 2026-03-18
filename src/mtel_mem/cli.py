from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters.base import load_manifest, validate_manifest
from .adapters.locomo import default_manifest_path as locomo_manifest_path
from .adapters.longmemeval_s import default_manifest_path as longmemeval_s_manifest_path
from .core.instability import compute_instability
from .core.rescore import load_ranked_hits_jsonl, load_query_records_jsonl, score_paths
from .core.validation import (
    build_validation_report,
    render_validation_markdown,
    validate_query_records,
    validate_ranked_hits,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MTEL-Mem command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-benchmarks", help="List built-in benchmark manifests")
    list_parser.set_defaults(func=cmd_list_benchmarks)

    validate_parser = subparsers.add_parser("validate-manifest", help="Validate one benchmark manifest")
    validate_parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    validate_parser.set_defaults(func=cmd_validate_manifest)

    score_generic = subparsers.add_parser("score", help="Score a bring-your-own target mapping + ranked trace pair")
    score_generic.add_argument("--targets", required=True, help="Query target mapping JSONL")
    score_generic.add_argument("--trace", required=True, help="Ranked trace JSONL")
    score_generic.add_argument("--top-k", type=int, default=60)
    score_generic.add_argument("--out-json", help="Optional output JSON path")
    score_generic.set_defaults(func=cmd_score)

    score_parser = subparsers.add_parser("score-example", help="Run the bundled example scorer")
    score_parser.add_argument(
        "--targets",
        default=str(Path(__file__).resolve().parents[2] / "examples" / "minimal_target_mappings.jsonl"),
    )
    score_parser.add_argument(
        "--trace",
        default=str(Path(__file__).resolve().parents[2] / "examples" / "minimal_ranked_trace.jsonl"),
    )
    score_parser.add_argument("--top-k", type=int, default=5)
    score_parser.add_argument("--out-json", help="Optional output JSON path")
    score_parser.set_defaults(func=cmd_score)

    suite_parser = subparsers.add_parser("validate-suite", help="Run the MTEL-Mem scorecard and validation suite")
    suite_parser.add_argument("--out-json", help="Optional JSON report path")
    suite_parser.add_argument("--out-md", help="Optional Markdown report path")
    suite_parser.add_argument("--tolerance", type=float, default=1e-9)
    suite_parser.set_defaults(func=cmd_validate_suite)

    return parser


def cmd_list_benchmarks(_: argparse.Namespace) -> None:
    payload = {
        "benchmarks": {
            "locomo": str(locomo_manifest_path()),
            "longmemeval_s": str(longmemeval_s_manifest_path()),
        }
    }
    print(json.dumps(payload, indent=2))


def cmd_validate_manifest(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    print(json.dumps(validate_manifest(manifest), indent=2))


def cmd_score(args: argparse.Namespace) -> None:
    query_records = load_query_records_jsonl(args.targets)
    ranked_hits = load_ranked_hits_jsonl(args.trace)
    schema_report = validate_query_records(query_records)
    trace_report = validate_ranked_hits(ranked_hits, query_records=query_records)
    if schema_report["failed_checks"] or trace_report["failed_checks"]:
        raise SystemExit(json.dumps({"schema": schema_report, "trace": trace_report}, indent=2))

    scored = score_paths(args.targets, args.trace, k=args.top_k)
    payload = {
        "schema": schema_report,
        "trace": trace_report,
        "aggregate": scored["aggregate"],
        "instability": {
            "raw_vs_source": compute_instability(scored["query_scores"], "raw", "source"),
            "raw_vs_canonical": compute_instability(scored["query_scores"], "raw", "canonical"),
            "source_vs_canonical": compute_instability(scored["query_scores"], "source", "canonical"),
        },
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def cmd_validate_suite(args: argparse.Namespace) -> None:
    report = build_validation_report(tolerance=args.tolerance)
    markdown = render_validation_markdown(report)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
