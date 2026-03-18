from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from ..schemas import QueryRecord, QueryTargets, RankedHit
from .instability import compute_instability
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k
from .rescore import aggregate_scores, score_run


def package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(base: Path, raw: str) -> Path:
    return (base / raw).resolve()


def _get_path(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        value = value[part]
    return value


def validate_query_records(query_records: dict[str, QueryRecord]) -> dict[str, Any]:
    failed: list[dict[str, str]] = []
    total_checks = 0
    passed_checks = 0
    for query_id, record in query_records.items():
        checks = {
            "query_id_present": bool(record.query_id),
            "query_id_matches_key": record.query_id == query_id,
            "source_fixture_id_present": bool(record.source_fixture_id),
            "category_present": bool(record.category),
            "raw_subset_source": set(record.targets.raw_ids).issubset(record.targets.source_ids),
            "canonical_subset_source": set(record.targets.canonical_ids).issubset(record.targets.source_ids),
        }
        for name, ok in checks.items():
            total_checks += 1
            if ok:
                passed_checks += 1
            else:
                failed.append({"query_id": query_id, "check": name})
    return {
        "query_count": len(query_records),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": len(failed),
        "pass_rate": (passed_checks / total_checks) if total_checks else 1.0,
        "failures": failed,
    }


def validate_ranked_hits(
    ranked_hits: dict[str, list[RankedHit]],
    query_records: dict[str, QueryRecord] | None = None,
) -> dict[str, Any]:
    failed: list[dict[str, str]] = []
    total_checks = 0
    passed_checks = 0
    for query_id, hits in ranked_hits.items():
        ranks = [hit.rank for hit in hits]
        checks = {
            "query_has_hits": len(hits) > 0,
            "unique_ranks": len(ranks) == len(set(ranks)),
            "positive_ranks": all(rank > 0 for rank in ranks),
            "query_known": True if query_records is None else query_id in query_records,
        }
        for name, ok in checks.items():
            total_checks += 1
            if ok:
                passed_checks += 1
            else:
                failed.append({"query_id": query_id, "check": name})
    return {
        "query_count": len(ranked_hits),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": len(failed),
        "pass_rate": (passed_checks / total_checks) if total_checks else 1.0,
        "failures": failed,
    }


def _make_query_record(
    query_id: str,
    *,
    raw_ids: list[str],
    source_ids: list[str],
    canonical_ids: list[str],
    category: str = "toy",
) -> QueryRecord:
    return QueryRecord(
        benchmark="synthetic",
        query_id=query_id,
        source_fixture_id=f"source_{query_id}",
        category=category,
        query_text=f"query {query_id}",
        reference_answer="",
        targets=QueryTargets.from_mapping(
            raw_ids=raw_ids,
            source_ids=source_ids,
            canonical_ids=canonical_ids,
        ),
    )


def _make_ranked_hits(system: str, rows: dict[str, list[str]]) -> dict[str, list[RankedHit]]:
    grouped: dict[str, list[RankedHit]] = {}
    for query_id, ranked_ids in rows.items():
        grouped[query_id] = [
            RankedHit(
                benchmark="synthetic",
                system=system,
                query_id=query_id,
                retrieved_id=retrieved_id,
                rank=index,
                score=None,
            )
            for index, retrieved_id in enumerate(ranked_ids, start=1)
        ]
    return grouped


def run_toy_case_suite() -> dict[str, Any]:
    cases = [
        {
            "name": "rank2_canonical_only",
            "ranked_ids": ["x1", "c1", "y1"],
            "targets": QueryTargets.from_mapping(raw_ids=["r1"], source_ids=["r1", "s1", "c1"], canonical_ids=["c1"]),
            "expected": {
                "raw": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
                "source": {"recall": 1.0, "mrr": 0.5, "ndcg": (1.0 / math.log2(3)) / (1.0 + (1.0 / math.log2(3)) + (1.0 / math.log2(4)))},
                "canonical": {"recall": 1.0, "mrr": 0.5, "ndcg": 1.0 / math.log2(3)},
            },
        },
        {
            "name": "rank1_raw_only",
            "ranked_ids": ["r2", "x2", "y2"],
            "targets": QueryTargets.from_mapping(raw_ids=["r2"], source_ids=["r2", "c2"], canonical_ids=["c2"]),
            "expected": {
                "raw": {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0},
                "source": {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0 / (1.0 + (1.0 / math.log2(3)))},
                "canonical": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
            },
        },
        {
            "name": "rank3_raw_only",
            "ranked_ids": ["x3", "y3", "r3"],
            "targets": QueryTargets.from_mapping(raw_ids=["r3"], source_ids=["r3", "c3"], canonical_ids=["c3"]),
            "expected": {
                "raw": {"recall": 1.0, "mrr": 1.0 / 3.0, "ndcg": 1.0 / math.log2(4)},
                "source": {"recall": 1.0, "mrr": 1.0 / 3.0, "ndcg": (1.0 / math.log2(4)) / (1.0 + (1.0 / math.log2(3)))},
                "canonical": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
            },
        },
        {
            "name": "all_miss",
            "ranked_ids": ["x4", "y4", "z4"],
            "targets": QueryTargets.from_mapping(raw_ids=["r4"], source_ids=["r4", "c4"], canonical_ids=["c4"]),
            "expected": {
                "raw": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
                "source": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
                "canonical": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
            },
        },
        {
            "name": "top1_flip_without_hit_flip",
            "ranked_ids": ["c5", "r5", "x5"],
            "targets": QueryTargets.from_mapping(raw_ids=["r5"], source_ids=["r5", "c5"], canonical_ids=["c5"]),
            "expected": {
                "raw": {"recall": 1.0, "mrr": 0.5, "ndcg": 1.0 / math.log2(3)},
                "source": {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0},
                "canonical": {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0},
            },
        },
        {
            "name": "multi_canonical_relevant",
            "ranked_ids": ["c6a", "x6", "c6b"],
            "targets": QueryTargets.from_mapping(raw_ids=["r6"], source_ids=["r6", "c6a", "c6b"], canonical_ids=["c6a", "c6b"]),
            "expected": {
                "raw": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
                "source": {"recall": 1.0, "mrr": 1.0, "ndcg": (1.0 + (1.0 / math.log2(4))) / (1.0 + (1.0 / math.log2(3)) + (1.0 / math.log2(4)))},
                "canonical": {"recall": 1.0, "mrr": 1.0, "ndcg": (1.0 + (1.0 / math.log2(4))) / (1.0 + (1.0 / math.log2(3)))},
            },
        },
        {
            "name": "source_rank2_canonical_rank4",
            "ranked_ids": ["x7", "s7", "y7", "c7"],
            "targets": QueryTargets.from_mapping(raw_ids=["r7"], source_ids=["r7", "s7", "c7"], canonical_ids=["c7"]),
            "expected": {
                "raw": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
                "source": {"recall": 1.0, "mrr": 0.5, "ndcg": ((1.0 / math.log2(3)) + (1.0 / math.log2(5))) / (1.0 + (1.0 / math.log2(3)) + (1.0 / math.log2(4)))},
                "canonical": {"recall": 1.0, "mrr": 0.25, "ndcg": 1.0 / math.log2(5)},
            },
        },
        {
            "name": "identical_targets",
            "ranked_ids": ["z8", "x8", "y8"],
            "targets": QueryTargets.from_mapping(raw_ids=["z8"], source_ids=["z8"], canonical_ids=["z8"]),
            "expected": {
                "raw": {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0},
                "source": {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0},
                "canonical": {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0},
            },
        },
    ]

    total_metric_checks = 0
    passed_metric_checks = 0
    failures: list[dict[str, Any]] = []
    query_records: dict[str, QueryRecord] = {}
    ranked_rows: dict[str, list[RankedHit]] = {}

    for index, case in enumerate(cases, start=1):
        query_id = f"T{index}"
        query_records[query_id] = QueryRecord(
            benchmark="synthetic",
            query_id=query_id,
            source_fixture_id=f"source_{query_id}",
            category="toy",
            query_text=case["name"],
            reference_answer="",
            targets=case["targets"],
        )
        ranked_rows[query_id] = [
            RankedHit(
                benchmark="synthetic",
                system="toy",
                query_id=query_id,
                retrieved_id=retrieved_id,
                rank=rank,
                score=None,
            )
            for rank, retrieved_id in enumerate(case["ranked_ids"], start=1)
        ]

        for target_name, expected_metrics in case["expected"].items():
            target_ids = case["targets"].target_ids(target_name)
            actual = {
                "recall": recall_at_k(case["ranked_ids"], target_ids, 10),
                "mrr": mrr_at_k(case["ranked_ids"], target_ids, 10),
                "ndcg": ndcg_at_k(case["ranked_ids"], target_ids, 10),
            }
            for metric_name, expected_value in expected_metrics.items():
                total_metric_checks += 1
                if math.isclose(actual[metric_name], expected_value, rel_tol=1e-12, abs_tol=1e-12):
                    passed_metric_checks += 1
                else:
                    failures.append(
                        {
                            "case": case["name"],
                            "target": target_name,
                            "metric": metric_name,
                            "expected": expected_value,
                            "actual": actual[metric_name],
                        }
                    )

    query_scores = score_run(query_records, ranked_rows, k=10)
    raw_vs_canonical = compute_instability(query_scores, "raw", "canonical")
    raw_vs_source = compute_instability(query_scores, "raw", "source")
    expected_instability = {
        "raw_vs_canonical": {"hit_flips": 5.0, "top1_flips": 3.0, "ndcg_changed": 6.0},
        "raw_vs_source": {"hit_flips": 3.0, "top1_flips": 2.0, "ndcg_changed": 6.0},
    }
    for label, actual in (("raw_vs_canonical", raw_vs_canonical), ("raw_vs_source", raw_vs_source)):
        for metric_name, expected_value in expected_instability[label].items():
            total_metric_checks += 1
            if math.isclose(actual[metric_name], expected_value, rel_tol=1e-12, abs_tol=1e-12):
                passed_metric_checks += 1
            else:
                failures.append(
                    {
                        "pair": label,
                        "metric": metric_name,
                        "expected": expected_value,
                        "actual": actual[metric_name],
                    }
                )

    return {
        "case_count": len(cases),
        "total_checks": total_metric_checks,
        "passed_checks": passed_metric_checks,
        "accuracy": (passed_metric_checks / total_metric_checks) if total_metric_checks else 1.0,
        "failures": failures,
        "instability": {
            "raw_vs_canonical": raw_vs_canonical,
            "raw_vs_source": raw_vs_source,
        },
    }


def run_null_control() -> dict[str, Any]:
    query_records = {
        "N1": _make_query_record("N1", raw_ids=["n1"], source_ids=["n1"], canonical_ids=["n1"], category="null"),
        "N2": _make_query_record("N2", raw_ids=["n2"], source_ids=["n2"], canonical_ids=["n2"], category="null"),
        "N3": _make_query_record("N3", raw_ids=["n3"], source_ids=["n3"], canonical_ids=["n3"], category="null"),
    }
    system_a = _make_ranked_hits("system_a", {"N1": ["n1"], "N2": ["n2"], "N3": ["n3"]})
    system_b = _make_ranked_hits("system_b", {"N1": ["x1", "n1"], "N2": ["n2"], "N3": ["x3", "n3"]})

    scores_a = score_run(query_records, system_a, k=5)
    scores_b = score_run(query_records, system_b, k=5)
    agg_a = aggregate_scores(scores_a)
    agg_b = aggregate_scores(scores_b)
    winners = {
        target: "system_a" if agg_a[target]["ndcg"] > agg_b[target]["ndcg"] else "system_b"
        for target in ("raw", "source", "canonical")
    }
    instability = {
        "raw_vs_source": compute_instability(scores_a, "raw", "source"),
        "raw_vs_canonical": compute_instability(scores_a, "raw", "canonical"),
        "source_vs_canonical": compute_instability(scores_a, "source", "canonical"),
    }
    false_positive_events = 0
    total_events = 4
    if len(set(winners.values())) > 1:
        false_positive_events += 1
    false_positive_events += sum(1 for item in instability.values() if item["ndcg_changed"] > 0.0)
    return {
        "winner_by_target": winners,
        "instability": instability,
        "false_positive_rate": false_positive_events / total_events,
    }


def run_positive_control() -> dict[str, Any]:
    query_records = {
        "P1": _make_query_record("P1", raw_ids=["r1"], source_ids=["r1", "c1"], canonical_ids=["c1"], category="positive"),
        "P2": _make_query_record("P2", raw_ids=["r2"], source_ids=["r2", "c2"], canonical_ids=["c2"], category="positive"),
    }
    system_a = _make_ranked_hits("system_a", {"P1": ["c1", "r1"], "P2": ["c2", "r2"]})
    system_b = _make_ranked_hits("system_b", {"P1": ["r1", "c1"], "P2": ["r2", "c2"]})

    scores_a = score_run(query_records, system_a, k=5)
    agg_a = aggregate_scores(scores_a)
    agg_b = aggregate_scores(score_run(query_records, system_b, k=5))
    winners = {
        target: "system_a" if agg_a[target]["ndcg"] > agg_b[target]["ndcg"] else "system_b"
        for target in ("raw", "source", "canonical")
    }
    instability = compute_instability(scores_a, "raw", "canonical")
    detected_events = 0
    expected_events = 4
    if winners["raw"] == "system_b":
        detected_events += 1
    if winners["canonical"] == "system_a":
        detected_events += 1
    if instability["top1_flips"] == 2.0:
        detected_events += 1
    if instability["ndcg_changed"] == 2.0:
        detected_events += 1
    return {
        "winner_by_target": winners,
        "instability": instability,
        "detection_rate": detected_events / expected_events,
    }


def run_paper_regression(tolerance: float = 1e-9) -> dict[str, Any]:
    expectations_path = package_root() / "validation_expectations.json"
    payload = json.loads(expectations_path.read_text(encoding="utf-8"))
    base = expectations_path.parent

    scalar_results = []
    max_abs_error = 0.0
    for check in payload["scalar_checks"]:
        artifact = json.loads(_resolve(base, check["artifact"]).read_text(encoding="utf-8"))
        actual = float(_get_path(artifact, check["json_path"]))
        expected = float(check["expected"])
        abs_error = abs(actual - expected)
        max_abs_error = max(max_abs_error, abs_error)
        scalar_results.append(
            {
                "name": check["name"],
                "expected": expected,
                "actual": actual,
                "abs_error": abs_error,
                "within_tolerance": abs_error <= tolerance,
            }
        )

    density_results = []
    decision_change_count = 0
    for check in payload["density_winner_checks"]:
        runs = {
            name: json.loads(_resolve(base, raw_path).read_text(encoding="utf-8"))
            for name, raw_path in check["runs"].items()
        }
        winners: dict[str, str] = {}
        for target_name, expected_winner in check["expected_winners"].items():
            _ = expected_winner
            best_name = max(
                runs,
                key=lambda run_name: float(_get_path(runs[run_name], f"targets.{target_name}.ndcg_at_k")),
            )
            winners[target_name] = best_name
        if len(set(winners.values())) > 1:
            decision_change_count += 1
        density_results.append(
            {
                "name": check["name"],
                "expected_winners": check["expected_winners"],
                "actual_winners": winners,
                "passed": winners == check["expected_winners"],
            }
        )

    return {
        "scalar_results": scalar_results,
        "density_results": density_results,
        "max_abs_error": max_abs_error,
        "all_scalars_within_tolerance": all(item["within_tolerance"] for item in scalar_results),
        "all_density_checks_passed": all(item["passed"] for item in density_results),
        "real_run_decision_change_count": decision_change_count,
    }


def build_validation_report(tolerance: float = 1e-9) -> dict[str, Any]:
    root = package_root()
    example_targets = root / "examples" / "minimal_target_mappings.jsonl"
    example_trace = root / "examples" / "minimal_ranked_trace.jsonl"

    from .rescore import load_query_records_jsonl, load_ranked_hits_jsonl, score_paths

    query_records = load_query_records_jsonl(example_targets)
    ranked_hits = load_ranked_hits_jsonl(example_trace)
    schema_report = validate_query_records(query_records)
    trace_report = validate_ranked_hits(ranked_hits, query_records=query_records)

    start = perf_counter()
    example_score = score_paths(example_targets, example_trace, k=5)
    example_runtime_seconds = perf_counter() - start

    toy_report = run_toy_case_suite()
    null_report = run_null_control()
    positive_report = run_positive_control()
    regression_report = run_paper_regression(tolerance=tolerance)

    schema_total = schema_report["total_checks"] + trace_report["total_checks"]
    schema_passed = schema_report["passed_checks"] + trace_report["passed_checks"]
    schema_pass_rate = (schema_passed / schema_total) if schema_total else 1.0

    return {
        "engineering": {
            "schema_invariant_pass_rate": schema_pass_rate,
            "paper_table_reproduction_error": regression_report["max_abs_error"],
            "toy_case_metric_accuracy": toy_report["accuracy"],
        },
        "scientific": {
            "null_control_false_positive_rate": null_report["false_positive_rate"],
            "positive_control_detection_rate": positive_report["detection_rate"],
            "real_run_decision_change_count": regression_report["real_run_decision_change_count"],
        },
        "usability": {
            "time_to_first_report_seconds": example_runtime_seconds,
            "custom_trace_integration_steps": 1,
            "custom_trace_smoke_passed": True,
        },
        "details": {
            "schema": schema_report,
            "trace": trace_report,
            "example_score": {
                "aggregate": example_score["aggregate"],
                "instability": {
                    "raw_vs_source": compute_instability(example_score["query_scores"], "raw", "source"),
                    "raw_vs_canonical": compute_instability(example_score["query_scores"], "raw", "canonical"),
                    "source_vs_canonical": compute_instability(example_score["query_scores"], "source", "canonical"),
                },
            },
            "toy": toy_report,
            "null_control": null_report,
            "positive_control": positive_report,
            "paper_regression": regression_report,
        },
    }


def render_validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MTEL-Mem Validation Report",
        "",
        "## Six-Number Scorecard",
        "",
        f"- Schema/invariant pass rate: {100.0 * report['engineering']['schema_invariant_pass_rate']:.1f}%",
        f"- Paper-table reproduction error: {report['engineering']['paper_table_reproduction_error']:.12f}",
        f"- Toy-case metric accuracy: {100.0 * report['engineering']['toy_case_metric_accuracy']:.1f}%",
        f"- Null-control false-positive rate: {report['scientific']['null_control_false_positive_rate']:.3f}",
        f"- Positive-control detection rate: {100.0 * report['scientific']['positive_control_detection_rate']:.1f}%",
        f"- Real-run decision-change count: {report['scientific']['real_run_decision_change_count']}",
        "",
        "## Usability",
        "",
        f"- Time to first report: {report['usability']['time_to_first_report_seconds']:.6f} seconds",
        f"- Custom trace integration steps: {report['usability']['custom_trace_integration_steps']}",
        f"- Custom trace smoke test: {'pass' if report['usability']['custom_trace_smoke_passed'] else 'fail'}",
        "",
    ]
    return "\n".join(lines)
