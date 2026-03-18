from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..schemas import QueryRecord, QueryTargets, RankedHit
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


def load_query_records_jsonl(path: str | Path) -> dict[str, QueryRecord]:
    records: dict[str, QueryRecord] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            query_id = str(payload["query_id"])
            records[query_id] = QueryRecord(
                benchmark=str(payload["benchmark"]),
                query_id=query_id,
                source_fixture_id=str(payload["source_fixture_id"]),
                category=str(payload.get("category", "")),
                query_text=str(payload.get("query_text", "")),
                reference_answer=str(payload.get("reference_answer", "")),
                targets=QueryTargets.from_mapping(
                    raw_ids=payload.get("raw_ids", []),
                    source_ids=payload.get("source_ids", []),
                    canonical_ids=payload.get("canonical_ids", []),
                ),
            )
    return records


def load_ranked_hits_jsonl(path: str | Path) -> dict[str, list[RankedHit]]:
    grouped: dict[str, list[RankedHit]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            hit = RankedHit(
                benchmark=str(payload["benchmark"]),
                system=str(payload["system"]),
                query_id=str(payload["query_id"]),
                retrieved_id=str(payload["retrieved_id"]),
                rank=int(payload["rank"]),
                score=float(payload["score"]) if payload.get("score") is not None else None,
            )
            grouped[hit.query_id].append(hit)

    for query_id, hits in grouped.items():
        grouped[query_id] = sorted(hits, key=lambda item: item.rank)
    return grouped


def score_paths(targets_path: str | Path, trace_path: str | Path, k: int = 60) -> dict[str, Any]:
    query_records = load_query_records_jsonl(targets_path)
    ranked_hits = load_ranked_hits_jsonl(trace_path)
    query_scores = score_run(query_records, ranked_hits, k=k)
    return {
        "query_scores": query_scores,
        "aggregate": aggregate_scores(query_scores),
    }


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


def score_query(ranked_ids: list[str], targets: QueryTargets, k: int) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for target_name in ("raw", "source", "canonical"):
        target_ids = targets.target_ids(target_name)
        scores[target_name] = {
            "recall": recall_at_k(ranked_ids, target_ids, k),
            "mrr": mrr_at_k(ranked_ids, target_ids, k),
            "ndcg": ndcg_at_k(ranked_ids, target_ids, k),
        }
    return scores


def score_run(query_records: dict[str, QueryRecord], ranked_hits: dict[str, list[RankedHit]], k: int = 60) -> dict[str, Any]:
    query_scores: dict[str, Any] = {}
    for query_id, query_record in query_records.items():
        hits = ranked_hits.get(query_id)
        if not hits:
            continue
        ranked_ids = [hit.retrieved_id for hit in hits]
        query_scores[query_id] = {
            "benchmark": query_record.benchmark,
            "query_id": query_record.query_id,
            "category": query_record.category,
            "query_text": query_record.query_text,
            "scores": score_query(ranked_ids, query_record.targets, k),
        }
    return query_scores


def aggregate_scores(query_scores: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for target_name in ("raw", "source", "canonical"):
        metrics = {"recall": [], "mrr": [], "ndcg": []}
        for row in query_scores.values():
            for metric_name, metric_value in row["scores"][target_name].items():
                if not _is_nan(metric_value):
                    metrics[metric_name].append(metric_value)
        summary[target_name] = {
            metric_name: (sum(values) / len(values) if values else float("nan"))
            for metric_name, values in metrics.items()
        }
        summary[target_name]["queries"] = float(len(metrics["ndcg"]))
    return summary
