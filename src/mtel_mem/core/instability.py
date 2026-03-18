from __future__ import annotations

from typing import Any


def compute_instability(query_scores: dict[str, Any], left: str, right: str) -> dict[str, float]:
    hit_flips = 0
    top1_flips = 0
    ndcg_changed = 0
    shared_queries = 0

    for row in query_scores.values():
        left_scores = row["scores"][left]
        right_scores = row["scores"][right]
        if any(value != value for value in (*left_scores.values(), *right_scores.values())):
            continue

        shared_queries += 1
        if left_scores["recall"] != right_scores["recall"]:
            hit_flips += 1
        if (left_scores["mrr"] == 1.0) != (right_scores["mrr"] == 1.0):
            top1_flips += 1
        if left_scores["ndcg"] != right_scores["ndcg"]:
            ndcg_changed += 1

    return {
        "shared_queries": float(shared_queries),
        "hit_flips": float(hit_flips),
        "top1_flips": float(top1_flips),
        "ndcg_changed": float(ndcg_changed),
        "change_rate": (ndcg_changed / shared_queries) if shared_queries else 0.0,
    }
