from __future__ import annotations

from math import log2


def recall_at_k(ranked_ids: list[str], target_ids: set[str] | frozenset[str], k: int) -> float:
    if not target_ids:
        return float("nan")
    return 1.0 if any(rid in target_ids for rid in ranked_ids[:k]) else 0.0


def mrr_at_k(ranked_ids: list[str], target_ids: set[str] | frozenset[str], k: int) -> float:
    if not target_ids:
        return float("nan")
    for idx, rid in enumerate(ranked_ids[:k], start=1):
        if rid in target_ids:
            return 1.0 / idx
    return 0.0


def dcg_binary(ranked_ids: list[str], target_ids: set[str] | frozenset[str], k: int) -> float:
    score = 0.0
    for idx, rid in enumerate(ranked_ids[:k], start=1):
        if rid in target_ids:
            score += 1.0 / log2(idx + 1)
    return score


def idcg_binary(num_relevant: int, k: int) -> float:
    return sum(1.0 / log2(idx + 1) for idx in range(1, min(num_relevant, k) + 1))


def ndcg_at_k(ranked_ids: list[str], target_ids: set[str] | frozenset[str], k: int) -> float:
    if not target_ids:
        return float("nan")
    ideal = idcg_binary(len(target_ids), k)
    if ideal == 0:
        return float("nan")
    return dcg_binary(ranked_ids, target_ids, k) / ideal
