from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


_LOCOMO_CATEGORY_MAP = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal-reasoning",
    4: "open-domain-knowledge",
    5: "adversarial",
}


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _to_records(frame_like: Any) -> list[dict[str, Any]]:
    if isinstance(frame_like, list):
        return [dict(item) for item in frame_like]
    if hasattr(frame_like, "to_dict"):
        return list(frame_like.to_dict(orient="records"))
    raise TypeError(f"unsupported frame-like object: {type(frame_like)!r}")


def _normalize_tuple_key(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        return tuple(int(part) for part in value)
    if isinstance(value, list):
        return tuple(int(part) for part in value)
    raise TypeError(f"expected tuple/list key, got: {value!r}")


def _normalize_raw_id(value: Any) -> str:
    text = str(value)
    return text if text.startswith("raw_") else f"raw_{text}"


def _normalize_reason_id(value: Any) -> str:
    text = str(value)
    return text if text.startswith("reason_") else f"reason_{text}"


def _extract_sample_id(session_pool: list[str]) -> str:
    if not session_pool:
        raise ValueError("PREMem QA record is missing session_pool")
    first = str(session_pool[0])
    marker = "_session_"
    if marker not in first:
        raise ValueError(f"unexpected LoCoMo session id format: {first}")
    return first.split(marker, 1)[0]


def _build_qa_lookup(qa_pickle_path: str | Path) -> dict[str, dict[str, Any]]:
    payload = _read_pickle(qa_pickle_path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in payload:
        question_id = str(row["question_id"])
        lookup[question_id] = {
            "question_id": question_id,
            "question": str(row["question"]),
            "answer": str(row.get("answer", "")),
            "question_type": str(row["question_type"]),
            "sample_id": _extract_sample_id(list(row["session_pool"])),
            "others": dict(row.get("others", {})),
        }
    return lookup


def _build_locomo_lookup(locomo_json_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    payload = _read_json(locomo_json_path)
    lookup: dict[str, list[dict[str, Any]]] = {}
    for sample in payload:
        sample_rows: list[dict[str, Any]] = []
        for qa in sample["qa"]:
            sample_rows.append(
                {
                    "question": str(qa["question"]),
                    "answer": str(qa.get("answer", "")),
                    "question_type": _LOCOMO_CATEGORY_MAP[int(qa["category"])],
                    "evidence": [str(item) for item in qa.get("evidence", [])],
                    "adversarial_answer": str(qa.get("adversarial_answer", "")),
                }
            )
        lookup[str(sample["sample_id"])] = sample_rows
    return lookup


def _match_locomo_entry(
    qa_row: dict[str, Any],
    locomo_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [row for row in locomo_rows if row["question"] == qa_row["question"]]
    if len(candidates) == 1:
        return candidates[0]

    by_type = [row for row in candidates if row["question_type"] == qa_row["question_type"]]
    if len(by_type) == 1:
        return by_type[0]

    if qa_row["question_type"] == "adversarial":
        adv = str(qa_row["others"].get("adversarial_answer", ""))
        by_adv = [row for row in by_type if row["adversarial_answer"] == adv]
        if len(by_adv) == 1:
            return by_adv[0]
    else:
        by_answer = [row for row in by_type if row["answer"] == qa_row["answer"]]
        if len(by_answer) == 1:
            return by_answer[0]

    raise ValueError(
        f"could not uniquely match LoCoMo QA for sample={qa_row['sample_id']} "
        f"question={qa_row['question']!r}"
    )


def _build_reason_lineage_from_file(path: str | Path) -> dict[str, dict[str, set[str]]]:
    payload = _read_json(path)
    lineage: dict[str, dict[str, set[str]]] = {}
    for query_id, rows in payload.items():
        lineage[query_id] = {
            _normalize_reason_id(reason_id): {_normalize_raw_id(raw_id) for raw_id in raw_ids}
            for reason_id, raw_ids in rows.items()
        }
    return lineage


def _build_reason_lineage_from_caches(
    raw_session_pool_path: str | Path,
    raw_embeddings_path: str | Path,
    reasoning_session_pool_path: str | Path,
) -> dict[str, dict[str, set[str]]]:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise ImportError(
            "PREMem cache lineage reconstruction requires numpy, pandas, and scikit-learn. "
            "Use a PREMem-style environment or provide --canonical-lineage-json instead."
        ) from exc

    raw_session_pools = _read_pickle(raw_session_pool_path)
    raw_embeddings = _read_pickle(raw_embeddings_path)
    reasoning_session_pools = _read_pickle(reasoning_session_pool_path)

    def find_optimal_labels(embeddings: Any) -> Any:
        vectors = np.asarray(embeddings)
        best_labels = np.zeros(len(vectors), dtype=np.int32)
        best_score = -1.0
        try:
            for k in range(2, min(10, len(vectors))):
                labels = KMeans(n_clusters=k, random_state=42).fit_predict(vectors)
                score = silhouette_score(vectors, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
        except ValueError:
            return np.zeros(len(vectors), dtype=np.int32)
        return best_labels

    lineage: dict[str, dict[str, set[str]]] = {}
    for query_id, raw_frame in raw_session_pools.items():
        if query_id not in reasoning_session_pools:
            continue

        raw_df = raw_frame if hasattr(raw_frame, "iloc") else pd.DataFrame(_to_records(raw_frame))
        reasoning_df = (
            reasoning_session_pools[query_id]
            if hasattr(reasoning_session_pools[query_id], "iloc")
            else pd.DataFrame(_to_records(reasoning_session_pools[query_id]))
        )
        embeddings = np.asarray(raw_embeddings[query_id])

        session_ids = raw_df["session_id"].unique()
        session_dates = raw_df.drop_duplicates(subset=["session_id"])["session_date"]
        sorted_index = pd.to_datetime(
            session_dates,
            format="%Y-%m-%d %A %H:%M:%S",
        ).argsort()
        session_ids = session_ids[sorted_index]

        cluster_members: dict[tuple[int, int], set[str]] = {}
        for session_idx, session_id in enumerate(session_ids):
            session_df = raw_df[raw_df["session_id"] == session_id]
            session_vectors = embeddings[session_df.index]
            labels = find_optimal_labels(session_vectors)
            for cluster_id in sorted({int(value) for value in labels.tolist()}):
                positions = [offset for offset, value in enumerate(labels.tolist()) if int(value) == cluster_id]
                raw_ids = {
                    _normalize_raw_id(item)
                    for item in session_df.iloc[positions]["id"].tolist()
                }
                cluster_members[(session_idx, cluster_id)] = raw_ids

        query_lineage: dict[str, set[str]] = {}
        for row in reasoning_df.to_dict(orient="records"):
            reason_id = _normalize_reason_id(row["id"])
            idx_key = _normalize_tuple_key(row["idx_key"])
            remain_key = _normalize_tuple_key(row["remain_key"])
            query_lineage[reason_id] = set(cluster_members.get(idx_key, set()))
            query_lineage[reason_id].update(cluster_members.get(remain_key, set()))
        lineage[str(query_id)] = query_lineage
    return lineage


def export_premem_locomo(
    *,
    premem_results_path: str | Path,
    qa_pickle_path: str | Path,
    locomo_json_path: str | Path,
    raw_session_pool_path: str | Path,
    out_targets_path: str | Path,
    out_trace_path: str | Path,
    system_name: str = "premem",
    benchmark_name: str = "premem_locomo",
    canonical_lineage_json: str | Path | None = None,
    raw_embeddings_path: str | Path | None = None,
    reasoning_session_pool_path: str | Path | None = None,
) -> dict[str, Any]:
    qa_lookup = _build_qa_lookup(qa_pickle_path)
    locomo_lookup = _build_locomo_lookup(locomo_json_path)
    raw_session_pools = _read_pickle(raw_session_pool_path)
    results = _read_jsonl(premem_results_path)

    if canonical_lineage_json:
        reason_lineage = _build_reason_lineage_from_file(canonical_lineage_json)
    else:
        if raw_embeddings_path is None or reasoning_session_pool_path is None:
            raise ValueError(
                "provide either canonical_lineage_json or both raw_embeddings_path and "
                "reasoning_session_pool_path"
            )
        reason_lineage = _build_reason_lineage_from_caches(
            raw_session_pool_path=raw_session_pool_path,
            raw_embeddings_path=raw_embeddings_path,
            reasoning_session_pool_path=reasoning_session_pool_path,
        )

    target_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    raw_covered = 0
    canonical_covered = 0

    for result in results:
        query_id = str(result["question_id"])
        qa_row = qa_lookup[query_id]
        locomo_row = _match_locomo_entry(qa_row, locomo_lookup[qa_row["sample_id"]])
        evidence_ids = set(locomo_row["evidence"])

        raw_records = _to_records(raw_session_pools[query_id])
        raw_target_ids = {
            _normalize_raw_id(row["id"])
            for row in raw_records
            if str(row.get("message_id", "")) in evidence_ids
        }
        canonical_target_ids = {
            reason_id
            for reason_id, linked_raw_ids in reason_lineage.get(query_id, {}).items()
            if linked_raw_ids & raw_target_ids
        }
        source_target_ids = set(raw_target_ids)
        source_target_ids.update(canonical_target_ids)
        source_fixture_id = (
            "|".join(sorted(evidence_ids))
            if evidence_ids
            else f"unannotated:{qa_row['sample_id']}:{query_id}"
        )

        if raw_target_ids:
            raw_covered += 1
        if canonical_target_ids:
            canonical_covered += 1

        target_rows.append(
            {
                "benchmark": benchmark_name,
                "query_id": query_id,
                "source_fixture_id": source_fixture_id,
                "category": str(result.get("question_type", qa_row["question_type"])),
                "query_text": str(result["question"]),
                "reference_answer": str(result.get("answer", qa_row["answer"])),
                "raw_ids": sorted(raw_target_ids),
                "source_ids": sorted(source_target_ids),
                "canonical_ids": sorted(canonical_target_ids),
            }
        )

        for rank, item in enumerate(result.get("retrieved_results", []), start=1):
            trace_rows.append(
                {
                    "benchmark": benchmark_name,
                    "system": system_name,
                    "query_id": query_id,
                    "retrieved_id": str(item["id"]),
                    "rank": rank,
                    "score": item.get("score"),
                }
            )

    _write_jsonl(out_targets_path, target_rows)
    _write_jsonl(out_trace_path, trace_rows)

    return {
        "queries": len(target_rows),
        "trace_rows": len(trace_rows),
        "raw_target_coverage": raw_covered,
        "canonical_target_coverage": canonical_covered,
        "out_targets": str(Path(out_targets_path).resolve()),
        "out_trace": str(Path(out_trace_path).resolve()),
    }
