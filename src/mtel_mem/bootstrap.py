from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any


LONGMEMEVAL_S_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_s_cleaned.json"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_locomo(
    locomo: list[dict[str, Any]],
    *,
    mode: str = "paperlite",
    sanitize_percent: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fixture: list[dict[str, Any]] = []
    eval_set: list[dict[str, Any]] = []
    skipped_no_evidence = 0
    skipped_missing_dialogues = 0

    def session_number(key: str) -> int:
        if not key.startswith("session_") or key.endswith("_date_time"):
            return 10**9
        rest = key[len("session_") :]
        if not rest.isdigit():
            return 10**9
        return int(rest)

    for sample in locomo:
        sample_id = str(sample.get("sample_id", "")).strip()
        if not sample_id:
            continue
        tenant_id = f"locomo_{sample_id}"
        conversation = sample.get("conversation", {})
        if not isinstance(conversation, dict):
            continue

        speaker_a = str(conversation.get("speaker_a", "")).strip()
        speaker_b = str(conversation.get("speaker_b", "")).strip()
        dia_to_fixture_idx: dict[str, int] = {}

        session_keys = [key for key in conversation.keys() if session_number(key) != 10**9]
        session_keys.sort(key=session_number)
        for session_key in session_keys:
            session_time = str(conversation.get(f"{session_key}_date_time", "")).strip()
            turns = conversation.get(session_key)
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                dia_id = str(turn.get("dia_id", "")).strip()
                speaker = str(turn.get("speaker", "")).strip()
                text = str(turn.get("text", "")).strip()
                if not dia_id or not text:
                    continue
                if mode == "paperlite":
                    content = (
                        f"[sample:{sample_id}] [dialog:{dia_id}] [time:{session_time}] "
                        f"[speaker_a:{speaker_a}] [speaker_b:{speaker_b}] "
                        f"{speaker}: {text}"
                    )
                else:
                    content = f"{speaker}: {text}" if speaker else text
                if sanitize_percent:
                    content = content.replace("%", " percent ")
                idx = len(fixture)
                dia_to_fixture_idx[dia_id] = idx
                fixture.append(
                    {
                        "tenant_id": tenant_id,
                        "content": content,
                        "tags": ["locomo", mode],
                        "tier": "episodic",
                    }
                )

        qa_items = sample.get("qa", [])
        if not isinstance(qa_items, list):
            continue
        for qa in qa_items:
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question", "")).strip()
            if not question:
                continue
            evidence = qa.get("evidence", [])
            if not isinstance(evidence, list) or not evidence:
                skipped_no_evidence += 1
                continue
            expected_indexes = sorted(
                {
                    dia_to_fixture_idx[eid]
                    for eid in evidence
                    if isinstance(eid, str) and eid in dia_to_fixture_idx
                }
            )
            if not expected_indexes:
                skipped_missing_dialogues += 1
                continue
            eval_set.append(
                {
                    "tenant_id": tenant_id,
                    "query": question,
                    "expected_fixture_indexes": expected_indexes,
                    "category": qa.get("category"),
                    "reference_answer": qa.get("answer", ""),
                }
            )

    stats = {
        "locomo_samples": len(locomo),
        "fixture_rows": len(fixture),
        "eval_rows": len(eval_set),
        "skipped_no_evidence": skipped_no_evidence,
        "skipped_missing_dialogues": skipped_missing_dialogues,
        "mode": mode,
        "sanitize_percent": sanitize_percent,
    }
    return fixture, eval_set, stats


def build_longmemeval_s(
    rows: list[dict[str, Any]],
    *,
    mode: str = "paperlite",
    sanitize_percent: bool = True,
    include_abstention: bool = False,
    fallback_to_answer_sessions: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fixture: list[dict[str, Any]] = []
    eval_set: list[dict[str, Any]] = []
    skipped_missing_question_id = 0
    skipped_abstention = 0
    skipped_missing_history = 0
    skipped_no_evidence = 0
    fallback_rows_used = 0

    for sample in rows:
        question_id = str(sample.get("question_id", "")).strip()
        if not question_id:
            skipped_missing_question_id += 1
            continue
        if not include_abstention and question_id.endswith("_abs"):
            skipped_abstention += 1
            continue

        haystack_sessions = sample.get("haystack_sessions", [])
        if not isinstance(haystack_sessions, list) or not haystack_sessions:
            skipped_missing_history += 1
            continue

        haystack_session_ids = sample.get("haystack_session_ids", [])
        haystack_dates = sample.get("haystack_dates", [])
        local_fixture: list[dict[str, Any]] = []
        local_answer_turn_indexes: list[int] = []
        local_session_to_turn_indexes: dict[str, list[int]] = {}
        question_type = str(sample.get("question_type", "")).strip()
        tenant_id = f"longmemeval_{question_id}"

        for session_idx, session in enumerate(haystack_sessions):
            if not isinstance(session, list):
                continue
            session_id = ""
            if isinstance(haystack_session_ids, list) and session_idx < len(haystack_session_ids):
                session_id = str(haystack_session_ids[session_idx]).strip()
            if not session_id:
                session_id = f"session_{session_idx + 1}"
            session_date = ""
            if isinstance(haystack_dates, list) and session_idx < len(haystack_dates):
                session_date = str(haystack_dates[session_idx]).strip()

            for turn_idx, turn in enumerate(session):
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role", "")).strip()
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                if mode == "paperlite":
                    rendered = (
                        f"[question:{question_id}] [session:{session_id}] "
                        f"[date:{session_date}] [turn:{turn_idx + 1}] "
                        f"[role:{role}] {content}"
                    )
                else:
                    rendered = f"{role}: {content}" if role else content
                if sanitize_percent:
                    rendered = rendered.replace("%", " percent ")

                local_index = len(local_fixture)
                local_fixture.append(
                    {
                        "tenant_id": tenant_id,
                        "content": rendered,
                        "tags": ["longmemeval", mode, question_type or "unknown_type"],
                        "tier": "episodic",
                    }
                )
                local_session_to_turn_indexes.setdefault(session_id, []).append(local_index)
                if turn.get("has_answer") is True:
                    local_answer_turn_indexes.append(local_index)

        if not local_fixture:
            skipped_missing_history += 1
            continue

        expected_local_indexes = sorted(set(local_answer_turn_indexes))
        if not expected_local_indexes and fallback_to_answer_sessions:
            answer_session_ids = sample.get("answer_session_ids", [])
            fallback: list[int] = []
            if isinstance(answer_session_ids, list):
                for answer_sid in answer_session_ids:
                    sid = str(answer_sid).strip()
                    if sid:
                        fallback.extend(local_session_to_turn_indexes.get(sid, []))
            expected_local_indexes = sorted(set(fallback))
            if expected_local_indexes:
                fallback_rows_used += 1

        if not expected_local_indexes:
            skipped_no_evidence += 1
            continue

        offset = len(fixture)
        fixture.extend(local_fixture)
        eval_set.append(
            {
                "tenant_id": tenant_id,
                "query": str(sample.get("question", "")).strip(),
                "expected_fixture_indexes": [offset + idx for idx in expected_local_indexes],
                "category": question_type,
                "reference_answer": str(sample.get("answer", "")).strip(),
                "question_id": question_id,
                "question_date": str(sample.get("question_date", "")).strip(),
            }
        )

    stats = {
        "longmemeval_rows": len(rows),
        "fixture_rows": len(fixture),
        "eval_rows": len(eval_set),
        "skipped_missing_question_id": skipped_missing_question_id,
        "skipped_abstention": skipped_abstention,
        "skipped_missing_history": skipped_missing_history,
        "skipped_no_evidence": skipped_no_evidence,
        "fallback_rows_used": fallback_rows_used,
        "mode": mode,
        "sanitize_percent": sanitize_percent,
        "include_abstention": include_abstention,
        "fallback_to_answer_sessions": fallback_to_answer_sessions,
    }
    return fixture, eval_set, stats


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _download_file(url: str, out_path: Path, *, force: bool = False) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return {"downloaded": False, "path": str(out_path)}

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(out_path)
    return {"downloaded": True, "path": str(out_path)}


def ensure_locomo(*, force: bool = False) -> dict[str, Any]:
    root = repo_root()
    source_path = root / "data" / "locomo10.json"
    fixture_path = root / "data" / "locomo10.paperlite.fixture.json"
    eval_path = root / "data" / "locomo10.paperlite.eval.json"
    stats_path = root / "data" / "locomo10.paperlite.stats.json"

    if not source_path.exists():
        return {"benchmark": "locomo", "status": "missing-source", "source_json": str(source_path)}

    if not force and fixture_path.exists() and eval_path.exists() and stats_path.exists():
        return {
            "benchmark": "locomo",
            "status": "ready",
            "source_json": str(source_path),
            "fixture_json": str(fixture_path),
            "eval_json": str(eval_path),
            "stats_json": str(stats_path),
            "generated": False,
        }

    rows = json.loads(source_path.read_text(encoding="utf-8"))
    fixture, eval_set, stats = build_locomo(rows)
    _write_json(fixture_path, fixture)
    _write_json(eval_path, eval_set)
    _write_json(stats_path, stats)
    return {
        "benchmark": "locomo",
        "status": "ready",
        "source_json": str(source_path),
        "fixture_json": str(fixture_path),
        "eval_json": str(eval_path),
        "stats_json": str(stats_path),
        "generated": True,
    }


def ensure_longmemeval_s(*, force: bool = False, download: bool = False) -> dict[str, Any]:
    root = repo_root()
    source_path = root / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
    fixture_path = root / "data" / "longmemeval" / "longmemeval_s_cleaned.paperlite.noabs.fixture.json"
    eval_path = root / "data" / "longmemeval" / "longmemeval_s_cleaned.paperlite.noabs.eval.json"
    stats_path = root / "data" / "longmemeval" / "longmemeval_s_cleaned.paperlite.noabs.stats.json"

    download_summary: dict[str, Any] | None = None
    if not source_path.exists() and download:
        download_summary = _download_file(LONGMEMEVAL_S_URL, source_path, force=force)

    if not source_path.exists():
        return {
            "benchmark": "longmemeval_s",
            "status": "missing-source",
            "source_json": str(source_path),
            "download_url": LONGMEMEVAL_S_URL,
        }

    if not force and fixture_path.exists() and eval_path.exists() and stats_path.exists():
        return {
            "benchmark": "longmemeval_s",
            "status": "ready",
            "source_json": str(source_path),
            "fixture_json": str(fixture_path),
            "eval_json": str(eval_path),
            "stats_json": str(stats_path),
            "generated": False,
            "download": download_summary,
        }

    rows = json.loads(source_path.read_text(encoding="utf-8"))
    fixture, eval_set, stats = build_longmemeval_s(rows)
    _write_json(fixture_path, fixture)
    _write_json(eval_path, eval_set)
    _write_json(stats_path, stats)
    return {
        "benchmark": "longmemeval_s",
        "status": "ready",
        "source_json": str(source_path),
        "fixture_json": str(fixture_path),
        "eval_json": str(eval_path),
        "stats_json": str(stats_path),
        "generated": True,
        "download": download_summary,
    }


def ensure_benchmark_assets(name: str, *, force: bool = False, download: bool = False) -> dict[str, Any]:
    if name == "locomo":
        return ensure_locomo(force=force)
    if name == "longmemeval_s":
        return ensure_longmemeval_s(force=force, download=download)
    return {"benchmark": name, "status": "unknown"}
