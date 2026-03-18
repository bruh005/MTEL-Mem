from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkManifest:
    manifest_path: Path
    benchmark_name: str
    display_name: str
    description: str
    source_json: Path
    fixture_json: Path
    eval_json: Path
    stats_json: Path
    trace_examples: tuple[Path, ...]
    expected_counts: dict[str, int]


def _resolve_path(base: Path, raw: str) -> Path:
    return (base / raw).resolve()


def load_manifest(path: str | Path) -> BenchmarkManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    return BenchmarkManifest(
        manifest_path=manifest_path,
        benchmark_name=str(payload["benchmark_name"]),
        display_name=str(payload.get("display_name", payload["benchmark_name"])),
        description=str(payload.get("description", "")),
        source_json=_resolve_path(base, payload["source_json"]),
        fixture_json=_resolve_path(base, payload["fixture_json"]),
        eval_json=_resolve_path(base, payload["eval_json"]),
        stats_json=_resolve_path(base, payload["stats_json"]),
        trace_examples=tuple(_resolve_path(base, item) for item in payload.get("trace_examples", [])),
        expected_counts={str(k): int(v) for k, v in payload.get("expected_counts", {}).items()},
    )


def validate_manifest(manifest: BenchmarkManifest) -> dict[str, Any]:
    paths = {
        "source_json": manifest.source_json,
        "fixture_json": manifest.fixture_json,
        "eval_json": manifest.eval_json,
        "stats_json": manifest.stats_json,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    trace_missing = [str(path) for path in manifest.trace_examples if not path.exists()]
    stats_payload = {}
    if manifest.stats_json.exists():
        stats_payload = json.loads(manifest.stats_json.read_text(encoding="utf-8"))

    counts = {
        "fixture_rows": int(stats_payload.get("fixture_rows", -1)),
        "eval_rows": int(stats_payload.get("eval_rows", -1)),
    }
    counts_match = all(
        counts.get(key, -1) == expected
        for key, expected in manifest.expected_counts.items()
        if counts.get(key, -1) >= 0
    )

    return {
        "benchmark_name": manifest.benchmark_name,
        "display_name": manifest.display_name,
        "missing_files": missing,
        "missing_trace_examples": trace_missing,
        "counts": counts,
        "expected_counts": manifest.expected_counts,
        "counts_match": counts_match,
        "manifest_path": str(manifest.manifest_path),
    }
