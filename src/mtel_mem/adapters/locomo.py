from __future__ import annotations

from pathlib import Path


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / "manifests" / "locomo.json"
