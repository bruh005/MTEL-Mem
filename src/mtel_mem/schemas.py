from __future__ import annotations

from dataclasses import dataclass, field


TARGET_NAMES = ("raw", "source", "canonical")


@dataclass(frozen=True)
class QueryTargets:
    raw_ids: frozenset[str] = frozenset()
    source_ids: frozenset[str] = frozenset()
    canonical_ids: frozenset[str] = frozenset()

    @classmethod
    def from_mapping(
        cls,
        *,
        raw_ids: list[str] | set[str] | tuple[str, ...],
        source_ids: list[str] | set[str] | tuple[str, ...],
        canonical_ids: list[str] | set[str] | tuple[str, ...],
    ) -> "QueryTargets":
        return cls(
            raw_ids=frozenset(raw_ids),
            source_ids=frozenset(source_ids),
            canonical_ids=frozenset(canonical_ids),
        )

    def target_ids(self, name: str) -> frozenset[str]:
        lookup = {
            "raw": self.raw_ids,
            "source": self.source_ids,
            "canonical": self.canonical_ids,
        }
        try:
            return lookup[name]
        except KeyError as exc:
            raise ValueError(f"unknown target name: {name}") from exc


@dataclass(frozen=True)
class QueryRecord:
    benchmark: str
    query_id: str
    source_fixture_id: str
    category: str
    query_text: str = ""
    reference_answer: str = ""
    targets: QueryTargets = field(default_factory=QueryTargets)


@dataclass(frozen=True)
class RankedHit:
    benchmark: str
    system: str
    query_id: str
    retrieved_id: str
    rank: int
    score: float | None = None
