"""Minimal internal schema required by the generic persistence primitive."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PersistenceSummary:
    numerator: int
    denominator: int
    proportion: float
