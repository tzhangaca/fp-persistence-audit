"""Deterministic fixed-seed generative red-team battery for release gating."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from fp_persistence import (
    AuditStatus,
    AuditValidationError,
    ExposureSpec,
    PermutationSpec,
    audit,
    flags_from_scores,
)
import fp_persistence.public_api as public_api
from fp_persistence.public_types import RatioState


GENERATIVE_SEED = 2026101001
VALID_GENERATED_CASES = 1024
MALFORMED_GENERATED_CASES = 1024


def _science(result):
    diagnostics = result.diagnostics
    return (
        result.status,
        result.reason,
        result.population,
        result.counts,
        result.estimates,
        result.reference,
        diagnostics.entity_legitimate_exposure_distribution,
        diagnostics.final_stratum_count,
        diagnostics.final_stratum_size_distribution,
        diagnostics.minimum_final_stratum_size,
        diagnostics.maximum_final_stratum_size,
        diagnostics.singleton_final_stratum_count,
        diagnostics.eligible_entity_count_from_exposure_distribution,
        diagnostics.eligible_entity_count_matches_population,
    )


def test_deterministic_valid_generative_battery(monkeypatch):
    rng = np.random.default_rng(GENERATIVE_SEED)
    original_entity_results = public_api._entity_results
    expected_structure: dict[str, object] = {}
    checked_frames = 0

    def checked_entity_results(frame, *, exposure):
        nonlocal checked_frames
        realized = tuple(
            int(value)
            for value in frame.groupby("final_stratum", sort=True)[
                "false_positive"
            ].sum()
        )
        assert realized == expected_structure["stratum_flags"]
        assert int(frame["false_positive"].sum()) == expected_structure["total_flags"]
        checked_frames += 1
        return original_entity_results(frame, exposure=exposure)

    monkeypatch.setattr(public_api, "_entity_results", checked_entity_results)

    for case_index in range(VALID_GENERATED_CASES):
        event_count = int(rng.integers(1, 51))
        entity_count = int(rng.integers(1, min(10, event_count) + 1))
        stratum_count = int(rng.integers(1, min(5, event_count) + 1))
        entities = [f"entity-{value}" for value in range(entity_count)]
        strata = [f"stratum-{value}" for value in range(stratum_count)]
        entity_values = [entities[value % entity_count] for value in range(event_count)]
        stratum_values = [strata[value % stratum_count] for value in range(event_count)]
        rng.shuffle(entity_values)
        rng.shuffle(stratum_values)
        scores = rng.integers(-2, 5, size=event_count).tolist()
        frame = pd.DataFrame(
            {
                "event_id": [f"case-{case_index:04d}-event-{value:02d}" for value in range(event_count)],
                "entity": entity_values,
                "stratum": stratum_values,
                "legitimate": np.ones(event_count, dtype=np.int8),
                "score": scores,
            }
        )
        n_select = int(rng.integers(0, event_count + 1))
        selection = flags_from_scores(
            frame,
            score_col="score",
            event_id_col="event_id",
            legitimate_col="legitimate",
            n_select=n_select,
        )
        frame["flag"] = frame["event_id"].map(selection.flags_by_event_id).astype(int)
        exposure_min = int(rng.integers(1, 6))
        if bool(rng.integers(0, 2)):
            exposure_max = None
        else:
            exposure_max = int(rng.integers(exposure_min, 8))
        exposure = ExposureSpec(exposure_min, exposure_max)
        permutation = PermutationSpec(3, 100000 + case_index)

        expected_structure["stratum_flags"] = tuple(
            int(value)
            for value in frame.groupby("stratum", sort=True)["flag"].sum()
        )
        expected_structure["total_flags"] = int(frame["flag"].sum())

        score_result = audit(
            frame,
            entity_col="entity",
            event_id_col="event_id",
            final_stratum_col="stratum",
            legitimate_col="legitimate",
            flag_selection=selection,
            exposure=exposure,
            permutation=permutation,
        )
        external_result = audit(
            frame,
            entity_col="entity",
            event_id_col="event_id",
            final_stratum_col="stratum",
            legitimate_col="legitimate",
            flag_col="flag",
            exposure=exposure,
            permutation=permutation,
        )
        shuffled_result = audit(
            frame.sample(frac=1, random_state=case_index).copy(),
            entity_col="entity",
            event_id_col="event_id",
            final_stratum_col="stratum",
            legitimate_col="legitimate",
            flag_col="flag",
            exposure=exposure,
            permutation=permutation,
        )

        exposures = frame.groupby("entity", sort=False).size()
        fp_totals = frame.groupby("entity", sort=False)["flag"].sum()
        eligible = exposures >= exposure_min
        if exposure_max is not None:
            eligible &= exposures <= exposure_max
        eligible_count = int(eligible.sum())
        repeated_count = int(((fp_totals >= 2) & eligible).sum())
        exposure_distribution = {
            int(value): int(count)
            for value, count in exposures.value_counts().sort_index().items()
        }

        assert external_result.population.eligible_entity_count == eligible_count
        assert external_result.diagnostics.entity_legitimate_exposure_distribution == exposure_distribution
        assert external_result.diagnostics.eligible_entity_count_from_exposure_distribution == eligible_count
        assert external_result.diagnostics.eligible_entity_count_matches_population is True
        assert external_result.counts.false_positive_event_count == n_select
        assert external_result.counts.repeated_fp_entity_count == repeated_count
        assert _science(score_result) == _science(external_result)
        assert _science(shuffled_result) == _science(external_result)

        if eligible_count == 0:
            assert external_result.status is AuditStatus.NOT_ESTIMABLE
            assert external_result.reason == "zero_eligible_entities"
            assert external_result.estimates.observed_repeat_fp_proportion is None
            assert external_result.estimates.observed_expected_ratio.state is RatioState.NOT_ESTIMABLE
            continue

        assert external_result.status is AuditStatus.ESTIMABLE
        observed = repeated_count / eligible_count
        reference = external_result.estimates.reference_expected_repeat_fp_proportion
        assert external_result.estimates.observed_repeat_fp_proportion == observed
        assert reference is not None and 0.0 <= reference <= 1.0
        assert external_result.reference.interval_lower is not None
        assert external_result.reference.interval_upper is not None
        assert 0.0 <= external_result.reference.interval_lower <= 1.0
        assert 0.0 <= external_result.reference.interval_upper <= 1.0
        assert external_result.reference.interval_lower <= external_result.reference.interval_upper
        assert external_result.estimates.excess_repeat_fp_percentage_points == 100 * (
            observed - reference
        )
        assert external_result.counts.excess_repeated_fp_entity_count == (
            repeated_count - reference * eligible_count
        )

    assert checked_frames >= VALID_GENERATED_CASES * 3


class _CustomHashable:
    def __hash__(self):
        return 7


class _CustomEquality:
    def __hash__(self):
        return 7

    def __eq__(self, other):
        raise RuntimeError("must be rejected before equality")


class _CustomPartialOrder:
    def __hash__(self):
        return 7

    def __lt__(self, other):
        return False


class _CustomNumeric:
    def __float__(self):
        return 1.0


MALFORMED_IDENTIFIER_VALUES = (
    None,
    True,
    float("nan"),
    float("inf"),
    float("-inf"),
    Decimal("NaN"),
    Decimal("sNaN"),
    Decimal("Infinity"),
    [],
    {},
    set(),
    frozenset(),
    _CustomHashable(),
    _CustomEquality(),
    _CustomPartialOrder(),
    _CustomNumeric(),
    ("nested", float("nan")),
    ("nested", frozenset({1})),
)

MALFORMED_SCORE_VALUES = MALFORMED_IDENTIFIER_VALUES + (
    1 + 2j,
    "1",
    b"1",
)

MALFORMED_BINARY_VALUES = (
    0.0,
    1.0,
    2,
    -1,
    None,
    float("nan"),
    float("inf"),
    Decimal("1"),
    "1",
    b"1",
    [],
    {},
    set(),
    frozenset(),
    _CustomHashable(),
    _CustomEquality(),
    _CustomPartialOrder(),
    _CustomNumeric(),
)


def _malformed_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": pd.Series(["e0", "e1", "e2", "e3"], dtype=object),
            "entity": pd.Series(["A", "A", "B", "B"], dtype=object),
            "stratum": pd.Series(["s0", "s0", "s1", "s1"], dtype=object),
            "legitimate": pd.Series([1, 1, 1, 1], dtype=object),
            "flag": pd.Series([1, 0, 1, 0], dtype=object),
            "score": pd.Series([4, 3, 2, 1], dtype=object),
        }
    )


def test_deterministic_malformed_generative_battery_has_no_raw_exception_leaks():
    roles = ("event_id", "entity", "stratum", "score", "flag", "legitimate")
    for case_index in range(MALFORMED_GENERATED_CASES):
        frame = _malformed_base()
        role = roles[case_index % len(roles)]
        domain = (
            MALFORMED_SCORE_VALUES
            if role == "score"
            else MALFORMED_BINARY_VALUES
            if role in {"flag", "legitimate"}
            else MALFORMED_IDENTIFIER_VALUES
        )
        bad_value = domain[(case_index // len(roles)) % len(domain)]
        values = frame[role].tolist()
        values[0] = bad_value
        frame[role] = pd.Series(values, dtype=object)
        with pytest.raises(AuditValidationError):
            if role == "score":
                flags_from_scores(
                    frame,
                    score_col="score",
                    event_id_col="event_id",
                    legitimate_col="legitimate",
                    n_select=2,
                )
            else:
                audit(
                    frame,
                    entity_col="entity",
                    event_id_col="event_id",
                    final_stratum_col="stratum",
                    legitimate_col="legitimate",
                    flag_col="flag",
                    exposure=ExposureSpec(2, 2),
                    permutation=PermutationSpec(2, GENERATIVE_SEED + case_index),
                )
