"""Independent exact-enumeration oracles for small conditional-null fixtures."""

from __future__ import annotations

from itertools import combinations, product
import math

import numpy as np
import pandas as pd
import pytest

from fp_persistence import ExposureSpec, PermutationSpec, audit


ORACLE_SEED = 2026101002
ORACLE_REPLICATES = 2000


def _frame(entities, strata, flags) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [f"e{index:02d}" for index in range(len(entities))],
            "entity": entities,
            "stratum": strata,
            "legitimate": [1] * len(entities),
            "flag": flags,
        }
    )


def _exact_reference(frame: pd.DataFrame, exposure: ExposureSpec) -> np.ndarray:
    """Enumerate the null without calling any production permutation function."""
    entities = list(dict.fromkeys(frame["entity"].tolist()))
    exposures = {
        entity: sum(value == entity for value in frame["entity"].tolist())
        for entity in entities
    }
    eligible = [
        entity
        for entity in entities
        if exposures[entity] >= exposure.min_legitimate_events
        and (
            exposure.max_legitimate_events is None
            or exposures[entity] <= exposure.max_legitimate_events
        )
    ]
    assert eligible
    groups = []
    for stratum in dict.fromkeys(frame["stratum"].tolist()):
        positions = [
            index for index, value in enumerate(frame["stratum"].tolist()) if value == stratum
        ]
        fixed_count = sum(int(frame.iloc[position]["flag"]) for position in positions)
        groups.append(tuple(combinations(positions, fixed_count)))
    values = []
    entity_values = frame["entity"].tolist()
    for allocation in product(*groups):
        selected = {position for group in allocation for position in group}
        repeated = 0
        for entity in eligible:
            entity_fp = sum(
                position in selected
                for position, value in enumerate(entity_values)
                if value == entity
            )
            repeated += int(entity_fp >= 2)
        values.append(repeated / len(eligible))
    return np.asarray(values, dtype=float)


FIXTURES = [
    pytest.param(
        _frame(
            ["A", "A", "B", "B", "C", "C"],
            ["one"] * 6,
            [1, 1, 0, 0, 1, 0],
        ),
        ExposureSpec(2, 2),
        id="one-stratum",
    ),
    pytest.param(
        _frame(
            ["A", "A", "B", "B", "C", "C", "D", "D"],
            ["left"] * 4 + ["right"] * 4,
            [1, 0, 1, 0, 1, 0, 1, 0],
        ),
        ExposureSpec(2, 2),
        id="two-strata-dispersed",
    ),
    pytest.param(
        _frame(
            ["A", "A", "B", "B", "C", "C", "D", "D"],
            ["left"] * 4 + ["right"] * 4,
            [1, 1, 0, 0, 1, 1, 0, 0],
        ),
        ExposureSpec(2, 2),
        id="two-strata-concentrated",
    ),
    pytest.param(
        _frame(
            ["A", "A", "B", "B", "C"],
            ["main", "main", "main", "main", "singleton"],
            [1, 0, 1, 0, 1],
        ),
        ExposureSpec(1, 2),
        id="singleton-stratum",
    ),
    pytest.param(
        _frame(
            ["A", "A", "B", "B"],
            ["left", "left", "right", "right"],
            [0, 0, 0, 0],
        ),
        ExposureSpec(2, None),
        id="zero-flags",
    ),
    pytest.param(
        _frame(
            ["A", "A", "B", "B"],
            ["left", "left", "right", "right"],
            [1, 1, 1, 1],
        ),
        ExposureSpec(2, None),
        id="all-flags",
    ),
    pytest.param(
        _frame(
            ["A", "A", "A", "B", "B", "C"],
            ["x", "x", "y", "x", "y", "y"],
            [1, 0, 1, 0, 1, 0],
        ),
        ExposureSpec(1, None),
        id="unequal-entity-exposures",
    ),
]


@pytest.mark.parametrize("frame,exposure", FIXTURES)
def test_production_monte_carlo_agrees_with_independent_exact_oracle(frame, exposure):
    exact = _exact_reference(frame, exposure)
    result = audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        legitimate_col="legitimate",
        flag_col="flag",
        exposure=exposure,
        permutation=PermutationSpec(ORACLE_REPLICATES, ORACLE_SEED),
    )
    exact_mean = float(exact.mean())
    exact_standard_error = float(exact.std(ddof=0) / math.sqrt(ORACLE_REPLICATES))
    tolerance = 4.0 * exact_standard_error + np.finfo(float).eps * 8
    assert result.estimates.reference_expected_repeat_fp_proportion == pytest.approx(
        exact_mean, abs=tolerance
    )
    if exact_standard_error == 0:
        assert result.estimates.reference_expected_repeat_fp_proportion == exact_mean


def test_quick_start_exact_and_fixed_seed_answers_are_preserved():
    entities = ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3
    frame = _frame(
        entities,
        ["day-1"] * 6 + ["day-2"] * 6,
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    )
    exact = _exact_reference(frame, ExposureSpec(3, 3))
    result = audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        legitimate_col="legitimate",
        flag_col="flag",
        exposure=ExposureSpec(3, 3),
        permutation=PermutationSpec(500, 2026091601),
    )
    assert len(exact) == 225
    assert float(exact.mean()) == pytest.approx(0.20)
    assert result.estimates.reference_expected_repeat_fp_proportion == pytest.approx(0.19)
