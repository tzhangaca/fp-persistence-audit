"""Metamorphic release gates for scientifically irrelevant transformations."""

from __future__ import annotations

from decimal import Decimal
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from fp_persistence import ExposureSpec, PermutationSpec, audit, flags_from_scores


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [f"e{index:02d}" for index in range(12)],
            "entity": ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3,
            "stratum": ["s0", "s1"] * 6,
            "legitimate": [1] * 12,
            "flag": [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            "score": [3, 3, 2, 1, 1, 0, 3, 3, 2, 1, 1, 0],
        }
    )


def _run(frame: pd.DataFrame, *, metadata=None, selection=None):
    return audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        legitimate_col="legitimate",
        flag_col=None if selection is not None else "flag",
        flag_selection=selection,
        exposure=ExposureSpec(3, 3),
        permutation=PermutationSpec(120, 2026091601),
        metadata=metadata,
    )


def _science(result):
    return (
        result.status,
        result.reason,
        result.population.legitimate_event_count,
        result.population.entity_count_before_exposure_filter,
        result.population.eligible_entity_count,
        result.counts,
        result.estimates,
        result.reference,
        result.diagnostics.entity_legitimate_exposure_distribution,
        result.diagnostics.final_stratum_count,
        result.diagnostics.final_stratum_size_distribution,
        result.diagnostics.minimum_final_stratum_size,
        result.diagnostics.maximum_final_stratum_size,
        result.diagnostics.singleton_final_stratum_count,
    )


def test_row_index_and_column_order_metamorphisms_are_exact():
    frame = _frame()
    expected = _science(_run(frame))
    for seed in range(12):
        transformed = frame.sample(frac=1, random_state=seed).copy()
        transformed.index = [f"arbitrary-{value * 17 + 3}" for value in range(len(frame))]
        transformed = transformed[list(reversed(transformed.columns))]
        assert _science(_run(transformed)) == expected


def test_appending_and_shuffling_nonlegitimate_rows_does_not_change_science():
    frame = _frame()
    excluded = pd.DataFrame(
        {
            "event_id": ["excluded-z", "excluded-a", "excluded-m"],
            "entity": [None, {"unsupported": "ignored"}, object()],
            "stratum": [None, frozenset({"ignored"}), object()],
            "legitimate": [0, False, np.int8(0)],
            "flag": [99, None, "ignored"],
            "score": [np.nan, object(), "ignored"],
        }
    )
    expected = _science(_run(frame))
    for seed in (1, 9, 33):
        augmented = pd.concat(
            [frame, excluded.sample(frac=1, random_state=seed)], ignore_index=True
        ).sample(frac=1, random_state=seed + 100)
        assert _science(_run(augmented)) == expected


def test_score_selection_and_external_flags_are_scientifically_identical():
    frame = _frame()
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        n_select=4,
    )
    score_result = _run(frame, selection=selection)
    explicit = frame.copy()
    explicit["flag"] = explicit["event_id"].map(selection.flags_by_event_id).astype(int)
    external_result = _run(explicit)
    assert _science(score_result) == _science(external_result)


def test_metadata_insertion_order_does_not_change_serialized_meaning():
    first = _run(_frame(), metadata={"alpha": 1, "nested": {"x": 2, "y": 3}}).to_dict()
    second = _run(_frame(), metadata={"nested": {"y": 3, "x": 2}, "alpha": 1}).to_dict()
    first["provenance"].pop("execution_timestamp_utc")
    second["provenance"].pop("execution_timestamp_utc")
    assert first == second


def test_equality_equivalent_strata_have_identical_science():
    frame = _frame()
    baseline = frame.assign(stratum=pd.Series([1] * 6 + [2] * 6, dtype=object))
    equivalent = frame.assign(
        stratum=pd.Series(
            [Decimal("1"), 1.0, 1, Decimal("1.00"), 1.0, 1]
            + [Decimal("2"), 2.0, 2, Decimal("2.00"), 2.0, 2],
            dtype=object,
        )
    )
    assert _science(_run(baseline)) == _science(_run(equivalent))


def test_identical_calls_repeat_exactly_and_do_not_consume_global_numpy_rng():
    frame = _frame()
    np.random.seed(8675309)
    before = np.random.get_state()
    first = _run(frame)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.random.seed(123)
    second = _run(frame)
    assert _science(first) == _science(second)


def test_audit_does_not_mutate_input():
    frame = _frame()
    before = frame.copy(deep=True)
    _run(frame, metadata={"purpose": "immutability"})
    pd.testing.assert_frame_equal(frame, before)


def test_pythonhashseed_does_not_change_scientific_output():
    script = r'''
import json
import pandas as pd
from fp_persistence import ExposureSpec, PermutationSpec, audit
frame = pd.DataFrame({
    "event_id": [f"e{i:02d}" for i in range(12)],
    "entity": ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3,
    "stratum": [("z", 2), ("a", 1)] * 6,
    "legitimate": [1] * 12,
    "flag": [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
})
result = audit(
    frame, entity_col="entity", event_id_col="event_id",
    final_stratum_col="stratum", legitimate_col="legitimate", flag_col="flag",
    exposure=ExposureSpec(3, 3), permutation=PermutationSpec(120, 2026091601),
).to_dict()
result["provenance"].pop("execution_timestamp_utc")
print(json.dumps(result, sort_keys=True))
'''
    outputs = []
    for seed in ("0", "1", "77", "random"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script], env=environment, text=True
            ).strip()
        )
    assert len(set(outputs)) == 1
