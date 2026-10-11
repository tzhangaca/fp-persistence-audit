"""Cross-product release gates for supported representations and modes."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, DefaultContext, FloatOperation, localcontext
from fractions import Fraction
from itertools import combinations
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from fp_persistence import (
    AuditStatus,
    AuditValidationError,
    ExposureSpec,
    FlagSelection,
    PermutationSpec,
    audit,
    flags_from_scores,
)


def _object_series(values, *, name=None) -> pd.Series:
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return pd.Series(array, name=name, dtype=object)


def _science(result):
    return (
        result.status,
        result.reason,
        result.population,
        result.counts,
        result.estimates,
        result.reference,
        result.diagnostics.entity_legitimate_exposure_distribution,
        result.diagnostics.final_stratum_count,
        result.diagnostics.final_stratum_size_distribution,
        result.diagnostics.minimum_final_stratum_size,
        result.diagnostics.maximum_final_stratum_size,
        result.diagnostics.singleton_final_stratum_count,
        result.diagnostics.eligible_entity_count_from_exposure_distribution,
        result.diagnostics.eligible_entity_count_matches_population,
    )


def _audit(frame: pd.DataFrame, *, selection=None, replicates=25, seed=314159):
    return audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        legitimate_col="legitimate",
        flag_col="flag" if selection is None else None,
        flag_selection=selection,
        exposure=ExposureSpec(2, 2),
        permutation=PermutationSpec(replicates, seed),
    )


EVENT_ID_REPRESENTATIONS = [
    pytest.param(["a", "b", "c", "d"], id="str"),
    pytest.param([1, 2, 3, 4], id="int"),
    pytest.param([b"a", b"b", b"c", b"d"], id="bytes"),
    pytest.param([UUID(int=value) for value in range(1, 5)], id="uuid"),
    pytest.param([date(2026, 1, value) for value in range(1, 5)], id="date"),
    pytest.param(
        [datetime(2026, 1, value, 1, 2, 3) for value in range(1, 5)],
        id="datetime",
    ),
    pytest.param(
        [pd.Timestamp(2026, 1, value, 1, 2, 3) for value in range(1, 5)],
        id="pandas-timestamp",
    ),
    pytest.param([0.25, 0.5, 0.75, 1.0], id="finite-float"),
    pytest.param(
        [Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1")],
        id="decimal",
    ),
    pytest.param([np.int8(1), np.int8(2), np.int8(3), np.int8(4)], id="np-int8"),
    pytest.param(
        [np.int32(1), np.int32(2), np.int32(3), np.int32(4)], id="np-int32"
    ),
    pytest.param(
        [np.int64(1), np.int64(2), np.int64(3), np.int64(4)], id="np-int64"
    ),
    pytest.param(
        [np.uint64(1), np.uint64(2), np.uint64(3), np.uint64(4)], id="np-uint64"
    ),
    pytest.param(
        [np.float16(0.25), np.float16(0.5), np.float16(0.75), np.float16(1)],
        id="np-float16",
    ),
    pytest.param(
        [np.float32(0.25), np.float32(0.5), np.float32(0.75), np.float32(1)],
        id="np-float32",
    ),
    pytest.param(
        [np.float64(0.25), np.float64(0.5), np.float64(0.75), np.float64(1)],
        id="np-float64",
    ),
    pytest.param([np.str_(value) for value in ("a", "b", "c", "d")], id="np-str"),
    pytest.param(
        [np.bytes_(value) for value in (b"a", b"b", b"c", b"d")],
        id="np-bytes",
    ),
    pytest.param([("x", value) for value in range(1, 5)], id="tuple"),
    pytest.param(
        [("x", (value, b"z")) for value in range(1, 5)], id="nested-tuple"
    ),
]


def _event_frame(event_ids) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": _object_series(event_ids),
            "entity": _object_series(["A", "A", "B", "B"]),
            "stratum": _object_series(["s0", "s0", "s1", "s1"]),
            "legitimate": [1, 1, 1, 1],
            "flag": [1, 0, 1, 0],
            "score": _object_series([4, 1, 3, 2]),
        }
    )


@pytest.mark.parametrize("event_ids", EVENT_ID_REPRESENTATIONS)
def test_event_id_representation_crosses_both_modes_and_row_order(event_ids):
    frame = _event_frame(event_ids)
    external = _audit(frame)
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        n_select=2,
    )
    score_mode = _audit(frame, selection=selection)

    explicit_flags = {
        event_id: int(flag)
        for event_id, flag in selection.flags_by_event_id.items()
    }
    equivalent = frame.copy()
    equivalent["flag"] = [explicit_flags[event_id] for event_id in frame["event_id"]]
    equivalent_external = _audit(equivalent)
    assert _science(score_mode) == _science(equivalent_external)
    assert external.status is AuditStatus.ESTIMABLE

    shuffled = frame.sample(frac=1, random_state=6102)
    shuffled_selection = flags_from_scores(
        shuffled,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        n_select=2,
    )
    assert shuffled_selection.selected_event_ids == selection.selected_event_ids
    pd.testing.assert_series_equal(
        shuffled_selection.flags_by_event_id, selection.flags_by_event_id
    )
    assert (
        shuffled_selection.selected_event_ids_sha256
        == selection.selected_event_ids_sha256
    )
    assert (
        shuffled_selection.flags_by_event_id_sha256
        == selection.flags_by_event_id_sha256
    )
    assert _science(_audit(shuffled)) == _science(external)
    assert _science(_audit(shuffled, selection=shuffled_selection)) == _science(
        score_mode
    )


@pytest.mark.parametrize(
    "event_ids",
    [
        pytest.param([("x", value) for value in range(1, 5)], id="tuple"),
        pytest.param(
            [("x", (value, b"z")) for value in range(1, 5)], id="nested-tuple"
        ),
    ],
)
@pytest.mark.parametrize(
    "hash_field", ["selected_event_ids_sha256", "flags_by_event_id_sha256"]
)
def test_tuple_event_id_hash_tampering_is_rejected(event_ids, hash_field):
    frame = _event_frame(event_ids)
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        n_select=2,
    )
    assert type(selection.flags_by_event_id.index) is pd.Index
    assert selection.flags_by_event_id.index.tolist() == sorted(event_ids)
    tampered = replace(selection, **{hash_field: "0" * 64})
    with pytest.raises(AuditValidationError, match=hash_field):
        _audit(frame, selection=tampered)


@pytest.mark.parametrize(
    "event_ids",
    [
        pytest.param([("x", value) for value in range(1, 5)], id="tuple"),
        pytest.param(
            [("x", (value, b"z")) for value in range(1, 5)], id="nested-tuple"
        ),
    ],
)
def test_flag_selection_reconstructs_inferred_multiindex_as_atomic_index(event_ids):
    selection = FlagSelection(
        flags_by_event_id=pd.Series(
            [1, 0, 1, 0],
            index=pd.MultiIndex.from_tuples(event_ids),
            name="false_positive",
        ),
        selected_event_ids=(event_ids[0], event_ids[2]),
        legitimate_event_count=4,
        n_select=2,
        selected_event_count=2,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        ordering="score_desc_event_id_asc",
        selected_event_ids_sha256="a" * 64,
        flags_by_event_id_sha256="b" * 64,
    )
    assert type(selection.flags_by_event_id.index) is pd.Index
    assert selection.flags_by_event_id.index.tolist() == event_ids


STRATUM_PAIRS = [
    pytest.param(1, 1.0, True, id="small-int-float-equal"),
    pytest.param(2**53 + 1, float(2**53), False, id="large-int-float-unequal"),
    pytest.param(Decimal("0.5"), 0.5, True, id="decimal-float-equal"),
    pytest.param(Decimal("0.1"), 0.1, False, id="decimal-float-unequal"),
    pytest.param(Decimal("-0"), -0.0, True, id="signed-zero-equal"),
    pytest.param(
        datetime(2026, 1, 2, 3, 4, 5),
        pd.Timestamp(2026, 1, 2, 3, 4, 5),
        True,
        id="datetime-timestamp-equal",
    ),
    pytest.param(
        datetime(2026, 1, 2, 3, tzinfo=timezone(timedelta(hours=-5))),
        pd.Timestamp("2026-01-02T08:00:00Z"),
        True,
        id="timezone-equivalent",
    ),
    pytest.param(
        date(2026, 1, 2),
        datetime(2026, 1, 2),
        False,
        id="date-datetime-distinct",
    ),
    pytest.param(
        ("x", Decimal("0.5")), ("x", 0.5), True, id="tuple-equal"
    ),
    pytest.param(("x", 1), ("x", 2), False, id="tuple-distinct"),
]


def _numeric_fraction(value) -> Fraction | None:
    if isinstance(value, np.integer):
        return Fraction(int(value), 1)
    if isinstance(value, np.floating):
        numerator, denominator = value.as_integer_ratio()
        return Fraction(int(numerator), int(denominator))
    if type(value) is int:
        return Fraction(value, 1)
    if type(value) is float:
        return Fraction(*value.as_integer_ratio())
    if type(value) is Decimal:
        numerator, denominator = value.as_integer_ratio()
        return Fraction(numerator, denominator)
    return None


def _independent_equal(left, right) -> bool:
    left_numeric = _numeric_fraction(left)
    right_numeric = _numeric_fraction(right)
    if left_numeric is not None or right_numeric is not None:
        return (
            left_numeric is not None
            and right_numeric is not None
            and left_numeric == right_numeric
        )
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _independent_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, datetime) and isinstance(right, datetime):
        if (left.tzinfo is None) != (right.tzinfo is None):
            return False
        if left.tzinfo is not None:
            return left.astimezone(timezone.utc) == right.astimezone(timezone.utc)
        return left == right
    if isinstance(left, date) or isinstance(right, date):
        return type(left) is type(right) and left == right
    return type(left) is type(right) and left == right


def _stratum_frame(left, right) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["e0", "e1", "e2", "e3"],
            "entity": ["A", "B", "A", "B"],
            "stratum": _object_series([left, left, right, right]),
            "legitimate": [1, 1, 1, 1],
            "flag": [1, 0, 1, 0],
        }
    )


class _PermutationRecorder:
    def __init__(self, calls):
        self.calls = calls

    def permutation(self, values):
        array = np.asarray(values)
        self.calls.append((len(array), int(array.sum())))
        return array.copy()


@pytest.mark.parametrize("left,right,expected_equal", STRATUM_PAIRS)
def test_pairwise_stratum_partition_fixed_counts_and_shuffle(
    left, right, expected_equal, monkeypatch
):
    assert _independent_equal(left, right) is expected_equal
    expected_count = 1 if expected_equal else 2
    expected_fixed_counts = [(4, 2)] if expected_equal else [(2, 1), (2, 1)]

    calls = []
    monkeypatch.setattr(
        "fp_persistence.public_api.np.random.default_rng",
        lambda seed: _PermutationRecorder(calls),
    )
    frame = _stratum_frame(left, right)
    result = _audit(frame, replicates=3)
    assert result.diagnostics.final_stratum_count == expected_count
    assert result.reference.final_stratum_count == expected_count
    for replicate in range(3):
        start = replicate * expected_count
        assert sorted(calls[start : start + expected_count]) == expected_fixed_counts

    calls.clear()
    shuffled = frame.sample(frac=1, random_state=991)
    shuffled_result = _audit(shuffled, replicates=3)
    assert _science(shuffled_result) == _science(result)
    for replicate in range(3):
        start = replicate * expected_count
        assert sorted(calls[start : start + expected_count]) == expected_fixed_counts


def _repeat_proportion(flags, entities=("A", "B", "A", "B")) -> Fraction:
    repeated = sum(
        sum(flag for flag, entity in zip(flags, entities) if entity == candidate) >= 2
        for candidate in ("A", "B")
    )
    return Fraction(repeated, 2)


def test_large_int_float_strata_have_known_fixed_count_reference():
    separate_reference = []
    for first in ((1, 0), (0, 1)):
        for second in ((1, 0), (0, 1)):
            separate_reference.append(_repeat_proportion(first + second))
    exact_separate = sum(separate_reference, Fraction()) / len(separate_reference)

    collapsed_reference = []
    for selected in combinations(range(4), 2):
        flags = tuple(int(index in selected) for index in range(4))
        collapsed_reference.append(_repeat_proportion(flags))
    exact_collapsed = sum(collapsed_reference, Fraction()) / len(collapsed_reference)
    assert exact_separate == Fraction(1, 4)
    assert exact_collapsed == Fraction(1, 6)

    frame = _stratum_frame(2**53 + 1, float(2**53))
    result = _audit(frame, replicates=4000, seed=2026091601)
    assert result.diagnostics.final_stratum_count == 2
    assert result.diagnostics.final_stratum_size_distribution == {2: 2}
    assert result.estimates.observed_repeat_fp_proportion == 0.5
    assert result.estimates.reference_expected_repeat_fp_proportion == pytest.approx(
        float(exact_separate), abs=0.03
    )
    assert abs(
        result.estimates.reference_expected_repeat_fp_proportion
        - float(exact_collapsed)
    ) > 0.04


SCORE_VALUES = [
    pytest.param("python-int", 2**53 + 1, id="python-int"),
    pytest.param("python-float", float(2**53), id="python-float"),
    pytest.param("numpy-int", np.int64(2**53 + 2), id="numpy-int"),
    pytest.param("numpy-float16", np.float16("0.1"), id="numpy-float16"),
    pytest.param("numpy-float32", np.float32("0.1"), id="numpy-float32"),
    pytest.param("numpy-float64", np.float64("0.1"), id="numpy-float64"),
    pytest.param("numpy-longdouble", np.longdouble("0.1"), id="numpy-longdouble"),
    pytest.param("decimal", Decimal("0.1"), id="decimal"),
]

DECIMAL_CONTEXTS = [
    pytest.param(28, False, id="default-context"),
    pytest.param(28, True, id="float-operation-trap"),
    pytest.param(6, False, id="changed-precision"),
]


def _exact_score(value) -> Fraction:
    result = _numeric_fraction(value)
    assert result is not None
    return result


@pytest.mark.parametrize("left_name,left", SCORE_VALUES)
@pytest.mark.parametrize("right_name,right", SCORE_VALUES)
@pytest.mark.parametrize("precision,float_operation_trap", DECIMAL_CONTEXTS)
def test_pairwise_score_ordering_uses_exact_context_independent_oracle(
    left_name,
    left,
    right_name,
    right,
    precision,
    float_operation_trap,
):
    del left_name, right_name
    left_exact = _exact_score(left)
    right_exact = _exact_score(right)
    expected = "a-left" if left_exact >= right_exact else "b-right"
    frame = pd.DataFrame(
        {
            "event_id": ["a-left", "b-right"],
            "score": _object_series([left, right]),
            "legitimate": [1, 1],
        }
    )
    with localcontext(DefaultContext) as context:
        context.prec = precision
        context.traps[FloatOperation] = float_operation_trap
        selection = flags_from_scores(
            frame,
            score_col="score",
            event_id_col="event_id",
            legitimate_col="legitimate",
            n_select=1,
        )
    assert selection.selected_event_ids == (expected,)
