"""Systematic release matrix for the documented finite public input domain."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import UUID

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


class CustomHashable:
    def __hash__(self):
        return 1


class CustomEquality:
    def __hash__(self):
        return 1

    def __eq__(self, other):
        raise RuntimeError("custom equality must never be invoked")


class CustomPartialOrder:
    def __init__(self, value=0):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __lt__(self, other):
        return False


class CustomUnstableRepr:
    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"unstable-{id(self)}"


def _base() -> pd.DataFrame:
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


def _audit(frame: pd.DataFrame):
    return audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        legitimate_col="legitimate",
        flag_col="flag",
        exposure=ExposureSpec(2, 2),
        permutation=PermutationSpec(5, 90210),
    )


SUPPORTED_IDENTIFIER_VALUES = [
    pytest.param(["a", "b", "c", "d"], id="str"),
    pytest.param([1, 2, 3, 4], id="int"),
    pytest.param([0.25, 0.5, 0.75, 1.0], id="finite-float"),
    pytest.param([np.int8(1), np.int8(2), np.int8(3), np.int8(4)], id="np-int8"),
    pytest.param([np.int32(1), np.int32(2), np.int32(3), np.int32(4)], id="np-int32"),
    pytest.param([np.int64(1), np.int64(2), np.int64(3), np.int64(4)], id="np-int64"),
    pytest.param(
        [np.float32(0.25), np.float32(0.5), np.float32(0.75), np.float32(1.0)],
        id="np-float32",
    ),
    pytest.param(
        [np.float64(0.25), np.float64(0.5), np.float64(0.75), np.float64(1.0)],
        id="np-float64",
    ),
    pytest.param(
        [Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1")],
        id="decimal",
    ),
    pytest.param([b"a", b"b", b"c", b"d"], id="bytes"),
    pytest.param([UUID(int=value) for value in range(1, 5)], id="uuid"),
    pytest.param([date(2026, 1, value) for value in range(1, 5)], id="date"),
    pytest.param(
        [datetime(2026, 1, value, tzinfo=timezone.utc) for value in range(1, 5)],
        id="datetime",
    ),
    pytest.param([("x", value) for value in range(1, 5)], id="tuple"),
    pytest.param([("x", (value, b"z")) for value in range(1, 5)], id="nested-tuple"),
]


@pytest.mark.parametrize("values", SUPPORTED_IDENTIFIER_VALUES)
@pytest.mark.parametrize("role", ["event_id", "entity", "stratum"])
def test_supported_identifier_domain_by_role(role, values):
    frame = _base()
    if role == "event_id":
        role_values = values
    else:
        role_values = [values[0], values[0], values[1], values[1]]
    frame[role] = pd.Series(role_values, dtype=object)
    assert _audit(frame).status is AuditStatus.ESTIMABLE


UNSUPPORTED_IDENTIFIER_VALUES = [
    pytest.param(True, id="bool"),
    pytest.param(None, id="none"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param(Decimal("NaN"), id="decimal-nan"),
    pytest.param(Decimal("sNaN"), id="decimal-snan"),
    pytest.param(Decimal("Infinity"), id="decimal-infinity"),
    pytest.param([], id="list"),
    pytest.param({}, id="dict"),
    pytest.param(set(), id="set"),
    pytest.param(frozenset(), id="frozenset"),
    pytest.param(CustomHashable(), id="custom-hashable"),
    pytest.param(CustomEquality(), id="custom-equality"),
    pytest.param(CustomPartialOrder(), id="custom-partial-order"),
    pytest.param(CustomUnstableRepr(), id="custom-unstable-repr"),
    pytest.param(("valid", float("nan")), id="nested-tuple-nonfinite"),
    pytest.param(("valid", frozenset({1})), id="nested-tuple-unsupported"),
]


@pytest.mark.parametrize("bad_value", UNSUPPORTED_IDENTIFIER_VALUES)
@pytest.mark.parametrize("role", ["event_id", "entity", "stratum"])
def test_unsupported_identifier_domain_by_role_raises_validation_error(role, bad_value):
    frame = _base()
    values = frame[role].astype(object).tolist()
    values[0] = bad_value
    frame[role] = pd.Series(values, dtype=object)
    with pytest.raises(AuditValidationError):
        _audit(frame)


SUPPORTED_SCORE_VALUES = [
    pytest.param([4, 3, 2, 1], id="int"),
    pytest.param([4.0, 3.0, 2.0, 1.0], id="float"),
    pytest.param([np.int8(4), np.int32(3), np.int64(2), np.int64(1)], id="np-int"),
    pytest.param(
        [np.float32(4), np.float32(3), np.float64(2), np.float64(1)], id="np-float"
    ),
    pytest.param(
        [Decimal("4"), Decimal("3"), Decimal("2"), Decimal("1")], id="decimal"
    ),
]


@pytest.mark.parametrize("scores", SUPPORTED_SCORE_VALUES)
def test_supported_score_domain(scores):
    frame = _base()
    frame["score"] = pd.Series(scores, dtype=object)
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        n_select=2,
    )
    assert selection.selected_event_count == 2


UNSUPPORTED_SCORE_VALUES = [
    pytest.param(True, id="bool"),
    pytest.param(1 + 2j, id="complex"),
    pytest.param(None, id="none"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param(Decimal("NaN"), id="decimal-nan"),
    pytest.param(Decimal("sNaN"), id="decimal-snan"),
    pytest.param(Decimal("Infinity"), id="decimal-infinity"),
    pytest.param("4", id="str"),
    pytest.param(b"4", id="bytes"),
    pytest.param(UUID(int=1), id="uuid"),
    pytest.param(date(2026, 1, 1), id="date"),
    pytest.param(datetime(2026, 1, 1), id="datetime"),
    pytest.param((4,), id="tuple"),
    pytest.param([], id="list"),
    pytest.param({}, id="dict"),
    pytest.param(set(), id="set"),
    pytest.param(frozenset(), id="frozenset"),
    pytest.param(CustomHashable(), id="custom-hashable"),
    pytest.param(CustomEquality(), id="custom-equality"),
    pytest.param(CustomPartialOrder(), id="custom-partial-order"),
    pytest.param(CustomUnstableRepr(), id="custom-unstable-repr"),
]


@pytest.mark.parametrize("bad_value", UNSUPPORTED_SCORE_VALUES)
def test_unsupported_score_domain_raises_validation_error(bad_value):
    frame = _base()
    scores = frame["score"].tolist()
    scores[0] = bad_value
    frame["score"] = pd.Series(scores, dtype=object)
    with pytest.raises(AuditValidationError):
        flags_from_scores(
            frame,
            score_col="score",
            event_id_col="event_id",
            legitimate_col="legitimate",
            n_select=2,
        )


SUPPORTED_BINARY_VALUES = [
    pytest.param([True, False, True, False], id="python-bool"),
    pytest.param([1, 0, 1, 0], id="python-int"),
    pytest.param([np.bool_(True), np.bool_(False), np.bool_(True), np.bool_(False)], id="np-bool"),
    pytest.param([np.int8(1), np.int32(0), np.int64(1), np.int8(0)], id="np-int"),
]


@pytest.mark.parametrize("values", SUPPORTED_BINARY_VALUES)
@pytest.mark.parametrize("role", ["flag", "legitimate"])
def test_supported_strict_binary_domain(role, values):
    frame = _base()
    frame[role] = pd.Series(values, dtype=object)
    assert _audit(frame).status in {AuditStatus.ESTIMABLE, AuditStatus.NOT_ESTIMABLE}


UNSUPPORTED_BINARY_VALUES = [
    pytest.param(0.0, id="float-zero"),
    pytest.param(1.0, id="float-one"),
    pytest.param(2, id="integer-two"),
    pytest.param(-1, id="integer-negative"),
    pytest.param(None, id="none"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinity"),
    pytest.param(Decimal("1"), id="decimal"),
    pytest.param("1", id="str"),
    pytest.param(b"1", id="bytes"),
    pytest.param(UUID(int=1), id="uuid"),
    pytest.param(date(2026, 1, 1), id="date"),
    pytest.param(datetime(2026, 1, 1), id="datetime"),
    pytest.param((1,), id="tuple"),
    pytest.param([], id="list"),
    pytest.param({}, id="dict"),
    pytest.param(set(), id="set"),
    pytest.param(frozenset(), id="frozenset"),
    pytest.param(CustomHashable(), id="custom-hashable"),
    pytest.param(CustomEquality(), id="custom-equality"),
    pytest.param(CustomPartialOrder(), id="custom-partial-order"),
    pytest.param(CustomUnstableRepr(), id="custom-unstable-repr"),
]


@pytest.mark.parametrize("bad_value", UNSUPPORTED_BINARY_VALUES)
@pytest.mark.parametrize("role", ["flag", "legitimate"])
def test_unsupported_strict_binary_domain_raises_validation_error(role, bad_value):
    frame = _base()
    values = frame[role].tolist()
    values[0] = bad_value
    frame[role] = pd.Series(values, dtype=object)
    with pytest.raises(AuditValidationError):
        _audit(frame)
