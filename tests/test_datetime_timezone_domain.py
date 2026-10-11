"""Release gates for the deliberately narrow datetime identifier domain."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import pytz
from dateutil import tz as dateutil_tz

from fp_persistence import (
    AuditStatus,
    AuditValidationError,
    ExposureSpec,
    FlagSelection,
    PermutationSpec,
    audit,
    flags_from_scores,
)


def _objects(values) -> pd.Series:
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return pd.Series(array, dtype=object)


def _object_index(values) -> pd.Index:
    return pd.Index(_objects(values).to_numpy(), dtype=object, tupleize_cols=False)


def _frame(*, event_ids=None, entities=None, strata=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": _objects(event_ids or ["e0", "e1", "e2", "e3"]),
            "entity": _objects(entities or ["A", "A", "B", "B"]),
            "stratum": _objects(strata or ["s0", "s0", "s1", "s1"]),
            "legitimate": [1, 1, 1, 1],
            "flag": [1, 0, 1, 0],
            "score": [4.0, 1.0, 3.0, 2.0],
        }
    )


def _audit(frame: pd.DataFrame, *, selection=None, replicates=31):
    return audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        legitimate_col="legitimate",
        flag_col="flag" if selection is None else None,
        flag_selection=selection,
        exposure=ExposureSpec(2, 2),
        permutation=PermutationSpec(replicates, 2026091601),
    )


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


def _dynamic_fold_values(*, pandas_timestamp: bool):
    zone = ZoneInfo("America/New_York")
    values = [
        datetime(2021, 11, 7, 0, 30, tzinfo=zone),
        datetime(2021, 11, 7, 1, 30, tzinfo=zone, fold=0),
        datetime(2021, 11, 7, 1, 30, tzinfo=zone, fold=1),
        datetime(2021, 11, 7, 2, 30, tzinfo=zone),
    ]
    if pandas_timestamp:
        return [pd.Timestamp(value) for value in values]
    return values


def _supported_event_id_cases():
    plus = timezone(timedelta(hours=5, minutes=30))
    minus = timezone(-timedelta(hours=7))
    return [
        pytest.param(
            [datetime(2026, 1, day, 1, 2, 3) for day in range(1, 5)],
            id="python-naive",
        ),
        pytest.param(
            [pd.Timestamp(2026, 1, day, 1, 2, 3) for day in range(1, 5)],
            id="timestamp-naive",
        ),
        pytest.param(
            [datetime(2026, 1, day, 1, 2, 3, tzinfo=timezone.utc) for day in range(1, 5)],
            id="python-utc",
        ),
        pytest.param(
            [pd.Timestamp(datetime(2026, 1, day, 1, 2, 3, tzinfo=timezone.utc)) for day in range(1, 5)],
            id="timestamp-utc",
        ),
        pytest.param(
            [datetime(2026, 1, day, 1, 2, 3, tzinfo=plus) for day in range(1, 5)],
            id="python-positive-fixed-offset",
        ),
        pytest.param(
            [pd.Timestamp(datetime(2026, 1, day, 1, 2, 3, tzinfo=minus)) for day in range(1, 5)],
            id="timestamp-negative-fixed-offset",
        ),
        pytest.param(
            [("outer", datetime(2026, 1, day, tzinfo=plus)) for day in range(1, 5)],
            id="tuple-fixed-offset",
        ),
        pytest.param(
            [("outer", (pd.Timestamp(datetime(2026, 1, day, tzinfo=minus)), day)) for day in range(1, 5)],
            id="nested-tuple-fixed-offset",
        ),
    ]


class _CustomFixedOffset(tzinfo):
    def utcoffset(self, dt):
        return timedelta(hours=2)

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "CUSTOM+02"


@pytest.mark.parametrize(
    "dynamic_value",
    [
        pytest.param(
            pytz.timezone("America/New_York").localize(datetime(2021, 11, 7, 1, 30), is_dst=True),
            id="pytz-regional",
        ),
        pytest.param(
            datetime(2021, 11, 7, 1, 30, tzinfo=dateutil_tz.gettz("America/New_York"), fold=0),
            id="dateutil-regional",
        ),
        pytest.param(
            datetime(2026, 1, 1, tzinfo=_CustomFixedOffset()),
            id="custom-tzinfo",
        ),
    ],
)
def test_other_dynamic_and_custom_timezone_representations_are_rejected(dynamic_value):
    values = [dynamic_value + timedelta(days=day) for day in range(4)]
    with pytest.raises(AuditValidationError, match="normalize values to UTC"):
        _audit(_frame(event_ids=values))


@pytest.mark.parametrize("pandas_timestamp", [False, True], ids=["datetime", "timestamp"])
def test_dynamic_zone_event_ids_rejected_in_external_audit_before_science(pandas_timestamp):
    frame = _frame(event_ids=_dynamic_fold_values(pandas_timestamp=pandas_timestamp))
    shuffled = frame.sample(frac=1, random_state=417)
    with pytest.raises(AuditValidationError, match="dynamic or custom timezone"):
        _audit(frame)
    with pytest.raises(AuditValidationError, match="dynamic or custom timezone"):
        _audit(shuffled)


@pytest.mark.parametrize("pandas_timestamp", [False, True], ids=["datetime", "timestamp"])
def test_dynamic_zone_score_tie_event_ids_rejected_for_both_row_orders(pandas_timestamp):
    frame = _frame(event_ids=_dynamic_fold_values(pandas_timestamp=pandas_timestamp))
    frame["score"] = 1.0
    for candidate in (frame, frame.sample(frac=1, random_state=911)):
        with pytest.raises(AuditValidationError, match="dynamic or custom timezone"):
            flags_from_scores(
                candidate,
                score_col="score",
                event_id_col="event_id",
                legitimate_col="legitimate",
                n_select=2,
            )


@pytest.mark.parametrize("pandas_timestamp", [False, True], ids=["datetime", "timestamp"])
def test_dynamic_zone_event_ids_rejected_in_flag_selection_audit(pandas_timestamp):
    values = _dynamic_fold_values(pandas_timestamp=pandas_timestamp)
    frame = _frame(event_ids=values)
    valid = flags_from_scores(
        _frame(),
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        n_select=2,
    )
    with pytest.raises(AuditValidationError, match="dynamic or custom timezone"):
        _audit(frame, selection=valid)


@pytest.mark.parametrize("pandas_timestamp", [False, True], ids=["datetime", "timestamp"])
def test_manually_constructed_flag_selection_rejects_dynamic_zone_ids(pandas_timestamp):
    values = _dynamic_fold_values(pandas_timestamp=pandas_timestamp)
    with pytest.raises(AuditValidationError, match="dynamic or custom timezone"):
        FlagSelection(
            flags_by_event_id=pd.Series([1, 0, 1, 0], index=_object_index(values)),
            selected_event_ids=(values[0], values[2]),
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


@pytest.mark.parametrize("pandas_timestamp", [False, True], ids=["datetime", "timestamp"])
@pytest.mark.parametrize("nesting", ["direct", "tuple", "nested"])
def test_dynamic_zone_entity_ids_rejected_recursively(pandas_timestamp, nesting):
    fold_values = _dynamic_fold_values(pandas_timestamp=pandas_timestamp)
    values = [fold_values[1], fold_values[1], fold_values[2], fold_values[2]]
    if nesting == "tuple":
        values = [("entity", value) for value in values]
    elif nesting == "nested":
        values = [("entity", (value, 1)) for value in values]
    with pytest.raises(AuditValidationError, match="dynamic or custom timezone"):
        _audit(_frame(entities=values))


@pytest.mark.parametrize("pandas_timestamp", [False, True], ids=["datetime", "timestamp"])
@pytest.mark.parametrize("nesting", ["direct", "tuple", "nested"])
def test_dynamic_zone_final_strata_rejected_recursively(pandas_timestamp, nesting):
    fold_values = _dynamic_fold_values(pandas_timestamp=pandas_timestamp)
    values = [fold_values[1], fold_values[1], fold_values[2], fold_values[2]]
    if nesting == "tuple":
        values = [("stratum", value) for value in values]
    elif nesting == "nested":
        values = [("stratum", (value, 1)) for value in values]
    with pytest.raises(AuditValidationError, match="dynamic or custom timezone"):
        _audit(_frame(strata=values), replicates=4000)


@pytest.mark.parametrize("event_ids", _supported_event_id_cases())
def test_supported_datetime_event_ids_cross_modes_shuffle_and_hashes(event_ids):
    frame = _frame(event_ids=event_ids)
    external = _audit(frame)
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        n_select=2,
    )
    score_mode = _audit(frame, selection=selection)
    explicit = frame.copy()
    explicit_flags = dict(selection.flags_by_event_id.items())
    explicit["flag"] = [int(explicit_flags[value]) for value in frame["event_id"]]
    assert _science(score_mode) == _science(_audit(explicit))

    shuffled = frame.sample(frac=1, random_state=1601)
    shuffled_selection = flags_from_scores(
        shuffled,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        n_select=2,
    )
    assert shuffled_selection.selected_event_ids == selection.selected_event_ids
    assert shuffled_selection.selected_event_ids_sha256 == selection.selected_event_ids_sha256
    assert shuffled_selection.flags_by_event_id_sha256 == selection.flags_by_event_id_sha256
    pd.testing.assert_series_equal(shuffled_selection.flags_by_event_id, selection.flags_by_event_id)
    assert _science(_audit(shuffled)) == _science(external)
    assert _science(_audit(shuffled, selection=shuffled_selection)) == _science(score_mode)


@pytest.mark.parametrize("nesting", ["direct", "tuple", "nested"])
def test_supported_fixed_offset_entity_ids_preserve_exposure_grouping(nesting):
    instant_a = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    instant_b = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    values = [
        instant_a,
        pd.Timestamp(instant_a.astimezone(timezone(timedelta(hours=3)))),
        instant_b.astimezone(timezone(timedelta(hours=-5))),
        pd.Timestamp(instant_b),
    ]
    if nesting == "tuple":
        values = [("entity", value) for value in values]
    elif nesting == "nested":
        values = [("entity", (value, 1)) for value in values]
    result = _audit(_frame(entities=values))
    assert result.status is AuditStatus.ESTIMABLE
    assert result.population.entity_count_before_exposure_filter == 2
    assert result.population.eligible_entity_count == 2
    assert result.diagnostics.entity_legitimate_exposure_distribution == {2: 2}


class _PermutationRecorder:
    def __init__(self):
        self.calls = []

    def permutation(self, values):
        array = np.asarray(values)
        self.calls.append((len(array), int(array.sum())))
        return array.copy()


@pytest.mark.parametrize("nesting", ["direct", "tuple", "nested"])
def test_supported_fixed_offset_strata_group_equal_instants_and_keep_fixed_counts(
    nesting, monkeypatch
):
    first = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    second = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    values = [
        first,
        pd.Timestamp(first.astimezone(timezone(timedelta(hours=5)))),
        second.astimezone(timezone(timedelta(hours=-7))),
        pd.Timestamp(second),
    ]
    if nesting == "tuple":
        values = [("stratum", value) for value in values]
    elif nesting == "nested":
        values = [("stratum", (value, 1)) for value in values]
    recorder = _PermutationRecorder()
    monkeypatch.setattr("fp_persistence.public_api.np.random.default_rng", lambda seed: recorder)
    frame = _frame(strata=values)
    result = _audit(frame, replicates=3)
    assert result.reference.final_stratum_count == 2
    assert result.diagnostics.final_stratum_size_distribution == {2: 2}
    assert recorder.calls == [(2, 1), (2, 1)] * 3

    recorder.calls.clear()
    shuffled = frame.sample(frac=1, random_state=2026)
    assert _science(_audit(shuffled, replicates=3)) == _science(result)
    assert recorder.calls == [(2, 1), (2, 1)] * 3


def test_equivalent_fixed_offset_instants_are_duplicate_event_ids():
    instant = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    equivalent = pd.Timestamp(instant.astimezone(timezone(timedelta(hours=-5))))
    frame = _frame(event_ids=[instant, equivalent, datetime(2026, 1, 2, tzinfo=timezone.utc), datetime(2026, 1, 3, tzinfo=timezone.utc)])
    with pytest.raises(AuditValidationError, match="event IDs must be unique"):
        _audit(frame)
    with pytest.raises(AuditValidationError, match="event IDs must be unique"):
        flags_from_scores(
            frame,
            score_col="score",
            event_id_col="event_id",
            legitimate_col="legitimate",
            n_select=2,
        )


def test_equivalent_utc_forms_are_supported_when_not_duplicate_ids():
    python_utc = [datetime(2026, 1, day, tzinfo=timezone.utc) for day in range(1, 5)]
    timestamp_utc = [pd.Timestamp(value) for value in python_utc]
    assert _science(_audit(_frame(event_ids=python_utc))) == _science(
        _audit(_frame(event_ids=timestamp_utc))
    )
