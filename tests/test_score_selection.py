from dataclasses import replace
from datetime import date
from decimal import Decimal
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from fp_persistence import (
    AuditValidationError,
    ExposureSpec,
    PermutationSpec,
    audit,
    flags_from_scores,
)


def _score_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["e5", "e1", "e3", "e2", "e4", "e6"],
            "entity": ["A", "A", "B", "B", "C", "C"],
            "score": [0.2, 0.9, 0.7, 0.7, 0.1, 0.4],
            "legitimate": [1, 1, 1, 1, 1, 1],
            "stratum": ["s1", "s1", "s2", "s2", "s1", "s2"],
        },
        index=[70, 3, 91, 14, 55, 8],
    )


def _select(frame: pd.DataFrame, n_select: int):
    return flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        n_select=n_select,
        legitimate_col="legitimate",
    )


def test_exact_zero_selection_returns_all_zero_flags():
    selection = _select(_score_frame(), 0)
    assert selection.selected_event_ids == ()
    assert selection.legitimate_event_count == 6
    assert selection.n_select == 0
    assert selection.selected_event_count == 0
    assert selection.flags_by_event_id.sum() == 0
    assert selection.flags_by_event_id.index.tolist() == ["e1", "e2", "e3", "e4", "e5", "e6"]


def test_exact_all_legitimate_selection_returns_all_one_flags():
    selection = _select(_score_frame(), 6)
    assert selection.selected_event_ids == ("e1", "e2", "e3", "e6", "e5", "e4")
    assert selection.selected_event_count == 6
    assert selection.flags_by_event_id.sum() == 6
    assert selection.flags_by_event_id.eq(1).all()


def test_intermediate_selection_uses_exact_caller_supplied_count():
    selection = _select(_score_frame(), 3)
    assert selection.selected_event_ids == ("e1", "e2", "e3")
    assert selection.n_select == 3
    assert selection.selected_event_count == 3
    assert int(selection.flags_by_event_id.sum()) == 3
    assert selection.ordering == "score_desc_event_id_asc"


def test_score_ties_are_resolved_by_event_id_ascending_not_row_order():
    frame = _score_frame().iloc[::-1].copy()
    selection = _select(frame, 3)
    assert selection.selected_event_ids == ("e1", "e2", "e3")


def test_integer_scores_preserve_order_above_binary64_exact_range():
    frame = pd.DataFrame(
        {
            "event_id": ["a-lower", "z-higher"],
            "score": [2**53, 2**53 + 1],
            "legitimate": [1, 1],
        }
    )
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        n_select=1,
        legitimate_col="legitimate",
    )
    assert selection.selected_event_ids == ("z-higher",)


def test_numpy_float64_and_python_integer_use_exact_numeric_ordering():
    frame = pd.DataFrame(
        {
            "event_id": ["a-lower", "z-higher"],
            "score": pd.Series([np.float64(2**53), 2**53 + 1], dtype=object),
            "legitimate": [1, 1],
        }
    )
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        n_select=1,
        legitimate_col="legitimate",
    )
    assert selection.selected_event_ids == ("z-higher",)


def test_numpy_float32_and_python_integer_use_exact_numeric_ordering():
    frame = pd.DataFrame(
        {
            "event_id": ["a-lower", "z-higher"],
            "score": pd.Series([np.float32(2**24), 2**24 + 1], dtype=object),
            "legitimate": [1, 1],
        }
    )
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        n_select=1,
        legitimate_col="legitimate",
    )
    assert selection.selected_event_ids == ("z-higher",)


def test_python_integer_and_numpy_integer_use_exact_numeric_ordering():
    frame = pd.DataFrame(
        {
            "event_id": ["a-lower", "z-higher"],
            "score": pd.Series([np.int64(10), 11], dtype=object),
            "legitimate": [1, 1],
        }
    )
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        n_select=1,
        legitimate_col="legitimate",
    )
    assert selection.selected_event_ids == ("z-higher",)


def test_decimal_and_python_float_use_exact_numeric_ordering():
    frame = pd.DataFrame(
        {
            "event_id": ["a-lower", "z-higher"],
            "score": pd.Series([Decimal("0.1"), 0.1], dtype=object),
            "legitimate": [1, 1],
        }
    )
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        n_select=1,
        legitimate_col="legitimate",
    )
    assert selection.selected_event_ids == ("z-higher",)


def test_true_cross_representation_numeric_tie_uses_event_id():
    frame = pd.DataFrame(
        {
            "event_id": ["z-first-row", "a-second-row"],
            "score": pd.Series([Decimal("0.5"), np.float32(0.5)], dtype=object),
            "legitimate": [1, 1],
        }
    )
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        n_select=1,
        legitimate_col="legitimate",
    )
    assert selection.selected_event_ids == ("a-second-row",)


def test_decimal_scores_preserve_high_precision_ordering():
    frame = pd.DataFrame(
        {
            "event_id": ["a-lower", "z-higher"],
            "score": [Decimal("1.0000000000000000001"), Decimal("1.0000000000000000002")],
            "legitimate": [1, 1],
        }
    )
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        n_select=1,
        legitimate_col="legitimate",
    )
    assert selection.selected_event_ids == ("z-higher",)


def test_nonlegitimate_high_scores_are_excluded_from_ranking():
    frame = _score_frame()
    extra = pd.DataFrame(
        {
            "event_id": ["fraud-high", "fraud-missing-score"],
            "entity": ["Z", "Z"],
            "score": [999.0, np.nan],
            "legitimate": [0, 0],
            "stratum": ["fraud", "fraud"],
        }
    )
    frame = pd.concat([extra, frame], ignore_index=True)
    selection = _select(frame, 2)
    assert selection.legitimate_event_count == 6
    assert selection.selected_event_ids == ("e1", "e2")
    assert "fraud-high" not in selection.flags_by_event_id.index


def test_row_shuffle_leaves_selected_ids_flags_and_hashes_identical():
    frame = _score_frame()
    original = _select(frame, 4)
    shuffled = _select(frame.sample(frac=1, random_state=2901), 4)
    assert shuffled.selected_event_ids == original.selected_event_ids
    pd.testing.assert_series_equal(shuffled.flags_by_event_id, original.flags_by_event_id)
    assert shuffled.selected_event_ids_sha256 == original.selected_event_ids_sha256
    assert shuffled.flags_by_event_id_sha256 == original.flags_by_event_id_sha256
    assert len(original.selected_event_ids_sha256) == 64
    assert len(original.flags_by_event_id_sha256) == 64


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf, None, "0.9", True])
def test_legitimate_scores_must_be_finite_numeric_values(bad_score):
    frame = _score_frame()
    frame["score"] = frame["score"].astype(object)
    frame.loc[frame.index[0], "score"] = bad_score
    with pytest.raises(AuditValidationError, match="finite numbers"):
        _select(frame, 2)


def test_duplicate_event_ids_are_rejected():
    frame = _score_frame()
    frame.loc[frame.index[-1], "event_id"] = "e1"
    with pytest.raises(AuditValidationError, match="unique"):
        _select(frame, 2)


def test_missing_event_ids_are_rejected():
    frame = _score_frame()
    frame.loc[frame.index[-1], "event_id"] = None
    with pytest.raises(AuditValidationError, match="nonmissing"):
        _select(frame, 2)


@pytest.mark.parametrize("column", ["event_id", "score", "legitimate"])
def test_score_selection_rejects_duplicate_dataframe_column_labels(column):
    frame = _score_frame()
    duplicated = pd.concat([frame, frame[[column]]], axis=1)
    with pytest.raises(AuditValidationError, match="column labels must be unique"):
        _select(duplicated, 2)


def test_mixed_incomparable_event_id_representations_are_rejected():
    frame = _score_frame()
    frame["event_id"] = [1, "e1", "e3", "e2", "e4", "e6"]
    with pytest.raises(AuditValidationError, match="consistently sortable"):
        _select(frame, 2)


def test_mixed_comparable_event_id_representations_are_rejected():
    frame = _score_frame()
    frame["event_id"] = pd.Series([1, 2.0, 3, 4, 5, 6], dtype=object).to_numpy()
    with pytest.raises(AuditValidationError, match="consistent representation"):
        _select(frame, 2)


class _PartialOrderEventId:
    def __init__(self, members):
        self.members = frozenset(members)

    def __hash__(self):
        return hash(self.members)

    def __eq__(self, other):
        return isinstance(other, _PartialOrderEventId) and self.members == other.members

    def __lt__(self, other):
        if not isinstance(other, _PartialOrderEventId):
            return NotImplemented
        return self.members < other.members


@pytest.mark.parametrize(
    "event_ids",
    [
        [frozenset({f"id-{index}"}) for index in range(6)],
        [_PartialOrderEventId({index}) for index in range(6)],
    ],
)
def test_score_selection_rejects_partial_order_event_ids(event_ids):
    frame = _score_frame()
    frame["event_id"] = event_ids
    with pytest.raises(AuditValidationError, match="unsupported identifier representation"):
        _select(frame, 2)


@pytest.mark.parametrize(
    "event_ids",
    [
        [1, 2, 3, 4, 5, 6],
        ["a", "b", "c", "d", "e", "f"],
        [UUID(int=value) for value in range(1, 7)],
        [date(2026, 1, value) for value in range(1, 7)],
    ],
)
def test_homogeneous_supported_event_id_representations_pass(event_ids):
    frame = _score_frame()
    frame["event_id"] = event_ids
    selection = _select(frame, 2)
    assert selection.selected_event_count == 2


@pytest.mark.parametrize("n_select", [-1, True])
def test_invalid_negative_or_boolean_selection_count_is_rejected(n_select):
    with pytest.raises(AuditValidationError, match="nonnegative integer"):
        _select(_score_frame(), n_select)


def test_selection_count_above_legitimate_count_is_rejected():
    with pytest.raises(AuditValidationError, match="must not exceed"):
        _select(_score_frame(), 7)


@pytest.mark.parametrize("bad_indicator", [2, -1, 0.0, "1", None])
def test_legitimate_indicator_is_strict_binary(bad_indicator):
    frame = _score_frame()
    frame["legitimate"] = frame["legitimate"].astype(object)
    frame.loc[frame.index[0], "legitimate"] = bad_indicator
    with pytest.raises(AuditValidationError, match="strict binary"):
        _select(frame, 2)


def test_helper_does_not_mutate_input_dataframe():
    frame = _score_frame()
    before = frame.copy(deep=True)
    _select(frame, 3)
    pd.testing.assert_frame_equal(frame, before)


def test_all_rows_are_legitimate_when_legitimate_column_is_omitted():
    frame = _score_frame().drop(columns=["legitimate"])
    selection = flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        n_select=2,
    )
    assert selection.legitimate_col is None
    assert selection.legitimate_event_count == len(frame)
    assert selection.selected_event_ids == ("e1", "e2")


def test_typed_hashes_distinguish_identifier_types_without_process_hashing():
    integer_frame = pd.DataFrame(
        {"event_id": [1, 2], "score": [1.0, 0.0], "legitimate": [1, 1]}
    )
    string_frame = pd.DataFrame(
        {"event_id": ["1", "2"], "score": [1.0, 0.0], "legitimate": [1, 1]}
    )
    integers = _select(integer_frame.assign(entity="x", stratum="s"), 1)
    strings = _select(string_frame.assign(entity="x", stratum="s"), 1)
    assert integers.selected_event_ids_sha256 != strings.selected_event_ids_sha256
    assert integers.flags_by_event_id_sha256 != strings.flags_by_event_id_sha256


def test_flag_selection_passes_directly_to_audit_without_dataframe_mutation():
    frame = _score_frame()
    before = frame.copy(deep=True)
    selection = _select(frame, 3)
    result = audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        exposure=ExposureSpec(2, 2),
        permutation=PermutationSpec(50, 31),
        flag_selection=selection,
        legitimate_col="legitimate",
    )
    assert result.diagnostics.flag_mode == "score_selection"
    assert result.provenance.score_selection_spec["n_select"] == 3
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    "hash_field",
    ["selected_event_ids_sha256", "flags_by_event_id_sha256"],
)
def test_audit_rejects_tampered_flag_selection_hashes(hash_field):
    frame = _score_frame()
    selection = _select(frame, 3)
    tampered = replace(selection, **{hash_field: "0" * 64})
    with pytest.raises(AuditValidationError, match=hash_field):
        audit(
            frame,
            entity_col="entity",
            event_id_col="event_id",
            final_stratum_col="stratum",
            exposure=ExposureSpec(2, 2),
            permutation=PermutationSpec(20, 31),
            flag_selection=tampered,
            legitimate_col="legitimate",
        )


def test_score_selection_and_equivalent_external_flags_are_scientifically_identical():
    frame = _score_frame()
    selection = _select(frame, 3)
    score_result = audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        exposure=ExposureSpec(2, 2),
        permutation=PermutationSpec(200, 2026091601),
        flag_selection=selection,
        legitimate_col="legitimate",
    )
    explicit = frame.copy()
    explicit["explicit_flag"] = explicit["event_id"].isin(
        selection.selected_event_ids
    ).astype("int8")
    external_result = audit(
        explicit,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        exposure=ExposureSpec(2, 2),
        permutation=PermutationSpec(200, 2026091601),
        flag_col="explicit_flag",
        legitimate_col="legitimate",
    )
    assert score_result.population == external_result.population
    assert score_result.counts == external_result.counts
    assert score_result.estimates == external_result.estimates
    assert score_result.reference == external_result.reference
    assert replace(
        score_result.diagnostics,
        flag_mode="external_flags",
        flag_col="explicit_flag",
    ) == external_result.diagnostics
    assert score_result.provenance.flag_mode == "score_selection"
    assert external_result.provenance.flag_mode == "external_flags"
    assert score_result.provenance.score_selection_spec is not None
    assert external_result.provenance.score_selection_spec is None
