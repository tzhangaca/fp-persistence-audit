from dataclasses import replace

import pandas as pd
import pytest

from fp_persistence import (
    AuditStatus,
    AuditValidationError,
    ExposureSpec,
    FlagSelection,
    PermutationSpec,
    audit,
)
from fp_persistence.public_types import RatioState


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [f"e{i:02d}" for i in range(9)],
            "entity": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "legitimate": [1] * 9,
            "flag": [1, 1, 0, 1, 0, 0, 0, 0, 0],
            "stratum": ["s1", "s2", "s1", "s2", "s1", "s2", "s1", "s2", "s1"],
        },
        index=[90, 12, 71, 5, 44, 30, 2, 88, 16],
    )


def _audit_external(frame: pd.DataFrame, *, exposure=None, permutation=None):
    return audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        exposure=exposure or ExposureSpec(3, 3),
        permutation=permutation or PermutationSpec(100, 19),
        flag_col="flag",
        legitimate_col="legitimate",
        metadata={"fixture": "generic-test"},
    )


def test_estimable_external_flag_audit_uses_frozen_metric_semantics():
    result = _audit_external(_base_frame())
    assert result.status is AuditStatus.ESTIMABLE
    assert result.reason is None
    assert result.population.eligible_entity_count == 3
    assert result.counts.false_positive_event_count == 3
    assert result.counts.repeated_fp_entity_count == 1
    assert result.estimates.observed_repeat_fp_proportion == pytest.approx(1 / 3)
    assert result.estimates.excess_repeat_fp_percentage_points == pytest.approx(
        100
        * (
            result.estimates.observed_repeat_fp_proportion
            - result.estimates.reference_expected_repeat_fp_proportion
        )
    )
    assert result.counts.excess_repeated_fp_entity_count == pytest.approx(
        1
        - result.estimates.reference_expected_repeat_fp_proportion
        * result.population.eligible_entity_count
    )
    assert result.reference.replicates == 100
    assert result.reference.seed == 19
    assert result.provenance.flag_mode == "external_flags"


def test_score_selection_aligns_only_by_event_id_and_copies_provenance():
    frame = _base_frame()
    flags = pd.Series(
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        index=["e08", "e00", "e07", "e06", "e01", "e05", "e04", "e03", "e02"],
    )
    selection = FlagSelection(
        flags_by_event_id=flags,
        selected_event_ids=("e00", "e01", "e03"),
        legitimate_event_count=9,
        n_select=3,
        selected_event_count=3,
        score_col="external_score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        ordering="score_desc_event_id_asc",
        selected_event_ids_sha256="a" * 64,
        flags_by_event_id_sha256="b" * 64,
    )
    result = audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        exposure=ExposureSpec(3, 3),
        permutation=PermutationSpec(20, 3),
        flag_selection=selection,
        legitimate_col="legitimate",
    )
    assert result.diagnostics.flag_mode == "score_selection"
    assert result.diagnostics.flag_col is None
    assert result.counts.repeated_fp_entity_count == 1
    assert result.provenance.flag_mode == "score_selection"
    assert result.provenance.score_selection_spec["n_select"] == 3
    assert result.provenance.score_selection_spec["selected_event_ids_sha256"] == "a" * 64
    assert "selected_event_ids" not in result.provenance.score_selection_spec
    assert "flags_by_event_id" not in result.provenance.score_selection_spec


def test_observed_zero_is_an_estimable_scientific_zero():
    frame = _base_frame()
    frame["flag"] = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    result = _audit_external(frame)
    assert result.status is AuditStatus.ESTIMABLE
    assert result.estimates.observed_repeat_fp_proportion == 0.0
    assert result.counts.repeated_fp_entity_count == 0
    assert result.estimates.observed_expected_ratio.state is RatioState.FINITE
    assert result.estimates.observed_expected_ratio.value == 0.0


def test_zero_reference_and_zero_observed_use_zero_over_zero_state():
    frame = pd.DataFrame(
        {
            "event_id": ["a1", "a2", "b1", "b2"],
            "entity": ["A", "A", "B", "B"],
            "flag": [1, 0, 1, 0],
            "stratum": ["one", "two", "three", "four"],
        }
    )
    result = audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        exposure=ExposureSpec(2, 2),
        permutation=PermutationSpec(7, 1),
        flag_col="flag",
    )
    assert result.estimates.observed_repeat_fp_proportion == 0.0
    assert result.estimates.reference_expected_repeat_fp_proportion == 0.0
    assert result.estimates.observed_expected_ratio.state is RatioState.UNDEFINED_ZERO_OVER_ZERO
    assert result.estimates.observed_expected_ratio.value is None


def test_sampled_zero_reference_with_positive_observed_uses_positive_infinity():
    frame = pd.DataFrame(
        {
            "event_id": ["a1", "a2", "b1", "b2"],
            "entity": ["A", "A", "B", "B"],
            "flag": [1, 1, 0, 0],
            "stratum": ["all"] * 4,
        }
    )
    result = audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        exposure=ExposureSpec(2, 2),
        permutation=PermutationSpec(1, 0),
        flag_col="flag",
    )
    assert result.estimates.observed_repeat_fp_proportion == 0.5
    assert result.estimates.reference_expected_repeat_fp_proportion == 0.0
    assert result.estimates.observed_expected_ratio.state is RatioState.POSITIVE_INFINITY
    assert result.estimates.observed_expected_ratio.value is None


def test_zero_eligible_entities_returns_not_estimable_without_running_permutation(monkeypatch):
    frame = _base_frame()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("permutation RNG must not be constructed")

    monkeypatch.setattr("fp_persistence.public_api.np.random.default_rng", fail_if_called)
    result = _audit_external(frame, exposure=ExposureSpec(4, 5))
    assert result.status is AuditStatus.NOT_ESTIMABLE
    assert result.reason == "zero_eligible_entities"
    assert result.population.eligible_entity_count == 0
    assert result.counts.repeated_fp_entity_count == 0
    assert result.counts.excess_repeated_fp_entity_count is None
    assert result.estimates.observed_repeat_fp_proportion is None
    assert result.estimates.reference_expected_repeat_fp_proportion is None
    assert result.estimates.excess_repeat_fp_percentage_points is None
    assert result.estimates.observed_expected_ratio.state is RatioState.NOT_ESTIMABLE
    assert result.reference.status == "NOT_RUN_NOT_ESTIMABLE"
    assert result.reference.replicates == 0


@pytest.mark.parametrize(
    "mutator,kwargs",
    [
        (lambda frame: frame, {"flag_col": None}),
        (lambda frame: frame, {"flag_selection": "not-a-selection"}),
        (lambda frame: frame.drop(columns=["entity"]), {}),
        (lambda frame: frame.assign(legitimate=[1] * 8 + [2]), {}),
        (lambda frame: frame.assign(flag=[1] * 8 + [2]), {}),
        (lambda frame: frame.assign(entity=["A"] * 8 + [None]), {}),
        (lambda frame: frame.assign(stratum=["s"] * 8 + [None]), {}),
    ],
)
def test_malformed_inputs_raise_without_a_result(mutator, kwargs):
    frame = mutator(_base_frame())
    call = dict(
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        exposure=ExposureSpec(3, 3),
        permutation=PermutationSpec(5, 1),
        flag_col="flag",
        legitimate_col="legitimate",
    )
    call.update(kwargs)
    if kwargs.get("flag_selection") is not None:
        call["flag_col"] = None
    with pytest.raises(AuditValidationError):
        audit(frame, **call)


def test_both_flag_sources_are_rejected():
    frame = _base_frame()
    selection = FlagSelection(
        flags_by_event_id=pd.Series(frame["flag"].to_numpy(), index=frame["event_id"]),
        selected_event_ids=("e00", "e01", "e03"),
        legitimate_event_count=9,
        n_select=3,
        selected_event_count=3,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        ordering="score_desc_event_id_asc",
        selected_event_ids_sha256="a" * 64,
        flags_by_event_id_sha256="b" * 64,
    )
    with pytest.raises(AuditValidationError, match="exactly one"):
        audit(
            frame,
            entity_col="entity",
            event_id_col="event_id",
            final_stratum_col="stratum",
            exposure=ExposureSpec(3, 3),
            permutation=PermutationSpec(5, 1),
            flag_col="flag",
            flag_selection=selection,
            legitimate_col="legitimate",
        )


def test_duplicate_and_mixed_event_ids_raise_validation_errors():
    duplicate = _base_frame()
    duplicate.loc[duplicate.index[-1], "event_id"] = "e00"
    with pytest.raises(AuditValidationError, match="unique"):
        _audit_external(duplicate)

    mixed = _base_frame()
    mixed["event_id"] = [1, "e01", "e02", "e03", "e04", "e05", "e06", "e07", "e08"]
    with pytest.raises(AuditValidationError, match="consistently sortable"):
        _audit_external(mixed)


def test_flag_selection_event_id_coverage_mismatch_is_rejected():
    frame = _base_frame()
    flags = pd.Series([1, 1, 0, 1, 0, 0, 0, 0, 0], index=["other"] + list(frame.event_id[1:]))
    selection = FlagSelection(
        flags_by_event_id=flags,
        selected_event_ids=("other", "e01", "e03"),
        legitimate_event_count=9,
        n_select=3,
        selected_event_count=3,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        ordering="score_desc_event_id_asc",
        selected_event_ids_sha256="a" * 64,
        flags_by_event_id_sha256="b" * 64,
    )
    with pytest.raises(AuditValidationError, match="one-to-one coverage"):
        audit(
            frame,
            entity_col="entity",
            event_id_col="event_id",
            final_stratum_col="stratum",
            exposure=ExposureSpec(3, 3),
            permutation=PermutationSpec(5, 1),
            flag_selection=selection,
            legitimate_col="legitimate",
        )


def test_only_legitimate_events_enter_exposure_counts_flags_strata_and_permutations():
    frame = pd.DataFrame(
        {
            "event_id": ["a1", "a2", "b1", "b2", "fraud-1", "fraud-2"],
            "entity": ["A", "A", "B", "B", None, "Z"],
            "legitimate": [1, 1, 1, 1, 0, 0],
            "flag": [1, 1, 0, 0, "ignored", "ignored"],
            "stratum": ["s", "s", "s", "s", None, "fraud-only"],
        }
    )
    result = audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        exposure=ExposureSpec(2, 2),
        permutation=PermutationSpec(20, 4),
        flag_col="flag",
        legitimate_col="legitimate",
    )
    assert result.population.input_event_count == 6
    assert result.population.legitimate_event_count == 4
    assert result.population.entity_count_before_exposure_filter == 2
    assert result.counts.false_positive_event_count == 2
    assert result.diagnostics.final_stratum_count == 1
    assert result.diagnostics.final_stratum_size_distribution == {4: 1}


def test_structural_diagnostics_losslessly_reconstruct_counts():
    result = _audit_external(_base_frame())
    exposure_distribution = result.diagnostics.entity_legitimate_exposure_distribution
    assert sum(exposure_distribution.values()) == result.population.entity_count_before_exposure_filter
    reconstructed = sum(
        count
        for exposure, count in exposure_distribution.items()
        if result.population.exposure_min <= exposure <= result.population.exposure_max
    )
    assert reconstructed == result.population.eligible_entity_count
    assert result.diagnostics.eligible_entity_count_from_exposure_distribution == reconstructed
    assert result.diagnostics.eligible_entity_count_matches_population
    size_distribution = result.diagnostics.final_stratum_size_distribution
    assert sum(size * count for size, count in size_distribution.items()) == (
        result.population.legitimate_event_count
    )
    assert sum(size_distribution.values()) == result.diagnostics.final_stratum_count


def test_dataframe_row_shuffle_leaves_all_scientific_outputs_and_diagnostics_identical():
    frame = _base_frame()
    original = _audit_external(frame, permutation=PermutationSpec(250, 2026091601))
    shuffled = _audit_external(
        frame.sample(frac=1, random_state=812).copy(),
        permutation=PermutationSpec(250, 2026091601),
    )
    assert shuffled.population == original.population
    assert shuffled.counts == original.counts
    assert shuffled.estimates == original.estimates
    assert shuffled.reference == original.reference
    assert shuffled.diagnostics == original.diagnostics
    assert replace(
        shuffled.provenance,
        execution_timestamp_utc=original.provenance.execution_timestamp_utc,
    ) == original.provenance
