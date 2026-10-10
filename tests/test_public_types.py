import json
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from fp_persistence.public_types import (
    AuditDiagnostics,
    AuditProvenance,
    AuditResult,
    AuditStatus,
    AuditValidationError,
    CountResult,
    EstimateResult,
    ExposureSpec,
    FlagSelection,
    PermutationSpec,
    PopulationResult,
    RatioState,
    RatioValue,
    ReferenceResult,
)


def _provenance(
    exposure: ExposureSpec,
    permutation: PermutationSpec,
    *,
    metadata=None,
) -> AuditProvenance:
    return AuditProvenance(
        package_name="fp-persistence-audit",
        package_version="1.0.0",
        result_schema_version="1.0",
        statistical_contract_commit="c95c58e5886d88db4ab9678227d7ea6089d2f178",
        statistical_contract_sha256=(
            "abf8af44e3e0a4869a2675b9a3fd96d372f7dd15612e1d5ae751995434062816"
        ),
        execution_timestamp_utc="2026-10-09T12:00:00Z",
        execution_profile="generic",
        flag_mode="external_flags",
        column_mapping={"entity": "entity_id", "flag": "is_flagged"},
        exposure_spec=exposure,
        context_source="resolved_final_strata",
        permutation_spec=permutation,
        score_selection_spec=None,
        python_version="3.10.19",
        numpy_version="2.2.5",
        pandas_version="2.3.3",
        user_metadata={} if metadata is None else metadata,
    )


def _estimable_result(*, ratio=None, metadata=None) -> AuditResult:
    exposure = ExposureSpec(2, 4)
    permutation = PermutationSpec(10, 7)
    return AuditResult(
        schema_version="1.0",
        status=AuditStatus.ESTIMABLE,
        reason=None,
        population=PopulationResult(
            input_event_count=8,
            legitimate_event_count=8,
            exposure_min=2,
            exposure_max=4,
            entity_count_before_exposure_filter=4,
            eligible_entity_count=3,
        ),
        counts=CountResult(
            false_positive_event_count=2,
            repeated_fp_entity_count=1,
            excess_repeated_fp_entity_count=0.4,
        ),
        estimates=EstimateResult(
            observed_repeat_fp_proportion=1 / 3,
            reference_expected_repeat_fp_proportion=0.2,
            excess_repeat_fp_percentage_points=13.333333333333332,
            observed_expected_ratio=ratio or RatioValue(RatioState.FINITE, 5 / 3),
        ),
        reference=ReferenceResult(
            status="COMPLETED",
            replicates=10,
            seed=7,
            rng="numpy.random.default_rng",
            final_stratum_count=2,
            interval_quantiles=(0.025, 0.975),
            quantile_method="linear",
            interval_lower=0.0,
            interval_upper=2 / 3,
        ),
        diagnostics=AuditDiagnostics(
            flag_mode="external_flags",
            entity_col="entity_id",
            flag_col="is_flagged",
            legitimate_col=None,
            event_id_col="event_id",
            final_stratum_col="stratum",
            all_required_values_valid=True,
            entity_legitimate_exposure_distribution={1: 1, 2: 2, 3: 1},
            final_stratum_count=2,
            final_stratum_size_distribution={4: 2},
            minimum_final_stratum_size=4,
            maximum_final_stratum_size=4,
            singleton_final_stratum_count=0,
            eligible_entity_count_from_exposure_distribution=3,
            eligible_entity_count_matches_population=True,
        ),
        provenance=_provenance(exposure, permutation, metadata=metadata),
    )


def _not_estimable_result() -> AuditResult:
    exposure = ExposureSpec(5, 19)
    permutation = PermutationSpec(10, 7)
    return AuditResult(
        schema_version="1.0",
        status=AuditStatus.NOT_ESTIMABLE,
        reason="zero_eligible_entities",
        population=PopulationResult(
            input_event_count=2,
            legitimate_event_count=2,
            exposure_min=5,
            exposure_max=19,
            entity_count_before_exposure_filter=2,
            eligible_entity_count=0,
        ),
        counts=CountResult(
            false_positive_event_count=0,
            repeated_fp_entity_count=0,
            excess_repeated_fp_entity_count=None,
        ),
        estimates=EstimateResult(
            observed_repeat_fp_proportion=None,
            reference_expected_repeat_fp_proportion=None,
            excess_repeat_fp_percentage_points=None,
            observed_expected_ratio=RatioValue(RatioState.NOT_ESTIMABLE, None),
        ),
        reference=ReferenceResult(
            status="NOT_RUN_NOT_ESTIMABLE",
            replicates=0,
            seed=7,
            rng="numpy.random.default_rng",
            final_stratum_count=1,
            interval_quantiles=(0.025, 0.975),
            quantile_method="linear",
            interval_lower=None,
            interval_upper=None,
        ),
        diagnostics=AuditDiagnostics(
            flag_mode="external_flags",
            entity_col="entity_id",
            flag_col="is_flagged",
            legitimate_col=None,
            event_id_col="event_id",
            final_stratum_col="stratum",
            all_required_values_valid=True,
            entity_legitimate_exposure_distribution={1: 2},
            final_stratum_count=1,
            final_stratum_size_distribution={2: 1},
            minimum_final_stratum_size=2,
            maximum_final_stratum_size=2,
            singleton_final_stratum_count=0,
            eligible_entity_count_from_exposure_distribution=0,
            eligible_entity_count_matches_population=True,
        ),
        provenance=_provenance(exposure, permutation),
    )


def test_public_value_objects_are_frozen_and_defensively_freeze_mappings():
    result = _estimable_result(metadata={"study": {"phase": "test"}})
    with pytest.raises(FrozenInstanceError):
        result.status = AuditStatus.NOT_ESTIMABLE
    with pytest.raises(TypeError):
        result.provenance.user_metadata["new"] = "value"
    with pytest.raises(TypeError):
        result.diagnostics.entity_legitimate_exposure_distribution[9] = 1


@pytest.mark.parametrize(
    "minimum,maximum",
    [(0, None), (True, None), (5, 4), (5, True), (1, 0)],
)
def test_exposure_spec_rejects_invalid_bounds(minimum, maximum):
    with pytest.raises(AuditValidationError):
        ExposureSpec(minimum, maximum)


def test_exposure_spec_accepts_inclusive_open_or_closed_upper_bound():
    assert ExposureSpec(5, 19) == ExposureSpec(5, 19)
    assert ExposureSpec(20, None).max_legitimate_events is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replicates": 0, "seed": 1},
        {"replicates": True, "seed": 1},
        {"replicates": 10, "seed": True},
        {"replicates": 10, "seed": 1, "interval_quantiles": (0.5, 0.5)},
        {"replicates": 10, "seed": 1, "interval_quantiles": (float("nan"), 0.9)},
        {"replicates": 10, "seed": 1, "quantile_method": "nearest"},
    ],
)
def test_permutation_spec_rejects_invalid_values(kwargs):
    with pytest.raises(AuditValidationError):
        PermutationSpec(**kwargs)


def test_permutation_spec_preserves_frozen_defaults():
    spec = PermutationSpec(2000, 2026091601)
    assert spec.interval_quantiles == (0.025, 0.975)
    assert spec.quantile_method == "linear"


@pytest.mark.parametrize(
    "state,value",
    [
        (RatioState.FINITE, 1.25),
        (RatioState.FINITE, 0.0),
        (RatioState.POSITIVE_INFINITY, None),
        (RatioState.UNDEFINED_ZERO_OVER_ZERO, None),
        (RatioState.NOT_ESTIMABLE, None),
    ],
)
def test_ratio_value_supports_every_explicit_state(state, value):
    ratio = RatioValue(state, value)
    assert ratio.state is state
    assert ratio.value == value


@pytest.mark.parametrize(
    "state,value",
    [
        (RatioState.FINITE, None),
        (RatioState.FINITE, float("inf")),
        (RatioState.POSITIVE_INFINITY, 1.0),
        (RatioState.UNDEFINED_ZERO_OVER_ZERO, 0.0),
        (RatioState.NOT_ESTIMABLE, 0.0),
    ],
)
def test_ratio_value_rejects_state_value_mismatches(state, value):
    with pytest.raises(AuditValidationError):
        RatioValue(state, value)


def test_finite_zero_and_undefined_ratio_serialize_differently():
    finite_zero = _estimable_result(ratio=RatioValue(RatioState.FINITE, 0.0)).to_dict()
    undefined = _estimable_result(
        ratio=RatioValue(RatioState.UNDEFINED_ZERO_OVER_ZERO, None)
    ).to_dict()
    assert finite_zero["estimates"]["observed_expected_ratio"] == {
        "state": "FINITE",
        "value": 0.0,
    }
    assert undefined["estimates"]["observed_expected_ratio"] == {
        "state": "UNDEFINED_ZERO_OVER_ZERO",
        "value": None,
    }


def test_not_estimable_is_completed_structured_result_with_null_estimates():
    payload = _not_estimable_result().to_dict()
    assert payload["status"] == "NOT_ESTIMABLE"
    assert payload["reason"] == "zero_eligible_entities"
    assert payload["population"]["eligible_entity_count"] == 0
    assert payload["counts"]["repeated_fp_entity_count"] == 0
    assert payload["counts"]["excess_repeated_fp_entity_count"] is None
    assert payload["estimates"]["observed_repeat_fp_proportion"] is None
    assert payload["reference"]["status"] == "NOT_RUN_NOT_ESTIMABLE"


def test_json_serialization_is_standard_compliant_and_contains_no_nonfinite_numbers():
    result = _estimable_result(ratio=RatioValue(RatioState.POSITIVE_INFINITY, None))

    def reject_constant(value):
        raise AssertionError(f"non-standard numeric constant: {value}")

    encoded = result.to_json()
    decoded = json.loads(encoded, parse_constant=reject_constant)
    assert decoded["estimates"]["observed_expected_ratio"] == {
        "state": "POSITIVE_INFINITY",
        "value": None,
    }
    assert ": NaN" not in encoded
    assert ": Infinity" not in encoded
    assert ": -Infinity" not in encoded


def test_csv_row_is_flat_and_uses_ratio_companion_and_canonical_json_columns():
    row = _estimable_result(metadata={"source": "fixture"}).to_csv_row()
    assert row["status"] == "ESTIMABLE"
    assert row["population_eligible_entity_count"] == 3
    assert row["observed_expected_ratio_state"] == "FINITE"
    assert row["observed_expected_ratio_value"] == pytest.approx(5 / 3)
    assert row["diagnostics_entity_legitimate_exposure_distribution_json"] == (
        '{"1":1,"2":2,"3":1}'
    )
    assert row["provenance_user_metadata_json"] == '{"source":"fixture"}'
    assert all(not isinstance(value, (dict, list, tuple)) for value in row.values())


@pytest.mark.parametrize(
    "metadata",
    [
        {"bad": {1, 2}},
        {"bad": float("nan")},
        {1: "non-string-key"},
        {"package_version": "override"},
    ],
)
def test_provenance_rejects_invalid_or_reserved_user_metadata(metadata):
    with pytest.raises(AuditValidationError):
        _provenance(ExposureSpec(1, None), PermutationSpec(5, 1), metadata=metadata)


def test_flag_selection_is_structural_and_does_not_alias_input_series():
    flags = pd.Series([1, 0], index=["event-1", "event-2"])
    selection = FlagSelection(
        flags_by_event_id=flags,
        selected_event_ids=("event-1",),
        legitimate_event_count=2,
        n_select=1,
        selected_event_count=1,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="is_legitimate",
        ordering="score_desc_event_id_asc",
        selected_event_ids_sha256="a" * 64,
        flags_by_event_id_sha256="b" * 64,
    )
    flags.iloc[0] = 0
    assert selection.flags_by_event_id.loc["event-1"] == 1
    with pytest.raises(FrozenInstanceError):
        selection.n_select = 0


def test_flag_selection_rejects_mixed_incomparable_event_ids():
    with pytest.raises(AuditValidationError, match="consistently sortable"):
        FlagSelection(
            flags_by_event_id=pd.Series([1, 0], index=[1, "event-2"]),
            selected_event_ids=(1,),
            legitimate_event_count=2,
            n_select=1,
            selected_event_count=1,
            score_col="score",
            event_id_col="event_id",
            legitimate_col=None,
            ordering="score_desc_event_id_asc",
            selected_event_ids_sha256="a" * 64,
            flags_by_event_id_sha256="b" * 64,
        )


def test_result_schema_has_stable_required_fields_only():
    payload = _estimable_result().to_dict()
    assert set(payload) == {
        "schema_version",
        "status",
        "reason",
        "population",
        "counts",
        "estimates",
        "reference",
        "diagnostics",
        "provenance",
    }
    assert set(payload["population"]) == {
        "input_event_count",
        "legitimate_event_count",
        "exposure_min",
        "exposure_max",
        "entity_count_before_exposure_filter",
        "eligible_entity_count",
    }
    assert set(payload["counts"]) == {
        "false_positive_event_count",
        "repeated_fp_entity_count",
        "excess_repeated_fp_entity_count",
    }
    assert set(payload["estimates"]) == {
        "observed_repeat_fp_proportion",
        "reference_expected_repeat_fp_proportion",
        "excess_repeat_fp_percentage_points",
        "observed_expected_ratio",
    }
    assert set(payload["reference"]) == {
        "status",
        "replicates",
        "seed",
        "rng",
        "final_stratum_count",
        "interval_quantiles",
        "quantile_method",
        "interval_lower",
        "interval_upper",
    }
    assert set(payload["diagnostics"]) == {
        "flag_mode",
        "entity_col",
        "flag_col",
        "legitimate_col",
        "event_id_col",
        "final_stratum_col",
        "all_required_values_valid",
        "entity_legitimate_exposure_distribution",
        "final_stratum_count",
        "final_stratum_size_distribution",
        "minimum_final_stratum_size",
        "maximum_final_stratum_size",
        "singleton_final_stratum_count",
        "eligible_entity_count_from_exposure_distribution",
        "eligible_entity_count_matches_population",
    }


def test_nonfinite_scientific_fields_are_rejected_before_serialization():
    with pytest.raises(AuditValidationError):
        CountResult(0, 0, float("nan"))
    with pytest.raises(AuditValidationError):
        EstimateResult(0.0, float("inf"), 0.0, RatioValue(RatioState.FINITE, 0.0))
