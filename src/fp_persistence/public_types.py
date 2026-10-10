"""Immutable public value objects for the v1.0 audit API.

This module contains configuration and result containers only.  It deliberately
does not implement audit statistics, score selection, or permutation logic.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Hashable, Literal, Mapping, TypeAlias

import pandas as pd


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class AuditValidationError(ValueError):
    """A public API value object is structurally invalid."""


class AuditStatus(str, Enum):
    """Completed scientific audit states."""

    ESTIMABLE = "ESTIMABLE"
    NOT_ESTIMABLE = "NOT_ESTIMABLE"


class RatioState(str, Enum):
    """Explicit states for the observed/reference ratio."""

    FINITE = "FINITE"
    POSITIVE_INFINITY = "POSITIVE_INFINITY"
    UNDEFINED_ZERO_OVER_ZERO = "UNDEFINED_ZERO_OVER_ZERO"
    NOT_ESTIMABLE = "NOT_ESTIMABLE"


def _require_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AuditValidationError(f"{name} must be an integer at least {minimum}")
    return value


def _finite_float_or_none(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditValidationError(f"{name} must be a finite number or None")
    converted = float(value)
    if not math.isfinite(converted):
        raise AuditValidationError(f"{name} must be finite when defined")
    return converted


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuditValidationError(f"{name} must be a nonempty string")
    return value


def _freeze_json(value: object, path: str) -> JsonValue | tuple[Any, ...] | Mapping[str, Any]:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AuditValidationError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AuditValidationError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise AuditValidationError(f"{path} must contain only JSON-compatible values")


def _freeze_int_distribution(value: Mapping[int, int], name: str) -> Mapping[int, int]:
    if not isinstance(value, Mapping):
        raise AuditValidationError(f"{name} must be a mapping")
    frozen: dict[int, int] = {}
    for key, count in value.items():
        _require_integer(key, f"{name} key")
        _require_integer(count, f"{name}[{key}]")
        if count == 0:
            raise AuditValidationError(f"{name} must omit unrealized zero-count categories")
        frozen[key] = count
    return MappingProxyType(dict(sorted(frozen.items())))


def _json_safe(value: object) -> JsonValue:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        rendered: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise AuditValidationError("serialized mapping keys must be strings or integers")
            rendered[str(key)] = _json_safe(item)
        return rendered
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AuditValidationError("result contains a non-finite numeric value")
        return value
    raise AuditValidationError(f"unsupported result value type: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_safe(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class ExposureSpec:
    min_legitimate_events: int
    max_legitimate_events: int | None

    def __post_init__(self) -> None:
        _require_integer(self.min_legitimate_events, "min_legitimate_events", minimum=1)
        if self.max_legitimate_events is not None:
            _require_integer(self.max_legitimate_events, "max_legitimate_events", minimum=1)
            if self.max_legitimate_events < self.min_legitimate_events:
                raise AuditValidationError(
                    "max_legitimate_events must be at least min_legitimate_events"
                )


@dataclass(frozen=True)
class PermutationSpec:
    replicates: int
    seed: int
    interval_quantiles: tuple[float, float] = (0.025, 0.975)
    quantile_method: Literal["linear"] = "linear"

    def __post_init__(self) -> None:
        _require_integer(self.replicates, "replicates", minimum=1)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise AuditValidationError("seed must be an integer and not a Boolean")
        if not isinstance(self.interval_quantiles, tuple) or len(self.interval_quantiles) != 2:
            raise AuditValidationError("interval_quantiles must be a two-item tuple")
        lower = _finite_float_or_none(self.interval_quantiles[0], "lower interval quantile")
        upper = _finite_float_or_none(self.interval_quantiles[1], "upper interval quantile")
        assert lower is not None and upper is not None
        if not 0 <= lower < upper <= 1:
            raise AuditValidationError("interval quantiles must satisfy 0 <= lower < upper <= 1")
        object.__setattr__(self, "interval_quantiles", (lower, upper))
        if self.quantile_method != "linear":
            raise AuditValidationError("quantile_method must be 'linear' in v1.0")


@dataclass(frozen=True)
class RatioValue:
    state: RatioState
    value: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.state, RatioState):
            raise AuditValidationError("state must be a RatioState")
        if self.state is RatioState.FINITE:
            finite = _finite_float_or_none(self.value, "finite ratio value")
            if finite is None:
                raise AuditValidationError("FINITE ratio state requires a value")
            object.__setattr__(self, "value", finite)
        elif self.value is not None:
            raise AuditValidationError(f"{self.state.value} ratio state requires value=None")


@dataclass(frozen=True)
class FlagSelection:
    flags_by_event_id: pd.Series
    selected_event_ids: tuple[Hashable, ...]
    legitimate_event_count: int
    n_select: int
    selected_event_count: int
    score_col: str
    event_id_col: str
    legitimate_col: str | None
    ordering: Literal["score_desc_event_id_asc"]
    selected_event_ids_sha256: str
    flags_by_event_id_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.flags_by_event_id, pd.Series):
            raise AuditValidationError("flags_by_event_id must be a pandas.Series")
        flags = self.flags_by_event_id.copy(deep=True)
        if flags.index.hasnans or not flags.index.is_unique:
            raise AuditValidationError("flag event IDs must be nonmissing and unique")
        try:
            sorted(flags.index.tolist())
        except TypeError as error:
            raise AuditValidationError("flag event IDs must be consistently sortable") from error
        if flags.isna().any() or not flags.isin([0, 1, False, True]).all():
            raise AuditValidationError("flags_by_event_id values must be strictly binary")
        _require_integer(self.legitimate_event_count, "legitimate_event_count")
        _require_integer(self.n_select, "n_select")
        _require_integer(self.selected_event_count, "selected_event_count")
        if len(flags) != self.legitimate_event_count:
            raise AuditValidationError("flags_by_event_id length must equal legitimate_event_count")
        if self.n_select > self.legitimate_event_count:
            raise AuditValidationError("n_select must not exceed legitimate_event_count")
        selected = tuple(self.selected_event_ids)
        if len(selected) != self.n_select or self.selected_event_count != self.n_select:
            raise AuditValidationError("selected-event counts must equal n_select")
        if len(set(selected)) != len(selected):
            raise AuditValidationError("selected_event_ids must be unique")
        if any(event_id not in flags.index for event_id in selected):
            raise AuditValidationError("selected_event_ids must occur in flags_by_event_id")
        if int(flags.sum()) != self.n_select:
            raise AuditValidationError("the binary flag total must equal n_select")
        _require_nonempty_string(self.score_col, "score_col")
        _require_nonempty_string(self.event_id_col, "event_id_col")
        if self.legitimate_col is not None:
            _require_nonempty_string(self.legitimate_col, "legitimate_col")
        if self.ordering != "score_desc_event_id_asc":
            raise AuditValidationError("ordering must be 'score_desc_event_id_asc'")
        sha256_pattern = re.compile(r"[0-9a-f]{64}")
        for name in ("selected_event_ids_sha256", "flags_by_event_id_sha256"):
            if not sha256_pattern.fullmatch(getattr(self, name)):
                raise AuditValidationError(f"{name} must be a lowercase SHA256 hex digest")
        try:
            flags.to_numpy(copy=False).setflags(write=False)
        except ValueError:
            pass
        object.__setattr__(self, "flags_by_event_id", flags)
        object.__setattr__(self, "selected_event_ids", selected)


@dataclass(frozen=True)
class PopulationResult:
    input_event_count: int
    legitimate_event_count: int
    exposure_min: int
    exposure_max: int | None
    entity_count_before_exposure_filter: int
    eligible_entity_count: int

    def __post_init__(self) -> None:
        _require_integer(self.input_event_count, "input_event_count")
        _require_integer(self.legitimate_event_count, "legitimate_event_count")
        _require_integer(self.exposure_min, "exposure_min", minimum=1)
        if self.exposure_max is not None:
            _require_integer(self.exposure_max, "exposure_max", minimum=1)
            if self.exposure_max < self.exposure_min:
                raise AuditValidationError("exposure_max must be at least exposure_min")
        _require_integer(
            self.entity_count_before_exposure_filter, "entity_count_before_exposure_filter"
        )
        _require_integer(self.eligible_entity_count, "eligible_entity_count")
        if self.legitimate_event_count > self.input_event_count:
            raise AuditValidationError("legitimate_event_count must not exceed input_event_count")
        if self.eligible_entity_count > self.entity_count_before_exposure_filter:
            raise AuditValidationError(
                "eligible_entity_count must not exceed entity_count_before_exposure_filter"
            )


@dataclass(frozen=True)
class CountResult:
    false_positive_event_count: int
    repeated_fp_entity_count: int
    excess_repeated_fp_entity_count: float | None

    def __post_init__(self) -> None:
        _require_integer(self.false_positive_event_count, "false_positive_event_count")
        _require_integer(self.repeated_fp_entity_count, "repeated_fp_entity_count")
        object.__setattr__(
            self,
            "excess_repeated_fp_entity_count",
            _finite_float_or_none(
                self.excess_repeated_fp_entity_count, "excess_repeated_fp_entity_count"
            ),
        )


@dataclass(frozen=True)
class EstimateResult:
    observed_repeat_fp_proportion: float | None
    reference_expected_repeat_fp_proportion: float | None
    excess_repeat_fp_percentage_points: float | None
    observed_expected_ratio: RatioValue

    def __post_init__(self) -> None:
        for name in (
            "observed_repeat_fp_proportion",
            "reference_expected_repeat_fp_proportion",
            "excess_repeat_fp_percentage_points",
        ):
            object.__setattr__(self, name, _finite_float_or_none(getattr(self, name), name))
        if not isinstance(self.observed_expected_ratio, RatioValue):
            raise AuditValidationError("observed_expected_ratio must be a RatioValue")


@dataclass(frozen=True)
class ReferenceResult:
    status: Literal["COMPLETED", "NOT_RUN_NOT_ESTIMABLE"]
    replicates: int
    seed: int
    rng: str
    final_stratum_count: int
    interval_quantiles: tuple[float, float]
    quantile_method: Literal["linear"]
    interval_lower: float | None
    interval_upper: float | None

    def __post_init__(self) -> None:
        if self.status not in {"COMPLETED", "NOT_RUN_NOT_ESTIMABLE"}:
            raise AuditValidationError("invalid reference status")
        _require_integer(self.replicates, "reference replicates")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise AuditValidationError("reference seed must be an integer and not a Boolean")
        _require_nonempty_string(self.rng, "rng")
        _require_integer(self.final_stratum_count, "final_stratum_count")
        if not isinstance(self.interval_quantiles, tuple) or len(self.interval_quantiles) != 2:
            raise AuditValidationError("interval_quantiles must be a two-item tuple")
        lower_q = _finite_float_or_none(self.interval_quantiles[0], "lower interval quantile")
        upper_q = _finite_float_or_none(self.interval_quantiles[1], "upper interval quantile")
        assert lower_q is not None and upper_q is not None
        if not 0 <= lower_q < upper_q <= 1:
            raise AuditValidationError("interval quantiles must satisfy 0 <= lower < upper <= 1")
        object.__setattr__(self, "interval_quantiles", (lower_q, upper_q))
        if self.quantile_method != "linear":
            raise AuditValidationError("quantile_method must be 'linear' in v1.0")
        interval_lower = _finite_float_or_none(self.interval_lower, "interval_lower")
        interval_upper = _finite_float_or_none(self.interval_upper, "interval_upper")
        object.__setattr__(self, "interval_lower", interval_lower)
        object.__setattr__(self, "interval_upper", interval_upper)
        if self.status == "COMPLETED":
            if self.replicates < 1 or interval_lower is None or interval_upper is None:
                raise AuditValidationError(
                    "COMPLETED reference requires replicates and finite interval endpoints"
                )
        elif interval_lower is not None or interval_upper is not None:
            raise AuditValidationError(
                "NOT_RUN_NOT_ESTIMABLE reference requires undefined interval endpoints"
            )


@dataclass(frozen=True)
class AuditDiagnostics:
    flag_mode: Literal["external_flags", "score_selection"]
    entity_col: str
    flag_col: str | None
    legitimate_col: str | None
    event_id_col: str
    final_stratum_col: str
    all_required_values_valid: bool
    entity_legitimate_exposure_distribution: Mapping[int, int]
    final_stratum_count: int
    final_stratum_size_distribution: Mapping[int, int]
    minimum_final_stratum_size: int | None
    maximum_final_stratum_size: int | None
    singleton_final_stratum_count: int
    eligible_entity_count_from_exposure_distribution: int
    eligible_entity_count_matches_population: bool

    def __post_init__(self) -> None:
        if self.flag_mode not in {"external_flags", "score_selection"}:
            raise AuditValidationError("invalid flag_mode")
        for name in ("entity_col", "event_id_col", "final_stratum_col"):
            _require_nonempty_string(getattr(self, name), name)
        if self.flag_col is not None:
            _require_nonempty_string(self.flag_col, "flag_col")
        if self.legitimate_col is not None:
            _require_nonempty_string(self.legitimate_col, "legitimate_col")
        if self.flag_mode == "external_flags" and self.flag_col is None:
            raise AuditValidationError("external_flags mode requires flag_col")
        if self.flag_mode == "score_selection" and self.flag_col is not None:
            raise AuditValidationError("score_selection mode requires flag_col=None")
        if not isinstance(self.all_required_values_valid, bool):
            raise AuditValidationError("all_required_values_valid must be Boolean")
        exposure_distribution = _freeze_int_distribution(
            self.entity_legitimate_exposure_distribution,
            "entity_legitimate_exposure_distribution",
        )
        stratum_distribution = _freeze_int_distribution(
            self.final_stratum_size_distribution, "final_stratum_size_distribution"
        )
        object.__setattr__(
            self, "entity_legitimate_exposure_distribution", exposure_distribution
        )
        object.__setattr__(self, "final_stratum_size_distribution", stratum_distribution)
        _require_integer(self.final_stratum_count, "final_stratum_count")
        _require_integer(self.singleton_final_stratum_count, "singleton_final_stratum_count")
        _require_integer(
            self.eligible_entity_count_from_exposure_distribution,
            "eligible_entity_count_from_exposure_distribution",
        )
        if not isinstance(self.eligible_entity_count_matches_population, bool):
            raise AuditValidationError(
                "eligible_entity_count_matches_population must be Boolean"
            )
        if sum(stratum_distribution.values()) != self.final_stratum_count:
            raise AuditValidationError(
                "final_stratum_size_distribution must reconstruct final_stratum_count"
            )
        if stratum_distribution.get(1, 0) != self.singleton_final_stratum_count:
            raise AuditValidationError(
                "singleton_final_stratum_count must match final_stratum_size_distribution"
            )
        if self.final_stratum_count == 0:
            if self.minimum_final_stratum_size is not None or self.maximum_final_stratum_size is not None:
                raise AuditValidationError("empty final strata require undefined minimum and maximum")
        else:
            if self.minimum_final_stratum_size is None or self.maximum_final_stratum_size is None:
                raise AuditValidationError("nonempty final strata require minimum and maximum sizes")
            _require_integer(self.minimum_final_stratum_size, "minimum_final_stratum_size", minimum=1)
            _require_integer(self.maximum_final_stratum_size, "maximum_final_stratum_size", minimum=1)
            if self.minimum_final_stratum_size != min(stratum_distribution):
                raise AuditValidationError("minimum_final_stratum_size does not match distribution")
            if self.maximum_final_stratum_size != max(stratum_distribution):
                raise AuditValidationError("maximum_final_stratum_size does not match distribution")


_PROVENANCE_FIELD_NAMES = {
    "package_name",
    "package_version",
    "result_schema_version",
    "statistical_contract_commit",
    "statistical_contract_sha256",
    "execution_timestamp_utc",
    "execution_profile",
    "flag_mode",
    "column_mapping",
    "exposure_spec",
    "context_source",
    "permutation_spec",
    "score_selection_spec",
    "python_version",
    "numpy_version",
    "pandas_version",
    "user_metadata",
}


@dataclass(frozen=True)
class AuditProvenance:
    package_name: str
    package_version: str
    result_schema_version: str
    statistical_contract_commit: str
    statistical_contract_sha256: str
    execution_timestamp_utc: str
    execution_profile: Literal["generic", "persistence_paper_v1"]
    flag_mode: Literal["external_flags", "score_selection"]
    column_mapping: Mapping[str, JsonValue]
    exposure_spec: ExposureSpec
    context_source: str
    permutation_spec: PermutationSpec
    score_selection_spec: Mapping[str, JsonValue] | None
    python_version: str
    numpy_version: str
    pandas_version: str
    user_metadata: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        for name in (
            "package_name",
            "package_version",
            "result_schema_version",
            "statistical_contract_commit",
            "statistical_contract_sha256",
            "execution_timestamp_utc",
            "context_source",
            "python_version",
            "numpy_version",
            "pandas_version",
        ):
            _require_nonempty_string(getattr(self, name), name)
        if self.execution_profile not in {"generic", "persistence_paper_v1"}:
            raise AuditValidationError("invalid execution_profile")
        if self.flag_mode not in {"external_flags", "score_selection"}:
            raise AuditValidationError("invalid provenance flag_mode")
        if not isinstance(self.exposure_spec, ExposureSpec):
            raise AuditValidationError("exposure_spec must be an ExposureSpec")
        if not isinstance(self.permutation_spec, PermutationSpec):
            raise AuditValidationError("permutation_spec must be a PermutationSpec")
        column_mapping = _freeze_json(self.column_mapping, "column_mapping")
        if not isinstance(column_mapping, Mapping):
            raise AuditValidationError("column_mapping must be a mapping")
        score_selection_spec = None
        if self.score_selection_spec is not None:
            score_selection_spec = _freeze_json(
                self.score_selection_spec, "score_selection_spec"
            )
            if not isinstance(score_selection_spec, Mapping):
                raise AuditValidationError("score_selection_spec must be a mapping or None")
        user_metadata = _freeze_json(self.user_metadata, "user_metadata")
        if not isinstance(user_metadata, Mapping):
            raise AuditValidationError("user_metadata must be a mapping")
        reserved = _PROVENANCE_FIELD_NAMES.intersection(user_metadata)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise AuditValidationError(
                f"user_metadata must not override reserved provenance fields: {names}"
            )
        object.__setattr__(self, "column_mapping", column_mapping)
        object.__setattr__(self, "score_selection_spec", score_selection_spec)
        object.__setattr__(self, "user_metadata", user_metadata)


@dataclass(frozen=True)
class AuditResult:
    schema_version: str
    status: AuditStatus
    reason: str | None
    population: PopulationResult
    counts: CountResult
    estimates: EstimateResult
    reference: ReferenceResult
    diagnostics: AuditDiagnostics
    provenance: AuditProvenance

    def __post_init__(self) -> None:
        _require_nonempty_string(self.schema_version, "schema_version")
        if not isinstance(self.status, AuditStatus):
            raise AuditValidationError("status must be an AuditStatus")
        for name, expected_type in (
            ("population", PopulationResult),
            ("counts", CountResult),
            ("estimates", EstimateResult),
            ("reference", ReferenceResult),
            ("diagnostics", AuditDiagnostics),
            ("provenance", AuditProvenance),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise AuditValidationError(f"{name} must be a {expected_type.__name__}")
        if self.schema_version != self.provenance.result_schema_version:
            raise AuditValidationError(
                "schema_version must match provenance.result_schema_version"
            )
        if self.status is AuditStatus.ESTIMABLE:
            if self.reason is not None:
                raise AuditValidationError("ESTIMABLE result requires reason=None")
            if self.population.eligible_entity_count < 1:
                raise AuditValidationError("ESTIMABLE result requires eligible entities")
            if self.reference.status != "COMPLETED":
                raise AuditValidationError("ESTIMABLE result requires a completed reference")
            if any(
                value is None
                for value in (
                    self.estimates.observed_repeat_fp_proportion,
                    self.estimates.reference_expected_repeat_fp_proportion,
                    self.estimates.excess_repeat_fp_percentage_points,
                    self.counts.excess_repeated_fp_entity_count,
                )
            ):
                raise AuditValidationError("ESTIMABLE result requires defined scientific fields")
            if self.estimates.observed_expected_ratio.state is RatioState.NOT_ESTIMABLE:
                raise AuditValidationError("ESTIMABLE result cannot use NOT_ESTIMABLE ratio state")
        else:
            if self.reason != "zero_eligible_entities":
                raise AuditValidationError(
                    "NOT_ESTIMABLE result requires reason='zero_eligible_entities'"
                )
            if self.population.eligible_entity_count != 0:
                raise AuditValidationError("NOT_ESTIMABLE result requires zero eligible entities")
            if self.counts.repeated_fp_entity_count != 0:
                raise AuditValidationError(
                    "NOT_ESTIMABLE result requires repeated_fp_entity_count=0"
                )
            if self.counts.excess_repeated_fp_entity_count is not None:
                raise AuditValidationError("NOT_ESTIMABLE excess entity count must be undefined")
            if any(
                value is not None
                for value in (
                    self.estimates.observed_repeat_fp_proportion,
                    self.estimates.reference_expected_repeat_fp_proportion,
                    self.estimates.excess_repeat_fp_percentage_points,
                )
            ):
                raise AuditValidationError("NOT_ESTIMABLE scientific estimates must be undefined")
            if self.estimates.observed_expected_ratio.state is not RatioState.NOT_ESTIMABLE:
                raise AuditValidationError("NOT_ESTIMABLE result requires NOT_ESTIMABLE ratio state")
            if self.reference.status != "NOT_RUN_NOT_ESTIMABLE":
                raise AuditValidationError(
                    "NOT_ESTIMABLE result requires a non-run reference"
                )

    def to_dict(self) -> dict[str, JsonValue]:
        rendered = _json_safe(self)
        assert isinstance(rendered, dict)
        return rendered

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def to_csv_row(self) -> dict[str, str | int | float | None]:
        population = self.population
        counts = self.counts
        estimates = self.estimates
        reference = self.reference
        diagnostics = self.diagnostics
        provenance = self.provenance
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "reason": self.reason,
            "population_input_event_count": population.input_event_count,
            "population_legitimate_event_count": population.legitimate_event_count,
            "population_exposure_min": population.exposure_min,
            "population_exposure_max": population.exposure_max,
            "population_entity_count_before_exposure_filter": population.entity_count_before_exposure_filter,
            "population_eligible_entity_count": population.eligible_entity_count,
            "counts_false_positive_event_count": counts.false_positive_event_count,
            "counts_repeated_fp_entity_count": counts.repeated_fp_entity_count,
            "counts_excess_repeated_fp_entity_count": counts.excess_repeated_fp_entity_count,
            "estimates_observed_repeat_fp_proportion": estimates.observed_repeat_fp_proportion,
            "estimates_reference_expected_repeat_fp_proportion": estimates.reference_expected_repeat_fp_proportion,
            "estimates_excess_repeat_fp_percentage_points": estimates.excess_repeat_fp_percentage_points,
            "observed_expected_ratio_state": estimates.observed_expected_ratio.state.value,
            "observed_expected_ratio_value": estimates.observed_expected_ratio.value,
            "reference_status": reference.status,
            "reference_replicates": reference.replicates,
            "reference_seed": reference.seed,
            "reference_rng": reference.rng,
            "reference_final_stratum_count": reference.final_stratum_count,
            "reference_interval_quantiles_json": _canonical_json(reference.interval_quantiles),
            "reference_quantile_method": reference.quantile_method,
            "reference_interval_lower": reference.interval_lower,
            "reference_interval_upper": reference.interval_upper,
            "diagnostics_flag_mode": diagnostics.flag_mode,
            "diagnostics_entity_col": diagnostics.entity_col,
            "diagnostics_flag_col": diagnostics.flag_col,
            "diagnostics_legitimate_col": diagnostics.legitimate_col,
            "diagnostics_event_id_col": diagnostics.event_id_col,
            "diagnostics_final_stratum_col": diagnostics.final_stratum_col,
            "diagnostics_all_required_values_valid": diagnostics.all_required_values_valid,
            "diagnostics_entity_legitimate_exposure_distribution_json": _canonical_json(
                diagnostics.entity_legitimate_exposure_distribution
            ),
            "diagnostics_final_stratum_count": diagnostics.final_stratum_count,
            "diagnostics_final_stratum_size_distribution_json": _canonical_json(
                diagnostics.final_stratum_size_distribution
            ),
            "diagnostics_minimum_final_stratum_size": diagnostics.minimum_final_stratum_size,
            "diagnostics_maximum_final_stratum_size": diagnostics.maximum_final_stratum_size,
            "diagnostics_singleton_final_stratum_count": diagnostics.singleton_final_stratum_count,
            "diagnostics_eligible_entity_count_from_exposure_distribution": diagnostics.eligible_entity_count_from_exposure_distribution,
            "diagnostics_eligible_entity_count_matches_population": diagnostics.eligible_entity_count_matches_population,
            "provenance_package_name": provenance.package_name,
            "provenance_package_version": provenance.package_version,
            "provenance_result_schema_version": provenance.result_schema_version,
            "provenance_statistical_contract_commit": provenance.statistical_contract_commit,
            "provenance_statistical_contract_sha256": provenance.statistical_contract_sha256,
            "provenance_execution_timestamp_utc": provenance.execution_timestamp_utc,
            "provenance_execution_profile": provenance.execution_profile,
            "provenance_flag_mode": provenance.flag_mode,
            "provenance_column_mapping_json": _canonical_json(provenance.column_mapping),
            "provenance_exposure_spec_min_legitimate_events": provenance.exposure_spec.min_legitimate_events,
            "provenance_exposure_spec_max_legitimate_events": provenance.exposure_spec.max_legitimate_events,
            "provenance_context_source": provenance.context_source,
            "provenance_permutation_spec_replicates": provenance.permutation_spec.replicates,
            "provenance_permutation_spec_seed": provenance.permutation_spec.seed,
            "provenance_permutation_spec_interval_quantiles_json": _canonical_json(
                provenance.permutation_spec.interval_quantiles
            ),
            "provenance_permutation_spec_quantile_method": provenance.permutation_spec.quantile_method,
            "provenance_score_selection_spec_json": (
                None
                if provenance.score_selection_spec is None
                else _canonical_json(provenance.score_selection_spec)
            ),
            "provenance_python_version": provenance.python_version,
            "provenance_numpy_version": provenance.numpy_version,
            "provenance_pandas_version": provenance.pandas_version,
            "provenance_user_metadata_json": _canonical_json(provenance.user_metadata),
        }


__all__ = [
    "AuditDiagnostics",
    "AuditProvenance",
    "AuditResult",
    "AuditStatus",
    "AuditValidationError",
    "CountResult",
    "EstimateResult",
    "ExposureSpec",
    "FlagSelection",
    "JsonValue",
    "PermutationSpec",
    "PopulationResult",
    "RatioState",
    "RatioValue",
    "ReferenceResult",
]
