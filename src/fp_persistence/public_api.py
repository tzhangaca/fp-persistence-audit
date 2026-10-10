"""Generic public single-run audit engine.

The engine accepts prespecified flags and resolved final context strata.  It
does not train models, select score thresholds, or construct context strata.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import platform
from typing import Any, Hashable, Mapping
from uuid import UUID

import numpy as np
import pandas as pd

from .persistence import compute_persistence_summary, compute_repeat_fp_by_entity
from .public_types import (
    AuditDiagnostics,
    AuditProvenance,
    AuditResult,
    AuditStatus,
    AuditValidationError,
    CountResult,
    EstimateResult,
    ExposureSpec,
    FlagSelection,
    JsonValue,
    PermutationSpec,
    PopulationResult,
    RatioState,
    RatioValue,
    ReferenceResult,
)


_RESULT_SCHEMA_VERSION = "1.0"
_STATISTICAL_CONTRACT_COMMIT = "c95c58e5886d88db4ab9678227d7ea6089d2f178"
_STATISTICAL_CONTRACT_SHA256 = (
    "abf8af44e3e0a4869a2675b9a3fd96d372f7dd15612e1d5ae751995434062816"
)
_RNG_NAME = "numpy.random.default_rng"


def _package_version() -> str:
    try:
        return importlib_metadata.version("fp-persistence-audit")
    except importlib_metadata.PackageNotFoundError:
        return "0.1.0"


def _validate_column_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuditValidationError(f"{name} must be a nonempty string")
    return value


def _strict_binary(series: pd.Series, name: str) -> pd.Series:
    values: list[int] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            values.append(int(value))
        elif isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
            values.append(int(value))
        else:
            raise AuditValidationError(f"{name} must contain only strict binary values")
    return pd.Series(values, index=series.index, dtype="int8", name=name)


def _require_nonmissing_hashable(series: pd.Series, name: str) -> None:
    if series.isna().any():
        raise AuditValidationError(f"{name} must be nonmissing")
    for value in series.tolist():
        try:
            hash(value)
        except TypeError as error:
            raise AuditValidationError(f"{name} values must be hashable") from error


def _require_consistently_sortable(values: list[Hashable], name: str) -> None:
    try:
        sorted(values)
    except (TypeError, ValueError) as error:
        raise AuditValidationError(f"{name} values must be consistently sortable") from error


def _canonical_event_id(value: Hashable) -> dict[str, JsonValue]:
    """Encode supported event-ID values with explicit, deterministic type tags."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AuditValidationError("event IDs must be finite when numeric")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise AuditValidationError("event IDs must be finite when numeric")
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, UUID):
        return {"type": "uuid", "value": str(value)}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "value": [_canonical_event_id(item) for item in value],
        }
    raise AuditValidationError(
        f"event ID type {type(value).__name__!r} has no canonical public encoding"
    )


def _typed_sha256(kind: str, items: list[JsonValue]) -> str:
    payload = {
        "encoding": "fp-persistence-event-id-v1",
        "kind": kind,
        "items": items,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_stratum_key(value: Hashable) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"


def _canonicalize_legitimate_events(
    legitimate: pd.DataFrame,
    *,
    entity_col: str,
    event_id_col: str,
    final_stratum_col: str,
    flags: pd.Series,
) -> pd.DataFrame:
    stratum_values = legitimate[final_stratum_col].tolist()
    stratum_keys: list[str] = []
    key_to_value: dict[str, Hashable] = {}
    for value in stratum_values:
        key = _stable_stratum_key(value)
        if key in key_to_value and key_to_value[key] != value:
            raise AuditValidationError(
                "final-stratum labels cannot be ordered deterministically without collision"
            )
        key_to_value[key] = value
        stratum_keys.append(key)

    event_ids = legitimate[event_id_col].tolist()
    order = sorted(range(len(legitimate)), key=lambda index: (stratum_keys[index], event_ids[index]))
    working = pd.DataFrame(
        {
            "entity": legitimate[entity_col].to_numpy(copy=True),
            "event_id": legitimate[event_id_col].to_numpy(copy=True),
            "final_stratum": stratum_keys,
            "legitimate": np.ones(len(legitimate), dtype=bool),
            "false_positive": flags.to_numpy(dtype="int8", copy=True),
        }
    )
    return working.iloc[order].reset_index(drop=True)


def _score_selection_provenance(selection: FlagSelection) -> dict[str, JsonValue]:
    return {
        "score_col": selection.score_col,
        "event_id_col": selection.event_id_col,
        "legitimate_col": selection.legitimate_col,
        "legitimate_event_count": selection.legitimate_event_count,
        "n_select": selection.n_select,
        "selected_event_count": selection.selected_event_count,
        "ordering": selection.ordering,
        "selected_event_ids_sha256": selection.selected_event_ids_sha256,
        "flags_by_event_id_sha256": selection.flags_by_event_id_sha256,
    }


def _build_provenance(
    *,
    entity_col: str,
    event_id_col: str,
    final_stratum_col: str,
    legitimate_col: str | None,
    flag_col: str | None,
    flag_mode: str,
    exposure: ExposureSpec,
    permutation: PermutationSpec,
    selection: FlagSelection | None,
    metadata: Mapping[str, JsonValue] | None,
) -> AuditProvenance:
    column_mapping: dict[str, JsonValue] = {
        "entity": entity_col,
        "event_id": event_id_col,
        "final_stratum": final_stratum_col,
        "legitimate": legitimate_col,
        "flag": flag_col,
    }
    return AuditProvenance(
        package_name="fp-persistence-audit",
        package_version=_package_version(),
        result_schema_version=_RESULT_SCHEMA_VERSION,
        statistical_contract_commit=_STATISTICAL_CONTRACT_COMMIT,
        statistical_contract_sha256=_STATISTICAL_CONTRACT_SHA256,
        execution_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        execution_profile="generic",
        flag_mode=flag_mode,
        column_mapping=column_mapping,
        exposure_spec=exposure,
        context_source="resolved_final_strata",
        permutation_spec=permutation,
        score_selection_spec=(
            None if selection is None else _score_selection_provenance(selection)
        ),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        user_metadata={} if metadata is None else metadata,
    )


def _entity_results(
    frame: pd.DataFrame,
    *,
    exposure: ExposureSpec,
) -> tuple[pd.DataFrame, Any]:
    entities = compute_repeat_fp_by_entity(
        frame,
        entity_column="entity",
        legitimate_column="legitimate",
        false_positive_column="false_positive",
        exposure_min=exposure.min_legitimate_events,
        exposure_max=exposure.max_legitimate_events,
        minimum_false_positives=2,
    )
    return entities, compute_persistence_summary(entities)


def _build_diagnostics(
    *,
    frame: pd.DataFrame,
    entities: pd.DataFrame,
    eligible_entity_count: int,
    flag_mode: str,
    entity_col: str,
    flag_col: str | None,
    legitimate_col: str | None,
    event_id_col: str,
    final_stratum_col: str,
    exposure: ExposureSpec,
) -> AuditDiagnostics:
    exposure_counts = {
        int(exposure_value): int(entity_count)
        for exposure_value, entity_count in entities["legitimate_exposure"]
        .value_counts()
        .sort_index()
        .items()
    }
    eligible_from_distribution = sum(
        entity_count
        for exposure_value, entity_count in exposure_counts.items()
        if exposure_value >= exposure.min_legitimate_events
        and (
            exposure.max_legitimate_events is None
            or exposure_value <= exposure.max_legitimate_events
        )
    )
    stratum_sizes = frame.groupby("final_stratum", sort=True).size()
    stratum_size_distribution = {
        int(size): int(count)
        for size, count in stratum_sizes.value_counts().sort_index().items()
    }
    final_stratum_count = int(len(stratum_sizes))
    return AuditDiagnostics(
        flag_mode=flag_mode,
        entity_col=entity_col,
        flag_col=flag_col,
        legitimate_col=legitimate_col,
        event_id_col=event_id_col,
        final_stratum_col=final_stratum_col,
        all_required_values_valid=True,
        entity_legitimate_exposure_distribution=exposure_counts,
        final_stratum_count=final_stratum_count,
        final_stratum_size_distribution=stratum_size_distribution,
        minimum_final_stratum_size=(None if stratum_sizes.empty else int(stratum_sizes.min())),
        maximum_final_stratum_size=(None if stratum_sizes.empty else int(stratum_sizes.max())),
        singleton_final_stratum_count=int((stratum_sizes == 1).sum()),
        eligible_entity_count_from_exposure_distribution=int(eligible_from_distribution),
        eligible_entity_count_matches_population=(
            int(eligible_from_distribution) == eligible_entity_count
        ),
    )


def _validate_and_prepare(
    data: pd.DataFrame,
    *,
    entity_col: str,
    event_id_col: str,
    final_stratum_col: str,
    exposure: ExposureSpec,
    permutation: PermutationSpec,
    flag_col: str | None,
    flag_selection: FlagSelection | None,
    legitimate_col: str | None,
    metadata: Mapping[str, JsonValue] | None,
) -> tuple[pd.DataFrame, str, AuditProvenance]:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise AuditValidationError("data must be a nonempty pandas.DataFrame")
    if not isinstance(exposure, ExposureSpec):
        raise AuditValidationError("exposure must be an ExposureSpec")
    if not isinstance(permutation, PermutationSpec):
        raise AuditValidationError("permutation must be a PermutationSpec")
    entity_col = _validate_column_name(entity_col, "entity_col")
    event_id_col = _validate_column_name(event_id_col, "event_id_col")
    final_stratum_col = _validate_column_name(final_stratum_col, "final_stratum_col")
    if legitimate_col is not None:
        legitimate_col = _validate_column_name(legitimate_col, "legitimate_col")
    if flag_col is not None:
        flag_col = _validate_column_name(flag_col, "flag_col")
    if (flag_col is None) == (flag_selection is None):
        raise AuditValidationError("exactly one of flag_col and flag_selection must be supplied")
    if flag_selection is not None and not isinstance(flag_selection, FlagSelection):
        raise AuditValidationError("flag_selection must be a FlagSelection")

    named_columns = [entity_col, event_id_col, final_stratum_col]
    if legitimate_col is not None:
        named_columns.append(legitimate_col)
    if flag_col is not None:
        named_columns.append(flag_col)
    if len(set(named_columns)) != len(named_columns):
        raise AuditValidationError("named roles must refer to distinct dataframe columns")
    missing = [column for column in named_columns if column not in data.columns]
    if missing:
        raise AuditValidationError(f"missing required dataframe columns: {sorted(missing)}")

    if legitimate_col is None:
        legitimate_mask = pd.Series(True, index=data.index)
    else:
        legitimate_mask = _strict_binary(data[legitimate_col], legitimate_col).astype(bool)

    _require_nonmissing_hashable(data[event_id_col], event_id_col)
    if data[event_id_col].duplicated().any():
        raise AuditValidationError("event IDs must be unique")
    _require_consistently_sortable(data[event_id_col].tolist(), "event IDs")

    legitimate = data.loc[legitimate_mask].copy()
    _require_nonmissing_hashable(legitimate[entity_col], entity_col)
    _require_nonmissing_hashable(legitimate[final_stratum_col], final_stratum_col)

    selection: FlagSelection | None = None
    if flag_col is not None:
        flag_mode = "external_flags"
        flags = _strict_binary(legitimate[flag_col], flag_col)
    else:
        assert flag_selection is not None
        selection = flag_selection
        flag_mode = "score_selection"
        if selection.event_id_col != event_id_col:
            raise AuditValidationError("FlagSelection event_id_col does not match audit event_id_col")
        legitimate_ids = legitimate[event_id_col].tolist()
        selection_ids = selection.flags_by_event_id.index.tolist()
        if len(selection_ids) != len(legitimate_ids) or set(selection_ids) != set(legitimate_ids):
            raise AuditValidationError(
                "FlagSelection must provide one-to-one coverage of legitimate event IDs"
            )
        selected_ids = set(selection.selected_event_ids)
        if not selected_ids <= set(legitimate_ids):
            raise AuditValidationError("selected event IDs must correspond to legitimate events")
        flagged_ids = set(
            selection.flags_by_event_id.index[
                selection.flags_by_event_id.astype(bool)
            ].tolist()
        )
        if flagged_ids != selected_ids:
            raise AuditValidationError(
                "FlagSelection selected_event_ids must match its binary flags"
            )
        flags = selection.flags_by_event_id.reindex(legitimate_ids)
        if flags.isna().any():
            raise AuditValidationError("FlagSelection alignment produced missing flags")
        flags = pd.Series(flags.to_numpy(dtype="int8"), index=legitimate.index)

    provenance = _build_provenance(
        entity_col=entity_col,
        event_id_col=event_id_col,
        final_stratum_col=final_stratum_col,
        legitimate_col=legitimate_col,
        flag_col=flag_col,
        flag_mode=flag_mode,
        exposure=exposure,
        permutation=permutation,
        selection=selection,
        metadata=metadata,
    )
    working = _canonicalize_legitimate_events(
        legitimate,
        entity_col=entity_col,
        event_id_col=event_id_col,
        final_stratum_col=final_stratum_col,
        flags=flags,
    )
    return working, flag_mode, provenance


def flags_from_scores(
    data: pd.DataFrame,
    *,
    score_col: str,
    event_id_col: str,
    n_select: int,
    legitimate_col: str | None = None,
) -> FlagSelection:
    """Construct exactly ``n_select`` deterministic flags from external scores."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise AuditValidationError("data must be a nonempty pandas.DataFrame")
    score_col = _validate_column_name(score_col, "score_col")
    event_id_col = _validate_column_name(event_id_col, "event_id_col")
    if legitimate_col is not None:
        legitimate_col = _validate_column_name(legitimate_col, "legitimate_col")
    named_columns = [score_col, event_id_col]
    if legitimate_col is not None:
        named_columns.append(legitimate_col)
    if len(set(named_columns)) != len(named_columns):
        raise AuditValidationError("named roles must refer to distinct dataframe columns")
    missing = [column for column in named_columns if column not in data.columns]
    if missing:
        raise AuditValidationError(f"missing required dataframe columns: {sorted(missing)}")
    if isinstance(n_select, bool) or not isinstance(n_select, int) or n_select < 0:
        raise AuditValidationError("n_select must be a nonnegative integer")

    if legitimate_col is None:
        legitimate_mask = pd.Series(True, index=data.index)
    else:
        legitimate_mask = _strict_binary(data[legitimate_col], legitimate_col).astype(bool)

    _require_nonmissing_hashable(data[event_id_col], event_id_col)
    if data[event_id_col].duplicated().any():
        raise AuditValidationError("event IDs must be unique")
    all_event_ids = data[event_id_col].tolist()
    _require_consistently_sortable(all_event_ids, "event IDs")
    for event_id in all_event_ids:
        _canonical_event_id(event_id)

    legitimate = data.loc[legitimate_mask]
    legitimate_event_count = int(len(legitimate))
    if n_select > legitimate_event_count:
        raise AuditValidationError("n_select must not exceed the legitimate-event count")

    scored_events: list[tuple[Hashable, float]] = []
    for event_id, score in zip(
        legitimate[event_id_col].tolist(), legitimate[score_col].tolist()
    ):
        if isinstance(score, (bool, np.bool_)) or not isinstance(
            score, (int, float, np.integer, np.floating, Decimal)
        ):
            raise AuditValidationError("scores on legitimate events must be finite numbers")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise AuditValidationError("scores on legitimate events must be finite numbers")
        scored_events.append((event_id, numeric_score))

    ranked = sorted(scored_events, key=lambda item: (-item[1], item[0]))
    selected_event_ids = tuple(event_id for event_id, _ in ranked[:n_select])
    selected_set = set(selected_event_ids)
    ordered_legitimate_ids = sorted(event_id for event_id, _ in scored_events)
    flag_values = [int(event_id in selected_set) for event_id in ordered_legitimate_ids]
    flags_by_event_id = pd.Series(
        flag_values,
        index=pd.Index(ordered_legitimate_ids, name=event_id_col),
        dtype="int8",
        name="false_positive",
    )

    selected_hash_items = [
        _canonical_event_id(event_id) for event_id in selected_event_ids
    ]
    flag_hash_items = [
        {
            "event_id": _canonical_event_id(event_id),
            "flag": flag,
        }
        for event_id, flag in zip(ordered_legitimate_ids, flag_values)
    ]
    return FlagSelection(
        flags_by_event_id=flags_by_event_id,
        selected_event_ids=selected_event_ids,
        legitimate_event_count=legitimate_event_count,
        n_select=n_select,
        selected_event_count=len(selected_event_ids),
        score_col=score_col,
        event_id_col=event_id_col,
        legitimate_col=legitimate_col,
        ordering="score_desc_event_id_asc",
        selected_event_ids_sha256=_typed_sha256(
            "selected_event_ids", selected_hash_items
        ),
        flags_by_event_id_sha256=_typed_sha256(
            "flags_by_event_id", flag_hash_items
        ),
    )


def audit(
    data: pd.DataFrame,
    *,
    entity_col: str,
    event_id_col: str,
    final_stratum_col: str,
    exposure: ExposureSpec,
    permutation: PermutationSpec,
    flag_col: str | None = None,
    flag_selection: FlagSelection | None = None,
    legitimate_col: str | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> AuditResult:
    """Audit repeated false positives using resolved final context strata."""
    working, flag_mode, provenance = _validate_and_prepare(
        data,
        entity_col=entity_col,
        event_id_col=event_id_col,
        final_stratum_col=final_stratum_col,
        exposure=exposure,
        permutation=permutation,
        flag_col=flag_col,
        flag_selection=flag_selection,
        legitimate_col=legitimate_col,
        metadata=metadata,
    )
    entities, observed = _entity_results(working, exposure=exposure)
    diagnostics = _build_diagnostics(
        frame=working,
        entities=entities,
        eligible_entity_count=observed.denominator,
        flag_mode=flag_mode,
        entity_col=entity_col,
        flag_col=flag_col,
        legitimate_col=legitimate_col,
        event_id_col=event_id_col,
        final_stratum_col=final_stratum_col,
        exposure=exposure,
    )
    population = PopulationResult(
        input_event_count=int(len(data)),
        legitimate_event_count=int(len(working)),
        exposure_min=exposure.min_legitimate_events,
        exposure_max=exposure.max_legitimate_events,
        entity_count_before_exposure_filter=int(len(entities)),
        eligible_entity_count=int(observed.denominator),
    )
    false_positive_event_count = int(working["false_positive"].sum())

    if observed.denominator == 0:
        return AuditResult(
            schema_version=_RESULT_SCHEMA_VERSION,
            status=AuditStatus.NOT_ESTIMABLE,
            reason="zero_eligible_entities",
            population=population,
            counts=CountResult(
                false_positive_event_count=false_positive_event_count,
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
                seed=permutation.seed,
                rng=_RNG_NAME,
                final_stratum_count=diagnostics.final_stratum_count,
                interval_quantiles=permutation.interval_quantiles,
                quantile_method=permutation.quantile_method,
                interval_lower=None,
                interval_upper=None,
            ),
            diagnostics=diagnostics,
            provenance=provenance,
        )

    group_positions = tuple(
        group.index.to_numpy(dtype=int)
        for _, group in working.groupby("final_stratum", sort=True)
    )
    original_flags = working["false_positive"].to_numpy(dtype="int8", copy=True)
    rng = np.random.default_rng(permutation.seed)
    reference_values = np.empty(permutation.replicates, dtype=float)
    for replicate in range(permutation.replicates):
        reassigned = original_flags.copy()
        for positions in group_positions:
            reassigned[positions] = rng.permutation(original_flags[positions])
        candidate = working.copy()
        candidate["false_positive"] = reassigned
        _, candidate_summary = _entity_results(candidate, exposure=exposure)
        reference_values[replicate] = candidate_summary.proportion

    reference_expectation = float(reference_values.mean())
    interval_lower, interval_upper = (
        float(value)
        for value in np.quantile(
            reference_values,
            permutation.interval_quantiles,
            method=permutation.quantile_method,
        )
    )
    observed_proportion = float(observed.proportion)
    if reference_expectation > 0:
        ratio = RatioValue(RatioState.FINITE, observed_proportion / reference_expectation)
    elif observed_proportion > 0:
        ratio = RatioValue(RatioState.POSITIVE_INFINITY, None)
    else:
        ratio = RatioValue(RatioState.UNDEFINED_ZERO_OVER_ZERO, None)
    excess_percentage_points = (observed_proportion - reference_expectation) * 100
    excess_entity_count = (
        int(observed.numerator) - reference_expectation * int(observed.denominator)
    )
    return AuditResult(
        schema_version=_RESULT_SCHEMA_VERSION,
        status=AuditStatus.ESTIMABLE,
        reason=None,
        population=population,
        counts=CountResult(
            false_positive_event_count=false_positive_event_count,
            repeated_fp_entity_count=int(observed.numerator),
            excess_repeated_fp_entity_count=excess_entity_count,
        ),
        estimates=EstimateResult(
            observed_repeat_fp_proportion=observed_proportion,
            reference_expected_repeat_fp_proportion=reference_expectation,
            excess_repeat_fp_percentage_points=excess_percentage_points,
            observed_expected_ratio=ratio,
        ),
        reference=ReferenceResult(
            status="COMPLETED",
            replicates=permutation.replicates,
            seed=permutation.seed,
            rng=_RNG_NAME,
            final_stratum_count=diagnostics.final_stratum_count,
            interval_quantiles=permutation.interval_quantiles,
            quantile_method=permutation.quantile_method,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
        ),
        diagnostics=diagnostics,
        provenance=provenance,
    )


__all__ = ["audit", "flags_from_scores"]
