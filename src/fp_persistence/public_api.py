"""Generic public single-run audit engine.

The engine accepts prespecified flags and resolved final context strata.  It
does not train models, select score thresholds, or construct context strata.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import hmac
import json
import math
import platform
from typing import Any, Hashable, Mapping
from uuid import UUID

import numpy as np
import pandas as pd

from .persistence import compute_persistence_summary, compute_repeat_fp_by_entity
from ._version import PACKAGE_VERSION
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
    _atomic_object_index,
    _normalize_identifier_value,
    _one_dimensional_object_array,
    _require_supported_identifier_values,
)


_RESULT_SCHEMA_VERSION = "1.0"
_STATISTICAL_CONTRACT_COMMIT = "c95c58e5886d88db4ab9678227d7ea6089d2f178"
_STATISTICAL_CONTRACT_SHA256 = (
    "abf8af44e3e0a4869a2675b9a3fd96d372f7dd15612e1d5ae751995434062816"
)
_RNG_NAME = "numpy.random.default_rng"


def _validate_column_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuditValidationError(f"{name} must be a nonempty string")
    return value


def _require_unique_dataframe_columns(data: pd.DataFrame) -> None:
    if not data.columns.is_unique:
        raise AuditValidationError("dataframe column labels must be unique")


def _strict_binary(series: pd.Series, name: str) -> pd.Series:
    values: list[int] = []
    for value in series.tolist():
        if type(value) is bool or isinstance(value, np.bool_):
            values.append(int(value))
        elif (
            (type(value) is int or isinstance(value, np.integer))
            and not isinstance(value, np.bool_)
            and int(value) in {0, 1}
        ):
            values.append(int(value))
        else:
            raise AuditValidationError(f"{name} must contain only strict binary values")
    return pd.Series(values, index=series.index, dtype="int8", name=name)


def _canonical_event_id(value: Hashable) -> dict[str, JsonValue]:
    """Encode supported event-ID values with explicit, deterministic type tags."""
    value, signature = _normalize_identifier_value(value, "event IDs")
    kind = signature[0]
    if kind == "int":
        return {"type": "int", "value": str(value)}
    if kind == "float":
        assert isinstance(value, float)
        return {"type": "float", "value": value.hex()}
    if kind == "str":
        return {"type": "str", "value": value}
    if kind == "bytes":
        return {"type": "bytes", "value": value.hex()}
    if kind == "decimal":
        return {"type": "decimal", "value": str(value)}
    if kind == "datetime":
        return {"type": "datetime", "value": value.isoformat()}
    if kind == "date":
        return {"type": "date", "value": value.isoformat()}
    if kind == "uuid":
        return {"type": "uuid", "value": str(value)}
    if kind == "tuple":
        assert isinstance(value, tuple)
        return {
            "type": "tuple",
            "value": [_canonical_event_id(item) for item in value],
        }
    raise AssertionError("validated event ID had no canonical encoding")


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


def _verify_flag_selection_hashes(selection: FlagSelection) -> None:
    """Verify typed selection hashes before copying them into result provenance."""
    selected_hash = _typed_sha256(
        "selected_event_ids",
        [_canonical_event_id(event_id) for event_id in selection.selected_event_ids],
    )
    ordered_flag_items = sorted(
        selection.flags_by_event_id.items(),
        key=lambda item: _normalize_identifier_value(item[0], "event IDs")[0],
    )
    flags_hash = _typed_sha256(
        "flags_by_event_id",
        [
            {
                "event_id": _canonical_event_id(event_id),
                "flag": int(flag),
            }
            for event_id, flag in ordered_flag_items
        ],
    )
    if not hmac.compare_digest(selected_hash, selection.selected_event_ids_sha256):
        raise AuditValidationError(
            "FlagSelection selected_event_ids_sha256 does not match selected_event_ids"
        )
    if not hmac.compare_digest(flags_hash, selection.flags_by_event_id_sha256):
        raise AuditValidationError(
            "FlagSelection flags_by_event_id_sha256 does not match flags_by_event_id"
        )


def _exact_numeric_value(
    value: int | float | np.integer | np.floating | Decimal,
) -> Fraction:
    """Return a context-independent exact key for a supported numeric score."""
    if isinstance(value, np.integer):
        return Fraction(int(value), 1)
    if isinstance(value, np.floating):
        numerator, denominator = value.as_integer_ratio()
        return Fraction(int(numerator), int(denominator))
    if type(value) is int:
        return Fraction(value, 1)
    if type(value) is float:
        return Fraction(*value.as_integer_ratio())
    assert isinstance(value, Decimal)
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def _stable_stratum_token(value: Hashable) -> JsonValue:
    """Encode a final-stratum label with equality-invariant built-in semantics."""
    value, signature = _normalize_identifier_value(value, "final-stratum labels")
    kind = signature[0]
    if kind in {"int", "float", "decimal"}:
        if kind == "decimal":
            exact = Fraction(value)
        elif kind == "float":
            exact = Fraction(*value.as_integer_ratio())
        else:
            exact = Fraction(value, 1)
        return {
            "type": "number",
            "numerator": str(exact.numerator),
            "denominator": str(exact.denominator),
        }
    if kind == "str":
        return {"type": "str", "value": value}
    if kind == "bytes":
        return {"type": "bytes", "value": value.hex()}
    if kind == "datetime":
        if value.tzinfo is not None and value.utcoffset() is not None:
            value = value.astimezone(timezone.utc)
        return {"type": "datetime", "value": value.isoformat()}
    if kind == "date":
        return {"type": "date", "value": value.isoformat()}
    if kind == "uuid":
        return {"type": "uuid", "value": str(value)}
    if kind == "tuple":
        assert isinstance(value, tuple)
        items = [_stable_stratum_token(item) for item in value]
        return {
            "type": "tuple",
            "value": items,
        }
    raise AssertionError("validated final-stratum label had no stable token")


def _stable_stratum_key(value: Hashable) -> str:
    token = _stable_stratum_token(value)
    return json.dumps(
        token,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonicalize_legitimate_events(
    legitimate: pd.DataFrame,
    *,
    entity_col: str,
    event_id_col: str,
    final_stratum_col: str,
    flags: pd.Series,
) -> pd.DataFrame:
    stratum_values = legitimate[final_stratum_col].tolist()
    group_by_value: dict[Hashable, int] = {}
    group_values: list[list[Hashable]] = []
    group_indices: list[list[int]] = []
    for index, value in enumerate(stratum_values):
        group = group_by_value.get(value)
        if group is None:
            group = len(group_values)
            group_by_value[value] = group
            group_values.append([])
            group_indices.append([])
        group_values[group].append(value)
        group_indices[group].append(index)

    event_ids = legitimate[event_id_col].tolist()
    event_sort_keys = [
        _normalize_identifier_value(event_id, "event IDs")[0]
        for event_id in event_ids
    ]
    stable_group_keys = [
        min(_stable_stratum_key(value) for value in values)
        for values in group_values
    ]

    for left in range(len(stable_group_keys)):
        for right in range(left + 1, len(stable_group_keys)):
            if stable_group_keys[left] == stable_group_keys[right]:
                left_value = group_values[left][0]
                right_value = group_values[right][0]
                if left_value != right_value:
                    raise AuditValidationError(
                        "final-stratum labels cannot be ordered deterministically without collision"
                    )
    ordered_groups = sorted(
        range(len(group_values)), key=stable_group_keys.__getitem__
    )
    group_ranks = {group: rank for rank, group in enumerate(ordered_groups)}
    stratum_keys = [0] * len(stratum_values)
    for group, indices in enumerate(group_indices):
        for index in indices:
            stratum_keys[index] = group_ranks[group]
    for left in range(len(group_values)):
        for right in range(left + 1, len(group_values)):
            if group_values[left][0] == group_values[right][0]:
                raise AuditValidationError(
                    "equal final-stratum labels were not resolved into one group"
                )

    order = sorted(
        range(len(legitimate)),
        key=lambda index: (stratum_keys[index], event_sort_keys[index]),
    )
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
        package_version=PACKAGE_VERSION,
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
    _require_unique_dataframe_columns(data)
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

    normalized_event_ids = _require_supported_identifier_values(
        data[event_id_col].tolist(),
        "event IDs",
        consistent_representation=True,
        total_order=True,
    )
    if len(set(normalized_event_ids)) != len(normalized_event_ids):
        raise AuditValidationError("event IDs must be unique")
    legitimate_positions = np.flatnonzero(legitimate_mask.to_numpy(dtype=bool))
    legitimate = data.iloc[legitimate_positions].copy()
    legitimate_event_ids = [
        normalized_event_ids[position] for position in legitimate_positions
    ]
    legitimate[event_id_col] = _one_dimensional_object_array(legitimate_event_ids)
    normalized_entities = _require_supported_identifier_values(
        legitimate[entity_col].tolist(),
        "entity IDs",
        consistent_representation=True,
        total_order=False,
    )
    normalized_strata = _require_supported_identifier_values(
        legitimate[final_stratum_col].tolist(),
        "final-stratum labels",
        consistent_representation=False,
        total_order=False,
    )
    legitimate[entity_col] = _one_dimensional_object_array(normalized_entities)
    legitimate[final_stratum_col] = _one_dimensional_object_array(normalized_strata)

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
        _verify_flag_selection_hashes(selection)
        alignment_index = _atomic_object_index(legitimate_ids, name=event_id_col)
        flags = selection.flags_by_event_id.reindex(alignment_index)
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
    _require_unique_dataframe_columns(data)
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

    normalized_event_ids = _require_supported_identifier_values(
        data[event_id_col].tolist(),
        "event IDs",
        consistent_representation=True,
        total_order=True,
    )
    if len(set(normalized_event_ids)) != len(normalized_event_ids):
        raise AuditValidationError("event IDs must be unique")
    for event_id in normalized_event_ids:
        _canonical_event_id(event_id)

    legitimate_positions = np.flatnonzero(legitimate_mask.to_numpy(dtype=bool))
    legitimate = data.iloc[legitimate_positions]
    legitimate_event_count = int(len(legitimate))
    if n_select > legitimate_event_count:
        raise AuditValidationError("n_select must not exceed the legitimate-event count")

    raw_legitimate_scores = legitimate[score_col].tolist()
    ordinary_float_fast_path = all(
        type(score) is float for score in raw_legitimate_scores
    )
    scored_events: list[tuple[Hashable, float | Fraction]] = []
    for position, score in zip(
        legitimate_positions, raw_legitimate_scores
    ):
        event_id = normalized_event_ids[position]
        score_type = type(score)
        if score_type is bool or isinstance(score, np.bool_) or not (
            score_type in {int, float, Decimal}
            or isinstance(score, (np.integer, np.floating))
        ):
            raise AuditValidationError("scores on legitimate events must be finite numbers")
        if isinstance(score, Decimal):
            finite = score.is_finite()
        elif isinstance(score, (float, np.floating)):
            finite = bool(np.isfinite(score))
        else:
            finite = True
        if not finite:
            raise AuditValidationError("scores on legitimate events must be finite numbers")
        comparison_key = score if ordinary_float_fast_path else _exact_numeric_value(score)
        scored_events.append((event_id, comparison_key))

    ranked = sorted(scored_events, key=lambda item: item[0])
    ranked.sort(key=lambda item: item[1], reverse=True)
    selected_event_ids = tuple(event_id for event_id, _ in ranked[:n_select])
    selected_set = set(selected_event_ids)
    ordered_legitimate_ids = sorted(event_id for event_id, _ in scored_events)
    flag_values = [int(event_id in selected_set) for event_id in ordered_legitimate_ids]
    flags_by_event_id = pd.Series(
        flag_values,
        index=_atomic_object_index(ordered_legitimate_ids, name=event_id_col),
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
