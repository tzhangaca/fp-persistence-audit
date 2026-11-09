import numpy as np
import pandas as pd
import pytest

from fp_persistence_adapter import (
    exact_legitimate_budget,
    exposure_table,
    resolve_residual_strata,
    time_and_amount_context,
)


def test_exact_budget_boundaries_and_validation():
    assert [exact_legitimate_budget(n) for n in (0, 24, 25, 74, 75, 100, 84756)] == [0, 0, 1, 1, 2, 2, 1695]
    for value in (-1, 1.5, True, "100"):
        with pytest.raises(ValueError):
            exact_legitimate_budget(value)


def test_exposure_table_counts_and_rejects_missing_entities():
    frame = pd.DataFrame({"entity": ["A", "A", "B"]})
    got = exposure_table(frame, "entity").set_index("entity")["legitimate_exposure"].to_dict()
    assert got == {"A": 2, "B": 1}
    with pytest.raises(ValueError, match="null entity"):
        exposure_table(pd.DataFrame({"entity": ["A", None]}), "entity")


def test_training_only_amount_bins_and_explicit_timezone():
    train = pd.Series(range(100))
    audit = pd.DataFrame({
        "timestamp": ["2024-01-01T06:00:00+00:00"],
        "amount": [500],
        "merchant_category": ["a"],
    })
    got, cuts = time_and_amount_context(audit, train)
    assert got.iloc[0].six_hour_block == 1
    assert got.iloc[0].amount_bin == 4
    assert len(cuts) == 4
    with pytest.raises(ValueError, match="timezone-aware"):
        time_and_amount_context(audit.assign(timestamp="2024-01-01 06:00:00"), train)


def test_explicit_mixed_timezone_offsets_normalize_deterministically():
    audit = pd.DataFrame({
        "timestamp": ["2025-01-01T00:00:00+00:00", "2024-12-31T19:00:00-05:00"],
        "amount": [1.0, 2.0],
        "merchant_category": ["a", "b"],
    })
    resolved, _ = time_and_amount_context(audit, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert resolved["audit_day"].tolist() == ["2025-01-01", "2025-01-01"]


def test_reserved_missing_context_sentinel_is_rejected():
    audit = pd.DataFrame({
        "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
        "amount": [1.0, 2.0],
        "merchant_category": ["__MISSING__", None],
    })
    with pytest.raises(ValueError, match="reserved coarse-context"):
        time_and_amount_context(audit, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_missing_coarse_context_and_no_coarse_mode():
    audit = pd.DataFrame({
        "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
        "amount": [1.0, 2.0],
        "merchant_category": [None, "retail"],
    })
    got, _ = time_and_amount_context(audit, pd.Series([0.0, 1.0, 2.0, 3.0]))
    assert got.coarse_context.tolist() == ["__MISSING__", "retail"]
    no_coarse, _ = time_and_amount_context(
        audit.drop(columns="merchant_category"),
        pd.Series([0.0, 1.0, 2.0, 3.0]),
        coarse_col=None,
    )
    assert no_coarse.coarse_context.eq("__NO_COARSE_CONTEXT__").all()


def test_residual_partition_no_reentry_and_minimum_boundary():
    frame = pd.DataFrame({
        "d": ["x"] * 60,
        "h": [0] * 60,
        "a": [0] * 20 + [1] * 20 + [2] * 20,
        "c": ["A"] * 20 + ["B"] * 10 + ["C"] * 10 + ["D"] * 20,
    })
    out = resolve_residual_strata(frame, "d", "h", "a", "c", 20)
    assert out.iloc[0].startswith("L0|")
    assert out.iloc[20].startswith("L1|") and out.iloc[20] == out.iloc[30]
    assert len(out.unique()) == 3
    assert out.value_counts().min() == 20


def test_residual_partition_row_shuffle_invariance():
    frame = pd.DataFrame({
        "d": ["2025-01-01"] * 60,
        "h": [0] * 60,
        "a": [0] * 20 + [1] * 20 + [2] * 20,
        "c": ["alpha"] * 20 + ["beta"] * 10 + ["gamma"] * 10 + ["delta"] * 20,
    }, index=range(100, 160))
    first = resolve_residual_strata(frame, "d", "h", "a", "c", 20)
    second = resolve_residual_strata(frame.sample(frac=1, random_state=42), "d", "h", "a", "c", 20)
    assert first.sort_index().equals(second.sort_index())


def test_residual_partition_rejects_undersized_final_residual():
    frame = pd.DataFrame({"d": ["x"] * 5, "h": [0] * 5, "a": [0] * 5, "c": ["A"] * 5})
    with pytest.raises(ValueError, match="unresolved final day"):
        resolve_residual_strata(frame, "d", "h", "a", "c", 20)


def test_residual_partition_rejects_missing_context_and_duplicate_index():
    frame = pd.DataFrame({"d": ["x", "x"], "h": [0, 0], "a": [0, 0], "c": ["A", None]})
    with pytest.raises(ValueError, match="missing context"):
        resolve_residual_strata(frame, "d", "h", "a", "c", 1)
    duplicate = frame.fillna("A").set_axis([0, 0])
    with pytest.raises(ValueError, match="duplicate index"):
        resolve_residual_strata(duplicate, "d", "h", "a", "c", 1)


def test_residual_partition_rejects_ambiguous_typed_context_labels():
    frame = pd.DataFrame({
        "d": ["x", "x"],
        "h": [0, 0],
        "a": [0, 0],
        "c": pd.Series([1, "1"], dtype="object"),
    })
    with pytest.raises(ValueError, match="ambiguous string representation"):
        resolve_residual_strata(frame, "d", "h", "a", "c", 1)
