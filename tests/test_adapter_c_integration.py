import pandas as pd

from fp_persistence import AuditStatus, ExposureSpec, PermutationSpec, audit
from fp_persistence_adapter import resolve_residual_strata, time_and_amount_context


def _fixture():
    entities = [entity for entity in "ABCD" for _ in range(3)]
    timestamps = [
        *(f"2025-01-01T{hour:02d}:00:00Z" for hour in range(6)),
        *(f"2025-01-02T{hour:02d}:00:00Z" for hour in range(6)),
    ]
    frame = pd.DataFrame({
        "event_id": [f"e{i:02d}" for i in range(12)],
        "entity": entities,
        "timestamp": timestamps,
        "amount": [10.0] * 12,
        "merchant_category": ["retail"] * 12,
        "legitimate": [1] * 12,
        "flag": [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    })
    augmented, _ = time_and_amount_context(frame, pd.Series([1.0, 10.0, 100.0]))
    augmented["final_stratum"] = resolve_residual_strata(augmented, minimum=2)
    return augmented


def _run(frame):
    return audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="final_stratum",
        legitimate_col="legitimate",
        flag_col="flag",
        exposure=ExposureSpec(3, 3),
        permutation=PermutationSpec(200, 2026091601),
        metadata={"adapter": "C"},
    )


def test_adapter_output_runs_directly_through_installed_public_api():
    result = _run(_fixture())
    assert result.status is AuditStatus.ESTIMABLE
    assert result.population.eligible_entity_count == 4
    assert result.estimates.observed_repeat_fp_proportion == 0.5


def test_adapter_plus_audit_is_row_order_invariant():
    frame = _fixture()
    first = _run(frame).to_dict()
    shuffled = frame.sample(frac=1, random_state=42)
    second = _run(shuffled).to_dict()
    first["provenance"].pop("execution_timestamp_utc")
    second["provenance"].pop("execution_timestamp_utc")
    assert first == second
