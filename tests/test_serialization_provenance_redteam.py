"""Integrated serialization and provenance tampering release gates."""

from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd
import pytest

import fp_persistence
from fp_persistence import (
    AuditValidationError,
    ExposureSpec,
    PermutationSpec,
    audit,
    flags_from_scores,
)


CONTRACT_COMMIT = "c95c58e5886d88db4ab9678227d7ea6089d2f178"
CONTRACT_SHA256 = "abf8af44e3e0a4869a2675b9a3fd96d372f7dd15612e1d5ae751995434062816"


def _frame():
    return pd.DataFrame(
        {
            "event_id": ["e0", "e1", "e2", "e3"],
            "entity": ["A", "A", "B", "B"],
            "stratum": ["s", "s", "s", "s"],
            "legitimate": [1, 1, 1, 1],
            "score": [4, 3, 2, 1],
        }
    )


def _selection(frame):
    return flags_from_scores(
        frame,
        score_col="score",
        event_id_col="event_id",
        legitimate_col="legitimate",
        n_select=2,
    )


def _audit_selection(frame, selection, *, metadata=None):
    return audit(
        frame,
        entity_col="entity",
        event_id_col="event_id",
        final_stratum_col="stratum",
        legitimate_col="legitimate",
        flag_selection=selection,
        exposure=ExposureSpec(2, 2),
        permutation=PermutationSpec(20, 1701),
        metadata=metadata,
    )


def test_integrated_json_csv_and_provenance_are_safe_and_consistent():
    frame = _frame()
    result = _audit_selection(frame, _selection(frame), metadata={"nested": {"ok": True}})
    as_dict = result.to_dict()
    encoded = result.to_json()
    parsed = json.loads(encoded)
    assert parsed == as_dict
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert result.provenance.package_version == fp_persistence.__version__ == "0.1.2"
    assert result.provenance.result_schema_version == result.schema_version == "1.0"
    assert result.provenance.statistical_contract_commit == CONTRACT_COMMIT
    assert result.provenance.statistical_contract_sha256 == CONTRACT_SHA256
    row = result.to_csv_row()
    assert row["status"] == "ESTIMABLE"
    assert (
        row["estimates_observed_repeat_fp_proportion"]
        == result.estimates.observed_repeat_fp_proportion
    )


def test_tampered_selected_ids_are_rejected():
    frame = _frame()
    selection = _selection(frame)
    tampered = replace(selection, selected_event_ids=("e0", "e2"))
    with pytest.raises(AuditValidationError):
        _audit_selection(frame, tampered)


def test_tampered_binary_flags_are_rejected():
    frame = _frame()
    selection = _selection(frame)
    changed = selection.flags_by_event_id.copy(deep=True)
    changed.loc["e1"] = 0
    changed.loc["e2"] = 1
    tampered = replace(selection, flags_by_event_id=changed)
    with pytest.raises(AuditValidationError):
        _audit_selection(frame, tampered)


def test_tampered_event_ids_are_rejected():
    frame = _frame()
    selection = _selection(frame)
    changed = selection.flags_by_event_id.copy(deep=True)
    changed.index = pd.Index(["e0", "e1", "e2", "other"], name="event_id")
    tampered = replace(selection, flags_by_event_id=changed)
    with pytest.raises(AuditValidationError):
        _audit_selection(frame, tampered)


@pytest.mark.parametrize("field", ["selected_event_ids_sha256", "flags_by_event_id_sha256"])
def test_tampered_hashes_are_recomputed_and_rejected(field):
    frame = _frame()
    selection = _selection(frame)
    tampered = replace(selection, **{field: "0" * 64})
    with pytest.raises(AuditValidationError, match="does not match"):
        _audit_selection(frame, tampered)


def test_reserved_provenance_metadata_is_rejected_before_result_creation():
    frame = _frame()
    with pytest.raises(AuditValidationError, match="reserved provenance"):
        _audit_selection(
            frame,
            _selection(frame),
            metadata={"package_version": "forged"},
        )
