"""Run a small deterministic false-positive persistence audit."""

import json

import pandas as pd

from fp_persistence import ExposureSpec, PermutationSpec, audit


events = pd.DataFrame(
    {
        "event_id": [f"e{i:02d}" for i in range(12)],
        "entity_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3,
        "legitimate": [1] * 12,
        "final_stratum": ["day-1"] * 6 + ["day-2"] * 6,
        "fp_flag": [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    }
)

result = audit(
    events,
    entity_col="entity_id",
    event_id_col="event_id",
    final_stratum_col="final_stratum",
    legitimate_col="legitimate",
    flag_col="fp_flag",
    exposure=ExposureSpec(3, 3),
    permutation=PermutationSpec(500, 2026091601),
    metadata={
        "entity_semantics": "payment instrument",
        "context_definition": "prespecified example strata",
    },
)

summary = {
    "eligible_entity_count": result.population.eligible_entity_count,
    "observed_persistence": result.estimates.observed_repeat_fp_proportion,
    "reference_expectation": result.estimates.reference_expected_repeat_fp_proportion,
    "excess_persistence_percentage_points": result.estimates.excess_repeat_fp_percentage_points,
    "final_stratum_count": result.diagnostics.final_stratum_count,
    "minimum_final_stratum_size": result.diagnostics.minimum_final_stratum_size,
    "maximum_final_stratum_size": result.diagnostics.maximum_final_stratum_size,
}

print(json.dumps(summary, indent=2, sort_keys=True))
