"""Compare dispersed and concentrated false-positive persistence."""

import json

import pandas as pd

from fp_persistence import ExposureSpec, PermutationSpec, audit


events = pd.DataFrame(
    {
        "event_id": [f"e{i:02d}" for i in range(12)],
        "entity_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3,
        "legitimate": [1] * 12,
        "final_stratum": ["day-1"] * 6 + ["day-2"] * 6,
    }
)

flags_dispersed = [
    1, 0, 0,  # A
    1, 0, 0,  # B
    1, 0, 0,  # C
    1, 0, 0,  # D
]
flags_concentrated = [
    1, 1, 0,  # A
    0, 0, 0,  # B
    1, 1, 0,  # C
    0, 0, 0,  # D
]

audit_results = {}
summaries = {}
for scenario, flags in {
    "dispersed": flags_dispersed,
    "concentrated": flags_concentrated,
}.items():
    scenario_events = events.assign(fp_flag=flags)
    result = audit(
        scenario_events,
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
    audit_results[scenario] = result
    summaries[scenario] = {
        "total_false_positive_flags": int(scenario_events["fp_flag"].sum()),
        "flags_by_final_stratum": {
            str(stratum): int(count)
            for stratum, count in scenario_events.groupby(
                "final_stratum", sort=True
            )["fp_flag"].sum().items()
        },
        "eligible_entity_count": result.population.eligible_entity_count,
        "observed_persistence": result.estimates.observed_repeat_fp_proportion,
        "reference_expectation": result.estimates.reference_expected_repeat_fp_proportion,
        "excess_persistence_percentage_points": result.estimates.excess_repeat_fp_percentage_points,
    }

print(json.dumps(summaries, indent=2, sort_keys=True))
