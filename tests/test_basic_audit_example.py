from pathlib import Path
import runpy

import pytest

from fp_persistence import AuditStatus


def test_dispersed_and_concentrated_example_controls_flag_burden():
    namespace = runpy.run_path(
        Path(__file__).resolve().parents[1] / "examples" / "basic_audit.py"
    )
    events = namespace["events"]
    flags_dispersed = namespace["flags_dispersed"]
    flags_concentrated = namespace["flags_concentrated"]
    results = namespace["audit_results"]

    assert sum(flags_dispersed) == sum(flags_concentrated) == 4
    counts_by_stratum = []
    for flags in (flags_dispersed, flags_concentrated):
        frame = events.assign(fp_flag=flags)
        counts_by_stratum.append(
            frame.groupby("final_stratum", sort=True)["fp_flag"].sum().to_dict()
        )
    assert counts_by_stratum == [
        {"day-1": 2, "day-2": 2},
        {"day-1": 2, "day-2": 2},
    ]

    dispersed = results["dispersed"]
    concentrated = results["concentrated"]
    assert dispersed.status is AuditStatus.ESTIMABLE
    assert concentrated.status is AuditStatus.ESTIMABLE
    assert dispersed.estimates.observed_repeat_fp_proportion == pytest.approx(0.0)
    assert concentrated.estimates.observed_repeat_fp_proportion == pytest.approx(0.5)
    assert (
        concentrated.estimates.excess_repeat_fp_percentage_points
        > dispersed.estimates.excess_repeat_fp_percentage_points
    )
