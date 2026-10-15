"""Opt-in validation against complete real input; no arbitrary parity threshold."""

import json
from pathlib import Path

import pytest

from der import run
from der.compare import compare_result, summarize_report


@pytest.mark.integration
def test_full_session_matlab_agreement(request):
    directory = request.config.getoption("--der-data-dir")
    if not directory:
        pytest.skip("Supply --der-data-dir or DER_FULL_DATA_DIR for full-session validation")
    result = run(directory)
    references = request.config.getoption("--der-reference-dir") or [None]
    detailed = request.config.getoption("--der-detailed-report")
    reports = [compare_result(result, reference, detailed=detailed) for reference in references]
    target = Path(request.config.getoption("--der-report"))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": result.summary(), "comparisons": reports}
    if not detailed:
        payload = summarize_report(payload)
    target.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"\nFull-session validation report: {target.resolve()}")
    minimum = request.config.getoption("--der-min-agreement")
    if minimum is not None:
        assert 0 <= minimum <= 1
    for report in reports:
        assert report["coverage"]["compared_files"] > 0
        score = report["aggregate"]["exact_label_agreement"]
        assert score is not None
        print(
            f"{report['reference']}: exact-label agreement = {score:.4%}; "
            f"artifact Jaccard = {report['aggregate']['artifact']['jaccard']}"
        )
        if minimum is not None:
            assert score >= minimum
