import json

import pytest
from scipy.io import savemat

from der.cli import main
from der.compare import agreement, format_summary, summarize_comparison, summarize_report


def example_report():
    rows = [
        {"file": "channel-a.mat", "cluster_id_differences": 0, **agreement([1, 2], [1, 1])},
        {"file": "channel-b.mat", "cluster_id_differences": 0, **agreement([1] * 8, [1] * 8)},
        {"file": "empty.mat", "cluster_id_differences": 0, **agreement([], [])},
    ]
    return {
        "reference": "MATLAB",
        "cluster_id_differences": 0,
        "coverage": {
            "compared_files": 3,
            "input_files": 103,
            "missing_files": [f"missing-{i}.mat" for i in range(100)],
        },
        "aggregate": agreement([1, 2] + [1] * 8, [1] * 10),
        "per_file": rows,
    }


def test_summary_pooled_agreement_differs_from_channel_median():
    compact = summarize_comparison(example_report())
    assert compact["aggregate"]["exact_label_agreement"] == 0.9
    stats = compact["channel_statistics"]
    assert stats["median_exact_label_agreement"] == 0.75
    assert stats["min_exact_label_agreement"] == 0.5
    assert stats["max_exact_label_agreement"] == 1
    assert stats["empty_channels"] == 1
    assert stats["lowest_agreement_channels"][0]["file"] == "channel-a.mat"
    assert compact["aggregate"]["artifact"]["reference_flagged_fraction"] == 0.1
    assert compact["aggregate"]["artifact"]["both_flagged_fraction"] == 0


def test_compact_omits_bulk_but_detailed_preserves_it():
    report = example_report()
    compact = summarize_comparison(report)
    detailed = summarize_comparison(report, detailed=True)
    assert "per_file" not in compact
    assert "confusion" not in compact["aggregate"]
    assert compact["coverage"]["missing_file_count"] == 100
    assert len(compact["coverage"]["missing_file_examples"]) == 5
    assert "missing_files" not in compact["coverage"]
    assert detailed["per_file"] == report["per_file"]
    assert detailed["aggregate"] == report["aggregate"]
    assert detailed["coverage"]["missing_files"] == report["coverage"]["missing_files"]
    assert "confusion" in report["aggregate"]  # original is not mutated


def test_default_size_stays_bounded_with_many_channels():
    report = example_report()
    small_size = len(json.dumps(summarize_comparison(report)))
    report["per_file"] = report["per_file"] * 1000
    report["coverage"]["missing_files"] *= 1000
    compact = summarize_comparison(report)
    assert len(compact["channel_statistics"]["lowest_agreement_channels"]) == 5
    assert len(json.dumps(compact)) < small_size + 1000


def test_terminal_summary_handles_empty_data():
    report = example_report()
    report["per_file"] = [report["per_file"][-1]]
    report["aggregate"] = agreement([], [])
    summary = format_summary(summarize_comparison(report))
    assert "n/a" in summary
    assert "Empty channels excluded" in summary


@pytest.mark.parametrize("detailed", [False, True])
def test_cli_compact_default_and_explicit_details(tmp_path, capsys, detailed):
    import numpy as np

    path = tmp_path / "times_mTest01_1.mat"
    savemat(
        path,
        {
            "spikes": np.zeros((3, 64)),
            "cluster_class": np.c_[np.ones(3), [100.0, 200.0, 300.0]],
            "detectionLabel": np.array([[1], [2], [1]]),
        },
    )
    report_path = tmp_path / "agreement.json"
    args = ["compare", str(tmp_path), "--threshold", "1", "--report", str(report_path)]
    if detailed:
        args.append("--detailed-report")
    assert main(args) == 0
    payload = json.loads(report_path.read_text())
    comparison = payload["comparisons"][0]
    assert ("per_file" in comparison) == detailed
    assert ("confusion" in comparison["aggregate"]) == detailed
    assert comparison["aggregate"]["exact_label_agreement"] == pytest.approx(2 / 3)
    assert ("missing_thresholds" in payload["summary"]) == detailed
    terminal = capsys.readouterr()
    assert "Exact-label agreement (pooled spikes): 66.67%" in terminal.err
    assert "Factor 7" in terminal.err
    assert terminal.out == ""


def test_json_stdout_remains_parseable(tmp_path, capsys):
    import numpy as np

    savemat(
        tmp_path / "times_mTest01_1.mat",
        {"spikes": np.zeros((1, 64)), "cluster_class": [[1.0, 100.0]], "detectionLabel": [[1]]},
    )
    assert main(["compare", str(tmp_path), "--threshold", "1"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["report_detail"] == "compact"
    assert "Exact-label agreement" in captured.err


def test_summarize_legacy_report_without_mat_files(tmp_path, capsys):
    report = example_report()
    # Legacy reports have confusion tables but lack the new flagged fractions.
    for metric in [report["aggregate"]["artifact"], *report["aggregate"]["factors"].values()]:
        for key in list(metric):
            if key.endswith("_fraction"):
                del metric[key]
    payload = {"summary": {"missing_thresholds": [str(i) for i in range(20)]}, "comparisons": [report]}
    original = tmp_path / "agreement.json"
    original.write_text(json.dumps(payload))
    before = original.read_bytes()
    output = tmp_path / "summary.json"
    assert main(["summarize-report", str(original), "--report", str(output)]) == 0
    compact = json.loads(output.read_text())
    assert compact["summary"]["missing_threshold_count"] == 20
    assert "per_file" not in compact["comparisons"][0]
    assert compact["comparisons"][0]["aggregate"]["artifact"]["reference_flagged_fraction"] == 0.1
    assert original.read_bytes() == before
    assert "90.00%" in capsys.readouterr().err
    assert summarize_report(compact) == compact
    with pytest.raises(SystemExit) as error:
        main(["summarize-report", str(original), "--report", str(original)])
    assert error.value.code == 2
    assert original.read_bytes() == before


def test_rejects_non_agreement_report():
    with pytest.raises(TypeError, match="comparisons"):
        summarize_report({"wrong": []})
