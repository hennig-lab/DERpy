import json

import numpy as np
import pytest
from scipy.io import savemat

from der import Config, run
from der.cli import main
from der.core import candidate_groups


@pytest.fixture
def recording(tmp_path):
    for name, channel in [("mA01", 1), ("mA02", 2), ("mB01", 9)]:
        savemat(
            tmp_path / f"times_{name}_{channel}.mat",
            {
                "spikes": np.ones((3, 64)),
                "cluster_class": np.c_[np.ones(3), np.array([100.0, 200.0, 300.0]) + channel * 0.001],
                "detectionLabel": np.ones((3, 1)),
            },
        )
    return tmp_path


@pytest.mark.parametrize("compatible", [True, False])
def test_progress_does_not_change_detection(recording, capsys, compatible):
    config = Config(matlab_compatible=compatible)
    quiet = run(recording, threshold=1, config=config, progress=False)
    assert capsys.readouterr().err == ""
    visible = run(recording, threshold=1, config=config, progress=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    for stage in (
        "Part I: candidate groups",
        "Part II: same-channel scan",
        "Part II: bundle scan",
        "Part II: bundle 1",
        "Part III: cluster pairs",
    ):
        assert stage in captured.err
    assert "6/6" in captured.err
    np.testing.assert_array_equal(quiet.labels, visible.labels)
    assert quiet.correlations == visible.correlations


@pytest.mark.parametrize("compatible, expected", [(True, 1), (False, 2)])
def test_candidate_progress_total_counts_exact_loop_iterations(monkeypatch, compatible, expected):
    calls = []

    def spy(iterable, **kwargs):
        items = list(iterable)
        calls.append((len(items), kwargs))
        return iter(items)

    monkeypatch.setattr("der.core.tqdm", spy)
    list(candidate_groups(np.array([0.0, 0.01, 10.0, 10.01]), 0.05, compatible, progress=True, desc="test"))
    assert calls[0][0] == expected
    assert calls[0][1]["disable"] is False
    assert calls[0][1]["unit"] == "group"


@pytest.mark.parametrize("flag", ["--progress", "--no-progress"])
def test_cli_progress_goes_to_stderr_only(recording, capsys, flag):
    assert main(["compare", str(recording), "--threshold", "1", flag]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["comparisons"][0]["aggregate"]["n_spikes"] == 9
    assert ("Part III: cluster pairs" in captured.err) == (flag == "--progress")


def test_progress_auto_detection_is_quiet_for_redirected_stderr(recording, capsys):
    run(recording, threshold=1)
    assert capsys.readouterr().err == ""
