import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from der import Config, load_session, run, save_labels
from der.cli import main
from der.compare import agreement, compare_files, compare_result
from der.core import candidate_groups, detect_across, detect_correlations, detect_within, pair_histogram
from der.io import Session, labels_by_file
from der.wavelets import features, haar_coefficients


def synthetic(times, channels, bundles=None, clusters=None, amplitudes=None):
    n = len(times)
    channels = np.asarray(channels)
    amps = np.full(n, 10.0) if amplitudes is None else np.asarray(amplitudes)
    shapes = np.tile(np.sin(np.arange(64) / 5), (n, 1))
    shapes[:, 19] = amps
    return Session(
        [],
        np.array(times, float),
        shapes,
        channels,
        np.array(bundles if bundles is not None else np.ones(n), int),
        np.array(clusters if clusters is not None else np.ones(n), int),
        np.ones(n),
        np.full(n, "MU"),
        np.zeros(n, int),
        np.arange(n),
        [],
        [],
    )


def write_channel(directory, name="times_mLT2aHa01_2285.mat", times=(3.0, 1.0, 2.0), labels=None):
    path = Path(directory) / name
    shapes = np.arange(len(times) * 64, dtype=float).reshape(len(times), 64)
    d = {
        "spikes": shapes,
        "cluster_class": np.c_[np.ones(len(times)), times],
        "par": {"sr": 30000, "tag": "preserve me"},
        "forced": np.zeros((1, len(times)), bool),
    }
    if labels is not None:
        d["detectionLabel"] = np.asarray(labels)[:, None]
    savemat(path, d)
    savemat(path.with_name(path.name[6:-4] + "_spikes.mat"), {"threshold": [2.0, 4.0]})
    return path


def test_haar_matches_independent_wavelet_library():
    import pywt

    rng = np.random.default_rng(42)
    for width in (40, 64):
        x = rng.normal(size=(30, width))
        expected = np.stack(
            [np.concatenate(pywt.wavedec(row, "haar", mode="symmetric", level=5))[:width] for row in x]
        )
        np.testing.assert_allclose(haar_coefficients(x), expected, atol=1e-14)


def test_two_waveforms_select_last_ten_coefficients():
    x = np.random.default_rng(1).normal(size=(2, 40))
    np.testing.assert_allclose(features(x), haar_coefficients(x)[:, 39:29:-1])


@pytest.mark.parametrize("same", [True, False])
def test_correlogram_against_brute_force(same):
    a = np.array([0.0, 0.1, 0.25, 0.75, 1.0, 1.25, 2.1, 3.0])
    b = a if same else np.array([0.1, 0.5, 1.5, 2.0, 3.25])
    hist = pair_histogram(a, b, 2.0, 0.5, same)
    expected = np.zeros(9, int)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if same and i == j:
                continue
            delta = y - x
            if abs(delta) < 2.25:
                offset = int(np.copysign(np.floor(abs(delta / 0.5) + 0.5), delta))
                expected[4 + offset] += 1
    np.testing.assert_array_equal(hist, expected)


def test_compatibility_terminal_candidate_and_window_boundary():
    assert list(candidate_groups(np.array([0.0, 0.02]), 0.05, True)) == []
    np.testing.assert_array_equal(next(iter(candidate_groups(np.array([0.0, 0.02]), 0.05, False))), [0, 1])
    groups = list(candidate_groups(np.array([0.0, 0.03, 0.06, 0.09]), 0.05, False))
    assert all(np.ptp(np.array([0.0, 0.03, 0.06, 0.09])[g]) <= 0.05 for g in groups)


def test_compatibility_chain_endpoint_matches_matlab_loop():
    groups = list(candidate_groups(np.array([0.0, 0.01, 0.02, 0.03]), 0.05, True))
    assert [g.tolist() for g in groups] == [[0, 1, 2], [1, 2]]


def test_stage_factors_compose_only_once():
    s = synthetic([0.0, 0.01, 0.02], [1, 9, 17], [1, 2, 3])
    labels = np.full(3, 7, dtype=int)
    detect_across(s, labels, Config(matlab_compatible=False))
    np.testing.assert_array_equal(labels, [14, 14, 14])


def test_missing_thresholds_explicitly_reported(tmp_path):
    path = write_channel(tmp_path)
    path.with_name(path.name[6:-4] + "_spikes.mat").unlink()
    with pytest.warns(UserWarning, match="threshold"):
        s = load_session(tmp_path)
    assert s.missing_thresholds == [path.name]
    assert np.isnan(s.threshold).all()
    assert (load_session(tmp_path, threshold=4).threshold == 4).all()


def test_across_bundle_shapes_and_prime_factor():
    s = synthetic([0.0, 0.01, 0.02], [1, 9, 17], [1, 2, 3])
    labels = np.ones(3, int)
    detect_across(s, labels, Config(matlab_compatible=False))
    np.testing.assert_array_equal(labels, [2, 2, 2])
    s.bundle[:] = 1
    labels[:] = 1
    detect_across(s, labels, Config(matlab_compatible=False))
    np.testing.assert_array_equal(labels, [1, 1, 1])


def test_same_channel_opposite_polarity_retains_higher_sn():
    s = synthetic([0.0, 0.1, 10.0], [1, 1, 1], clusters=[1, 2, 1], amplitudes=[10.0, -5.0, 10.0])
    labels = np.ones(3, int)
    detect_within(s, labels, Config(matlab_compatible=False))
    np.testing.assert_array_equal(labels, [1, 3, 1])


def test_same_channel_cluster_zero_removed():
    s = synthetic([0.0, 0.1], [1, 1], clusters=[0, 1], amplitudes=[-10.0, 5.0])
    s.unit[0] = "A"
    labels = np.ones(2, int)
    detect_within(s, labels, Config())
    np.testing.assert_array_equal(labels, [3, 1])


def test_same_bundle_shapes_keep_higher_sn_and_use_named_bundle():
    s = synthetic([0.0, 0.01], [2285, 2290], [1, 1], amplitudes=[10.0, 5.0])
    labels = np.ones(2, int)
    detect_within(s, labels, Config(matlab_compatible=False))
    np.testing.assert_array_equal(labels, [1, 5])


def correlation_fixture(different_bundles=False):
    # Repeated synchronous events plus distributed background lags => z > 5.
    a = np.arange(100, dtype=float) * 100
    b = a.copy()
    b[:80] += 0.1
    b[80:] += np.linspace(1.0, 19.0, 20)
    order = np.argsort(np.r_[a, b], kind="stable")
    s = synthetic(
        np.r_[a, b][order],
        np.r_[np.ones(100, int), np.full(100, 2)][order],
        np.r_[np.ones(100, int), np.full(100, 2 if different_bundles else 1)][order],
        amplitudes=np.r_[np.full(100, 10.0), np.full(100, 5.0)][order],
    )
    return s


@pytest.mark.parametrize("different", [True, False])
def test_significant_correlation_deletion_policy(different):
    s = correlation_fixture(different)
    labels = np.ones(len(s.time), int)
    records = detect_correlations(s, labels, Config(matlab_compatible=False))
    assert any(r["z"] > 5 for r in records)
    assert np.sum(labels[s.channel == 2] == 7) == 80
    assert np.sum(labels[s.channel == 1] == 7) == (80 if different else 0)


def test_zero_background_variance_is_zero_z():
    s = synthetic([0.0, 0.1], [1, 2], [1, 2])
    labels = np.ones(2, int)
    records = detect_correlations(s, labels, Config(matlab_compatible=False))
    assert all(r["z"] == 0 for r in records)
    assert (labels == 1).all()


def test_io_sort_restore_and_preserve_payload(tmp_path):
    original = write_channel(tmp_path, labels=[7, 1, 5])
    s = load_session(tmp_path)
    np.testing.assert_array_equal(s.time, [1, 2, 3])
    assert (s.threshold == 3).all()
    out = tmp_path / "output"
    paths = save_labels(s, np.array([2, 3, 5]), out)
    a, b = loadmat(original), loadmat(paths[0])
    np.testing.assert_array_equal(b["detectionLabel"].ravel(), [5, 2, 3])
    for k in ("spikes", "cluster_class", "forced"):
        np.testing.assert_array_equal(a[k], b[k])
    assert loadmat(paths[0], simplify_cells=True)["par"]["tag"] == "preserve me"
    np.testing.assert_array_equal(a["detectionLabel"].ravel(), [7, 1, 5])
    with pytest.raises(FileExistsError):
        save_labels(s, np.ones(3), out)
    with pytest.raises(ValueError):
        save_labels(s, np.ones(3), tmp_path)


def test_existing_labels_never_used_as_algorithm_input(tmp_path):
    path = write_channel(tmp_path, labels=[1, 1, 1])
    a = run(tmp_path)
    d = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    d["detectionLabel"] = np.full((3, 1), 210)
    savemat(path, d)
    b = run(tmp_path)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_comparison_metrics_and_invalid_labels():
    result = agreement([1, 2, 7, 35], [1, 1, 14, 35])
    assert result["exact_label_agreement"] == 0.5
    assert result["artifact"]["tp"] == 2
    assert result["artifact"]["fn"] == 1
    assert result["factors"]["7"]["agreement"] == 1
    assert agreement([1], [1])["artifact"]["jaccard"] is None
    with pytest.raises(ValueError):
        agreement([4], [1])


def test_comparison_validates_event_identity(tmp_path):
    first = write_channel(tmp_path, labels=[1, 2, 3])
    other = tmp_path / "other"
    other.mkdir()
    second = write_channel(other, times=[1, 2, 3], labels=[1, 2, 3])
    with pytest.raises(ValueError, match="identical"):
        compare_files(first, second)
    with pytest.raises(ValueError, match="identical"):
        compare_result(run(tmp_path), other)


def test_duplicate_channels_rejected(tmp_path):
    write_channel(tmp_path)
    write_channel(tmp_path, name="times_other01_2285.mat")
    with pytest.raises(ValueError, match="Duplicate"):
        load_session(tmp_path)


def test_empty_channel_supported(tmp_path):
    write_channel(tmp_path, times=[])
    result = run(tmp_path)
    assert result.summary()["n_spikes"] == 0


def test_cli_reports_and_comparison_gate(tmp_path):
    write_channel(tmp_path, labels=[1, 1, 1])
    report = tmp_path / "report.json"
    assert main(["compare", str(tmp_path), "--report", str(report), "--min-agreement", "1"]) == 0
    assert report.exists()
    assert main(["run", str(tmp_path), "--output", str(tmp_path / "out"), "--report", str(report)]) == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"bin_width_ms": 0}, {"hist_limit_ms": 1.1}, {"amplitude_sample": 64}, {"across_threshold": -1}],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.integration
def test_uploaded_runs(example_data_dir, tmp_path):
    first = Path(example_data_dir) / "times_mLT2aHa01_2285.mat"
    second = Path(example_data_dir) / "times_mLT2aHa01_2285(1).mat"
    report = compare_files(first, second)
    assert report["n_spikes"] == 6209
    assert report["cluster_id_differences"] == 2
    # Smoke-test the provided channel, NEVER require single-channel label parity.
    with pytest.warns(UserWarning, match="threshold"):
        result = run(example_data_dir, pattern=first.name)
    assert len(result.labels) == 6209
    assert set(np.unique(result.labels)) <= {
        np.prod(c, dtype=int) for n in range(5) for c in itertools.combinations((2, 3, 5, 7), n)
    }
    assert len(labels_by_file(result.session, result.labels)[0]) == 6209
    output = save_labels(result.session, result.labels, tmp_path / "saved")[0]
    original, saved = loadmat(first), loadmat(output)

    def equal(a, b):
        assert a.dtype == b.dtype and a.shape == b.shape
        if a.dtype.names:
            for field in a.dtype.names:
                equal(a[field], b[field])
        elif a.dtype.kind == "O":
            for x, y in zip(a.flat, b.flat):
                equal(x, y)
        else:
            np.testing.assert_array_equal(a, b)

    for key in original:
        if not key.startswith("__") and key != "detectionLabel":
            equal(original[key], saved[key])
