import numpy as np
import pytest
from scipy.io import savemat

from der import Config, load_session, run
from der.cli import main
from der.core import detect_within
from der.wavelets import features


@pytest.mark.parametrize("compatible", [True, False])
def test_exact_blackrock_probe_layout_is_two_bundles(tmp_path, monkeypatch, capsys, compatible):
    expected = {}
    for prefix, start in [("mLT2aHa", 2285), ("mRT2aHa", 2293)]:
        for wire in range(1, 9):
            channel = start + wire - 1
            expected[channel] = prefix
            waveform = np.ones((1, 64))
            waveform[0, 0] = channel  # Identify the channel in comparison batches.
            savemat(
                tmp_path / f"times_{prefix}{wire:02}_{channel}.mat",
                {"spikes": waveform, "cluster_class": [[1, wire * 0.001]]},
            )
    session = load_session(tmp_path, threshold=1, matlab_compatible=compatible)
    assert session.bundle_names == {1: "mLT2aHa", 2: "mRT2aHa"}
    for bundle, channels in [(1, range(2285, 2293)), (2, range(2293, 2301))]:
        assert set(session.channel[session.bundle == bundle]) == set(channels)

    compared = []

    def checked_features(shapes):
        channels = set(shapes[:, 0].astype(int))
        # No within-bundle waveform comparison may mix the two physical probes.
        assert len({expected[ch] for ch in channels}) == 1
        compared.append(channels)
        return features(shapes)

    monkeypatch.setattr("der.core.features", checked_features)
    labels = np.ones(len(session.time), dtype=int)
    detect_within(session, labels, Config(matlab_compatible=compatible), progress=True)
    assert any({2288, 2289} <= channels for channels in compared)
    assert any({2296, 2297} <= channels for channels in compared)
    assert {expected[next(iter(channels))] for channels in compared} == {"mLT2aHa", "mRT2aHa"}
    progress = capsys.readouterr().err
    assert "bundle 1 mLT2aHa (channels 2285-2292)" in progress
    assert "bundle 2 mRT2aHa (channels 2293-2300)" in progress
    assert "2/2" in progress


def write_minimal(path):
    savemat(path, {"spikes": np.ones((1, 64)), "cluster_class": [[1.0, 100.0]], "detectionLabel": [[1]]})


@pytest.mark.parametrize("mode", ["auto", "relative"])
def test_numeric_only_names_make_two_relative_bundles(tmp_path, mode):
    for channel in range(2285, 2301):
        write_minimal(tmp_path / f"times_{channel}.mat")
    session = load_session(tmp_path, threshold=1, bundle_grouping=mode)
    assert session.bundle_grouping == "relative"
    assert session.channel_origin == 2285
    assert session.channels_per_bundle == 8
    assert set(session.channel[session.bundle == 1]) == set(range(2285, 2293))
    assert set(session.channel[session.bundle == 2]) == set(range(2293, 2301))
    assert session.bundle_names == {1: "relative 2285-2292", 2: "relative 2293-2300"}


def test_mixed_names_fall_back_consistently_for_all_channels(tmp_path):
    write_minimal(tmp_path / "times_mLT2aHa01_2285.mat")
    write_minimal(tmp_path / "times_channel_2293.mat")
    s = load_session(tmp_path, threshold=1)
    assert s.bundle_grouping == "relative"
    assert dict(zip(s.channel, s.bundle)) == {2285: 1, 2293: 2}


def test_relative_can_override_named_probe_prefix(tmp_path):
    write_minimal(tmp_path / "times_same01_2285.mat")
    write_minimal(tmp_path / "times_same02_2293.mat")
    assert len(load_session(tmp_path, threshold=1).bundle_names) == 1
    result = run(tmp_path, threshold=1, config=Config(bundle_grouping="relative"), progress=False)
    assert len(result.session.bundle_names) == 2
    assert result.summary()["bundle_layout"] == {
        "method": "relative",
        "channel_origin": 2285,
        "channels_per_bundle": 8,
        "n_bundles": 2,
    }


def test_filename_mode_requires_names(tmp_path):
    write_minimal(tmp_path / "times_2285.mat")
    with pytest.raises(ValueError, match="Filename grouping requires"):
        load_session(tmp_path, threshold=1, bundle_grouping="filename")


@pytest.mark.parametrize(
    "size, expected",
    [(8, {2285: 1, 2287: 1, 2292: 1, 2293: 2, 2301: 3}), (4, {2285: 1, 2287: 1, 2292: 2, 2293: 3, 2301: 5})],
)
def test_missing_channels_keep_their_slots(tmp_path, size, expected):
    for channel in expected:
        write_minimal(tmp_path / f"times_{channel}.mat")
    s = load_session(tmp_path, threshold=1, channels_per_bundle=size)
    assert dict(zip(s.channel, s.bundle)) == expected
    assert set(s.channel) == set(expected)  # Original channel IDs are not rewritten.


def test_cli_accepts_relative_grouping_and_bundle_size(tmp_path, capsys):
    import json

    write_minimal(tmp_path / "times_2285.mat")
    write_minimal(tmp_path / "times_2289.mat")
    assert (
        main(
            [
                "compare",
                str(tmp_path),
                "--threshold",
                "1",
                "--bundle-grouping",
                "relative",
                "--channels-per-bundle",
                "4",
                "--no-progress",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["bundle_layout"]["n_bundles"] == 2
    assert payload["parameters"]["bundle_grouping"] == "relative"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bundle_grouping": "invalid"},
        {"channels_per_bundle": 0},
        {"channels_per_bundle": 2.5},
        {"channels_per_bundle": True},
    ],
)
def test_invalid_grouping_configuration(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
