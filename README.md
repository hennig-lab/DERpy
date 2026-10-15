# DER algorithm

This is a Python port of the [MATLAB DER algorithm](https://github.com/hennig-lab/DER), adapted for Blackrock data.

Duplicate Event Removal algorithm — artifact detection in human single-unit recordings.

This project is licensed under the Mozilla Public License Version 2.0 (MPL 2.0). See `LICENSE` for details.

Original code (Version 1.0, released February 2021): Copyright © 2021 Gert Dehnen, Marcel S. Kehl, Florian Mormann, and the University of Bonn Medical Center. Blackrock support was added by Jay Hennig in Version 1.1, released October 2025.

## How to use the code

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), download this repository, and run the following from the repository directory to install Python and the required packages:

```bash
uv sync --locked
```

To run automated artifact detection on Blackrock data processed with Wave_clus:

```bash
uv run der run /path/to/data --output /path/to/output
```

Provide the complete set of `times_*.mat` files from one recording. The code reads spike waveforms and cluster assignments from these files, along with detection thresholds from associated `*_spikes.mat` files when available. Timestamps must be in milliseconds, and waveforms must contain 64 samples.

This writes new `times_*.mat` files to the output directory with a `detectionLabel` field. The original input files are unchanged, and no spikes are removed. Existing output files will not be overwritten.

Bundles are identified from channel names, such as `mLT2aHa` in `times_mLT2aHa01_2285.mat`. If probe names are unavailable, channels are grouped in blocks of eight starting from the smallest channel number. Use `--bundle-grouping relative` to explicitly select this rule, or `--channels-per-bundle` to change the bundle size.

Progress bars show progress within each detection stage. Use `--progress` to force them on or `--no-progress` to disable them.

To compare Python results with MATLAB `detectionLabel` values already stored in the input files:

```bash
uv run der compare /path/to/data --report agreement.json
```

This reruns Python DER without writing MAT files and produces summary agreement statistics. To use a separate MATLAB reference directory, add `--reference /path/to/matlab-output`. Add `--detailed-report` to include all per-channel statistics and confusion tables.

## Structure of the code

The detection pipeline is structured in three parts, implemented in `src/der/core.py`:

* **Part I — Detection of artifacts across different wire bundles.** `detect_across` compares waveforms of nearly simultaneous events recorded on different bundles.
* **Part II — Detection of duplicate spikes within a wire bundle.** `detect_within` identifies opposite-polarity duplicates within a channel and similar events across channels of the same bundle.
* **Part III — Detection of suspicious cross-correlations.** `detect_correlations` calculates cluster-pair correlograms and labels events contributing to suspicious central bins.

### Data structures

`load_session` in `src/der/io.py` collects the information required for detection into a `Session` object. This includes each event's timestamp, waveform, channel ID, bundle ID, cluster ID, unit class, and detection threshold. The pipeline returns a `Result` containing the session and detection labels. `save_labels` writes the labels into copies of the original times files, preserving their other variables and original spike order.

This implementation supports Blackrock Wave_clus output; Neuralynx and Combinato input workflows are not included.

Detected events are labeled individually for each part of the pipeline using prime factors. The detection label is the product of the following factors; a label of 1 means the event was not flagged.

| Factor | Part detected |
| :--- | :--- |
| 2 | Across bundles (Part I) |
| 3 | Within the same channel (Part II) |
| 5 | Within the same bundle, but different channels (Part II) |
| 7 | Based on the cross-correlograms (Part III) |

To run tests: `uv run pytest -c pytest.ini tests -q`.

## References

Cite as:

Dehnen, G.; Kehl, M.S.; Darcher, A.; Müller, T.T.; Macke, J.H.; Borger, V.; Surges, R.; Mormann, F. Duplicate Detection of Spike Events: A Relevant Problem in Human Single-Unit Recordings. *Brain Sciences* 2021, 11, 761. [https://doi.org/10.3390/brainsci11060761](https://doi.org/10.3390/brainsci11060761)
