# DER for Python — Blackrock

Python port of the three-stage Duplicate Event Removal algorithm in
[hennig-lab/DER](https://github.com/hennig-lab/DER), source revision
`a44e98329c2abed47ede9e2328aa466134bf48e0`.
Supports Blackrock Wave_clus `times_*.mat` files, using **uv**.
No MATLAB, Neuralynx tools, or raw Blackrock reader is required.
This starts from sorted Wave_clus output; it does not perform spike sorting.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then in this folder:

```bash
uv sync --locked
uv run der run /path/to/session --output /path/to/new-output --report run.json
```

Use the **complete input set for one recording/run**. The three stages compare
channels; running a single channel cannot reproduce labels from a multichannel run.
Do not mix two DER runs or recordings in one directory. Named-probe filenames have the form
`times_<bundle><two-digit-wire>_<channel-number>.mat`, for example
`times_mLT2aHa01_2285.mat`. Bundle names are inferred from the prefix before the
two-digit wire number. Duplicate channel numbers are rejected.

By default (`--bundle-grouping auto`), grouping uses probe prefixes when every
filename supplies one. For example, `mLT2aHa01_2285` through
`mLT2aHa08_2292` are bundle `mLT2aHa`, and `mRT2aHa01_2293` through
`mRT2aHa08_2300` are bundle `mRT2aHa`. Acquisition channel numbers do not define
bundle boundaries in this mode. Part II progress bars include the bundle name.

Numeric-only filenames such as `times_2285.mat`, or generic names such as
`times_channel_2285.mat`, are also supported. If any file lacks a probe prefix,
auto mode uses relative channel-number blocks for the entire session:

```text
bundle ID = (original channel - smallest channel in this session) // 8 + 1
```

With smallest channel 2285, the ranges are 2285–2292 and 2293–2300. The original
channel IDs, filenames and MAT contents are retained; only bundle assignment uses
the offset. Missing channels leave gaps rather than shifting later channels into
another bundle. This fallback assumes the smallest channel is the first channel
of a probe and probe channels occupy contiguous numeric ranges.

Force relative grouping even when probe prefixes are present:

```bash
uv run der compare /data/session --bundle-grouping relative --report agreement.json
```

Set `--channels-per-bundle 4` for a different probe size (default 8). Use
`--bundle-grouping filename` to require valid probe prefixes in every filename.
All three detection stages use the same resolved assignment in both compatibility
modes. The terminal log and JSON `summary.bundle_layout` report the method used,
relative origin, bundle size and number of bundles. In Python, configure these via
`Config(bundle_grouping="relative", channels_per_bundle=8)` or the same keyword
arguments to `load_session`.

Timestamps in `cluster_class[:, 1]` are **milliseconds**. `cluster_class` must be
N×2 (cluster ID, timestamp), and `spikes` N×64. Cluster 0 is unclassified/artifact;
positive clusters are multi-units (subject to the legacy loader quirk below).
Inputs can be unsorted: original row order is restored when saving labels.
MATLAB v5–v7.2 MAT files are supported, including compressed files; v7.3/HDF5
files must first be converted using MATLAB `save(..., '-v7')`.

Noise thresholds are read as the median finite `threshold` from each companion
`<channel-name>_<channel-number>_spikes.mat`, exactly where the MATLAB loader looks.
If companions are missing, thresholds remain NaN and a warning is issued.
Provide companions when assessing agreement. An explicit `--threshold VALUE`
can supply a known common threshold; it overrides all channel thresholds.
The `par.stdmin` field is a threshold multiplier, not the companion's voltage
threshold, and is deliberately not substituted.

The command writes copies of the input MAT files with only `detectionLabel`
replaced. Other variables, including nested `par` and `gui_status`, are carried
through SciPy MAT serialization. Original files are untouched. Existing output
files are refused. Output is atomic **per file**, not transactional across the session.
Companion threshold files are not copied; retain the source directory and report.

Label 1 means unflagged. Factors can occur together:

| Factor | Detection stage |
| --- | --- |
| 2 | Similar synchronous events across bundles |
| 3 | Opposite-polarity duplicate detections in one channel |
| 5 | Similar synchronous events in different channels within a bundle |
| 7 | Significant zero-lag cross-correlation |

The detector labels events; it never deletes them. Counts for factors overlap.

## Progress bars

`run` and `compare` show tqdm bars automatically in an interactive terminal.
Each bar includes the completed/total count, percentage, elapsed time, rate, and
an estimated remaining time once enough work has completed.

- Part I: candidate temporal groups across bundles.
- Part II: cluster indexing, then channels, then bundles. Nested bars show
  close spike pairs in the current channel and candidate groups in the current
  bundle (including its channel-number range). The outer bundle bar tracks
  completed bundles; the inner bar tracks progress through a large bundle.
- Part III: cluster indexing and cluster-pair comparisons, including autocorrelations.

Force bars on (e.g. when stderr is redirected or terminal detection fails):

```bash
uv run der compare /path/to/session --report agreement.json --progress
```

Disable them with `--no-progress`. Bars go to stderr, so JSON output stays valid.
In Python, use `run(..., progress=True)` or `progress=False`; the default `None`
auto-detects terminal output. No parameter JSON changes are needed.
Counters reflect work units rather than equally expensive chunks: estimates may
change as larger groups or bundles are processed. A bar updates after each unit
finishes; one large waveform group or histogram can temporarily hold the count
steady. Initial Numba compilation is also included in elapsed time.

## Compare against MATLAB on a full recording

If the input files already contain MATLAB `detectionLabel` values:

```bash
uv run der compare /path/to/full-session --report agreement.json
```

To compare the Python result against two independently labeled MATLAB runs:

```bash
uv run der compare /path/to/full-session \
  --reference /path/to/matlab-run-1 \
  --reference /path/to/matlab-run-2 \
  --report agreement.json
```

Each reference directory must use the canonical input filenames. A reference may
contain a subset of channels; coverage and missing files are reported explicitly.
Each matched reference file is checked for **identical waveforms,
timestamps, and row order** before comparing labels. Differences in cluster IDs
are counted explicitly: these are changes to detector inputs and can affect
agreement. Python is always run on the
full primary directory, regardless of reference coverage. Existing labels are
never used as algorithm inputs. This comparison writes JSON only, not labeled MAT files.

By default, JSON reports are compact: coverage and changed cluster IDs; pooled
exact-label agreement; binary artifact agreement, precision, recall, F1 and
Jaccard; and the same binary metrics for each prime factor. Each includes
the fraction of all compared spikes flagged by the reference, Python, and both.
Channel statistics show the unweighted median and range of exact-label agreement
across nonempty channels, and the five channels with the lowest agreement.
Pooled agreement is weighted by spike counts, so it can differ from the channel
median. Empty channels are counted but excluded from channel statistics.

A short table of these results is also printed to the terminal (stderr, keeping
stdout valid JSON when no report path is supplied). The terminal displays
percentages; JSON rates are fractions from 0 to 1. Missing reference and threshold
filename lists are limited to five examples with total counts.

Add `--detailed-report` to retain every per-channel statistic, label confusion
table, and full missing-filename list:

```bash
uv run der compare /path/to/full-session --report agreement-detailed.json --detailed-report
```

To summarize an existing report from an earlier version **without rerunning DER**:

```bash
uv run der summarize-report agreement.json --report agreement-summary.json
```

This prints the same terminal summary and writes compact JSON. The original report
is preserved. No recording files are needed.

MATLAB is the **reference**,
not assumed ground truth. Null indicates an undefined metric (e.g. no positives).
Artifact agreement alone can look high when most events are unflagged, so inspect
Jaccard/recall and the per-factor results as well.

An optional `--min-agreement 0.95` causes exit status 1 if a comparison falls
below a user-chosen exact-label threshold. There is no default numerical gate:
the expected tolerance must be calibrated to full-session MATLAB variability.
Exit status 2 denotes an input/configuration error.

Compare two existing labeled files without running DER (filenames may differ):

```bash
uv run der compare-labels /path/run1/times_channel.mat /path/run2/times_channel.mat \
  --report matlab-repeatability.json
```

`compare-labels` also accepts `--detailed-report` to include its confusion table.

This is the appropriate baseline calculation for the two supplied example files.
The examples cannot validate the across-channel detector: they contain one channel
whose labels were obtained using many other, unavailable channels.

## Tests

```bash
uv run pytest -c pytest.ini tests -q
uv run pytest -c pytest.ini tests -q --example-data-dir /path/to/the/two-uploaded-files
uv run pytest -c pytest.ini -s tests/test_full_session.py \
  --der-data-dir /path/to/full-session \
  --der-reference-dir /path/to/matlab-run-1 \
  --der-reference-dir /path/to/matlab-run-2 \
  --der-report full-session-validation.json
```

`DER_FULL_DATA_DIR` and `DER_EXAMPLE_DATA_DIR` can replace the corresponding
options. Add `--der-min-agreement 0.95` to explicitly enforce a parity tolerance.
Add `--der-detailed-report` to retain full per-channel details in the pytest report.
Without external paths, real-data tests are skipped, while synthetic tests run.
Tests cover independent PyWavelets Haar coefficients, brute-force correlograms,
stage-specific deletion policies, missing/empty input handling, input-label
independence, MAT round-tripping, row order, and comparison metrics.
Full-session tests produce a report and do not modify their inputs.

## Compatibility and corrected behavior

Default mode (`matlab_compatible=true`) preserves consequential source quirks to
make comparison useful. It is **not a promise of bitwise MATLAB parity**.
`--corrected` applies the fixes below; its output is deliberately a different
algorithm and should be assessed separately.

| Behavior | Default compatibility mode | `--corrected` |
| --- | --- | --- |
| Part II bundle grouping | Resolved probe IDs (auto/filename/relative) | Resolved probe IDs (auto/filename/relative) |
| Candidate group loop | Omits final candidate edge, including chain endpoint | Includes final edge |
| Part I polarity | Reproduces MATLAB linear-index operation (sample 20 and element 1) | Flips whole opposite-polarity waveform |
| Part II S/N amplitude | Reproduces channel-local mask applied to bundle prefix | Correct channel-and-cluster amplitude |
| Part II artifact branch | Preserves ineffective cell-vs-char check | Marks both events when either is artifact |
| Correlation scan | Excludes final globally sorted eligible event | Includes final event |
| Noncontiguous cluster IDs | Reproduces 1…number-of-unique-IDs classification | All positive IDs are MU |

Both modes resolve an upstream undefined/stale variable: for Part III equal or
NaN signal-to-noise values, the second cluster in sorted `(channel,cluster)` order
is selected for labeling. The MATLAB implementation can instead error or reuse
a previous pair's target. This deliberate deterministic fallback is recorded here;
missing companion thresholds can therefore materially affect parity.
The port also accepts unsorted input safely, handles empty channels, and restores
original row order, whereas the MATLAB writer assumes already sorted rows.
As of version 0.1.3, both modes also fix the upstream Part II assumption that
bundles align with numeric channel blocks 1–8, 9–16, and so on. This assumption
does not hold for Blackrock acquisition IDs. Earlier Python versions preserved
that bug in compatibility mode. Rerun detection to apply this fix; existing labels
are not changed merely by updating the package.
Version 0.1.4 adds explicit relative-number grouping and the automatic fallback
for filenames without probe names.

Numerical details preserved: five-level symmetric Haar decomposition, first 40
samples for waveform comparisons, first 40 coefficients of the padded transform,
10 coefficients ranked by the original KS statistic, sample standard deviation
(`ddof=1`), stable sorting with reversed coefficient ties, half-away-from-zero
histogram rounding, strict central-bin comparisons, z=0 for zero background
variance, and all-cluster correlations including auto-correlations. The latter
means the correlation stage is not restricted to cross-channel pairs.

No random-number calls occur in this upstream DER revision. This implementation
is deterministic for fixed inputs/configuration/software versions. The supplied
MATLAB runs still provide a useful empirical variability baseline; they do not
establish where that variability arose. MATLAB wavelet-extension settings can
also affect parity; this port explicitly uses symmetric extension.

## Configuration and Python API

All algorithm parameters are available in `Config`, with the MATLAB defaults.
For CLI use, pass a JSON object with any subset of fields via `--config config.json`:

```json
{"z_threshold": 5.0, "bin_width_ms": 0.5, "hist_limit_ms": 20.0}
```

```python
from der import Config, run, save_labels
from der.compare import compare_result

result = run("/data/session", config=Config())
report = compare_result(result)  # against saved MATLAB labels, in memory
save_labels(result.session, result.labels, "/data/python-output")
```

Additional Config fields: `across_threshold=14.4`, `across_window_ms=0.05`,
`min_across_events=3`, `within_threshold=8.5`, `within_window_ms=0.05`,
`same_channel_window_ms=0.65`, `amplitude_sample=19` (**zero based**),
`correlation_scope="all"` (or `"within-bundle"`, MATLAB option 1),
`skip_labeled_correlations=false`, and `matlab_compatible=true`.
Correlations include previously flagged events by default, matching the source.

Numba accelerates bounded-lag histograms; first execution includes compilation.
Correlations are computed pair by pair, avoiding a dense clusters×clusters×lags
matrix and any spikes×spikes distance matrix. All waveforms are loaded in memory;
this is not an out-of-core implementation. Very large recordings may need
substantial RAM and computation time.

Optional diagnostic zero-lag z-score heatmap:

```bash
uv sync --locked --extra plots
uv run der compare /path/to/session --report agreement.json --plot central-z.png
```

The MATLAB GUI and individual cross-correlogram example figures are not ported;
tqdm progress bars, JSON diagnostics, and the optional heatmap replace those side effects.

## Attribution

MPL-2.0; see `LICENSE` and `NOTICE`. Cite Dehnen et al. (2021),
*Duplicate Detection of Spike Events: A Relevant Problem in Human Single-Unit
Recordings*, Brain Sciences 11, 761,
[doi:10.3390/brainsci11060761](https://doi.org/10.3390/brainsci11060761).
