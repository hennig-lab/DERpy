# SPDX-License-Identifier: MPL-2.0
"""Blackrock MAT input/output. All times are milliseconds, as in Wave_clus."""

import logging
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


@dataclass
class Session:
    files: list[Path]
    time: np.ndarray
    shapes: np.ndarray
    channel: np.ndarray
    bundle: np.ndarray
    cluster: np.ndarray
    threshold: np.ndarray
    unit: np.ndarray
    file_index: np.ndarray
    row: np.ndarray
    reference: list[np.ndarray | None]
    missing_thresholds: list[str]
    bundle_names: dict[int, str] = field(default_factory=dict)
    bundle_grouping: str = "filename"
    channel_origin: int | None = None
    channels_per_bundle: int | None = None


def read_mat(path, **kwargs):
    try:
        return loadmat(path, **kwargs)
    except NotImplementedError as exc:
        raise ValueError(f"{path}: MATLAB v7.3/HDF5 is not supported; save as -v7 first") from exc


def _channel_layout(files, grouping, size):
    if grouping not in ("auto", "filename", "relative"):
        raise ValueError("bundle_grouping must be auto, filename, or relative")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("channels_per_bundle must be a positive integer")
    parsed, seen = [], set()
    for path in files:
        match = re.fullmatch(r"times_(?:(.+)_)?(\d+)\.mat", path.name)
        if not match:
            raise ValueError(
                f"Invalid channel filename: {path.name}; expected times_<channel>.mat "
                "or times_<name>_<channel>.mat"
            )
        name, ch = match.group(1), int(match.group(2))
        if ch <= 0 or ch in seen:
            raise ValueError(f"Duplicate or nonpositive channel ID {ch}; use one run per directory")
        seen.add(ch)
        bundle_name = name[:-2] if name and re.fullmatch(r".+\d{2}", name) else None
        parsed.append((ch, bundle_name))
    if grouping == "auto":
        grouping = "filename" if all(name is not None for _, name in parsed) else "relative"
    if grouping == "filename" and any(name is None for _, name in parsed):
        raise ValueError("Filename grouping requires a <bundle><two-digit-wire> prefix in every file")
    origin = min(ch for ch, _ in parsed) if grouping == "relative" else None
    names, layout = {}, []
    for ch, name in parsed:
        if grouping == "relative":
            bundle = (ch - origin) // size + 1
            start = origin + (bundle - 1) * size
            names[bundle] = f"relative {start}-{start + size - 1}"
        else:
            bundle = next((i for i, n in names.items() if n == name), len(names) + 1)
            names[bundle] = name
        layout.append((ch, bundle))
    return layout, names, grouping, origin


def load_session(
    directory,
    *,
    pattern="times_*.mat",
    threshold=None,
    matlab_compatible=True,
    bundle_grouping="auto",
    channels_per_bundle=8,
):
    """Load one complete recording, never combining separate DER runs.

    Existing detectionLabel values are references only and never inputs.
    Duplicate channel IDs (e.g. copied runs in one folder) are rejected.
    """
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise ValueError(f"No files matching {pattern!r} in {directory}")
    arrays = {
        k: []
        for k in ("time", "shapes", "channel", "bundle", "cluster", "threshold", "unit", "file_index", "row")
    }
    layout, bundle_names, grouping, origin = _channel_layout(files, bundle_grouping, channels_per_bundle)
    logging.getLogger(__name__).info(
        "Bundle grouping: %s (%d bundles%s)",
        grouping,
        len(bundle_names),
        f", origin {origin}, {channels_per_bundle} channels per bundle" if origin is not None else "",
    )
    refs, missing = [], []
    for fi, (path, (ch, bundle)) in enumerate(zip(files, layout)):
        d = read_mat(path, variable_names=["spikes", "cluster_class", "detectionLabel"])
        if "spikes" not in d or "cluster_class" not in d:
            raise ValueError(f"{path.name}: missing spikes or cluster_class")
        shapes = np.asarray(d["spikes"], dtype=float)
        cc = np.asarray(d["cluster_class"], dtype=float)
        if cc.ndim != 2 or cc.shape[1] != 2 or shapes.shape != (len(cc), 64):
            raise ValueError(f"{path.name}: expected cluster_class (N,2) and spikes (N,64)")
        if not np.isfinite(cc).all() or not np.isfinite(shapes).all():
            raise ValueError(f"{path.name}: timestamps, clusters and waveforms must be finite")
        if np.any(cc[:, 0] != np.floor(cc[:, 0])) or np.any(cc[:, 0] < 0):
            raise ValueError(f"{path.name}: cluster IDs must be nonnegative integers")
        th = float(threshold) if threshold is not None else np.nan
        if threshold is None:
            companion = path.with_name(path.name[6:-4] + "_spikes.mat")
            if companion.exists():
                td = read_mat(companion, variable_names=["threshold"])
                vals = np.asarray(td.get("threshold", []), dtype=float).ravel()
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    th = float(np.median(vals))
        if np.isfinite(th) and th <= 0:
            raise ValueError(f"{path.name}: threshold must be positive")
        if not np.isfinite(th):
            missing.append(path.name)
        n = len(cc)
        cl = cc[:, 0].astype(np.int64)
        # Original loader iterates 1:numel(unique(clusterID)), not actual IDs.
        mu = (cl >= 1) & (cl <= len(np.unique(cl))) if matlab_compatible else cl > 0
        values = {
            "time": cc[:, 1],
            "shapes": shapes,
            "channel": np.full(n, ch),
            "bundle": np.full(n, bundle),
            "cluster": cl,
            "threshold": np.full(n, th),
            "unit": np.where(mu, "MU", "A"),
            "file_index": np.full(n, fi),
            "row": np.arange(n),
        }
        for k, value in values.items():
            arrays[k].append(value)
        ref = d.get("detectionLabel")
        if ref is not None:
            ref = np.asarray(ref).ravel()
            if len(ref) != n or not np.isfinite(ref).all() or np.any(ref < 1) or np.any(ref != np.floor(ref)):
                raise ValueError(f"{path.name}: invalid detectionLabel")
            ref = ref.astype(np.int64)
        refs.append(ref)
    data = {k: np.concatenate(v) for k, v in arrays.items()}
    # MATLAB loader concatenates channels in numeric order before stable time sort.
    order = np.lexsort((data["row"], data["channel"], data["time"]))
    if missing:
        warnings.warn(
            f"Missing finite companion thresholds for {len(missing)} channel(s); "
            "S/N comparisons will follow MATLAB NaN behavior. Supply *_spikes.mat or --threshold.",
            stacklevel=2,
        )
    return Session(
        files=files,
        reference=refs,
        missing_thresholds=missing,
        bundle_names=bundle_names,
        bundle_grouping=grouping,
        channel_origin=origin,
        channels_per_bundle=channels_per_bundle if grouping == "relative" else None,
        **{k: v[order] for k, v in data.items()},
    )


def labels_by_file(session, labels):
    if np.shape(labels) != np.shape(session.time):
        raise ValueError("Labels must match session length")
    result = []
    for fi in range(len(session.files)):
        mask = session.file_index == fi
        result.append(np.asarray(labels)[mask][np.argsort(session.row[mask])])
    return result


def save_labels(session, labels, output_directory):
    """Copy full MAT payloads, replacing only detectionLabel; atomic per file.

    Output is MATLAB v5 (through v7.2 compatible). Input directory must differ.
    Existing outputs are refused, avoiding silent overwrite of previous runs.
    """
    out = Path(output_directory).resolve()
    if any(p.parent.resolve() == out for p in session.files):
        raise ValueError("Output directory must differ from the input directory")
    targets = [out / p.name for p in session.files]
    if any(p.exists() for p in targets):
        raise FileExistsError("Output times files already exist; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    for path, target, lab in zip(session.files, targets, labels_by_file(session, labels)):
        payload = {k: v for k, v in read_mat(path).items() if not k.startswith("__")}
        payload["detectionLabel"] = lab[:, None]
        fd, tmp = tempfile.mkstemp(dir=out, suffix=".mat")
        os.close(fd)
        try:
            savemat(tmp, payload, do_compression=True, appendmat=False)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return targets
