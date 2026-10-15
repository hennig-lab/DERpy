# SPDX-License-Identifier: MPL-2.0
"""Three-stage DER port. No random state or random sampling is used."""

import logging
import time
from dataclasses import asdict, dataclass
from itertools import combinations, combinations_with_replacement

import numpy as np
from numba import njit
from tqdm import tqdm

from .io import Session, load_session
from .wavelets import features

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    across_threshold: float = 14.4
    across_window_ms: float = 0.05
    min_across_events: int = 3
    within_threshold: float = 8.5
    within_window_ms: float = 0.05
    same_channel_window_ms: float = 0.65
    amplitude_sample: int = 19  # zero-based equivalent of MATLAB sample 20
    hist_limit_ms: float = 20.0
    bin_width_ms: float = 0.5
    z_threshold: float = 5.0
    correlation_scope: str = "all"  # or "within-bundle" (MATLAB option 1)
    skip_labeled_correlations: bool = False
    matlab_compatible: bool = True
    bundle_grouping: str = "auto"
    channels_per_bundle: int = 8

    def __post_init__(self):
        if self.bundle_grouping not in ("auto", "filename", "relative"):
            raise ValueError("bundle_grouping must be auto, filename, or relative")
        if (
            isinstance(self.channels_per_bundle, bool)
            or not isinstance(self.channels_per_bundle, int)
            or self.channels_per_bundle <= 0
        ):
            raise ValueError("channels_per_bundle must be a positive integer")
        for key in (
            "across_window_ms",
            "within_window_ms",
            "same_channel_window_ms",
            "hist_limit_ms",
            "bin_width_ms",
        ):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("across_threshold", "within_threshold", "z_threshold"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        if self.min_across_events < 2 or not 0 <= self.amplitude_sample < 64:
            raise ValueError("min_across_events must be >=2; amplitude_sample must be in [0,63]")
        ratio = self.hist_limit_ms / self.bin_width_ms
        if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
            raise ValueError("hist_limit_ms must be a multiple of bin_width_ms")
        if self.correlation_scope not in ("all", "within-bundle"):
            raise ValueError("Unknown correlation_scope")


@dataclass
class Result:
    session: Session
    labels: np.ndarray
    config: Config
    correlations: list[dict]
    elapsed_seconds: float

    def summary(self):
        return {
            "n_files": len(self.session.files),
            "n_spikes": len(self.labels),
            "n_labeled": int(np.sum(self.labels != 1)),
            "factors": {str(p): int(np.sum(self.labels % p == 0)) for p in (2, 3, 5, 7)},
            "missing_thresholds": self.session.missing_thresholds,
            "config": asdict(self.config),
            "elapsed_seconds": self.elapsed_seconds,
            "bundle_layout": {
                "method": self.session.bundle_grouping,
                "channel_origin": self.session.channel_origin,
                "channels_per_bundle": self.session.channels_per_bundle,
                "n_bundles": len(self.session.bundle_names),
            },
        }


def _progress(iterable, *, desc, progress=None, total=None, unit="it", position=0, leave=True):
    """stderr bars: auto-detect terminals unless explicitly enabled/disabled."""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        disable=None if progress is None else not progress,
        dynamic_ncols=True,
        position=position,
        leave=leave,
    )


def candidate_groups(
    times, window, matlab_compatible=True, *, progress=None, desc=None, position=0, leave=True
):
    """Overlapping temporal groups, matching the MATLAB loop over adjacent edges."""
    edges = np.flatnonzero(np.diff(times) <= window)
    stop = len(edges) - 1 if matlab_compatible else len(edges)
    positions = range(max(0, stop))
    if desc is not None:
        positions = _progress(
            positions, desc=desc, progress=progress, unit="group", position=position, leave=leave
        )
    for pos in positions:
        end = pos
        chain_stop = len(edges) - 1 if matlab_compatible else len(edges)
        while end + 1 < chain_stop and edges[end + 1] == edges[end] + 1:
            end += 1
        first, last = edges[pos], edges[end] + 1
        last = min(last, np.searchsorted(times, times[first] + window, side="right") - 1)
        if last > first:
            yield np.arange(first, last + 1)


def _sn(session, indices, sample):
    with np.errstate(invalid="ignore", divide="ignore"):
        return abs(np.nanmedian(session.shapes[indices, sample])) / session.threshold[indices[0]]


def detect_across(session, labels, config, *, progress=None):
    marked = np.zeros(len(labels), dtype=bool)
    minimum = config.min_across_events * (config.min_across_events - 1) // 2
    for group in candidate_groups(
        session.time,
        config.across_window_ms,
        config.matlab_compatible,
        progress=progress,
        desc="Part I: candidate groups",
    ):
        pairs = np.asarray(
            [(i, j) for i, j in combinations(group, 2) if session.bundle[i] != session.bundle[j]],
            dtype=np.int64,
        ).reshape(-1, 2)
        if len(pairs) < minimum:
            continue
        a, b = session.shapes[pairs[:, 0], :40].copy(), session.shapes[pairs[:, 1], :40]
        if config.matlab_compatible:
            # Upstream uses linear index 20, then negates linear index 1.
            # Preserve this for comparison; it does NOT flip whole waveforms.
            if np.sign(a.ravel(order="F")[19]) != np.sign(b.ravel(order="F")[19]):
                a[0, 0] *= -1
        else:
            flip = np.sign(a[:, config.amplitude_sample]) != np.sign(b[:, config.amplitude_sample])
            a[flip] *= -1
        distances = []
        for x, y in zip(a, b):
            coeff = features(np.stack((x, y)))
            distances.append(np.linalg.norm(coeff[0] - coeff[1]))
        if np.median(distances) <= config.across_threshold:
            marked[group] = True
    labels[marked] *= 2


def detect_within(session, labels, config, *, progress=None):
    sample = config.amplitude_sample
    marked_same = np.zeros(len(labels), dtype=bool)
    marked_bundle = np.zeros(len(labels), dtype=bool)
    clusters = {
        tuple(key): np.flatnonzero((session.channel == key[0]) & (session.cluster == key[1]))
        for key in _progress(
            np.unique(np.c_[session.channel, session.cluster], axis=0),
            desc="Part II: indexing clusters",
            progress=progress,
            unit="cluster",
        )
    }
    for ch in _progress(
        np.unique(session.channel), desc="Part II: same-channel scan", progress=progress, unit="channel"
    ):
        idx = np.flatnonzero(session.channel == ch)
        close = np.flatnonzero(np.diff(session.time[idx]) <= config.same_channel_window_ms)
        for k in _progress(
            close, desc=f"Part II: channel {ch}", progress=progress, unit="pair", position=1, leave=False
        ):
            i, j = idx[k], idx[k + 1]
            if abs(np.sign(session.shapes[i, sample]) - np.sign(session.shapes[j, sample])) != 2:
                continue
            if session.cluster[i] == 0 or session.cluster[j] == 0:
                if session.cluster[i] == 0:
                    marked_same[i] = True
                if session.cluster[j] == 0:
                    marked_same[j] = True
            elif session.unit[i] == session.unit[j]:
                sn1 = _sn(session, clusters[ch, session.cluster[i]], sample)
                sn2 = _sn(session, clusters[ch, session.cluster[j]], sample)
                marked_same[i if sn1 <= sn2 else j] = True
            else:
                marked_same[j if session.unit[i] == "SU" else i] = True
    labels[marked_same] *= 3
    # Blackrock channel numbers are acquisition IDs, not bundle boundaries.
    # Use the same resolved grouping as Parts I and III in both modes.
    groups = session.bundle
    for bundle in _progress(np.unique(groups), desc="Part II: bundle scan", progress=progress, unit="bundle"):
        idx = np.flatnonzero(groups == bundle)
        if len(np.unique(session.channel[idx])) < 2:
            continue
        name = session.bundle_names.get(int(bundle), "")
        description = f"Part II: bundle {bundle} {name} (channels {session.channel[idx].min()}-{session.channel[idx].max()})"
        for group in candidate_groups(
            session.time[idx],
            config.within_window_ms,
            config.matlab_compatible,
            progress=progress,
            desc=description,
            position=1,
            leave=False,
        ):
            pairs = np.asarray(
                [
                    (idx[i], idx[j])
                    for i, j in combinations(group, 2)
                    if session.channel[idx[i]] != session.channel[idx[j]]
                ],
                dtype=np.int64,
            ).reshape(-1, 2)
            if not len(pairs):
                continue
            coeff = features(
                np.concatenate((session.shapes[pairs[:, 0], :40], session.shapes[pairs[:, 1], :40]))
            )
            if (
                np.median(np.linalg.norm(coeff[: len(pairs)] - coeff[len(pairs) :], axis=1))
                > config.within_threshold
            ):
                continue
            for i, j in pairs:
                ui, uj = session.unit[i], session.unit[j]
                if not config.matlab_compatible and (ui == "A" or uj == "A"):
                    marked_bundle[[i, j]] = True
                elif ui == uj:
                    if config.matlab_compatible:
                        # Upstream applies a channel-local cluster mask to the
                        # prefix of the entire current bundle table.
                        def upstream_sn(event, idx=idx):
                            local_ids = session.cluster[idx[session.channel[idx] == session.channel[event]]]
                            selected = idx[np.flatnonzero(local_ids == session.cluster[event])]
                            return (
                                abs(np.nanmedian(session.shapes[selected, sample])) / session.threshold[event]
                            )

                        sn1, sn2 = upstream_sn(i), upstream_sn(j)
                    else:
                        sn1 = _sn(session, clusters[session.channel[i], session.cluster[i]], sample)
                        sn2 = _sn(session, clusters[session.channel[j], session.cluster[j]], sample)
                    marked_bundle[i if sn1 <= sn2 else j] = True
                else:
                    # Upstream cell-versus-char artifact check is always false.
                    marked_bundle[j if ui == "SU" else i] = True
    labels[marked_bundle] *= 5


@njit(cache=True)
def pair_histogram(a, b, limit, width, same):
    """Bounded-lag histogram without an N x N spike difference matrix."""
    center = round(limit / width)
    hist = np.zeros(2 * center + 1, dtype=np.int64)
    radius = limit + width / 2
    left = 0
    for i in range(len(a)):
        if same:
            left = i + 1
        else:
            while left < len(b) and b[left] - a[i] <= -radius:
                left += 1
        j = left
        while j < len(b) and b[j] - a[i] < radius:
            delta = b[j] - a[i]
            # MATLAB round: halves away from zero, unlike np.round.
            offset = int(np.floor(abs(delta / width) + 0.5))
            if delta < 0:
                offset = -offset
            hist[center + offset] += 1
            if same:
                hist[center - offset] += 1
            j += 1
    return hist


def _has_neighbor(a, b, radius):
    if not len(b):
        return np.zeros(len(a), dtype=bool)
    pos = np.searchsorted(b, a)
    lo, hi = np.maximum(pos - 1, 0), np.minimum(pos, len(b) - 1)
    return np.minimum(abs(a - b[lo]), abs(a - b[hi])) < radius


def detect_correlations(session, labels, config, *, progress=None):
    eligible = np.ones(len(labels), dtype=bool)
    if config.skip_labeled_correlations:
        eligible &= labels == 1
    active = np.flatnonzero(eligible)
    if config.matlab_compatible and len(active):
        # MATLAB while-loop excludes the final globally sorted event.
        eligible[active[-1]] = False
    keys = np.unique(np.c_[session.channel, session.cluster], axis=0)
    indices = [
        np.flatnonzero((session.channel == ch) & (session.cluster == cl))
        for ch, cl in _progress(keys, desc="Part III: indexing clusters", progress=progress, unit="cluster")
    ]
    times = [session.time[idx[eligible[idx]]] for idx in indices]
    # Same global central-bin participation filter used by MATLAB.
    central = np.zeros(len(labels), dtype=bool)
    active = np.flatnonzero(eligible)
    close = np.flatnonzero(np.diff(session.time[active]) < config.bin_width_ms / 2)
    central[active[close]] = True
    central[active[close + 1]] = True
    marked = np.zeros(len(labels), dtype=bool)
    records = []
    pairs = combinations_with_replacement(range(len(keys)), 2)
    for k, l in _progress(
        pairs,
        total=len(keys) * (len(keys) + 1) // 2,
        desc="Part III: cluster pairs",
        progress=progress,
        unit="pair",
    ):
        ia, ib = indices[k], indices[l]
        same_bundle = session.bundle[ia[0]] == session.bundle[ib[0]]
        if config.correlation_scope == "within-bundle" and (not same_bundle or keys[k, 0] == keys[l, 0]):
            continue
        hist = pair_histogram(times[k], times[l], config.hist_limit_ms, config.bin_width_ms, k == l)
        center = len(hist) // 2
        background = np.r_[hist[:center], hist[center + 1 :]]
        sd = background.std(ddof=1)
        z = float((hist[center] - background.mean()) / sd) if sd else 0.0
        records.append(
            {
                "channel_a": int(keys[k, 0]),
                "cluster_a": int(keys[k, 1]),
                "channel_b": int(keys[l, 0]),
                "cluster_b": int(keys[l, 1]),
                "central_count": int(hist[center]),
                "z": z,
            }
        )
        if z <= config.z_threshold:
            continue
        a, b = ia[central[ia]], ib[central[ib]]
        ta, tb = session.time[a], session.time[b]
        ma = _has_neighbor(ta, tb, config.bin_width_ms / 2)
        mb = _has_neighbor(tb, ta, config.bin_width_ms / 2)
        ua, ub = session.unit[ia[0]], session.unit[ib[0]]
        if ua == "A" or ub == "A" or not same_bundle:
            marked[a[ma]], marked[b[mb]] = True, True
        if same_bundle and ua == ub and ua in ("SU", "MU"):
            if k == l:
                marked[a[1:][np.diff(ta) < config.bin_width_ms / 2]] = True
            else:
                sn1, sn2 = (
                    _sn(session, ia, config.amplitude_sample),
                    _sn(session, ib, config.amplitude_sample),
                )
                # Upstream leaves the target undefined/stale for ties or NaN.
                # Use a reproducible second-cluster fallback instead.
                marked[a[ma] if sn1 < sn2 else b[mb]] = True
        elif same_bundle and {ua, ub} == {"SU", "MU"}:
            marked[a[ma] if ua == "MU" else b[mb]] = True
    labels[marked] *= 7
    return records


def run(directory=None, *, session=None, config=None, pattern="times_*.mat", threshold=None, progress=None):
    """Run all three stages; return labels without modifying source files."""
    config = config or Config()
    start = time.perf_counter()
    if session is None:
        session = load_session(
            directory,
            pattern=pattern,
            threshold=threshold,
            matlab_compatible=config.matlab_compatible,
            bundle_grouping=config.bundle_grouping,
            channels_per_bundle=config.channels_per_bundle,
        )
    labels = np.ones(len(session.time), dtype=np.int64)
    log.info("Part I: across bundles (%d spikes)", len(labels))
    detect_across(session, labels, config, progress=progress)
    log.info("Part II: within bundles")
    detect_within(session, labels, config, progress=progress)
    log.info("Part III: cross-correlations")
    correlations = detect_correlations(session, labels, config, progress=progress)
    return Result(session, labels, config, correlations, time.perf_counter() - start)
