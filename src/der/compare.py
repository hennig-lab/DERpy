# SPDX-License-Identifier: MPL-2.0
"""Agreement reports: MATLAB labels are references, not ground truth."""

from pathlib import Path

import numpy as np

from .io import labels_by_file, read_mat


def binary_metrics(reference, predicted):
    reference, predicted = np.asarray(reference, bool), np.asarray(predicted, bool)
    tp = int(np.sum(reference & predicted))
    tn = int(np.sum(~reference & ~predicted))
    fp = int(np.sum(~reference & predicted))
    fn = int(np.sum(reference & ~predicted))

    def divide(a, b):
        return float(a / b) if b else None

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "agreement": divide(tp + tn, len(reference)),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "jaccard": divide(tp, tp + fp + fn),
        "reference_positive": tp + fn,
        "predicted_positive": tp + fp,
        "reference_flagged_fraction": divide(tp + fn, len(reference)),
        "predicted_flagged_fraction": divide(tp + fp, len(reference)),
        "both_flagged_fraction": divide(tp, len(reference)),
    }


def agreement(reference, predicted):
    reference, predicted = np.asarray(reference).ravel(), np.asarray(predicted).ravel()
    if reference.shape != predicted.shape:
        raise ValueError("Reference and prediction lengths differ")
    for lab in (reference, predicted):
        if not np.isfinite(lab).all() or np.any(lab < 1) or np.any(lab != np.floor(lab)):
            raise ValueError("Labels must be positive integers")
        rest = lab.astype(np.int64).copy()
        for p in (2, 3, 5, 7):
            rest[rest % p == 0] //= p
        if np.any(rest != 1):
            raise ValueError("Labels must be square-free products of 2,3,5,7")
    pairs, counts = np.unique(np.c_[reference, predicted], axis=0, return_counts=True)
    return {
        "n_spikes": len(reference),
        "exact_label_agreement": float(np.mean(reference == predicted)) if len(reference) else None,
        "artifact": binary_metrics(reference != 1, predicted != 1),
        "factors": {str(p): binary_metrics(reference % p == 0, predicted % p == 0) for p in (2, 3, 5, 7)},
        "confusion": [
            {"reference": int(a), "predicted": int(b), "count": int(n)} for (a, b), n in zip(pairs, counts)
        ],
    }


def summarize_comparison(report, *, detailed=False):
    """Bound the default report size while retaining spike-pooled metrics.

    Channel statistics are unweighted across nonempty channels. Detailed mode
    additionally retains all per-file metrics, missing filenames and confusion tables.
    """
    rows = report["per_file"]
    nonempty = [r for r in rows if r["exact_label_agreement"] is not None]
    scores = [r["exact_label_agreement"] for r in nonempty]
    worst = sorted(nonempty, key=lambda r: (r["exact_label_agreement"], r["file"]))[:5]
    statistics = {
        "nonempty_channels": len(nonempty),
        "empty_channels": len(rows) - len(nonempty),
        "median_exact_label_agreement": float(np.median(scores)) if scores else None,
        "min_exact_label_agreement": min(scores) if scores else None,
        "max_exact_label_agreement": max(scores) if scores else None,
        "lowest_agreement_channels": [
            {
                "file": r["file"],
                "n_spikes": r["n_spikes"],
                "exact_label_agreement": r["exact_label_agreement"],
                "artifact_jaccard": r["artifact"]["jaccard"],
            }
            for r in worst
        ],
    }

    def add_rates(metric):
        n = sum(metric[k] for k in ("tp", "tn", "fp", "fn"))
        return {
            **metric,
            "reference_flagged_fraction": (metric["tp"] + metric["fn"]) / n if n else None,
            "predicted_flagged_fraction": (metric["tp"] + metric["fp"]) / n if n else None,
            "both_flagged_fraction": metric["tp"] / n if n else None,
        }

    aggregate = {
        **report["aggregate"],
        "artifact": add_rates(report["aggregate"]["artifact"]),
        "factors": {p: add_rates(m) for p, m in report["aggregate"]["factors"].items()},
    }
    output = {**report, "aggregate": aggregate, "channel_statistics": statistics}
    missing = report["coverage"]["missing_files"]
    output["coverage"] = {**report["coverage"], "missing_file_count": len(missing)}
    if not detailed:
        output.pop("per_file")
        output["aggregate"] = {k: v for k, v in aggregate.items() if k != "confusion"}
        output["coverage"].pop("missing_files")
        output["coverage"]["missing_file_examples"] = missing[:5]
    return output


def summarize_report(payload):
    """Compact a previously generated agreement report without reading MAT files."""
    if not isinstance(payload, dict) or not isinstance(payload.get("comparisons"), list):
        raise TypeError("Expected an agreement report containing a comparisons array")
    output = {**payload, "report_detail": "compact"}
    output["comparisons"] = [
        summarize_comparison(c) if "per_file" in c else c for c in payload["comparisons"]
    ]
    if "summary" in payload:
        output["summary"] = dict(payload["summary"])
        if "missing_thresholds" in output["summary"]:
            missing = output["summary"].pop("missing_thresholds")
            output["summary"]["missing_threshold_count"] = len(missing)
            output["summary"]["missing_threshold_examples"] = missing[:5]
    return output


def format_summary(report):
    """Human-readable statistics; fractions in JSON are displayed as percentages."""

    def pct(value):
        return f"{value:.2%}" if value is not None else "n/a"

    aggregate = report["aggregate"]
    coverage = report["coverage"]
    lines = [
        f"Comparison: {report['reference']}",
        (
            f"  Coverage: {coverage['compared_files']}/{coverage['input_files']} channels; "
            f"{aggregate['n_spikes']:,} spikes; {coverage['missing_file_count']} missing references"
        ),
        f"  Changed cluster IDs: {report['cluster_id_differences']:,}",
        f"  Exact-label agreement (pooled spikes): {pct(aggregate['exact_label_agreement'])}",
        "  Stage       Agreement   Ref flagged   Python flagged   Both flagged   Jaccard",
    ]
    for name, metric in [
        ("Artifact", aggregate["artifact"]),
        *[(f"Factor {p}", aggregate["factors"][str(p)]) for p in (2, 3, 5, 7)],
    ]:
        lines.append(
            f"  {name:<10} {pct(metric['agreement']):>9} "
            f"{pct(metric['reference_flagged_fraction']):>13} "
            f"{pct(metric['predicted_flagged_fraction']):>16} "
            f"{pct(metric['both_flagged_fraction']):>14} {pct(metric['jaccard']):>9}"
        )
    stats = report["channel_statistics"]
    lines.append(
        f"  Per-channel exact agreement: median {pct(stats['median_exact_label_agreement'])}; "
        f"range {pct(stats['min_exact_label_agreement'])}–{pct(stats['max_exact_label_agreement'])}"
    )
    if stats["empty_channels"]:
        lines.append(f"  Empty channels excluded from channel statistics: {stats['empty_channels']}")
    if stats["lowest_agreement_channels"]:
        lines.append("  Lowest-agreement channels:")
        lines.extend(
            f"    {r['file']}: {pct(r['exact_label_agreement'])} ({r['n_spikes']:,} spikes)"
            for r in stats["lowest_agreement_channels"]
        )
    lines.append("  Flagged percentages use all compared spikes; Jaccard uses the union of flagged spikes.")
    return "\n".join(lines)


def compare_result(result, reference_directory=None, *, detailed=False):
    """Compare all input channels, or a validated subset in another run directory."""
    rows, refs, preds, missing = [], [], [], []
    for fi, (path, predicted) in enumerate(
        zip(result.session.files, labels_by_file(result.session, result.labels))
    ):
        cluster_differences = 0
        if reference_directory is None:
            ref = result.session.reference[fi]
        else:
            other = Path(reference_directory) / path.name
            if not other.exists():
                missing.append(path.name)
                continue
            d = read_mat(other, variable_names=["cluster_class", "spikes", "detectionLabel"])
            original = read_mat(path, variable_names=["cluster_class", "spikes"])
            if (
                "cluster_class" not in d
                or "spikes" not in d
                or not np.array_equal(d["spikes"], original["spikes"])
                or d["cluster_class"].shape != original["cluster_class"].shape
                or not np.array_equal(d["cluster_class"][:, 1], original["cluster_class"][:, 1])
            ):
                raise ValueError(f"Reference does not contain identical input events: {path.name}")
            cluster_differences = int(np.sum(d["cluster_class"][:, 0] != original["cluster_class"][:, 0]))
            ref = d.get("detectionLabel")
        if ref is None:
            missing.append(path.name)
            continue
        ref = np.asarray(ref).ravel()
        rows.append(
            dict(file=path.name, cluster_id_differences=cluster_differences, **agreement(ref, predicted))
        )
        refs.append(ref)
        preds.append(predicted)
    if not rows:
        raise ValueError("No matching channels with reference detectionLabel fields")
    report = {
        "reference": str(reference_directory) if reference_directory else "input detectionLabel",
        "coverage": {
            "compared_files": len(rows),
            "input_files": len(result.session.files),
            "missing_files": missing,
        },
        "cluster_id_differences": sum(r["cluster_id_differences"] for r in rows),
        "aggregate": agreement(np.concatenate(refs), np.concatenate(preds)),
        "per_file": rows,
    }
    return summarize_comparison(report, detailed=detailed)


def compare_files(first, second, *, detailed=False):
    a, b = (
        read_mat(p, variable_names=["cluster_class", "spikes", "detectionLabel"]) for p in (first, second)
    )
    if any(k not in d for d in (a, b) for k in ("cluster_class", "spikes", "detectionLabel")):
        raise ValueError("Both files require cluster_class, spikes, and detectionLabel")
    if (
        not np.array_equal(a["spikes"], b["spikes"])
        or a["cluster_class"].shape != b["cluster_class"].shape
        or not np.array_equal(a["cluster_class"][:, 1], b["cluster_class"][:, 1])
    ):
        raise ValueError("Files do not contain identical input events in identical order")
    report = dict(
        reference=str(first),
        prediction=str(second),
        cluster_id_differences=int(np.sum(a["cluster_class"][:, 0] != b["cluster_class"][:, 0])),
        **agreement(a["detectionLabel"], b["detectionLabel"]),
    )
    if not detailed:
        report.pop("confusion")
    return report
