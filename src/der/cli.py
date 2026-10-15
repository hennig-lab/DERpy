# SPDX-License-Identifier: MPL-2.0
import argparse
import importlib.metadata
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .compare import compare_files, compare_result, format_summary, summarize_report
from .core import Config, run
from .io import save_labels


def _emit(payload, report):
    content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if report:
        target = Path(report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    else:
        print(content, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Blackrock Duplicate Event Removal (times are milliseconds)")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "compare"):
        p = sub.add_parser(command)
        p.add_argument("directory", type=Path, help="Complete times-file set from ONE session/run")
        p.add_argument("--pattern", default="times_*.mat")
        p.add_argument(
            "--threshold", type=float, help="Explicit common detection threshold (otherwise companion files)"
        )
        p.add_argument("--config", type=Path, help="JSON object of Config fields")
        p.add_argument(
            "--bundle-grouping",
            choices=("auto", "filename", "relative"),
            default=None,
            help="auto: use probe prefixes when all files have them, otherwise relative channel blocks",
        )
        p.add_argument(
            "--channels-per-bundle",
            type=int,
            default=None,
            help="Channels per relative-number bundle (default 8)",
        )
        p.add_argument("--corrected", action="store_true", help="Fix documented upstream indexing quirks")
        p.add_argument("--report", type=Path, help="JSON report path (otherwise stdout)")
        p.add_argument("--plot", type=Path, help="Optional zero-lag z-score PNG (install plots extra)")
        p.add_argument(
            "--progress",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Show tqdm bars (default: auto-detect terminal; --no-progress disables)",
        )
        if command == "run":
            p.add_argument(
                "--output", type=Path, required=True, help="New output directory; originals untouched"
            )
        else:
            p.add_argument(
                "--detailed-report",
                action="store_true",
                help="Include all channel metrics, missing filenames and confusion tables",
            )
            p.add_argument(
                "--reference", type=Path, action="append", help="Another MATLAB run directory; repeatable"
            )
            p.add_argument(
                "--min-agreement", type=float, help="Optional exact-label agreement gate, from 0 to 1"
            )
    p = sub.add_parser("compare-labels", help="Compare two labeled files without rerunning DER")
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--detailed-report", action="store_true", help="Include the label confusion table")
    p = sub.add_parser("summarize-report", help="Compact an existing agreement JSON without rerunning DER")
    p.add_argument("input", type=Path)
    p.add_argument("--report", type=Path, help="Compact JSON path (otherwise stdout)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        if args.command == "summarize-report":
            if args.report and args.report.resolve() == args.input.resolve():
                raise ValueError("Choose a different output path to preserve the original report")
            payload = summarize_report(json.loads(args.input.read_text()))
            for comparison in payload["comparisons"]:
                print(format_summary(comparison), file=sys.stderr)
            _emit(payload, args.report)
            return 0
        if args.command == "compare-labels":
            _emit(compare_files(args.first, args.second, detailed=args.detailed_report), args.report)
            return 0
        options = json.loads(args.config.read_text()) if args.config else {}
        if args.bundle_grouping is not None:
            options["bundle_grouping"] = args.bundle_grouping
        if args.channels_per_bundle is not None:
            options["channels_per_bundle"] = args.channels_per_bundle
        if args.corrected:
            options["matlab_compatible"] = False
        config = Config(**options)
        if args.command == "compare" and args.min_agreement is not None and not 0 <= args.min_agreement <= 1:
            raise ValueError("--min-agreement must be in [0,1]")
        result = run(
            args.directory,
            config=config,
            threshold=args.threshold,
            pattern=args.pattern,
            progress=args.progress,
        )
        payload = {
            "summary": result.summary(),
            "versions": {
                p: importlib.metadata.version(p) for p in ("der-blackrock", "numpy", "scipy", "numba")
            },
        }
        status = 0
        if args.command == "run":
            payload["output_files"] = [
                str(p) for p in save_labels(result.session, result.labels, args.output)
            ]
        else:
            payload["comparisons"] = [
                compare_result(result, ref, detailed=args.detailed_report)
                for ref in (args.reference or [None])
            ]
            payload["report_detail"] = "detailed" if args.detailed_report else "compact"
            if not args.detailed_report:
                payload = summarize_report(payload)
            for comparison in payload["comparisons"]:
                print(format_summary(comparison), file=sys.stderr)
            if args.min_agreement is not None:
                passed = all(
                    c["aggregate"]["exact_label_agreement"] is not None
                    and c["aggregate"]["exact_label_agreement"] >= args.min_agreement
                    for c in payload["comparisons"]
                )
                payload["gate"] = {"min_agreement": args.min_agreement, "passed": passed}
                status = 0 if passed else 1
        if args.plot:
            from .plot import plot_correlations

            args.plot.parent.mkdir(parents=True, exist_ok=True)
            plot_correlations(result, args.plot)
        payload["parameters"] = asdict(config)
        _emit(payload, args.report)
        return status
    except (ValueError, TypeError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"der: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
