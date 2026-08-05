"""Single argparse entry point, thin wrapper around api.py."""

from __future__ import annotations

import argparse
import json
import sys
from enum import IntEnum
from pathlib import Path

from rope_dev_tools import api
from rope_dev_tools.spec import load_spec
from rope_dev_tools.validation.schema_types import report_all_passed


class ExitCode(IntEnum):
    OK = 0
    SPEC_OR_MANIFEST_INVALID = 1
    EXPORT_FAILED = 2
    CHECK_FAILED = 3


def _print_progress(index: int, total: int, check_id: str) -> None:
    print(f"[{index + 1}/{total}] running check {check_id!r}", flush=True)


def _cmd_export(args) -> int:
    try:
        spec = load_spec(args.spec)
    except Exception as e:
        print(f"could not load spec {args.spec!r}: {e}", file=sys.stderr)
        return ExitCode.SPEC_OR_MANIFEST_INVALID

    try:
        result = api.export_model(
            spec, args.out_dir,
            suite=args.suite, wrapper=args.wrapper, package_root=args.package_root,
            build_dir=args.build_dir, driver_path=args.driver_path,
            skip_validation=args.skip_validation,
            skip_conversion_check=args.skip_conversion_check,
            force=args.force, progress=None if args.quiet else _print_progress,
        )
    except Exception as e:
        print(f"export failed: {e}", file=sys.stderr)
        return ExitCode.EXPORT_FAILED

    print(f"wrote manifest: {result.manifest_path}")
    if result.passed is not None:
        print(f"wrote validation report: {result.report_path}")
        if not result.passed:
            print("one or more checks failed; see the report for details", file=sys.stderr)
            return ExitCode.CHECK_FAILED
    return ExitCode.OK


def _cmd_verify(args) -> int:
    try:
        if args.check_only:
            from rope_dev_tools.validation.runner import recheck_report
            from rope_dev_tools.validation.schema_types import ValidationSuite

            suite = ValidationSuite.from_json(args.suite)
            report = json.loads(Path(args.check_only).read_text())
            rechecked = recheck_report(report, suite)
            Path(args.check_only).write_text(json.dumps(rechecked, indent=2) + "\n")
            print(f"rechecked report written to {args.check_only}")
            return ExitCode.OK if report_all_passed(rechecked) else ExitCode.CHECK_FAILED

        grid = json.loads(Path(args.grid).read_text()) if args.grid else None
        result = api.verify_model(
            args.exported_dir, args.suite,
            wrapper=args.wrapper, grid=grid, package_root=args.package_root, build_dir=args.build_dir,
            driver_path=args.driver_path, only_check_ids=args.only_check_ids,
            progress=None if args.quiet else _print_progress,
        )
    except Exception as e:
        print(f"verify failed: {e}", file=sys.stderr)
        return ExitCode.CHECK_FAILED

    print(f"wrote validation report: {result.report_path}")
    if not result.passed:
        print("one or more checks failed; see the report for details", file=sys.stderr)
        return ExitCode.CHECK_FAILED
    return ExitCode.OK


def _cmd_mark_validated(args) -> int:
    try:
        manifest = api.mark_validated(args.exported_dir, report_path=args.report)
    except Exception as e:
        print(f"mark-validated failed: {e}", file=sys.stderr)
        return ExitCode.SPEC_OR_MANIFEST_INVALID
    print(f"manifest validated={manifest['validated']}")
    return ExitCode.OK


def _cmd_manifest_upgrade(args) -> int:
    try:
        api.upgrade_manifest(args.manifest)
    except Exception as e:
        print(f"manifest-upgrade failed: {e}", file=sys.stderr)
        return ExitCode.SPEC_OR_MANIFEST_INVALID
    print(f"upgraded {args.manifest}")
    return ExitCode.OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rope-dev-tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="Export a model spec to an artifact directory + manifest")
    p_export.add_argument("--spec", required=True, help="path/to/spec.json or module_or_path:ATTR")
    p_export.add_argument("--out-dir", required=True)
    p_export.add_argument("--suite", default=None, help="Validation suite JSON to run by default")
    p_export.add_argument("--wrapper", default=None,
                           help="module_or_path:function for wrapper-mode verification "
                                "(default: exported-dir mode)")
    p_export.add_argument("--package-root", default=None)
    p_export.add_argument("--build-dir", default=None,
                           help="directory containing the built rope binary/library (flat, or with "
                                "bin/+lib/ subdirs); overrides package-root's layout guessing")
    p_export.add_argument("--driver-path", default=None,
                           help="Local space-weather driver CSV/.swbin for exported-dir mode "
                                "(the online driver-cache fetch path isn't implemented yet)")
    p_export.add_argument("--skip-validation", action="store_true")
    p_export.add_argument("--skip-conversion-check", action="store_true")
    p_export.add_argument("--force", action="store_true", help="Overwrite a non-empty --out-dir")
    p_export.add_argument("--quiet", action="store_true",
                           help="suppress per-check validation progress lines")
    p_export.set_defaults(func=_cmd_export)

    p_verify = sub.add_parser("verify", help="Run a validation suite against an exported directory")
    p_verify.add_argument("--exported-dir", required=True)
    p_verify.add_argument("--suite", required=True)
    p_verify.add_argument("--wrapper", default=None)
    p_verify.add_argument("--grid", default=None,
                           help="Path to a GridSpec JSON file. Required with --wrapper; "
                                "exported-dir mode reads it from model_manifest.json instead")
    p_verify.add_argument("--package-root", default=None)
    p_verify.add_argument("--build-dir", default=None,
                           help="directory containing the built rope binary/library (flat, or with "
                                "bin/+lib/ subdirs); overrides package-root's layout guessing")
    p_verify.add_argument("--driver-path", default=None,
                           help="Local space-weather driver CSV/.swbin "
                                "(the online driver-cache fetch path isn't implemented yet)")
    p_verify.add_argument("--check-only", default=None,
                           help="Re-evaluate an existing report's pass/fail without re-running inference")
    p_verify.add_argument("--only-check", action="append", default=None, dest="only_check_ids",
                           help="restrict to this check id (repeatable); default: every check in the suite")
    p_verify.add_argument("--quiet", action="store_true",
                           help="suppress per-check progress lines")
    p_verify.set_defaults(func=_cmd_verify)

    p_mark = sub.add_parser("mark-validated",
                             help="Flip validated=true using an existing report, without re-exporting")
    p_mark.add_argument("--exported-dir", required=True)
    p_mark.add_argument("--report", default=None, help="Defaults to <exported-dir>/validation_report.json")
    p_mark.set_defaults(func=_cmd_mark_validated)

    p_upgrade = sub.add_parser("manifest-upgrade",
                                help="Migrate a legacy-shape manifest to registry shape in place")
    p_upgrade.add_argument("manifest")
    p_upgrade.set_defaults(func=_cmd_manifest_upgrade)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
