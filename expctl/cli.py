from __future__ import annotations

import argparse
import sys
from pathlib import Path

from expctl.scaffold import create_project_scaffold
from expctl.workflows import (
    load_runtime_context,
    run_build_splits,
    run_doctor,
    run_evaluate,
    run_make_submission,
    run_register,
    run_train,
    run_validate_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expctl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new expctl project scaffold")
    init_parser.add_argument("path", nargs="?", default=".")
    init_parser.add_argument("--force", action="store_true")

    for command_name, help_text in (
        ("validate-config", "Validate an experiment config"),
        ("build-splits", "Build validation split artifacts"),
        ("train", "Run cross-validation training"),
        ("evaluate", "Compute or refresh summary metrics"),
        ("register", "Register an experiment result"),
        ("doctor", "Check adapter/config/runtime readiness"),
    ):
        command_parser = subparsers.add_parser(command_name, help=help_text)
        command_parser.add_argument("--config", required=True)
        command_parser.add_argument("--repo-root", default=".")

    evaluate_parser = subparsers.choices["evaluate"]
    evaluate_parser.add_argument("--predictions")

    submission_parser = subparsers.add_parser(
        "make-submission",
        help="Validate and write a submission CSV from predictions",
    )
    submission_parser.add_argument("--config", required=True)
    submission_parser.add_argument("--repo-root", default=".")
    submission_parser.add_argument("--predictions", required=True)
    submission_parser.add_argument("--output")
    submission_parser.add_argument("--test-data")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            created = create_project_scaffold(Path(args.path), force=bool(args.force))
            for path in created:
                print(path)
            return 0

        repo_root = Path(args.repo_root).resolve()
        config_path = Path(args.config).resolve()

        if args.command == "doctor":
            result = run_doctor(config_path, repo_root=repo_root)
            for warning in result.warnings:
                print(f"WARNING: {warning}", file=sys.stderr)
            for issue in result.issues:
                print(f"ERROR: {issue}", file=sys.stderr)
            if result.is_healthy:
                print("Doctor checks passed.")
                return 0
            return 1

        context = load_runtime_context(config_path, repo_root=repo_root)
        for warning in context.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)

        if args.command == "validate-config":
            result = run_validate_config(context)
        elif args.command == "build-splits":
            result = run_build_splits(context)
        elif args.command == "train":
            result = run_train(context)
        elif args.command == "evaluate":
            predictions_path = Path(args.predictions).resolve() if args.predictions else None
            result = run_evaluate(context, predictions_path=predictions_path)
        elif args.command == "register":
            result = run_register(context)
        elif args.command == "make-submission":
            result = run_make_submission(
                context,
                predictions_path=Path(args.predictions).resolve(),
                output_path=Path(args.output).resolve() if args.output else None,
                test_path=Path(args.test_data).resolve() if args.test_data else None,
            )
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2

        for key, value in result.items():
            print(f"{key}: {value}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
