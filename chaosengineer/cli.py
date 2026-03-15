"""CLI entry point for ChaosEngineer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chaosengineer.testing.runner import ScenarioRunner


def main():
    parser = argparse.ArgumentParser(
        prog="chaosengineer",
        description="General-purpose parallel experimentation framework",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Test command: run scenarios
    test_parser = subparsers.add_parser("test", help="Run test scenarios")
    test_parser.add_argument(
        "scenario",
        nargs="?",
        help="Path to scenario YAML file. If omitted, runs all shipped scenarios.",
    )
    test_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".chaosengineer/test-output"),
        help="Directory for test output",
    )

    # Run command: execute a workload
    run_parser = subparsers.add_parser("run", help="Run a workload")
    run_parser.add_argument(
        "workload",
        type=Path,
        help="Path to workload spec markdown file",
    )
    run_parser.add_argument(
        "--llm-backend",
        choices=["claude-code", "sdk"],
        default="claude-code",
        help="LLM backend for coordinator decisions (default: claude-code)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".chaosengineer/output"),
        help="Directory for run output",
    )

    # Version
    subparsers.add_parser("version", help="Print version")

    args = parser.parse_args()

    if args.command == "version":
        from chaosengineer import __version__
        print(f"chaosengineer {__version__}")

    elif args.command == "test":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        runner = ScenarioRunner(output_dir=args.output_dir)

        if args.scenario:
            result = runner.run_scenario(Path(args.scenario))
            _print_scenario_result(result)
            sys.exit(0 if result.passed else 1)
        else:
            # Run all shipped scenarios
            scenarios_dir = Path(__file__).parent / "testing" / "scenarios"
            all_passed = True
            for path in sorted(scenarios_dir.glob("*.yaml")):
                result = runner.run_scenario(path)
                _print_scenario_result(result)
                if not result.passed:
                    all_passed = False
            sys.exit(0 if all_passed else 1)

    elif args.command == "run":
        from chaosengineer.workloads.parser import parse_workload_spec
        from chaosengineer.llm import create_decision_maker

        args.output_dir.mkdir(parents=True, exist_ok=True)
        spec = parse_workload_spec(args.workload)

        llm_dir = args.output_dir / "llm_decisions"
        llm_dir.mkdir(parents=True, exist_ok=True)

        dm = create_decision_maker(args.llm_backend, spec, llm_dir)
        print(f"Created {args.llm_backend} decision maker for workload: {spec.name}")
        print("(Full coordinator integration is Sub-project C)")

    else:
        parser.print_help()


def _print_scenario_result(result):
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.scenario_name}")
    print(f"  Best metric: {result.final_best_metric}")
    print(f"  Experiments: {result.total_experiments}")
    if result.errors:
        for error in result.errors:
            print(f"  ERROR: {error}")


if __name__ == "__main__":
    main()
