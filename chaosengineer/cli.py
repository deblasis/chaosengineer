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
