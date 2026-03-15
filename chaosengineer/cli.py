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
        choices=["claude-code", "sdk", "scripted"],
        default="claude-code",
        help="LLM backend for coordinator decisions (default: claude-code)",
    )
    run_parser.add_argument(
        "--executor",
        choices=["subagent", "scripted"],
        default="subagent",
        help="Executor backend (default: subagent)",
    )
    run_parser.add_argument(
        "--mode",
        choices=["sequential", "parallel"],
        default="sequential",
        help="Execution mode (default: sequential)",
    )
    run_parser.add_argument(
        "--scripted-results",
        type=Path,
        help="YAML file or folder with canned results (required for --executor=scripted)",
    )
    run_parser.add_argument(
        "--scripted-plans",
        type=Path,
        help="YAML file with scripted dimension plans (required for --llm-backend=scripted)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".chaosengineer/output"),
        help="Directory for run output",
    )
    run_parser.add_argument(
        "--initial-baseline",
        type=float,
        default=None,
        help="Override initial baseline metric value",
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
        _execute_run(args)

    else:
        parser.print_help()


def _execute_run(args):
    """Execute a workload run with the specified backends."""
    import uuid
    from chaosengineer.workloads.parser import parse_workload_spec
    from chaosengineer.llm import create_decision_maker
    from chaosengineer.execution import create_executor
    from chaosengineer.core.coordinator import Coordinator
    from chaosengineer.core.budget import BudgetTracker
    from chaosengineer.metrics.logger import EventLogger
    from chaosengineer.core.models import Baseline

    if args.executor == "scripted" and args.scripted_results is None:
        print("Error: --scripted-results is required when using --executor=scripted", file=sys.stderr)
        sys.exit(1)

    if args.llm_backend == "scripted" and args.scripted_plans is None:
        print("Error: --scripted-plans is required when using --llm-backend=scripted", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = parse_workload_spec(args.workload)

    # Generate a single run_id for both coordinator and executor
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    if args.llm_backend == "scripted":
        from chaosengineer.workloads.plan_loader import load_scripted_plans
        from chaosengineer.testing.simulator import ScriptedDecisionMaker
        plans = load_scripted_plans(args.scripted_plans)
        dm = ScriptedDecisionMaker(plans)
    else:
        llm_dir = args.output_dir / "llm_decisions"
        llm_dir.mkdir(parents=True, exist_ok=True)
        dm = create_decision_maker(args.llm_backend, spec, llm_dir)
    executor = create_executor(
        args.executor, spec, args.output_dir, args.mode,
        scripted_results=args.scripted_results,
        run_id=run_id,
    )
    logger = EventLogger(args.output_dir / "events.jsonl")
    budget = BudgetTracker(spec.budget)

    # TODO: initial baseline should come from workload spec or be auto-detected
    initial_baseline = Baseline(
        commit="HEAD",
        metric_value=float("inf") if spec.metric_direction == "lower" else float("-inf"),
        metric_name=spec.primary_metric,
    )

    coordinator = Coordinator(
        spec=spec,
        decision_maker=dm,
        executor=executor,
        logger=logger,
        budget=budget,
        initial_baseline=initial_baseline,
        run_id=run_id,
    )

    print(f"Starting run: {spec.name}")
    print(f"  LLM backend: {args.llm_backend}")
    print(f"  Executor: {args.executor} ({args.mode})")
    print(f"  Output: {args.output_dir}")

    coordinator.run()

    print(f"\nRun complete:")
    print(f"  Best metric: {coordinator.best_baseline.metric_value}")
    print(f"  Experiments: {coordinator.budget.experiments_run}")
    print(f"  Cost: ${coordinator.budget.spent_usd:.2f}")


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
