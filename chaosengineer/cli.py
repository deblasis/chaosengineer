"""CLI entry point for ChaosEngineer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from chaosengineer.testing.runner import ScenarioRunner


def _build_parser() -> argparse.ArgumentParser:
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
    run_parser.add_argument("workload", type=Path, help="Path to workload spec markdown file")
    run_parser.add_argument("--llm-backend", choices=["claude-code", "sdk", "scripted"], default="claude-code",
                            help="LLM backend for coordinator decisions (default: claude-code)")
    run_parser.add_argument("--executor", choices=["subagent", "scripted"], default="subagent",
                            help="Executor backend (default: subagent)")
    run_parser.add_argument("--mode", choices=["sequential", "parallel"], default="sequential",
                            help="Execution mode (default: sequential)")
    run_parser.add_argument("--scripted-results", type=Path,
                            help="YAML file or folder with canned results (required for --executor=scripted)")
    run_parser.add_argument("--scripted-plans", type=Path,
                            help="YAML file with scripted dimension plans (required for --llm-backend=scripted)")
    run_parser.add_argument("--output-dir", type=Path, default=Path(".chaosengineer/output"),
                            help="Directory for run output")
    run_parser.add_argument("--initial-baseline", type=float, default=None,
                            help="Override initial baseline metric value")
    run_parser.add_argument("--force-fresh", action="store_true",
                            help="Skip run guard prompt, start fresh even if resumable session exists")
    run_parser.add_argument("--tui", action="store_true", default=False,
                            help="Enable TUI dashboard (toggle with 't' during run)")

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume a partially-completed run")
    resume_parser.add_argument("output_dir", type=str, help="Path to output directory with events.jsonl")
    resume_parser.add_argument("workload", type=Path, help="Path to workload spec markdown file")
    resume_parser.add_argument("--llm-backend", choices=["claude-code", "sdk", "scripted"], default="claude-code",
                               help="LLM backend for coordinator decisions (default: claude-code)")
    resume_parser.add_argument("--executor", choices=["subagent", "scripted"], default="subagent",
                               help="Executor backend (default: subagent)")
    resume_parser.add_argument("--mode", choices=["sequential", "parallel"], default="sequential",
                               help="Execution mode (default: sequential)")
    resume_parser.add_argument("--scripted-results", type=Path,
                               help="YAML file or folder with canned results (required for --executor=scripted)")
    resume_parser.add_argument("--scripted-plans", type=Path,
                               help="YAML file with scripted dimension plans (required for --llm-backend=scripted)")
    resume_parser.add_argument("--add-cost", type=float, default=0, help="Add USD to cost budget")
    resume_parser.add_argument("--add-experiments", type=int, default=0, help="Add to experiment budget")
    resume_parser.add_argument("--add-time", type=float, default=0, help="Add seconds to time budget")
    resume_parser.add_argument("--restart-iteration", action="store_true",
                               help="Discard partial iteration, restart from scratch")
    resume_parser.add_argument("--tui", action="store_true", default=False,
                               help="Enable TUI dashboard (toggle with 't' during run)")

    # Monitor command: attach to a running bus
    monitor_parser = subparsers.add_parser("monitor", help="Monitor a running experiment via bus")
    monitor_parser.add_argument("bus_url", help="URL of the chaos-bus (e.g., http://127.0.0.1:50051)")
    monitor_parser.add_argument("--run-id", default="", help="Specific run to monitor")

    # Version
    subparsers.add_parser("version", help="Print version")

    return parser


def main():
    parser = _build_parser()
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

    elif args.command == "monitor":
        _execute_monitor(args)

    elif args.command == "run":
        _execute_run(args)

    elif args.command == "resume":
        _execute_resume(args)

    else:
        parser.print_help()


def detect_baseline(spec: "WorkloadSpec") -> float:
    """Run the workload once to measure the initial baseline metric."""
    import subprocess

    print("No baseline specified. Running workload to measure initial baseline...")

    result = subprocess.run(
        spec.execution_command, shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error: baseline execution failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        spec.metric_parse_command, shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error: baseline metric parse failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    try:
        value = float(result.stdout.strip())
    except ValueError:
        print(
            f"Error: could not parse baseline metric from output: {result.stdout.strip()!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Initial baseline: {spec.primary_metric} = {value}")
    return value


def _check_resumable_session(output_dir: Path) -> dict | None:
    """Check if output_dir has a resumable (non-completed) event log."""
    import json
    events_path = output_dir / "events.jsonl"
    if not events_path.exists():
        return None

    run_info = None
    is_completed = False
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("event") == "run_started":
                run_info = entry
            elif entry.get("event") == "run_completed":
                is_completed = True

    if run_info and not is_completed:
        return run_info
    return None


def _find_bus_binary() -> Path | None:
    """Locate the chaos-bus binary: env var > repo path > system PATH."""
    import shutil
    env_path = os.environ.get("CHAOS_BUS_BIN")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    # Development: relative to repo root (cli.py is in chaosengineer/)
    repo_root = Path(__file__).parent.parent
    dev_path = repo_root / "bus" / "chaos-bus"
    if dev_path.is_file():
        return dev_path
    found = shutil.which("chaos-bus")
    if found:
        return Path(found)
    return None


def _start_bus(output_file: Path) -> tuple[subprocess.Popen | None, str | None]:
    """Start the bus binary and return (process, bus_url) or (None, None)."""
    binary = _find_bus_binary()
    if binary is None:
        return None, None

    proc = subprocess.Popen(
        [str(binary), "--port", "0", "--output-file", str(output_file),
         "--shutdown-delay", "5s"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        port_line = proc.stdout.readline()
        if not port_line:
            proc.kill()
            return None, None
        port_data = json.loads(port_line)
        bus_url = f"http://127.0.0.1:{port_data['port']}"

        # Poll healthz until ready (up to 5 seconds)
        import urllib.request
        import time
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{bus_url}/healthz", timeout=1)
                return proc, bus_url
            except Exception:
                time.sleep(0.1)

        proc.kill()
        return None, None
    except Exception:
        proc.kill()
        return None, None


def _execute_run(args):
    """Execute a workload run with the specified backends."""
    # Run guard: check for resumable session
    if not getattr(args, "force_fresh", False):
        run_info = _check_resumable_session(args.output_dir)
        if run_info is not None:
            from chaosengineer.cli_menu import select
            from chaosengineer.core.snapshot import build_snapshot
            snap = build_snapshot(args.output_dir / "events.jsonl")
            dims = len(snap.dimensions_explored)
            best = snap.active_baselines[0].metric_value if snap.active_baselines else "?"

            choice = select(
                f"Found existing run ({dims} dimensions explored, best: {best})",
                [
                    "Resume previous run",
                    "Start fresh (archive existing)",
                    "Cancel",
                ],
            )
            if choice == 0:  # Resume
                print(f"\n  chaosengineer resume {args.output_dir}\n")
                sys.exit(0)
            elif choice == 1:  # Start fresh
                import shutil
                from datetime import datetime
                bak = args.output_dir.parent / f"{args.output_dir.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                shutil.move(str(args.output_dir), str(bak))
                print(f"Archived existing run to {bak}")
            else:  # Cancel
                sys.exit(0)

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
    from chaosengineer.metrics.publisher import EventPublisher
    log_path = args.output_dir / "events.jsonl"
    bus_proc, bus_url = _start_bus(log_path)
    logger = EventPublisher(bus_url=bus_url, fallback_path=log_path)
    if getattr(args, "tui", False):
        from chaosengineer.tui.bridge import EventBridge
        from chaosengineer.tui.pause_gate import PauseGate
        bridge = EventBridge()
        pause_gate = PauseGate()
        logger = EventPublisher(bus_url=bus_url, fallback_path=log_path, bridge=bridge)
    budget = BudgetTracker(spec.budget)

    # Resolve initial baseline: CLI flag > workload spec > auto-detect
    if args.initial_baseline is not None:
        metric_value = args.initial_baseline
    elif spec.baseline_metric_value is not None:
        metric_value = spec.baseline_metric_value
    elif args.executor == "scripted":
        print(
            "Error: --executor=scripted requires a baseline. "
            "Use --initial-baseline or add ## Baseline to the workload spec.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        metric_value = detect_baseline(spec)

    initial_baseline = Baseline(
        commit="HEAD",
        metric_value=metric_value,
        metric_name=spec.primary_metric,
    )

    from chaosengineer.core.pause import PauseController
    from chaosengineer.core.status import StatusDisplay

    pause_controller = PauseController()
    status_display = StatusDisplay()
    pause_controller.set_executor(executor)

    coordinator = Coordinator(
        spec=spec,
        decision_maker=dm,
        executor=executor,
        logger=logger,
        budget=budget,
        initial_baseline=initial_baseline,
        run_id=run_id,
        pause_controller=pause_controller,
        status_display=status_display,
    )

    print(f"Starting run: {spec.name}")
    print(f"  LLM backend: {args.llm_backend}")
    print(f"  Executor: {args.executor} ({args.mode})")
    print(f"  Output: {args.output_dir}")

    pause_controller.install()
    try:
        if getattr(args, "tui", False):
            import threading
            from chaosengineer.tui.views import ViewManager
            view_manager = ViewManager(bridge, pause_gate, pause_controller,
                                        coordinator, status_display)
            coordinator._view_manager = view_manager
            coordinator._pause_gate = pause_gate

            coord_done = threading.Event()

            def run_coordinator():
                try:
                    coordinator.run()
                finally:
                    coord_done.set()

            coord_thread = threading.Thread(target=run_coordinator, daemon=True)
            coord_thread.start()
            view_manager.run(coord_done)
            coord_thread.join()
        else:
            coordinator.run()
    finally:
        pause_controller.uninstall()
        if bus_proc:
            bus_proc.terminate()

    print(f"\nRun complete:")
    print(f"  Best metric: {coordinator.best_baseline.metric_value}")
    print(f"  Experiments: {coordinator.budget.experiments_run}")
    print(f"  Cost: ${coordinator.budget.spent_usd:.2f}")


def _execute_resume(args):
    """Execute the resume subcommand."""
    import json
    from chaosengineer.core.snapshot import build_snapshot, StopReason
    from chaosengineer.core.budget import BudgetTracker
    from chaosengineer.core.models import Baseline, BudgetConfig
    from chaosengineer.core.coordinator import Coordinator
    from chaosengineer.metrics.logger import EventLogger

    output_dir = Path(args.output_dir)
    events_path = output_dir / "events.jsonl"

    if not events_path.exists():
        print(f"Error: No events.jsonl found in {output_dir}")
        sys.exit(1)

    snapshot = build_snapshot(events_path)

    if snapshot.stop_reason == StopReason.COMPLETED:
        print("Run already completed. Nothing to resume.")
        sys.exit(0)

    # Apply budget extensions
    bc = snapshot.budget_config
    if args.add_cost > 0:
        bc = BudgetConfig(
            max_api_cost=(bc.max_api_cost or 0) + args.add_cost,
            max_experiments=bc.max_experiments,
            max_wall_time_seconds=bc.max_wall_time_seconds,
            max_plateau_iterations=bc.max_plateau_iterations,
        )
    if args.add_experiments > 0:
        bc = BudgetConfig(
            max_api_cost=bc.max_api_cost,
            max_experiments=(bc.max_experiments or 0) + args.add_experiments,
            max_wall_time_seconds=bc.max_wall_time_seconds,
            max_plateau_iterations=bc.max_plateau_iterations,
        )
    if args.add_time > 0:
        bc = BudgetConfig(
            max_api_cost=bc.max_api_cost,
            max_experiments=bc.max_experiments,
            max_wall_time_seconds=(bc.max_wall_time_seconds or 0) + args.add_time,
            max_plateau_iterations=bc.max_plateau_iterations,
        )
    snapshot.budget_config = bc

    # Check if budget is still exhausted
    budget_tracker = BudgetTracker.from_snapshot(
        config=bc,
        experiments_run=snapshot.total_experiments_run,
        cost_spent=snapshot.total_cost_usd,
        elapsed_offset=snapshot.elapsed_time,
        consecutive_no_improvement=snapshot.consecutive_no_improvement,
    )
    budget_tracker.start()
    if budget_tracker.is_exhausted():
        print("Error: Budget is still exhausted after extensions.")
        print("Use --add-cost, --add-experiments, or --add-time to extend.")
        sys.exit(1)

    # Print resume summary
    dims = len(snapshot.dimensions_explored)
    best = snapshot.active_baselines[0].metric_value if snapshot.active_baselines else "?"
    print(f"Resuming run {snapshot.run_id} — {dims} dimensions explored, best: {best}, ${snapshot.total_cost_usd:.2f} spent")

    # Crash warning
    if snapshot.stop_reason == StopReason.CRASHED:
        print("\nWarning: This run appears to have crashed (no clean stop event).")
        print("Review the event log before continuing.")
        try:
            resp = input("Continue? [y/N] ").strip().lower()
        except EOFError:
            resp = ""
        if resp != "y":
            sys.exit(0)

    # Wire up backends (same pattern as _execute_run)
    from chaosengineer.workloads.parser import parse_workload_spec
    from chaosengineer.llm import create_decision_maker
    from chaosengineer.execution import create_executor

    spec = parse_workload_spec(args.workload)

    # Validate workload spec hash
    if snapshot.workload_spec_hash and snapshot.workload_spec_hash != spec.spec_hash():
        print(f"Warning: Workload spec has changed since original run.")
        print(f"  Original: {snapshot.workload_spec_hash}")
        print(f"  Current:  {spec.spec_hash()}")
        print(f"  Results may not be comparable.")

    if args.llm_backend == "scripted":
        if args.scripted_plans is None:
            print("Error: --scripted-plans is required when using --llm-backend=scripted", file=sys.stderr)
            sys.exit(1)
        from chaosengineer.workloads.plan_loader import load_scripted_plans
        from chaosengineer.testing.simulator import ScriptedDecisionMaker
        plans = load_scripted_plans(args.scripted_plans)
        dm = ScriptedDecisionMaker(plans)
    else:
        llm_dir = output_dir / "llm_decisions"
        llm_dir.mkdir(parents=True, exist_ok=True)
        dm = create_decision_maker(args.llm_backend, spec, llm_dir)

    if args.executor == "scripted" and args.scripted_results is None:
        print("Error: --scripted-results is required when using --executor=scripted", file=sys.stderr)
        sys.exit(1)

    executor = create_executor(
        args.executor, spec, output_dir, args.mode,
        scripted_results=getattr(args, "scripted_results", None),
        run_id=snapshot.run_id,
    )
    from chaosengineer.metrics.publisher import EventPublisher
    bus_proc, bus_url = _start_bus(events_path)
    logger = EventPublisher(bus_url=bus_url, fallback_path=events_path)
    if getattr(args, "tui", False):
        from chaosengineer.tui.bridge import EventBridge
        from chaosengineer.tui.pause_gate import PauseGate
        bridge = EventBridge()
        pause_gate = PauseGate()
        logger = EventPublisher(bus_url=bus_url, fallback_path=events_path, bridge=bridge)

    from chaosengineer.core.pause import PauseController
    from chaosengineer.core.status import StatusDisplay

    pause_controller = PauseController()
    status_display = StatusDisplay()
    pause_controller.set_executor(executor)

    coordinator = Coordinator(
        spec=spec,
        decision_maker=dm,
        executor=executor,
        logger=logger,
        budget=budget_tracker,
        initial_baseline=snapshot.active_baselines[0] if snapshot.active_baselines else Baseline("HEAD", 0.0, spec.primary_metric),
        run_id=snapshot.run_id,
        pause_controller=pause_controller,
        status_display=status_display,
    )
    extensions = {}
    if args.add_cost > 0:
        extensions["add_cost"] = args.add_cost
    if args.add_experiments > 0:
        extensions["add_experiments"] = args.add_experiments
    if args.add_time > 0:
        extensions["add_time"] = args.add_time
    pause_controller.install()
    try:
        if getattr(args, "tui", False):
            import threading
            from chaosengineer.tui.views import ViewManager
            view_manager = ViewManager(bridge, pause_gate, pause_controller,
                                        coordinator, status_display)
            coordinator._view_manager = view_manager
            coordinator._pause_gate = pause_gate

            coord_done = threading.Event()

            def run_coordinator():
                try:
                    coordinator.resume_from_snapshot(
                        snapshot, restart_iteration=args.restart_iteration,
                        budget_extensions=extensions or None,
                    )
                finally:
                    coord_done.set()

            coord_thread = threading.Thread(target=run_coordinator, daemon=True)
            coord_thread.start()
            view_manager.run(coord_done)
            coord_thread.join()
        else:
            coordinator.resume_from_snapshot(
                snapshot, restart_iteration=args.restart_iteration,
                budget_extensions=extensions or None,
            )
    finally:
        pause_controller.uninstall()
        if bus_proc:
            bus_proc.terminate()

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


def _execute_monitor(args):
    """Attach to a running chaos-bus and display a read-only TUI dashboard."""
    from unittest.mock import MagicMock

    from chaosengineer.tui.monitor import MonitorClient
    from chaosengineer.tui.app import ChaosApp
    from chaosengineer.tui.pause_gate import PauseGate

    client = MonitorClient(bus_url=args.bus_url, run_id=args.run_id)
    client.start()

    # Create stub objects -- the TUI is read-only so these are never used.
    pause_gate = PauseGate()
    stub_coordinator = MagicMock()
    stub_pause_controller = MagicMock()
    stub_pause_controller.pause_requested = False

    app = ChaosApp(
        bridge=client.bridge,
        pause_gate=pause_gate,
        coordinator=stub_coordinator,
        pause_controller=stub_pause_controller,
        readonly=True,
    )
    try:
        app.run()
    finally:
        client.stop()


if __name__ == "__main__":
    main()
