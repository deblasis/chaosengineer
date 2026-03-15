"""Constructs task markdown files for Claude Code subagents."""

from __future__ import annotations

from pathlib import Path

from chaosengineer.core.interfaces import ExperimentTask
from chaosengineer.workloads.parser import WorkloadSpec


class TaskPacketBuilder:
    """Generates task markdown files that instruct subagents."""

    def build(
        self,
        task: ExperimentTask,
        spec: WorkloadSpec,
        worktree_path: Path,
        result_file: Path,
        run_id: str,
        output_dir: Path,
    ) -> Path:
        """Build a task markdown file and write it to the output directory.

        Returns the path to the written task file.
        """
        params_block = "\n".join(
            f"  {k}: {v}" for k, v in task.params.items()
        )

        if spec.modifiable_files:
            files_block = "\n".join(f"- {f}" for f in spec.modifiable_files)
        else:
            files_block = "Any files"

        constraints_block = spec.constraints_text if spec.constraints_text else "None"

        if spec.metric_parse_command:
            metric_step = f'5. Parse metrics by running: `{spec.metric_parse_command}`'
        else:
            metric_step = f"5. Extract the primary metric '{spec.primary_metric}' from the command output"

        if spec.time_budget_seconds is not None:
            time_block = f"Complete within {spec.time_budget_seconds} seconds"
        else:
            time_block = "No time limit — run to completion"

        branch_name = f"chaosengineer/{run_id}/{task.experiment_id}"
        param_summary = ", ".join(f"{k}={v}" for k, v in task.params.items())

        content = f"""# Experiment: {task.experiment_id}

## Objective
Apply the following parameter changes and run the experiment command.
Report the results as a JSON file.

## Parameters
{params_block}

## Working Directory
You are working in a git worktree at: {worktree_path}
Branch: {branch_name}

## Modifiable Files
{files_block}

## Constraints
{constraints_block}

## Instructions
1. Study the codebase to understand how the parameters above map to code
2. Modify the relevant files to apply these parameter values
3. Commit your changes with message: "experiment {task.experiment_id}: {param_summary}"
4. Run the experiment command: `{task.command}`
{metric_step}
6. Write results to: {result_file}

## Time Budget
{time_block}

## Result Format
Write ONLY this JSON to {result_file}:

```json
{{
  "primary_metric": <float>,
  "secondary_metrics": {{"metric_name": <float>, ...}},
  "artifacts": ["path/to/artifact", ...],
  "commit_hash": "<git commit hash of your changes>",
  "error_message": null
}}
```

If the experiment fails, set error_message to a description of what went wrong.
Primary metric name: {spec.primary_metric} ({spec.metric_direction} is better)
"""
        task_file = output_dir / task.experiment_id / "task.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(content)
        return task_file
