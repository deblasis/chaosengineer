"""Git worktree lifecycle management for experiment isolation."""

from __future__ import annotations

import subprocess
from pathlib import Path


class WorktreeManager:
    """Creates and cleans up git worktrees for experiment isolation."""

    def __init__(self, repo_root: Path):
        self._repo_root = repo_root
        self._worktrees_dir = repo_root / ".chaosengineer" / "worktrees"

    def create(
        self, baseline_commit: str, run_id: str, experiment_id: str
    ) -> Path:
        """Create a worktree with a named branch.

        Returns the worktree path.
        """
        worktree_path = self._worktrees_dir / experiment_id
        branch_name = f"chaosengineer/{run_id}/{experiment_id}"

        self._worktrees_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                "git", "worktree", "add",
                str(worktree_path),
                "-b", branch_name,
                baseline_commit,
            ],
            capture_output=True,
            text=True,
            cwd=str(self._repo_root),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create worktree for {experiment_id}: {result.stderr}"
            )

        return worktree_path

    def cleanup(self, worktree_path: Path) -> None:
        """Remove a worktree. The branch persists."""
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_path), "--force"],
            capture_output=True,
            text=True,
            cwd=str(self._repo_root),
        )
