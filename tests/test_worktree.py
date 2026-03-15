"""Tests for WorktreeManager."""

import subprocess
from pathlib import Path
from unittest.mock import patch, call

import pytest

from chaosengineer.execution.worktree import WorktreeManager


class TestWorktreeCreate:
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_creates_worktree_with_branch(self, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        mgr = WorktreeManager(repo_root=repo_root)

        path = mgr.create("abc123", "run-abcd1234", "exp-0-0")

        assert path == repo_root / ".chaosengineer" / "worktrees" / "exp-0-0"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "worktree" in cmd
        assert "add" in cmd
        assert "-b" in cmd
        assert "chaosengineer/run-abcd1234/exp-0-0" in cmd
        assert "abc123" in cmd
        # git commands run from repo root
        assert mock_run.call_args[1].get("cwd") == str(repo_root)

    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_create_raises_on_git_failure(self, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="fatal: error"
        )
        mgr = WorktreeManager(repo_root=tmp_path)

        with pytest.raises(RuntimeError, match="Failed to create worktree"):
            mgr.create("abc123", "run-abcd1234", "exp-0-0")


class TestWorktreeCleanup:
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_removes_worktree(self, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mgr = WorktreeManager(repo_root=tmp_path)
        worktree_path = tmp_path / ".chaosengineer" / "worktrees" / "exp-0-0"

        mgr.cleanup(worktree_path)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "worktree" in cmd
        assert "remove" in cmd
        assert str(worktree_path) in cmd

    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_cleanup_ignores_missing_worktree(self, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="not a working tree"
        )
        mgr = WorktreeManager(repo_root=tmp_path)
        worktree_path = tmp_path / ".chaosengineer" / "worktrees" / "exp-0-0"

        # Should not raise
        mgr.cleanup(worktree_path)
