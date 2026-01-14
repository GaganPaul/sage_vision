import subprocess
import sys
import pytest

from sagevision import cli


def test_cli_detects_tkinter_failure(monkeypatch, capsys):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "tk init failed"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())

    old_argv = sys.argv
    sys.argv = ["sagevision", "--gui"]
    try:
        cli.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert "Failed to start Tkinter GUI" in captured.out
    assert "tk init failed" in captured.out
