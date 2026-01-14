"""Module entrypoint for running the SageVision GUI with ``python -m sagevision.gui``.

This performs a safe Tkinter initialization check (helpful on macOS) before
launching the GUI and prints a short diagnostic if Tk cannot be initialized.
"""
from __future__ import annotations

import subprocess
import sys

from .gui import launch


def _check_tkinter() -> bool:
    check_code = (
        "import tkinter as tk;"
        "root = tk.Tk();"
        "root.destroy();"
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
    return proc.returncode == 0


if __name__ == "__main__":
    if not _check_tkinter():
        print("Failed to initialize Tkinter. On macOS, consider using a conda environment and installing Tcl/Tk via conda-forge (see README).")
        sys.exit(1)
    launch()
