"""Simple CLI for SageVision (placeholder).

Usage:
    python -m sagevision.cli --input path/to/video
"""
import argparse

from sagevision.pipeline.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(prog="sagevision")
    parser.add_argument("--input", "-i", required=False, help="Path to input video")
    parser.add_argument("--gui", action="store_true", help="Launch the Tkinter GUI")
    args = parser.parse_args()

    if args.gui:
        try:
            # Run Tcl/Tk init in a subprocess first to avoid aborting this process
            # (on macOS a failed Tk init can call abort() in native code).
            import subprocess
            import sys

            check_code = (
                "import tkinter as tk;"
                "root = tk.Tk();"
                "root.destroy();"
                "print('OK')"
            )
            proc = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
            if proc.returncode != 0:
                print("Failed to start Tkinter GUI (tkinter initialization failed).")
                if proc.stdout:
                    print("Stdout:", proc.stdout.strip())
                if proc.stderr:
                    print("Stderr:", proc.stderr.strip())
                print("Suggested actions: ensure Tcl/Tk is installed and your Python is linked to it (see README).")
                return

            from sagevision.gui import launch

            launch()
        except Exception as e:
            print(f"Failed to launch GUI: {e}")
        return

    if not args.input:
        parser.error("Either --input or --gui must be provided")

    pipeline = Pipeline()
    summary = pipeline.run(args.input)
    print(summary)


if __name__ == "__main__":
    main()
