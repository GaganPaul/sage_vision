"""Simple Tkinter GUI to run SageVision pipeline.

This provides a lightweight desktop UI to select a video file and run the
`Pipeline.run` method. Long-running work is executed in a background thread
and UI updates are properly marshaled to the main thread.

Improvements made:
- Reuse a single Pipeline instance to reduce startup overhead
- Input validation and helpful error messages
- Optional Stop button (uses Pipeline.stop() if implemented)
- Clear output button
- Robust handling of progress values in [0,1] or [0,100]
"""
from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
from typing import Optional

from sagevision.pipeline import Pipeline

logger = logging.getLogger(__name__)


class SageVisionApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("SageVision — GUI")
        master.geometry("700x420")

        self.video_path_var = tk.StringVar()

        tk.Label(master, text="SageVision", font=("Helvetica", 16, "bold")).pack(pady=8)

        frm = tk.Frame(master)
        frm.pack(fill=tk.X, padx=12)

        tk.Entry(frm, textvariable=self.video_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(frm, text="Browse…", command=self.browse).pack(side=tk.LEFT, padx=6)

        btn_frm = tk.Frame(master)
        btn_frm.pack(fill=tk.X, padx=12, pady=8)

        self.run_btn = tk.Button(btn_frm, text="Run Pipeline", command=self.run_pipeline)
        self.run_btn.pack(side=tk.LEFT)

        self.stop_btn = tk.Button(btn_frm, text="Stop", command=self.stop_pipeline, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)

        self.clear_btn = tk.Button(btn_frm, text="Clear Output", command=self.clear_output)
        self.clear_btn.pack(side=tk.RIGHT)

        self.progress_label = tk.Label(btn_frm, text="Idle")
        self.progress_label.pack(side=tk.LEFT, padx=8)

        # Visual progress bar (ttk.Progressbar)
        self.progress_bar = ttk.Progressbar(btn_frm, orient="horizontal", mode="determinate", length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=8)

        self.output = scrolledtext.ScrolledText(master, wrap=tk.WORD, height=16)
        self.output.pack(fill=tk.BOTH, padx=12, pady=8, expand=True)

        # Reuse a Pipeline instance to reduce per-run overhead
        self.pipeline = Pipeline()
        self._is_running = False
        self._stop_requested = False

    def browse(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", ("*.mp4", "*.mov", "*.avi", "*.mkv")), ("All files", "*")],
        )
        if path:
            self.video_path_var.set(path)

    def clear_output(self):
        self.output.delete("1.0", tk.END)

    def stop_pipeline(self):
        # Best-effort stop: if Pipeline implements a 'stop' method, call it.
        self._stop_requested = True
        if hasattr(self.pipeline, "stop"):
            try:
                self.pipeline.stop()
                self.output.insert(tk.END, "[info] Stop requested — pipeline.stop() called\n")
                self.output.see("end")
            except Exception as e:
                logger.exception("Failed to stop pipeline")
                self.output.insert(tk.END, f"[error] Failed to stop pipeline: {e}\n")
                self.output.see("end")
        else:
            self.output.insert(tk.END, "[info] Stop requested — but Pipeline.stop() is not implemented\n")
            self.output.see("end")

    def run_pipeline(self):
        path = self.video_path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please select a video file first.")
            return

        if not os.path.isfile(path):
            messagebox.showerror("File not found", f"The file does not exist:\n{path}")
            return

        if self._is_running:
            messagebox.showinfo("Already running", "Pipeline is already running.")
            return

        # Disable run button and spawn background thread
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_label.config(text="Running…")
        self.output.insert(tk.END, f"[info] Starting pipeline on: {path}\n")
        self.output.see("end")
        self._is_running = True
        self._stop_requested = False

        def progress_cb(stage: str, percent: Optional[float], message: Optional[str]):
            # Called from worker thread — marshal UI updates to main thread
            self.master.after(0, self._update_progress, stage, percent, message)

        def worker():
            try:
                res = self.pipeline.run(path, progress_callback=progress_cb)
            except Exception as e:
                logger.exception("Error during pipeline run")
                res = f"[Error running pipeline] {e}"
            # Schedule UI update back on main thread
            self.master.after(0, self._done, res)

        threading.Thread(target=worker, daemon=True).start()

    def _update_progress(self, stage: str, percent: Optional[float], message: Optional[str]):
        # Normalize percent to 0..100. Accept inputs in 0..1 or 0..100 range.
        pct_display = None
        try:
            if percent is None:
                pct_display = None
            else:
                # If percent seems fractional (0..1), scale to 0..100
                val = float(percent)
                if 0.0 <= val <= 1.0:
                    pct_display = int(val * 100)
                else:
                    pct_display = int(max(0, min(100, val)))
        except Exception:
            pct_display = None

        if pct_display is not None:
            self.progress_label.config(text=f"{stage} — {pct_display}%")
            try:
                self.progress_bar["value"] = pct_display
            except Exception:
                pass
        else:
            self.progress_label.config(text=f"{stage}")

        if message:
            self.output.insert(tk.END, f"[{stage}] {message}\n")
            self.output.see("end")

    def _done(self, result: str):
        # Ensure progress bar ends at 100% when completed successfully
        try:
            self.progress_bar["value"] = 100
        except Exception:
            pass
        self.output.insert(tk.END, str(result) + "\n")
        self.progress_label.config(text="Done")
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._is_running = False
        self._stop_requested = False


def launch():
    root = tk.Tk()
    app = SageVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
