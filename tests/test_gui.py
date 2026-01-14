import pytest

try:
    import tkinter as tk  # type: ignore
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


@pytest.mark.skipif(not TK_AVAILABLE, reason="Tkinter not available")
def test_gui_import():
    from sagevision.gui import gui

    assert hasattr(gui, "launch")
    assert callable(gui.launch)
