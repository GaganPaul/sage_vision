import sys
import types

# Create a fake tkinter environment to safely test GUI logic in headless CI
fake_tk = types.SimpleNamespace()

class _FakeWidget:
    def __init__(self, master=None, **kwargs):
        self._props = {}
    def pack(self, *args, **kwargs):
        pass
    def config(self, **kwargs):
        self._props.update(kwargs)
    def insert(self, *args, **kwargs):
        pass
    def delete(self, *args, **kwargs):
        pass
    def see(self, *args, **kwargs):
        pass

class FakeProgressbar(_FakeWidget):
    def __init__(self, master=None, orient=None, mode=None, length=None):
        super().__init__(master)
        self._props = {"value": 0}
    def __setitem__(self, k, v):
        self._props[k] = v
    def __getitem__(self, k):
        return self._props.get(k, 0)

class FakeScrolledText(_FakeWidget):
    def __init__(self, master=None, wrap=None, height=None):
        super().__init__(master)

# Minimal fake tkinter API
fake_tk.Frame = _FakeWidget
fake_tk.Label = _FakeWidget
fake_tk.Button = _FakeWidget
fake_tk.Entry = _FakeWidget

class FakeStringVar:
    def __init__(self):
        self._val = ""
    def set(self, v):
        self._val = v
    def get(self):
        return self._val

fake_tk.StringVar = FakeStringVar
fake_tk.Tk = lambda : types.SimpleNamespace(title=lambda *a, **k: None, geometry=lambda *a, **k: None, after=lambda *a, **k: None)
fake_tk.WORD = 'word'
fake_tk.LEFT = 'left'
fake_tk.RIGHT = 'right'
fake_tk.BOTH = 'both'
fake_tk.END = 'end'
fake_tk.W = 'w'
fake_tk.X = 'x'

fake_ttk = types.SimpleNamespace(Progressbar=FakeProgressbar)
fake_scrolledtext = types.SimpleNamespace(ScrolledText=FakeScrolledText)

sys.modules['tkinter'] = types.ModuleType('tkinter')
# populate attributes
for k, v in fake_tk.__dict__.items():
    setattr(sys.modules['tkinter'], k, v)
# Add filedialog and messagebox minimal APIs
setattr(sys.modules['tkinter'], 'filedialog', types.SimpleNamespace(askopenfilename=lambda *a, **k: ""))
setattr(sys.modules['tkinter'], 'messagebox', types.SimpleNamespace(showwarning=lambda *a, **k: None, showerror=lambda *a, **k: None))
sys.modules['tkinter.ttk'] = types.ModuleType('tkinter.ttk')
setattr(sys.modules['tkinter.ttk'], 'Progressbar', FakeProgressbar)
sys.modules['tkinter.scrolledtext'] = types.ModuleType('tkinter.scrolledtext')
setattr(sys.modules['tkinter.scrolledtext'], 'ScrolledText', FakeScrolledText)

from sagevision.gui.gui import SageVisionApp

root = sys.modules['tkinter'].Tk()
app = SageVisionApp(root)

# simulate a progress update and ensure progressbar updated
app._update_progress("parsing", 0.5, "halfway")
assert int(app.progress_bar['value']) == 50

# done should set to 100
app._done("ok")
assert int(app.progress_bar['value']) == 100
