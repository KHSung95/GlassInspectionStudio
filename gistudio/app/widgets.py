"""Reusable widgets: auto-generated parameter forms, background workers, small helpers."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLineEdit, QMessageBox, QSpinBox,
                               QWidget)

from ..algorithms.base import ParamSpec


class ParamForm(QWidget):
    """Builds an editor for a list of ParamSpec; emits `changed` on any edit."""
    changed = Signal()

    def __init__(self, specs: list[ParamSpec], values: dict | None = None, parent=None):
        super().__init__(parent)
        self.specs = specs
        self.w = {}
        lay = QFormLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        values = values or {}
        for s in specs:
            v = values.get(s.name, s.default)
            if s.type == "int":
                e = QSpinBox(); e.setRange(int(s.min if s.min is not None else -10**9), int(s.max if s.max is not None else 10**9))
                e.setSingleStep(int(s.step or 1)); e.setValue(int(v)); e.valueChanged.connect(self.changed)
            elif s.type == "float":
                e = QDoubleSpinBox(); e.setDecimals(4)
                e.setRange(float(s.min if s.min is not None else -1e9), float(s.max if s.max is not None else 1e9))
                e.setSingleStep(float(s.step or 0.1)); e.setValue(float(v)); e.valueChanged.connect(self.changed)
            elif s.type == "bool":
                e = QCheckBox(); e.setChecked(bool(v)); e.toggled.connect(self.changed)
            elif s.type == "choice":
                e = QComboBox(); e.addItems([str(c) for c in s.choices]); e.setCurrentText(str(v))
                e.currentTextChanged.connect(self.changed)
            else:
                e = QLineEdit(str(v)); e.editingFinished.connect(self.changed)
            e.setToolTip(s.help)
            self.w[s.name] = e
            lay.addRow(s.name, e)

    def values(self) -> dict:
        out = {}
        for s in self.specs:
            e = self.w[s.name]
            out[s.name] = (e.value() if isinstance(e, (QSpinBox, QDoubleSpinBox)) else e.isChecked()
                           if isinstance(e, QCheckBox) else e.currentText() if isinstance(e, QComboBox) else e.text())
        return out

    def set_values(self, d: dict) -> None:
        for k, v in d.items():
            e = self.w.get(k)
            if e is None:
                continue
            if isinstance(e, (QSpinBox,)):
                e.setValue(int(v))
            elif isinstance(e, QDoubleSpinBox):
                e.setValue(float(v))
            elif isinstance(e, QCheckBox):
                e.setChecked(bool(v))
            elif isinstance(e, QComboBox):
                e.setCurrentText(str(v))
            else:
                e.setText(str(v))


class Worker(QThread):
    """Runs fn(progress, stop_flag) in a background thread so the GUI never blocks."""
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            res = self.fn(lambda i, n, m="": self.progress.emit(int(i), int(n), str(m)), lambda: self._stop)
            self.done.emit(res)
        except Exception:
            self.failed.emit(traceback.format_exc())


def error_box(parent, title: str, text: str) -> None:
    m = QMessageBox(parent); m.setIcon(QMessageBox.Critical); m.setWindowTitle(title)
    short = text.strip().splitlines()[-1] if text.strip() else text
    m.setText(short); m.setDetailedText(text); m.exec()
