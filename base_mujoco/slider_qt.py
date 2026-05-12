import numpy as np
from PyQt5 import QtWidgets, QtCore


def _ensure_qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class MultiSliderClass(QtWidgets.QWidget):
    """
    Qt-based multi-slider, drop-in replacement for the Tkinter version.
    On macOS + Jupyter, run `%gui qt` in a cell first so the Qt event loop
    is integrated with IPython and coexists with GLFW (mujoco_viewer).
    """
    def __init__(
        self,
        n_slider      = 10,
        title         = "Multiple Sliders",
        window_width  = 500,
        window_height = None,
        x_offset      = 500,
        y_offset      = 100,
        slider_width  = 400,
        label_texts   = None,
        slider_mins   = None,
        slider_maxs   = None,
        slider_vals   = None,
        resolution    = None,
        resolutions   = None,
        verbose       = True,
    ):
        self._app = _ensure_qapp()
        super().__init__()

        self.n_slider     = n_slider
        self.title        = title
        self.label_texts  = label_texts
        self.slider_mins  = slider_mins
        self.slider_maxs  = slider_maxs
        self.slider_vals  = slider_vals
        self.resolution   = resolution
        self.resolutions  = resolutions
        self.verbose      = verbose
        self.slider_width = slider_width

        self.window_width  = window_width
        self.window_height = window_height if window_height is not None else n_slider * 40

        self.slider_values = np.zeros(self.n_slider)
        self._steps        = np.zeros(self.n_slider)
        self._mins         = np.zeros(self.n_slider)

        self._build_ui(x_offset, y_offset)
        self.show()

        # Pump events a bit so the window appears immediately
        for _ in range(20):
            self._app.processEvents()

    def _build_ui(self, x_offset, y_offset):
        self.setWindowTitle(self.title)
        self.setGeometry(x_offset, y_offset, self.window_width, self.window_height)

        outer = QtWidgets.QVBoxLayout(self)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        grid = QtWidgets.QGridLayout(inner)

        self.sliders   = []
        self.readouts  = []
        for i in range(self.n_slider):
            smin = 0   if self.slider_mins is None else float(self.slider_mins[i])
            smax = 100 if self.slider_maxs is None else float(self.slider_maxs[i])
            sval = 50  if self.slider_vals is None else float(self.slider_vals[i])
            if self.resolutions is not None:
                step = float(self.resolutions[i])
            elif self.resolution is not None:
                step = float(self.resolution)
            else:
                step = (smax - smin) / 100.0

            self._mins[i]  = smin
            self._steps[i] = step

            label_txt = (f"Slider {i:02d}" if self.label_texts is None
                         else f"[{i}/{self.n_slider}]{self.label_texts[i]}")
            label = QtWidgets.QLabel(label_txt)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            n_steps = max(1, int(round((smax - smin) / step)))
            slider.setMinimum(0)
            slider.setMaximum(n_steps)
            slider.setValue(int(round((sval - smin) / step)))
            slider.setFixedWidth(self.slider_width)

            readout = QtWidgets.QLabel(f"{sval:.3f}")
            readout.setMinimumWidth(60)

            slider.valueChanged.connect(self._make_cb(i))
            self.slider_values[i] = sval

            grid.addWidget(label,   i, 0)
            grid.addWidget(slider,  i, 1)
            grid.addWidget(readout, i, 2)
            self.sliders.append(slider)
            self.readouts.append(readout)

    def _make_cb(self, idx):
        def _cb(int_val):
            val = self._mins[idx] + int_val * self._steps[idx]
            self.slider_values[idx] = val
            self.readouts[idx].setText(f"{val:.3f}")
            if self.verbose:
                print(f"slider_idx:[{idx}] slider_value:[{val:.3f}]")
        return _cb

    # ── interface compatible with the Tk version ────────────────────────
    def update(self):
        if self.is_window_exists():
            self._app.processEvents()

    def run(self):
        self._app.exec_()

    def is_window_exists(self):
        try:
            return self.isVisible()
        except RuntimeError:
            return False

    def get_slider_values(self):
        return self.slider_values

    def set_slider_values(self, slider_values):
        self.slider_values = np.array(slider_values, dtype=float)
        for i, v in enumerate(self.slider_values):
            int_val = int(round((v - self._mins[i]) / self._steps[i]))
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(int_val)
            self.sliders[i].blockSignals(False)
            self.readouts[i].setText(f"{v:.3f}")

    def set_slider_value(self, slider_idx, slider_value):
        self.slider_values[slider_idx] = float(slider_value)
        int_val = int(round((slider_value - self._mins[slider_idx]) / self._steps[slider_idx]))
        self.sliders[slider_idx].blockSignals(True)
        self.sliders[slider_idx].setValue(int_val)
        self.sliders[slider_idx].blockSignals(False)
        self.readouts[slider_idx].setText(f"{slider_value:.3f}")

    def close(self):
        try:
            for _ in range(20):
                self._app.processEvents()
            super().close()
        except Exception:
            pass
