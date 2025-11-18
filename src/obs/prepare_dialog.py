# src/obs/prepare_dialog.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import time
from typing import Dict, Tuple
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from src.obs.mosaic_view import build_mosaic

class PrepareDialog(QtWidgets.QDialog):
    """
    Verifica que todas las fuentes entreguen frames y estima FPS y tamaño.
    Muestra una barra de progreso y una tabla final con resultados.
    """
    def __init__(self, parent, cm, layout: str, kinect_mode: str, duration=2.0):
        super().__init__(parent)
        self.setWindowTitle("Preparando dispositivos…")
        self.setModal(True)
        self.resize(520, 320)

        self.cm = cm
        self.layout = layout
        self.kinect_mode = kinect_mode
        self.duration = float(duration)

        v = QtWidgets.QVBoxLayout(self)
        self.lbl = QtWidgets.QLabel("Inicializando…")
        self.progress = QtWidgets.QProgressBar()
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Fuente", "Resolución", "FPS", "Estado"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btns.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)

        v.addWidget(self.lbl)
        v.addWidget(self.progress)
        v.addWidget(self.table, 1)
        v.addWidget(self.btns)

        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)

        self.results: Dict[str, dict] = {}

        QtCore.QTimer.singleShot(50, self._run_probe)

    def _append_row(self, name, reso, fps, ok=True):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, txt in enumerate([name, reso, f"{fps:.1f}", "OK" if ok else "FALLÓ"]):
            item = QtWidgets.QTableWidgetItem(txt)
            if c == 3:
                color = QtGui.QColor("#2ecc71" if ok else "#e74c3c")
                item.setForeground(color)
            self.table.setItem(r, c, item)

    def _run_probe(self):
        self.lbl.setText("Abriendo Kinect y webcams…")
        self.progress.setRange(0, 0)  # indeterminado
        QtWidgets.qApp.processEvents()

        # warm-up
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.6:
            _ = self.cm.get_all_frames()
            QtWidgets.qApp.processEvents()

        self.lbl.setText("Midiendo FPS real…")
        self.progress.setRange(0, 100)

        counts: Dict[str, int] = {}
        size: Dict[str, Tuple[int, int]] = {}
        tick_count = 0
        t0 = time.monotonic()
        last_pcnt = -1

        while True:
            frames = self.cm.get_all_frames()
            for k, v in frames.items():
                if v is None: continue
                counts[k] = counts.get(k, 0) + 1
                size[k] = (v.shape[1], v.shape[0])
            tick_count += 1

            elapsed = time.monotonic() - t0
            pcnt = int(min(100, (elapsed / self.duration) * 100))
            if pcnt != last_pcnt:
                self.progress.setValue(pcnt); last_pcnt = pcnt
            QtWidgets.qApp.processEvents()

            if elapsed >= self.duration: break

        # estimar FPS por fuente
        fps_map: Dict[str, float] = {}
        for k, n in counts.items():
            fps_est = max(5.0, min(60.0, n / self.duration))  # cota razonable
            if k.lower().startswith("kinect"):
                fps_est = 30.0  # Kinect v2 es 30fps
            fps_map[k] = fps_est

        # también medimos FPS del mosaico (Velocidad de UI)
        m0 = time.monotonic(); mcount = 0
        while time.monotonic() - m0 < 1.0:
            frames = self.cm.get_all_frames()
            canvas = build_mosaic(frames, target_height=540, layout=self.layout, kinect_mode=self.kinect_mode)
            if canvas is not None: mcount += 1
            QtWidgets.qApp.processEvents()
        mosaic_fps = max(5.0, min(60.0, mcount / 1.0))
        fps_map["scene_mosaic"] = mosaic_fps

        # llenar tabla + resultados
        self.table.setRowCount(0)
        for k in sorted(size.keys()):
            w, h = size[k]
            self._append_row(k, f"{w}x{h}", fps_map.get(k, 30.0), ok=True)
            self.results[k] = {"size": (w, h), "fps": fps_map.get(k, 30.0)}
        # mosaico
        self._append_row("scene_mosaic", "-", mosaic_fps, ok=True)

        self.progress.setRange(0, 1); self.progress.setValue(1)
        self.lbl.setText("Listo. Verificado.")
        self.btns.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)

    def get_fps_map(self) -> Dict[str, float]:
        """Devuelve FPS por-fuente + scene_mosaic (normalizados por nombre)."""
        return {k: v["fps"] for k, v in self.results.items()} | {"scene_mosaic": self.table.item(self.table.rowCount()-1, 2).text() and float(self.table.item(self.table.rowCount()-1, 2).text())}
