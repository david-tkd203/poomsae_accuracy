# src/obs/settings_dialog.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Callable, Optional
from dataclasses import asdict

from PyQt5 import QtCore, QtGui, QtWidgets
from pathlib import Path

from src.obs.config import ObsSettings


class ObsSettingsDialog(QtWidgets.QDialog):
    """
    Diálogo de configuración para OBS Poomsae.
    Cubre TODOS los campos actuales de ObsSettings (config.py). Si luego agregas más,
    este diálogo es fácil de extender.
    """

    def __init__(self, cfg: ObsSettings, apply_callback: Optional[Callable[[ObsSettings], None]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajustes de OBS Poomsae")
        self.setModal(True)
        self.cfg = cfg
        self.apply_callback = apply_callback

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)

        # --- Tabs ---
        self._build_tab_general()
        self._build_tab_recording()
        self._build_tab_webcams()

        # --- Botonera ---
        self.btn_apply = QtWidgets.QPushButton("Aplicar")
        self.btn_save_close = QtWidgets.QPushButton("Guardar y cerrar")
        self.btn_cancel = QtWidgets.QPushButton("Cancelar")

        self.btn_apply.clicked.connect(lambda: self._apply(save=False))
        self.btn_save_close.clicked.connect(lambda: self._apply(save=True))
        self.btn_cancel.clicked.connect(self.reject)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_save_close)
        btns.addWidget(self.btn_cancel)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.tabs, 1)
        lay.addLayout(btns)

        self._load_from_cfg()

    # ----------------------- General -----------------------
    def _build_tab_general(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        f.setLabelAlignment(QtCore.Qt.AlignRight)

        self.cmb_layout = QtWidgets.QComboBox()
        self.cmb_layout.addItems(["studio", "grid"])

        self.cmb_kmode = QtWidgets.QComboBox()
        self.cmb_kmode.addItems(["streams", "composite"])

        self.chk_include_webcams = QtWidgets.QCheckBox("Incluir webcams")
        self.spn_target_height = QtWidgets.QSpinBox()
        self.spn_target_height.setRange(240, 2160)
        self.spn_target_height.setSingleStep(60)

        f.addRow("Layout:", self.cmb_layout)
        f.addRow("Kinect modo:", self.cmb_kmode)
        f.addRow("", self.chk_include_webcams)
        f.addRow("Altura objetivo (px):", self.spn_target_height)

        self.tabs.addTab(w, "General")

    # ----------------------- Grabación -----------------------
    def _build_tab_recording(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        f.setLabelAlignment(QtCore.Qt.AlignRight)

        self.spn_record_fps = QtWidgets.QSpinBox()
        self.spn_record_fps.setRange(5, 120)

        self.cmb_record_codec = QtWidgets.QComboBox()
        # Códecs típicos que funcionan con OpenCV en Windows
        self.cmb_record_codec.addItems(["mp4v", "XVID", "MJPG", "H264"])

        self.dsb_record_scale = QtWidgets.QDoubleSpinBox()
        self.dsb_record_scale.setRange(0.1, 1.0)
        self.dsb_record_scale.setSingleStep(0.05)
        self.dsb_record_scale.setDecimals(3)

        self.chk_record_mosaic = QtWidgets.QCheckBox("Grabar mosaico")
        self.ed_out_root = QtWidgets.QLineEdit()
        self.btn_browse_out = QtWidgets.QPushButton("…")

        def _choose_out():
            base = self.ed_out_root.text().strip() or "recordings"
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Elegir carpeta de salida", base)
            if d:
                self.ed_out_root.setText(d)
        self.btn_browse_out.clicked.connect(_choose_out)

        row_out = QtWidgets.QHBoxLayout()
        row_out.addWidget(self.ed_out_root, 1)
        row_out.addWidget(self.btn_browse_out)

        f.addRow("FPS:", self.spn_record_fps)
        f.addRow("Códec:", self.cmb_record_codec)
        f.addRow("Escala (0.1–1.0):", self.dsb_record_scale)
        f.addRow("", self.chk_record_mosaic)
        f.addRow("Carpeta salida:", row_out)

        self.tabs.addTab(w, "Grabación")

    # ----------------------- Webcams -----------------------
    def _build_tab_webcams(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        f.setLabelAlignment(QtCore.Qt.AlignRight)

        self.spn_max_probe = QtWidgets.QSpinBox()
        self.spn_max_probe.setRange(1, 32)

        self.spn_consec_fail = QtWidgets.QSpinBox()
        self.spn_consec_fail.setRange(1, 10)

        self.ed_excl_indices = QtWidgets.QLineEdit()
        self.ed_excl_names = QtWidgets.QLineEdit()

        f.addRow("Máx. prueba índices:", self.spn_max_probe)
        f.addRow("Cortes tras fallos consecutivos:", self.spn_consec_fail)
        f.addRow("Excluir por índice (ej: 0,2,5):", self.ed_excl_indices)
        f.addRow("Excluir si nombre contiene (csv):", self.ed_excl_names)

        info = QtWidgets.QLabel("Sugerencia: en notebooks, la integrada suele ser índice 0 o 1.")
        info.setStyleSheet("color:#999;")
        f.addRow("", info)

        self.tabs.addTab(w, "Webcams")

    # ----------------------- Helpers -----------------------
    def _load_from_cfg(self):
        # General
        self.cmb_layout.setCurrentText(self.cfg.layout)
        self.cmb_kmode.setCurrentText(self.cfg.kinect_mode)
        self.chk_include_webcams.setChecked(bool(self.cfg.include_webcams))
        self.spn_target_height.setValue(int(self.cfg.target_height))

        # Grabación
        self.spn_record_fps.setValue(int(self.cfg.record_fps))
        idx = max(0, self.cmb_record_codec.findText(self.cfg.record_codec))
        self.cmb_record_codec.setCurrentIndex(idx)
        self.dsb_record_scale.setValue(float(self.cfg.record_scale))
        self.chk_record_mosaic.setChecked(bool(self.cfg.record_mosaic))
        self.ed_out_root.setText(str(self.cfg.out_root))

        # Webcams
        self.spn_max_probe.setValue(int(self.cfg.max_probe))
        self.spn_consec_fail.setValue(int(self.cfg.consecutive_fail_stop))
        self.ed_excl_indices.setText(",".join(str(i) for i in list(self.cfg.exclude_indices)))
        self.ed_excl_names.setText(",".join(self.cfg.exclude_name_contains))

    def _collect_to_cfg(self):
        # General
        self.cfg.layout = self.cmb_layout.currentText().strip()
        self.cfg.kinect_mode = self.cmb_kmode.currentText().strip()
        self.cfg.include_webcams = bool(self.chk_include_webcams.isChecked())
        self.cfg.target_height = int(self.spn_target_height.value())

        # Grabación
        self.cfg.record_fps = int(self.spn_record_fps.value())
        self.cfg.record_codec = self.cmb_record_codec.currentText().strip()
        self.cfg.record_scale = float(self.dsb_record_scale.value())
        self.cfg.record_mosaic = bool(self.chk_record_mosaic.isChecked())
        self.cfg.out_root = self.ed_out_root.text().strip() or "recordings"

        # Webcams
        self.cfg.max_probe = int(self.spn_max_probe.value())
        self.cfg.consecutive_fail_stop = int(self.spn_consec_fail.value())
        # índices
        excl_idx = []
        txt = self.ed_excl_indices.text().strip()
        if txt:
            for tok in txt.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    excl_idx.append(int(tok))
        self.cfg.exclude_indices = excl_idx
        # nombres
        excl_names = []
        txtn = self.ed_excl_names.text().strip()
        if txtn:
            excl_names = [t.strip() for t in txtn.split(",") if t.strip()]
        self.cfg.exclude_name_contains = excl_names

    def _apply(self, save: bool):
        self._collect_to_cfg()
        if save:
            try:
                self.cfg.save()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Guardar configuración",
                                              f"No fue posible guardar obs_settings.json:\n{e}")
        if self.apply_callback:
            try:
                self.apply_callback(self.cfg)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Aplicar",
                                              f"No se pudieron aplicar cambios en vivo:\n{e}")
        if save:
            self.accept()
