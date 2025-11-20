# src/obs/obs_window.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sys, time
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from src.obs.config import ObsSettings
from src.obs.video_recorder import MultiRecorder
from src.obs.mosaic_view import build_mosaic
from src.obs.camera_manager import CameraManager
from src.kinect.kinect_backend import KinectPoomsaeCapture
from src.obs.prepare_dialog import PrepareDialog
from src.obs.settings_dialog import ObsSettingsDialog  # <-- NUEVO: diálogo real de ajustes

# --------- Opcionales: 3D y análisis -------
_PC_IMPORT_ERR = ""
_AN_IMPORT_ERR = ""

_HAS_PC = False
PointCloudWidget = None
try:
    from src.obs.pointcloud_qt import PointCloudWidget as _PCW
    PointCloudWidget = _PCW
    _HAS_PC = True
except Exception as _e_pc:
    _PC_IMPORT_ERR = str(_e_pc)
    PointCloudWidget = None
    _HAS_PC = False

_HAS_ANALYTICS = False
LivePoomsaeEngine = None
generate_simple_report = None
try:
    from src.obs.analysis_bridge import LivePoomsaeEngine as _LPE, generate_simple_report as _GSR
    LivePoomsaeEngine = _LPE
    generate_simple_report = _GSR
    _HAS_ANALYTICS = True
except Exception as _e_an:
    _AN_IMPORT_ERR = str(_e_an)
    LivePoomsaeEngine = None
    generate_simple_report = None
    _HAS_ANALYTICS = False


def cv_to_qimage(img: np.ndarray) -> QtGui.QImage:
    if img is None:
        return QtGui.QImage()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888).copy()


class ObsWindow(QtWidgets.QMainWindow):
    TICK_MS = 25

    def __init__(self, cfg: ObsSettings):
        super().__init__()
        self.setWindowTitle("OBS Poomsae · Kinect v2 + Webcams")
        try:
            self.showMaximized()
        except Exception:
            self.resize(1280, 720)

        self.cfg = cfg
        self.recorder: Optional[MultiRecorder] = None
        self.recording = False
        self._last_canvas = None
        self._last_frames: Dict[str, np.ndarray] = {}
        self._rec_t0 = None
        self._last_session: Optional[Path] = None
        self._last_report: Optional[Path] = None

        # Estado de layout dinámico (se sincroniza con cfg más abajo)
        self._grid_override: Optional[Tuple[int, int]] = None   # (rows, cols) o None=Auto
        self._max_tiles: int = 6
        self._prefer_aspect: float = 16/9
        self._max_cols: int = 6

        # Backend / Manager
        kbackend = KinectPoomsaeCapture(
            enable_color=True, enable_depth=True, enable_ir=True,
            enable_audio=True, enable_body=True, enable_body_index=True
        )
        self.cm = CameraManager(
            kinect_backend=kbackend,
            kinect_mode=self.cfg.kinect_mode,
            max_probe=self.cfg.max_probe,
            consecutive_fail_stop=self.cfg.consecutive_fail_stop,
            exclude_indices=self.cfg.exclude_indices,
            exclude_name_contains=self.cfg.exclude_name_contains,
        )

        # Analítica en vivo (opcional)
        self.live_engine = LivePoomsaeEngine() if _HAS_ANALYTICS else None

        # UI central
        self._build_toolbar()
        self._build_dock_sources()
        self._build_dock_panel()

        # Lienzo central
        self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background:#1c1c1f;")
        self.setCentralWidget(self.image_label)
        self.status = self.statusBar()
        self.status.showMessage("Listo")

        # Sincroniza estado con cfg (mosaico/grilla/etc.)
        self._apply_cfg_to_state_and_toolbar()

        # Loop UI
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.TICK_MS)
        self._visible_sources: set[str] = set()
        self._update_title()

    # ---------- UI ----------
    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        self.act_rec = QtWidgets.QAction("REC 00:00", self)
        self.act_rec.setCheckable(True)
        self.act_snap = QtWidgets.QAction("Screenshot", self)
        self.act_cfg  = QtWidgets.QAction("Ajustes", self)
        self.act_layout = QtWidgets.QAction(f"Layout: {self.cfg.layout}", self)
        self.act_kmode  = QtWidgets.QAction(f"Kinect: {self.cfg.kinect_mode}", self)
        self.act_webcams = QtWidgets.QAction("Webcams ON", self)
        self.act_webcams.setCheckable(True)
        self.act_webcams.setChecked(self.cfg.include_webcams)

        tb.addAction(self.act_rec); tb.addSeparator()
        tb.addAction(self.act_snap); tb.addSeparator()
        tb.addAction(self.act_layout); tb.addAction(self.act_kmode); tb.addAction(self.act_webcams); tb.addSeparator()

        # --- Controles de mosaico responsivo ---
        self.cmb_grid = QtWidgets.QComboBox()
        self.cmb_grid.addItems(["Auto", "1x1", "2x2", "3x3", "4x4"])
        self.cmb_grid.setCurrentIndex(0)
        wa_grid = QtWidgets.QWidgetAction(self); wa_grid.setDefaultWidget(self.cmb_grid); tb.addAction(wa_grid)

        self.spn_max_tiles = QtWidgets.QSpinBox()
        self.spn_max_tiles.setRange(1, 36)
        self.spn_max_tiles.setValue(self._max_tiles)
        self.spn_max_tiles.setToolTip("Máximo de fuentes visibles en el mosaico")
        wa_max = QtWidgets.QWidgetAction(self); wa_max.setDefaultWidget(self.spn_max_tiles); tb.addAction(wa_max)

        tb.addSeparator()
        tb.addAction(self.act_cfg)

        self.act_rec.triggered.connect(self._toggle_record)
        self.act_snap.triggered.connect(self._screenshot)
        self.act_cfg.triggered.connect(self._open_settings)
        self.act_layout.triggered.connect(self._toggle_layout)
        self.act_kmode.triggered.connect(self._toggle_kinect_mode)
        self.act_webcams.triggered.connect(self._toggle_webcams)

        self.cmb_grid.currentTextChanged.connect(self._on_grid_changed)
        self.spn_max_tiles.valueChanged.connect(self._on_max_tiles_changed)

    def _apply_cfg_to_state_and_toolbar(self):
        """Sincroniza self.* y la barra con valores en cfg."""
        # Mosaico / layout (si no existen en cfg, usa defaults locales)
        self._prefer_aspect = float(getattr(self.cfg, "prefer_aspect", self._prefer_aspect))
        self._max_cols = int(getattr(self.cfg, "max_cols", self._max_cols))
        self._max_tiles = int(getattr(self.cfg, "max_tiles", self._max_tiles))
        grid_rc = getattr(self.cfg, "grid_override", (None, None))
        self._grid_override = grid_rc if (grid_rc and grid_rc[0] and grid_rc[1]) else None

        # Ajusta toolbar
        self.spn_max_tiles.blockSignals(True)
        self.spn_max_tiles.setValue(self._max_tiles)
        self.spn_max_tiles.blockSignals(False)

        # Combo grilla
        def rc_to_txt(rc):
            if not rc or not rc[0] or not rc[1]:
                return "Auto"
            m = {(1,1): "1x1", (2,2): "2x2", (3,3): "3x3", (4,4): "4x4"}
            return m.get((int(rc[0]), int(rc[1])), "Auto")

        want_txt = rc_to_txt(grid_rc)
        if self.cmb_grid.findText(want_txt) >= 0:
            self.cmb_grid.blockSignals(True)
            self.cmb_grid.setCurrentText(want_txt)
            self.cmb_grid.blockSignals(False)

        # Otros toggles
        self.act_webcams.setChecked(bool(getattr(self.cfg, "include_webcams", self.cfg.include_webcams)))
        self.act_layout.setText(f"Layout: {self.cfg.layout}")
        self.act_kmode.setText(f"Kinect: {self.cfg.kinect_mode}")

    def _on_grid_changed(self, txt: str):
        mapping = {"Auto": None, "1x1": (1, 1), "2x2": (2, 2), "3x3": (3, 3), "4x4": (4, 4)}
        self._grid_override = mapping.get(txt, None)
        # Persistimos en cfg si se puede (si no está en dataclass, save lo omitirá)
        self.cfg.grid_override = (None, None) if self._grid_override is None else self._grid_override
        try:
            self.cfg.save()
        except Exception:
            pass

    def _on_max_tiles_changed(self, v: int):
        self._max_tiles = int(v)
        self.cfg.max_tiles = int(v)
        try:
            self.cfg.save()
        except Exception:
            pass

    def _build_dock_sources(self):
        dock = QtWidgets.QDockWidget("Fuentes", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(6,6,6,6)
        self.list_sources = QtWidgets.QListWidget()
        self.list_sources.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        v.addWidget(QtWidgets.QLabel("Mostrar/Ocultar (check):"))
        v.addWidget(self.list_sources, 1)
        row = QtWidgets.QHBoxLayout()
        self.btn_select_all = QtWidgets.QPushButton("Mostrar todas")
        self.btn_clear = QtWidgets.QPushButton("Ocultar todas")
        row.addWidget(self.btn_select_all); row.addWidget(self.btn_clear)
        v.addLayout(row)
        dock.setWidget(w)
        self.btn_select_all.clicked.connect(self._select_all_sources)
        self.btn_clear.clicked.connect(self._clear_sources)

    def _build_dock_panel(self):
        dock = QtWidgets.QDockWidget("Panel", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)

        # --- Tab 1: Poomsae en vivo ---
        w_live = QtWidgets.QWidget(); vl = QtWidgets.QVBoxLayout(w_live); vl.setContentsMargins(8,8,8,8)
        self.tbl_live = QtWidgets.QTableWidget(0, 2)
        self.tbl_live.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.tbl_live.horizontalHeader().setStretchLastSection(True)
        self.tbl_live.verticalHeader().setVisible(False)
        self.tbl_live.setShowGrid(True)
        self.btn_live_reset = QtWidgets.QPushButton("Limpiar tabla")
        vl.addWidget(self.tbl_live, 1); vl.addWidget(self.btn_live_reset)
        self.btn_live_reset.clicked.connect(lambda: self.tbl_live.setRowCount(0))

        # --- Tab 2: Reporte ---
        w_rep = QtWidgets.QWidget(); vr = QtWidgets.QVBoxLayout(w_rep); vr.setContentsMargins(8,8,8,8)
        self.lbl_session = QtWidgets.QLabel("Sesión: —")
        self.btn_report = QtWidgets.QPushButton("Generar reporte (HTML) desde la última sesión")
        self.btn_open_report = QtWidgets.QPushButton("Abrir último reporte")
        self.btn_open_report.setEnabled(False)
        vr.addWidget(self.lbl_session); vr.addWidget(self.btn_report); vr.addWidget(self.btn_open_report); vr.addStretch(1)

        self.btn_report.setEnabled(_HAS_ANALYTICS and (generate_simple_report is not None))
        self.btn_report.clicked.connect(self._gen_report_clicked)
        self.btn_open_report.clicked.connect(self._open_last_report)

        # --- Tab 3: 3D ---
        w_3d = QtWidgets.QWidget(); v3 = QtWidgets.QVBoxLayout(w_3d); v3.setContentsMargins(8,8,8,8)
        if _HAS_PC and PointCloudWidget is not None:
            self.pc_widget = PointCloudWidget()

            # Config por defecto
            if hasattr(self.pc_widget, 'set_fov'): self.pc_widget.set_fov(90)
            if hasattr(self.pc_widget, 'set_clip'): self.pc_widget.set_clip(0.01, 12000.0)
            if hasattr(self.pc_widget, 'set_auto_fit'): self.pc_widget.set_auto_fit(True)
            if hasattr(self.pc_widget, 'set_accumulate'): self.pc_widget.set_accumulate(True, 1_800_000)
            if hasattr(self.pc_widget, 'set_voxel_size'): self.pc_widget.set_voxel_size(0.02)

            ctr = QtWidgets.QHBoxLayout()
            self.spn_dec = QtWidgets.QSpinBox(); self.spn_dec.setRange(1,6); self.spn_dec.setValue(3)
            self.spn_psz = QtWidgets.QDoubleSpinBox(); self.spn_psz.setRange(1.0, 10.0); self.spn_psz.setValue(3.0); self.spn_psz.setSingleStep(0.5)
            self.chk_color = QtWidgets.QCheckBox("Color desde RGB"); self.chk_color.setChecked(True)
            self.chk_mask_bodies = QtWidgets.QCheckBox("Solo cuerpos (BodyIndex)"); self.chk_mask_bodies.setChecked(True)
            self.chk_color_bidx = QtWidgets.QCheckBox("Color por persona (BodyIndex)"); self.chk_color_bidx.setChecked(True)

            # Si color_by_bidx está activo, bloquea "Color desde RGB"
            def _sync_color_checks():
                use_bidx = self.chk_color_bidx.isChecked()
                self.chk_color.setEnabled(not use_bidx)
            self.chk_color_bidx.toggled.connect(_sync_color_checks)
            _sync_color_checks()

            ctr.addWidget(QtWidgets.QLabel("Decimate:")); ctr.addWidget(self.spn_dec)
            ctr.addWidget(QtWidgets.QLabel("Tamaño punto:")); ctr.addWidget(self.spn_psz)
            ctr.addWidget(self.chk_mask_bodies)
            ctr.addWidget(self.chk_color_bidx)
            ctr.addWidget(self.chk_color)
            ctr.addStretch(1)
            v3.addLayout(ctr); v3.addWidget(self.pc_widget, 1)

            def _on_psz(v):
                if hasattr(self.pc_widget, 'set_point_size'):
                    self.pc_widget.set_point_size(v)
            self.spn_psz.valueChanged.connect(_on_psz)
        else:
            self.pc_widget = None
            msg = QtWidgets.QLabel(
                "Visor 3D deshabilitado.\n\n"
                f"Motivo: {'pyqtgraph.opengl no disponible' if not _HAS_PC else 'Error importando PointCloudWidget'}\n"
                f"{_PC_IMPORT_ERR if not _HAS_PC else ''}"
            )
            msg.setStyleSheet("color:#bbb; padding:16px;")
            msg.setAlignment(QtCore.Qt.AlignCenter)
            v3.addWidget(msg, 1)

        tabs.addTab(w_live, "Poomsae (en vivo)")
        tabs.addTab(w_rep, "Reporte")
        tabs.addTab(w_3d,  "3D")

        dock.setWidget(tabs)
        self._tabs = tabs

    # ---------- actions ----------
    def _toggle_layout(self):
        self.cfg.layout = "grid" if self.cfg.layout == "studio" else "studio"
        self.act_layout.setText(f"Layout: {self.cfg.layout}")
        self.cfg.save(); self._update_title()

    def _toggle_kinect_mode(self):
        self.cfg.kinect_mode = "composite" if self.cfg.kinect_mode == "streams" else "streams"
        self.cm.kinect_mode = self.cfg.kinect_mode
        self.act_kmode.setText(f"Kinect: {self.cfg.kinect_mode}")
        self.cfg.save(); self._update_title()

    def _toggle_webcams(self):
        self.cfg.include_webcams = self.act_webcams.isChecked()
        self.cfg.save(); self._update_title()

    def _toggle_record(self, is_on: bool):
        if is_on:
            prep = PrepareDialog(self, self.cm, layout=self.cfg.layout, kinect_mode=self.cfg.kinect_mode, duration=2.0)
            if prep.exec_() != QtWidgets.QDialog.Accepted:
                self.act_rec.setChecked(False)
                return
            fps_map = prep.get_fps_map()

            try:
                self.recorder = MultiRecorder(
                    out_root=self.cfg.out_root,
                    fps_default=self.cfg.record_fps,
                    codec=self.cfg.record_codec,
                    scale=self.cfg.record_scale,
                    record_mosaic=self.cfg.record_mosaic,
                    fps_map=fps_map
                )
                session = self.recorder.start()
                self._last_session = Path(session)
                if hasattr(self, "lbl_session"):
                    self.lbl_session.setText(f"Sesión: {self._last_session.name}")

                try:
                    self.cm._ensure_kinect_open()
                    kb = self.cm._kinect or self.cm._kinect_backend
                    kb_out = Path(session) / "kinect_backend"; kb_out.mkdir(exist_ok=True)
                    kb.start_recording(output_dir=str(kb_out),
                                       codec=self.cfg.record_codec,
                                       fps=int(fps_map.get("kinect_rgb", 30)),
                                       scale=1.0)
                    pc_dir = Path(session) / "pointcloud"
                    kb.start_pointcloud_recording(
                        out_dir=str(pc_dir),
                        every_n=int(getattr(self.cfg, "pc_every_n", 10)),
                        decimate=int(getattr(self.cfg, "pc_decimate", 2)),
                        colorize=bool(getattr(self.cfg, "pc_colorize", True)),
                        prefer_mapper=bool(getattr(self.cfg, "pc_prefer_mapper", True))
                    )
                except Exception as e:
                    self.status.showMessage(f"Kinect backend no pudo iniciar grabación: {e}")

                if self._last_canvas is not None:
                    self.recorder.write_frames(self._last_frames, self._last_canvas)

                self._rec_t0 = time.monotonic()
                self.status.showMessage(f"Grabando… → {session}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo iniciar la grabación:\n{e}")
                self.act_rec.setChecked(False)
                self.recorder = None
                return
            self.recording = True

        else:
            try:
                kb = self.cm._kinect or self.cm._kinect_backend
                if kb and getattr(kb, "_pc_record_dir", None) is not None:
                    kb.stop_pointcloud_recording()
                if kb and getattr(kb, "recording", False):
                    kb.stop_recording()
            except Exception:
                pass

            session = None
            if self.recorder:
                session = self.recorder.stop()
                if session:
                    self._last_session = Path(session)
                    if hasattr(self, "lbl_session"):
                        self.lbl_session.setText(f"Sesión: {self._last_session.name}")

            self.recording = False
            self.recorder = None
            self._rec_t0 = None
            self.act_rec.setText("REC 00:00")
            self.status.showMessage(f"Grabación guardada en {session}")
            try:
                if session and os.name == "nt":
                    os.startfile(str(session))
            except Exception:
                pass

    def _screenshot(self):
        p = Path("obs_screenshots"); p.mkdir(exist_ok=True)
        if self._last_canvas is None:
            self.status.showMessage("No hay imagen aún."); return
        from time import strftime
        out = p / f"snap_{strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(out), self._last_canvas)
        self.status.showMessage(f"Screenshot: {out}")

    def _open_settings(self):
        """Abre el diálogo de Ajustes y aplica cambios EN VIVO al cerrar/aplicar."""
        def _apply_now(new_cfg: ObsSettings):
            # 1) reconstruir CameraManager con el backend actual
            old_kb = self.cm._kinect_backend or self.cm._kinect
            try:
                self.cm.release_all()
            except Exception:
                pass
            self.cm = CameraManager(
                kinect_backend=old_kb,
                kinect_mode=new_cfg.kinect_mode,
                max_probe=new_cfg.max_probe,
                consecutive_fail_stop=new_cfg.consecutive_fail_stop,
                exclude_indices=new_cfg.exclude_indices,
                exclude_name_contains=new_cfg.exclude_name_contains,
            )

            # 2) actualizar acciones / estado UI
            self.act_layout.setText(f"Layout: {new_cfg.layout}")
            self.act_kmode.setText(f"Kinect: {new_cfg.kinect_mode}")
            self.act_webcams.setChecked(bool(new_cfg.include_webcams))
            self._apply_cfg_to_state_and_toolbar()
            self._update_title()
            self.status.showMessage("Ajustes aplicados.")

        dlg = ObsSettingsDialog(self.cfg, apply_callback=_apply_now, parent=self)
        dlg.exec_()

    def _select_all_sources(self):
        self._visible_sources.clear()
        for i in range(self.list_sources.count()):
            self.list_sources.item(i).setCheckState(QtCore.Qt.Checked)

    def _clear_sources(self):
        self._visible_sources = set(["(ninguna)"])
        for i in range(self.list_sources.count()):
            self.list_sources.item(i).setCheckState(QtCore.Qt.Unchecked)

    # ---------- Reporte ----------
    def _gen_report_clicked(self):
        if not _HAS_ANALYTICS or generate_simple_report is None:
            QtWidgets.QMessageBox.warning(self, "Reporte", "Función de reporte no disponible (analysis_bridge faltante).")
            return
        if not self._last_session or not self._last_session.exists():
            QtWidgets.QMessageBox.warning(self, "Reporte", "No hay una sesión reciente.")
            return
        try:
            out = generate_simple_report(self._last_session)
            self.status.showMessage(f"Reporte generado: {out}")
            self._last_report = out
            self.btn_open_report.setEnabled(True)
            try:
                os.startfile(str(out))
            except Exception:
                pass
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Reporte", f"No se pudo generar el reporte:\n{e}")

    def _open_last_report(self):
        p = self._last_report
        if p and Path(p).exists():
            try:
                os.startfile(str(p))
            except Exception:
                pass

    # ---------- loop ----------
    def _tick(self):
        frames = self.cm.get_all_frames()

        # refrescar lista de fuentes si cambió
        names = list(frames.keys())
        if self.list_sources.count() != len(names) or any(self.list_sources.item(i).text() != names[i] for i in range(len(names))):
            self.list_sources.blockSignals(True)
            self.list_sources.clear()
            for n in names:
                it = QtWidgets.QListWidgetItem(n)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked if (not self._visible_sources or n in self._visible_sources) else QtCore.Qt.Unchecked)
                self.list_sources.addItem(it)
            self.list_sources.blockSignals(False)
            self.list_sources.itemChanged.connect(self._on_source_check)

        # filtros de visibilidad
        show_frames: Dict[str, np.ndarray] = {}
        for k, v in frames.items():
            if not self.cfg.include_webcams and k.lower().startswith("webcam"):
                continue
            if self._visible_sources and k not in self._visible_sources:
                continue
            show_frames[k] = v

        # tamaño disponible (ancho/alto reales del label para responsividad)
        avail_w = max(0, int(self.image_label.width()))
        avail_h = max(0, int(self.image_label.height())) or self.cfg.target_height

        # compositar responsivo (grid override + max tiles)
        canvas = build_mosaic(
            show_frames,
            target_height=avail_h,
            layout=self.cfg.layout,
            kinect_mode=self.cfg.kinect_mode,
            target_width=avail_w if avail_w > 0 else None,
            grid=self._grid_override,
            max_tiles=self._max_tiles,
            prefer_aspect=self._prefer_aspect,
            max_cols=self._max_cols,
            show_labels=bool(getattr(self.cfg, "show_labels", True))
        )

        if canvas is not None:
            self._last_canvas = canvas
            self._last_frames = show_frames
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(cv_to_qimage(canvas)))
            self._update_title(extra=f"{len(show_frames)} fuente(s) | {canvas.shape[1]}x{canvas.shape[0]}")
            if self.recording and self.recorder:
                self.recorder.write_frames(show_frames, mosaic=canvas)

        # --- LIVE ANALYSIS + 3D ---
        try:
            kb = self.cm._kinect or self.cm._kinect_backend
            if kb and getattr(kb, "kinect", None) is not None:
                fd = kb.get_frame()
                if _HAS_ANALYTICS and self.live_engine and fd and fd.get("biomechanics"):
                    self.live_engine.update_from_fd(fd)
                    rows = self.live_engine.as_table_rows()
                    self._populate_live_table(rows)

                if _HAS_PC and self.pc_widget is not None and self._tabs.currentIndex() == 2:
                    dec = int(self.spn_dec.value()) if hasattr(self, "spn_dec") else 2
                    use_rgb = bool(self.chk_color.isChecked()) if hasattr(self, "chk_color") else True
                    mask_bodies = bool(self.chk_mask_bodies.isChecked()) if hasattr(self, "chk_mask_bodies") else False
                    color_by_bidx = bool(self.chk_color_bidx.isChecked()) if hasattr(self, "chk_color_bidx") else False

                    pc = kb.get_point_cloud(
                        decimate=dec,
                        colorize=use_rgb,
                        prefer_mapper=True,
                        mask_to_bodies=mask_bodies,
                        color_by_bidx=color_by_bidx
                    )
                    if pc and pc.get("points") is not None:
                        self.pc_widget.update_cloud(pc["points"], pc.get("colors"))
        except Exception:
            # evita romper el loop UI por fallas de análisis/3D
            pass

        # contador HH:MM:SS
        if self.recording and self._rec_t0 is not None:
            elapsed = int(time.monotonic() - self._rec_t0)
            hh, rem = divmod(elapsed, 3600); mm, ss = divmod(rem, 60)
            text = f"REC {hh:02d}:{mm:02d}:{ss:02d}" if hh else f"REC {mm:02d}:{ss:02d}"
            self.act_rec.setText(text)

    def _populate_live_table(self, rows: List[Tuple[str, str]]):
        self.tbl_live.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.tbl_live.setItem(i, 0, QtWidgets.QTableWidgetItem(str(k)))
            self.tbl_live.setItem(i, 1, QtWidgets.QTableWidgetItem(str(v)))

    def _on_source_check(self, item: QtWidgets.QListWidgetItem):
        name = item.text()
        if item.checkState() == QtCore.Qt.Checked:
            self._visible_sources.discard(name)
        else:
            self._visible_sources.add(name)

    def _update_title(self, extra: str = ""):
        txt = f"OBS Poomsae · {self.cfg.layout} · Kinect {self.cfg.kinect_mode}"
        if extra:
            txt += f" · {extra}"
        self.setWindowTitle(txt)

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        try:
            if self.recording and self.recorder:
                self.recorder.stop()
            kb = self.cm._kinect or self.cm._kinect_backend
            if kb and getattr(kb, "_pc_record_dir", None) is not None:
                kb.stop_pointcloud_recording()
            if kb and getattr(kb, "recording", False):
                kb.stop_recording()
        finally:
            self.cm.release_all()
        return super().closeEvent(e)
