from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import numpy as np
from pathlib import Path
from src.model.stance_classifier import StanceClassifier
from src.model.infer import SegmentInfer
from src.obs.obs_window import OBSWindow
from src.obs.camera_manager import CameraManager
from src.obs.mosaic_view import MosaicView
from src.obs.video_recorder import VideoRecorder
from src.obs.overlay_manager import OverlayManager
from src.obs.report_viewer import ReportViewer
from src.obs.config import OBSConfig
import sys
from PyQt5.QtWidgets import QVBoxLayout, QCheckBox

class OBSApp:
    def __init__(self):
        self.config = OBSConfig()
        self.win = OBSWindow()
        self.cam_manager = CameraManager()
        # Actualizar la UI con cámaras detectadas después de crear la ventana
        if hasattr(self.win, 'cam_list'):
            for chk in self.win.cam_checks:
                chk.setParent(None)
            self.win.cam_checks = []
            for cam in self.cam_manager.cameras:
                chk = QCheckBox(cam)
                chk.setChecked(False)
                self.win.cam_checks.append(chk)
                self.win.cam_list.addWidget(chk)
        self.overlay = OverlayManager()
        self.recorder = VideoRecorder(self.config.video_dir)
        self.report_viewer = ReportViewer()
        self.mosaic = None
        self.evaluando = False
        self.frames = {}
        self.metrics = {}
        self.angles = {}
        self.name = ""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.refresh_interval = int(getattr(self.config, 'refresh_interval', 33))
        # Cargar modelos ML (ajusta las rutas según tu estructura)
        self.stance_model = StanceClassifier.load(Path("data/models/stance_classifier.pkl"))
        self.segment_model = SegmentInfer("data/models/segment_model.pkl")
        # Sugerencias de mejora visual:
        # - Puedes agregar barras gráficas de confianza usando cv2.rectangle en OverlayManager.
        # - Resalta advertencias (faltan features) con fondo rojo o icono.
        # - Agrupa métricas biomecánicas y ML en secciones separadas del overlay.
        self.setup_connections()

    def setup_connections(self):
        self.win.start_btn.clicked.connect(self.start_evaluation)
        self.win.stop_btn.clicked.connect(self.stop_evaluation)

    def start_evaluation(self):
        self.name = self.win.name_input.text()
        cams = [chk.text() for chk in self.win.cam_checks if chk.isChecked()]
        self.mosaic = MosaicView(cams)
        self.mosaic.show()
        for cam in cams:
            self.recorder.start_recording(cam)
        self.evaluando = True
        self.timer.start(self.refresh_interval)

    def stop_evaluation(self):
        self.evaluando = False
        self.timer.stop()
        self.recorder.stop_recording()
        # Exportar automáticamente el reporte biomecánico
        self.export_report()
        # Generar sugerencias automáticas (aprendizaje automático básico)
        suggestions = self.generate_suggestions()
        report_text = "Reporte exportado automáticamente.\n\nSugerencias automáticas:\n" + suggestions
        # Visualización gráfica avanzada del reporte
        self.report_viewer.show_report(report_text, metrics=self.metrics)
        self.report_viewer.show()

    def generate_suggestions(self):
        # Analiza métricas y ángulos para detectar patrones y generar recomendaciones
        errores = []
        angulos_fuera = []
        for cam, metrics in self.metrics.items():
            for k, v in metrics.items():
                if 'error' in k.lower() and v:
                    errores.append(f"{cam}: {k} -> {v}")
        for cam, angles in self.angles.items():
            for k, v in angles.items():
                if v < 30 or v > 170:
                    angulos_fuera.append(f"{cam}: {k} = {v:.1f}°")
        sugerencias = []
        if errores:
            sugerencias.append("- Corrige los siguientes errores detectados:")
            sugerencias += [f"  {e}" for e in errores]
        if angulos_fuera:
            sugerencias.append("- Revisa los ángulos fuera de rango (30°-170°):")
            sugerencias += [f"  {a}" for a in angulos_fuera]
        if not sugerencias:
            sugerencias.append("- No se detectaron errores ni ángulos fuera de rango. ¡Buen trabajo!")
        return '\n'.join(sugerencias)

    def export_report(self):
        # Exportar métricas y ángulos a CSV, Excel y JSON
        import pandas as pd
        import json
        report_data = []
        for cam in self.metrics:
            row = {'cam': cam, 'name': self.name}
            row.update(self.metrics[cam])
            row.update({f'angle_{k}': v for k, v in self.angles.get(cam, {}).items()})
            report_data.append(row)
        df = pd.DataFrame(report_data)
        # CSV
        df.to_csv(f'reporte_{self.name}.csv', index=False)
        # Excel
        df.to_excel(f'reporte_{self.name}.xlsx', index=False)
        # JSON
        with open(f'reporte_{self.name}.json', 'w', encoding='utf-8') as fjson:
            json.dump(report_data, fjson, ensure_ascii=False, indent=2)

    def update_frames(self):
        if not self.evaluando or not self.mosaic:
            return
        for cam in self.mosaic.cam_names:
            try:
                frame = self.cam_manager.get_frame(cam)
                if frame is not None:
                    metrics = self.metrics.get(cam, {})
                    angles = self.angles.get(cam, {})
                    # --- INFERENCIA ML ---
                    # Selección y formateo de features
                    feat_row = {**metrics, **angles}
                    missing_feats = []
                    # Postura
                    try:
                        X_stance = np.array([feat_row.get(c, 0.0) for c in self.stance_model.feature_names], float).reshape(1,-1)
                        for idx, c in enumerate(self.stance_model.feature_names):
                            if c not in feat_row:
                                missing_feats.append(c)
                        clase_postura = self.stance_model.predict(X_stance)[0]
                        proba_postura = self.stance_model.predict_proba(X_stance)[0].max()
                    except Exception:
                        clase_postura = "?"
                        proba_postura = 0.0
                    # Segmento
                    try:
                        clase_segmento, proba_segmento = self.segment_model.predict_row(feat_row)
                    except Exception:
                        clase_segmento, proba_segmento = "?", 0.0
                    # Personalización visual avanzada
                    metrics_ml = dict(metrics)
                    # Nombres amigables y colores
                    metrics_ml["Postura"] = (f"{clase_postura}", (0,128,255))
                    metrics_ml["Confianza postura"] = (f"{proba_postura*100:.1f}%", (0,200,0) if proba_postura>0.7 else (0,0,255))
                    metrics_ml["Segmento"] = (f"{clase_segmento}", (255,128,0))
                    metrics_ml["Confianza segmento"] = (f"{proba_segmento*100:.1f}%", (0,200,0) if proba_segmento>0.7 else (0,0,255))
                    # Barra de confianza (visual)
                    metrics_ml["Barra postura"] = (int(proba_postura*100), (0,200,0) if proba_postura>0.7 else (0,0,255))
                    metrics_ml["Barra segmento"] = (int(proba_segmento*100), (0,200,0) if proba_segmento>0.7 else (0,0,255))
                    # Features relevantes
                    for k in list(metrics_ml.keys()):
                        if k not in ["Postura","Confianza postura","Segmento","Confianza segmento","Barra postura","Barra segmento"] and k not in angles:
                            metrics_ml[f"{k}"] = (metrics_ml.pop(k), (128,128,128))
                    # Advertencia si faltan datos
                    if missing_feats:
                        metrics_ml["Faltan features"] = (", ".join(missing_feats), (0,0,255))
                    overlay_frame = self.overlay.draw_metrics(frame, metrics=metrics_ml, angles=angles, name=self.name)
                    self.mosaic.update_frame(cam, overlay_frame)
                else:
                    self.mosaic.update_frame(cam, None, message="Cámara no responde")
            except Exception as e:
                self.mosaic.update_frame(cam, None, message=f"Error: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    obs_app = OBSApp()
    obs_app.win.show()
    sys.exit(app.exec_())
