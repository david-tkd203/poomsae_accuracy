#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal: Análisis en tiempo real de poomsae con Kinect (híbrido/mock)
"""

import sys, time, os
import cv2
from pathlib import Path

# Añadir src al sys.path si no está
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
from kinect.poomsae_analyzer import PoomsaeReport

# Selección automática de backend
backend_name = None
try:
    from kinect.kinect_backend import KinectPoomsaeCapture as KinectReal, detect_kinect_device
    if detect_kinect_device():
        KinectPoomsaeCapture = KinectReal
        from kinect.kinect_backend_hybrid import PoomsaeMovementAnalyzer
        backend_name = "real"
    else:
        raise ImportError
except Exception:
    try:
        from kinect.kinect_backend_hybrid import KinectPoomsaeCapture, PoomsaeMovementAnalyzer
        backend_name = "hibrido"
    except ImportError:
        from kinect.kinect_backend_mock import KinectPoomsaeCapture, PoomsaeMovementAnalyzer
        backend_name = "mock"

def main():
    print(f"\n==== Análisis de Poomsae en tiempo real ({backend_name}) ====")
    capture = KinectPoomsaeCapture()
    analyzer = PoomsaeMovementAnalyzer(poomsae_name="8yang")
    report = PoomsaeReport(poomsae_name="8yang")
    biomech_report = []
    if not capture.initialize():
        print("No se pudo inicializar el backend de captura.")
        sys.exit(1)

    print("Presiona 'r' para grabar, 's' para guardar, 'q' para salir.")
    recording = False
    while True:
        frame_data = capture.get_frame()
        if frame_data is None:
            print("Error capturando frame.")
            break
        # Selección de landmarks según backend
        landmarks = None
        pose_confidence = 0.0
        # Kinect real: usar primer cuerpo si existe
        if 'bodies' in frame_data and frame_data['bodies']:
            landmarks = frame_data['bodies'][0]
            pose_confidence = 1.0
        # Híbrido/mock: usar landmarks directos
        elif 'landmarks' in frame_data:
            landmarks = frame_data['landmarks']
            pose_confidence = frame_data.get('pose_confidence', 0.0)
        # Análisis biomecánico solo si hay landmarks
        results = {'discounts': [], 'angles': {}, 'errors': []}
        if landmarks is not None:
            results = analyzer.analyze(landmarks, frame_data.get('depth'))
        # Overlay visual
        vis_frame = capture.visualize_frame(frame_data)
        analyzer.draw_overlay(vis_frame, results)
        # Ajuste de tamaño de ventana para visualización cómoda y flexible
        win_name = "Poomsae Kinect Live"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # Control de modo de ventana
        # 'f' = pantalla completa, 'n' = normal, 'm' = miniatura
        if not hasattr(main, "window_mode"):
            main.window_mode = "normal"
        # Redimensionar según modo
        if main.window_mode == "mini":
            vis_frame = cv2.resize(vis_frame, (480, 270))
            cv2.resizeWindow(win_name, 480, 270)
        elif main.window_mode == "fullscreen":
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            if vis_frame.shape[1] > 1280:
                scale = 1280 / vis_frame.shape[1]
                new_size = (int(vis_frame.shape[1]*scale), int(vis_frame.shape[0]*scale))
                vis_frame = cv2.resize(vis_frame, new_size)
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, vis_frame)
        # Guardar resultados si está grabando
        if recording:
            report.add_frame(
                frame_data.get('frame_number', 0),
                frame_data.get('timestamp', 0),
                results['discounts'],
                results['errors'],
                results['angles'],
                pose_confidence
            )
            # Guardar biomecánica de todos los cuerpos
            if 'biomechanics' in frame_data:
                for idx, biomech in enumerate(frame_data['biomechanics']):
                    biomech_report.append({
                        'frame': frame_data.get('frame_number', 0),
                        'timestamp': frame_data.get('timestamp', 0),
                        'body_idx': idx,
                        **biomech['metrics'],
                        **{f"angle_{k}": v for k, v in biomech['angles'].items()}
                    })
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r') and not recording:
            print("🔴 Grabando...")
            capture.start_recording()
            recording = True
        if key == ord('s') and recording:
            print("⏹️  Deteniendo grabación y guardando reporte...")
            capture.stop_recording()
            recording = False
            ts = time.strftime("%Y%m%d_%H%M%S")
            json_path = Path(f"report_{ts}.json")
            csv_path = Path(f"report_{ts}.csv")
            report.save_json(json_path)
            report.save_csv(csv_path)
            report.summary()
            report.frames.clear()
            # Exportar biomecánica
            biomech_json = Path(f"biomechanics_{ts}.json")
            biomech_csv = Path(f"biomechanics_{ts}.csv")
            import json, csv
            with open(biomech_json, 'w') as f:
                json.dump(biomech_report, f, indent=2)
            if biomech_report:
                keys = biomech_report[0].keys()
                with open(biomech_csv, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for row in biomech_report:
                        writer.writerow(row)
            print(f"✅ Biomecánica exportada: {biomech_json}, {biomech_csv}")
            biomech_report.clear()
        # Cambiar modo de ventana
        if key == ord('f'):
            main.window_mode = "fullscreen"
            print("Modo pantalla completa")
        if key == ord('n'):
            main.window_mode = "normal"
            print("Modo ventana normal")
        if key == ord('m'):
            main.window_mode = "mini"
            print("Modo miniatura")
    capture.cleanup()
    cv2.destroyAllWindows()
    print("\nAnálisis finalizado.")

if __name__ == "__main__":
    main()
