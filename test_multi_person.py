#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Prueba Rápida - Detección Múltiple de Personas
=========================================================

Prueba rápida de detección de múltiples personas mostrando
la visualización en tiempo real.
"""

import cv2
import sys
from pathlib import Path

# Importar el backend
sys.path.append(str(Path(__file__).parent))
from src.pose.multi_person_backend import MultiPersonPoseDetector, visualize_multi_person


def test_multi_person_detection(video_path: str, max_persons: int = 2):
    """
    Prueba rápida de detección múltiple con visualización en ventana
    
    Args:
        video_path: Ruta al video
        max_persons: Número máximo de personas a detectar
    """
    print("="*80)
    print(" 🧪 PRUEBA DE DETECCIÓN DE MÚLTIPLES PERSONAS")
    print("="*80)
    print(f"\n📹 Video: {video_path}")
    print(f"👥 Detectando hasta {max_persons} personas")
    print("\n⌨️  Controles:")
    print("   • ESPACIO: Pausar/Reanudar")
    print("   • Q: Salir")
    print("="*80)
    print()
    
    # Inicializar detector
    detector = MultiPersonPoseDetector(
        max_num_persons=max_persons,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video cargado: {total_frames} frames @ {fps:.1f} FPS\n")
    
    frame_idx = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Fin del video")
                break
            
            # Detectar personas
            tracked_persons = detector.process_frame(frame)
            
            # Visualizar
            vis_frame = visualize_multi_person(frame, tracked_persons)
            
            # Info en pantalla
            info = f"Frame: {frame_idx}/{total_frames} | Personas: {len(tracked_persons)}"
            cv2.putText(vis_frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Mostrar IDs detectados
            if tracked_persons:
                ids_text = "IDs: " + ", ".join([str(pid) for pid in tracked_persons.keys()])
                cv2.putText(vis_frame, ids_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            current_frame = vis_frame
            frame_idx += 1
            
            # Mostrar progreso
            if frame_idx % 50 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Procesando... {progress:.1f}% ({frame_idx}/{total_frames} frames)")
        
        # Mostrar frame
        cv2.imshow('Multi-Person Detection Test', current_frame)
        
        # Controles de teclado
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            print("\n⚠️  Saliendo...")
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                print("⏸️  Pausado")
            else:
                print("▶️  Reanudando")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    print("\n" + "="*80)
    print(" ✅ PRUEBA COMPLETADA")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_multi_person.py <video_path> [num_persons]")
        print("\nEjemplo:")
        print("   python test_multi_person.py data/raw_videos/8yang/duo_video.mp4 2")
        sys.exit(1)
    
    video_path = sys.argv[1]
    num_persons = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    if not Path(video_path).exists():
        print(f"❌ Error: No se encontró el video {video_path}")
        sys.exit(1)
    
    test_multi_person_detection(video_path, num_persons)
