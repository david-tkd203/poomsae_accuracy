#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Alternativo Kinect (Sin PyK4A)
======================================

Backend de prueba que simula funcionalidad de Kinect para desarrollo
cuando PyK4A no está disponible.
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
import cv2

from pathlib import Path
class PoomsaeMovementAnalyzer:
    """Mock de analizador de movimientos para pruebas automáticas"""
    def __init__(self, poomsae_name="8yang"):
        self.poomsae_name = poomsae_name
    def analyze(self, landmarks, depth):
        # Simula errores biomecánicos y descuentos
        t = time.time()
        errors = []
        discounts = []
        angles = {f"angle_{i}": 150 + 10 * np.sin(t + i) for i in range(5)}
        if np.sin(t) > 0.5:
            errors.append("Desbalance")
            discounts.append(0.1)
        if np.cos(t) < -0.5:
            errors.append("Postura incorrecta")
            discounts.append(0.1)
        if np.sin(t*0.7) > 0.7:
            errors.append("Alineación grave")
            discounts.append(0.3)
        return { 'discounts': discounts, 'angles': angles, 'errors': errors }

    def draw_overlay(self, frame, results):
        y = 30
        for err in results['errors']:
            cv2.putText(frame, f"Error: {err}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y += 30
        for k, v in results['angles'].items():
            cv2.putText(frame, f"{k}: {v:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
            y += 22
        if results['discounts']:
            cv2.putText(frame, f"Descuento total: {sum(results['discounts']):.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)

class KinectPoomsaeCaptureMock:
    """Mock de KinectPoomsaeCapture para desarrollo sin hardware"""
    
    def __init__(self, **kwargs):
        print("⚠️  Usando Mock de Kinect (PyK4A no disponible)")
        self.frame_count = 0
        self.start_time = time.time()
        
        # Usar webcam como backup si está disponible
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                print("📹 Usando webcam como fuente de video")
                self.has_camera = True
            else:
                self.has_camera = False
        except:
            self.has_camera = False
            
        if not self.has_camera:
            print("📺 Generando frames sintéticos")
    
    def initialize(self) -> bool:
        """Inicialización mock"""
        print("✅ Mock Kinect inicializado")
        return True
    
    def get_frame(self) -> Optional[Dict[str, Any]]:
        """Genera frame mock con datos simulados"""
        self.frame_count += 1
        timestamp = time.time()
        
        # Generar frame RGB
        if self.has_camera and self.cap.isOpened():
            ret, rgb_frame = self.cap.read()
            if not ret:
                rgb_frame = self._generate_synthetic_frame()
        else:
            rgb_frame = self._generate_synthetic_frame()
        
        # Generar depth sintético
        h, w = rgb_frame.shape[:2]
        depth_frame = np.random.randint(500, 2000, (h, w), dtype=np.uint16)
        
        # Landmarks simulados (formato MediaPipe)
        landmarks = self._generate_synthetic_landmarks()
        
        return {
            'rgb': rgb_frame,
            'depth': depth_frame,
            'landmarks': landmarks,
            'pose_confidence': 0.85 + 0.1 * np.sin(timestamp),
            'timestamp': timestamp,
            'frame_number': self.frame_count
        }
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """Genera frame RGB sintético"""
        # Frame de 640x480 con gradiente y figura humana básica
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Gradiente de fondo
        for y in range(480):
            frame[y, :, 0] = int(50 + 100 * y / 480)  # Azul
            frame[y, :, 1] = int(30 + 80 * y / 480)   # Verde
        
        # Figura humana simple
        center_x, center_y = 320, 240
        
        # Cabeza
        cv2.circle(frame, (center_x, center_y - 100), 30, (255, 200, 150), -1)
        
        # Torso
        cv2.rectangle(frame, (center_x - 40, center_y - 60), 
                     (center_x + 40, center_y + 60), (100, 150, 200), -1)
        
        # Brazos
        cv2.line(frame, (center_x - 40, center_y - 20), 
                (center_x - 80, center_y + 20), (255, 200, 150), 8)
        cv2.line(frame, (center_x + 40, center_y - 20), 
                (center_x + 80, center_y + 20), (255, 200, 150), 8)
        
        # Piernas
        cv2.line(frame, (center_x - 20, center_y + 60), 
                (center_x - 30, center_y + 150), (100, 100, 200), 8)
        cv2.line(frame, (center_x + 20, center_y + 60), 
                (center_x + 30, center_y + 150), (100, 100, 200), 8)
        
        return frame
    
    def _generate_synthetic_landmarks(self) -> Dict[int, Tuple[float, float, float, float]]:
        """Genera landmarks sintéticos"""
        landmarks = {}
        
        # Posiciones base normalizadas (similar a MediaPipe)
        base_positions = {
            0: (0.5, 0.15),    # Cabeza
            11: (0.4, 0.35),   # Hombro izquierdo
            12: (0.6, 0.35),   # Hombro derecho
            13: (0.35, 0.45),  # Codo izquierdo
            14: (0.65, 0.45),  # Codo derecho
            15: (0.3, 0.55),   # Muñeca izquierda
            16: (0.7, 0.55),   # Muñeca derecha
            23: (0.42, 0.6),   # Cadera izquierda
            24: (0.58, 0.6),   # Cadera derecha
            25: (0.4, 0.75),   # Rodilla izquierda
            26: (0.6, 0.75),   # Rodilla derecha
            27: (0.38, 0.9),   # Tobillo izquierdo
            28: (0.62, 0.9),   # Tobillo derecho
        }
        
        # Agregar variación temporal
        t = time.time() * 0.5
        
        for lm_id, (base_x, base_y) in base_positions.items():
            # Agregar pequeño movimiento
            x = base_x + 0.02 * np.sin(t + lm_id)
            y = base_y + 0.01 * np.cos(t * 1.2 + lm_id)
            z = -0.5 + 0.1 * np.sin(t * 0.8 + lm_id)  # Profundidad simulada
            visibility = 0.8 + 0.15 * np.sin(t * 2 + lm_id)
            
            landmarks[lm_id] = (x, y, z, max(0.1, min(0.99, visibility)))
        
        return landmarks
    
    def visualize_frame(self, frame_data: Dict[str, Any]) -> np.ndarray:
        """Visualiza frame con información mock"""
        rgb_frame = frame_data['rgb'].copy()
        depth_frame = frame_data['depth']
        landmarks = frame_data['landmarks']
        
        # Dibujar landmarks si están disponibles
        if landmarks:
            h, w = rgb_frame.shape[:2]
            
            for lm_id, (x, y, z, visibility) in landmarks.items():
                if visibility > 0.5:
                    px, py = int(x * w), int(y * h)
                    
                    # Color basado en profundidad
                    depth_color = int(255 * min(1.0, abs(z)))
                    color = (0, 255 - depth_color, depth_color)
                    
                    cv2.circle(rgb_frame, (px, py), 5, color, -1)
                    
                    # Mostrar ID para landmarks principales
                    if lm_id in [0, 11, 12, 15, 16, 23, 24, 27, 28]:
                        cv2.putText(rgb_frame, str(lm_id), (px + 7, py),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Crear depth visualizable
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_frame, alpha=0.08), cv2.COLORMAP_JET
        )
        
        # Redimensionar depth
        if depth_colormap.shape[:2] != rgb_frame.shape[:2]:
            depth_colormap = cv2.resize(depth_colormap, (rgb_frame.shape[1], rgb_frame.shape[0]))
        
        # Combinar lado a lado
        combined = np.hstack((rgb_frame, depth_colormap))
        
        # Información en pantalla
        info_text = [
            f"MOCK KINECT - Frame: {frame_data['frame_number']}",
            f"Pose Confidence: {frame_data['pose_confidence']:.2f}",
            f"Landmarks: {len(landmarks)}",
            "PyK4A no disponible - usando simulacion"
        ]
        
        y_pos = 30
        for text in info_text:
            cv2.putText(combined, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 25
        
        return combined
    
    def start_recording(self, output_dir: str = "mock_recordings"):
        """Mock de grabación"""
        print(f"🔴 Mock: Iniciando grabación en {output_dir}")
        self.recording = True
    
    def stop_recording(self):
        """Mock de parar grabación"""
        print("⏹️  Mock: Grabación detenida")
        self.recording = False
        return Path("mock_recordings")
    
    def get_stats(self) -> Dict[str, Any]:
        """Stats mock"""
        elapsed = time.time() - self.start_time
        return {
            'frames_captured': self.frame_count,
            'frames_with_pose': self.frame_count,
            'avg_fps': self.frame_count / elapsed if elapsed > 0 else 0,
            'start_time': self.start_time
        }
    
    def cleanup(self):
        """Limpieza mock"""
        if self.has_camera and hasattr(self, 'cap'):
            self.cap.release()
        print("🔄 Mock Kinect limpiado")

# Función de detección mock
def detect_kinect_device() -> bool:
    """Mock de detección - siempre retorna True para permitir desarrollo"""
    print("⚠️  Mock: Simulando Kinect detectada")
    return True

# Alias para mantener compatibilidad
KinectPoomsaeCapture = KinectPoomsaeCaptureMock

def calibrate_poomsae_area(capture_device) -> Dict[str, Any]:
    """Mock de calibración"""
    print("🎯 Mock: Calibración simulada")
    return {
        'center_position': (0.5, 0.4, 0.8),
        'area_bounds': {'width': 0.6, 'height': 0.8},
        'distance_to_sensor': 1.5
    }
