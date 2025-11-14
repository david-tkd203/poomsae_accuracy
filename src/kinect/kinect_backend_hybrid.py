#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Híbrido Kinect - OpenCV + MediaPipe
===========================================

Este backend usa webcam para RGB y simula depth data, 
proporcionando funcionalidad similar a Kinect para desarrollo.
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

class PoomsaeMovementAnalyzer:
    """Analizador de movimientos para poomsae Pal Yang usando landmarks y depth"""
    def __init__(self, poomsae_name="8yang"):
        self.poomsae_name = poomsae_name
        # Definición de articulaciones clave (MediaPipe)
        self.joints = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        # Reglas de descuento
        self.rules = [
            { 'name': 'Desbalance', 'func': self.detect_balance_error, 'discount': 0.1 },
            { 'name': 'Postura incorrecta', 'func': self.detect_posture_error, 'discount': 0.1 },
            { 'name': 'Falta de extensión', 'func': self.detect_extension_error, 'discount': 0.1 },
            { 'name': 'Alineación grave', 'func': self.detect_grave_alignment_error, 'discount': 0.3 }
        ]

    def analyze(self, landmarks: dict, depth_frame: np.ndarray) -> dict:
        """Analiza biomecánica y calcula descuentos, robusto ante landmarks faltantes"""
        results = { 'discounts': [], 'angles': {}, 'errors': [] }
        # Calcular ángulos articulares
        for joint in ['elbow', 'shoulder', 'knee', 'hip', 'ankle']:
            for side in ['left', 'right']:
                name = f"{side}_{joint}"
                idxs = self._get_angle_indices(joint, side)
                # Validar que los índices existen y los landmarks no son None
                if (
                    len(idxs) == 3 and
                    all(i in landmarks and landmarks[i] is not None for i in idxs)
                ):
                    try:
                        angle = self._calculate_angle(
                            landmarks[idxs[0]], landmarks[idxs[1]], landmarks[idxs[2]]
                        )
                        if not (0 <= angle <= 180):
                            continue  # descartar ángulos fuera de rango
                        results['angles'][name] = angle
                    except Exception as e:
                        # Si ocurre error, ignorar ese ángulo
                        continue
        # Aplicar reglas de descuento
        for rule in self.rules:
            try:
                if rule['func'](landmarks, results['angles'], depth_frame):
                    results['discounts'].append(rule['discount'])
                    results['errors'].append(rule['name'])
            except Exception:
                continue
        return results

    def _get_angle_indices(self, joint, side):
        # Devuelve los índices de los landmarks para el ángulo
        if joint == 'elbow':
            return [self.joints[f'{side}_shoulder'], self.joints[f'{side}_elbow'], self.joints[f'{side}_wrist']]
        if joint == 'shoulder':
            return [self.joints[f'{side}_hip'], self.joints[f'{side}_shoulder'], self.joints[f'{side}_elbow']]
        if joint == 'knee':
            return [self.joints[f'{side}_hip'], self.joints[f'{side}_knee'], self.joints[f'{side}_ankle']]
        if joint == 'hip':
            return [self.joints[f'{side}_shoulder'], self.joints[f'{side}_hip'], self.joints[f'{side}_knee']]
        if joint == 'ankle':
            return [self.joints[f'{side}_knee'], self.joints[f'{side}_ankle'], self.joints[f'{side}_foot']] if f'{side}_foot' in self.joints else []
        return []

    def _calculate_angle(self, a, b, c):
        # Ángulo entre tres puntos (en radianes), robusto ante valores nulos
        try:
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            cx, cy = c[0], c[1]
            ba = np.array([ax-bx, ay-by])
            bc = np.array([cx-bx, cy-by])
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            if norm_ba == 0 or norm_bc == 0:
                return None
            cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        except Exception:
            return None

    def detect_balance_error(self, landmarks, angles, depth):
        # Desbalance: diferencia de altura entre hombros/caderas
        try:
            l_sh = landmarks[self.joints['left_shoulder']][1]
            r_sh = landmarks[self.joints['right_shoulder']][1]
            l_hip = landmarks[self.joints['left_hip']][1]
            r_hip = landmarks[self.joints['right_hip']][1]
            return abs(l_sh - r_sh) > 0.08 or abs(l_hip - r_hip) > 0.08
        except:
            return False

    def detect_posture_error(self, landmarks, angles, depth):
        # Postura incorrecta: espalda muy inclinada
        try:
            l_sh = landmarks[self.joints['left_shoulder']]
            r_sh = landmarks[self.joints['right_shoulder']]
            l_hip = landmarks[self.joints['left_hip']]
            r_hip = landmarks[self.joints['right_hip']]
            torso_angle = self._calculate_angle(l_sh, l_hip, r_hip)
            return torso_angle < 150
        except:
            return False

    def detect_extension_error(self, landmarks, angles, depth):
        # Falta de extensión: codo o rodilla muy flexionado
        for k in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
            if k in angles and angles[k] < 160:
                return True
        return False

    def detect_grave_alignment_error(self, landmarks, angles, depth):
        # Grave: ángulo de cadera o rodilla fuera de rango
        for k in ['left_hip', 'right_hip', 'left_knee', 'right_knee']:
            if k in angles and (angles[k] < 120 or angles[k] > 180):
                return True
        return False

    def draw_overlay(self, frame, results):
        # Overlay de errores y ángulos
        y = 30
        for err in results['errors']:
            cv2.putText(frame, f"Error: {err}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y += 30
        for k, v in results['angles'].items():
            cv2.putText(frame, f"{k}: {v:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
            y += 22
        if results['discounts']:
            cv2.putText(frame, f"Descuento total: {sum(results['discounts']):.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)

class KinectPoomsaeCapture:
    """Captura híbrida usando webcam + depth simulado"""
    
    def __init__(self, **kwargs):
        print("🎥 Iniciando captura híbrida (Webcam + Depth simulado)")
        
        # Configuración
        self.width = 640
        self.height = 480
        self.fps = kwargs.get('fps', 30)
        
        # Captura de video
        self.cap = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # MediaPipe para pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Buffer para grabación
        self.recording = False
        self.recorded_frames = []
        
        # Stats
        self.stats = {
            'frames_captured': 0,
            'frames_with_pose': 0,
            'avg_fps': 0.0,
            'start_time': time.time()
        }
    
    def initialize(self) -> bool:
        """Inicializa la captura"""
        try:
            # Intentar diferentes índices de cámara
            for i in range(3):
                self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows DirectShow
                if self.cap.isOpened():
                    # Configurar resolución
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    
                    # Test de captura
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"✅ Webcam inicializada (índice {i}): {frame.shape}")
                        return True
                    else:
                        self.cap.release()
                        self.cap = None
                        
            print("❌ No se pudo inicializar ninguna webcam")
            return False
            
        except Exception as e:
            print(f"❌ Error inicializando webcam: {e}")
            return False
    
    def get_frame(self) -> Optional[Dict[str, Any]]:
        """Captura frame con RGB + depth simulado + pose"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, rgb_frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        timestamp = time.time()
        
        # Redimensionar si es necesario
        if rgb_frame.shape[:2] != (self.height, self.width):
            rgb_frame = cv2.resize(rgb_frame, (self.width, self.height))
        
        # Generar depth simulado basado en imagen
        depth_frame = self._generate_depth_from_rgb(rgb_frame)
        
        # Detectar pose
        rgb_for_pose = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_for_pose)
        
        landmarks = {}
        pose_confidence = 0.0
        
        if pose_results.pose_landmarks:
            landmarks = self._extract_landmarks(pose_results.pose_landmarks)
            pose_confidence = self._calculate_confidence(pose_results.pose_landmarks)
            self.stats['frames_with_pose'] += 1
        
        self.stats['frames_captured'] += 1
        
        # Calcular FPS
        elapsed = timestamp - self.stats['start_time']
        self.stats['avg_fps'] = self.stats['frames_captured'] / elapsed if elapsed > 0 else 0
        
        frame_data = {
            'rgb': rgb_frame,
            'depth': depth_frame,
            'landmarks': landmarks,
            'pose_confidence': pose_confidence,
            'timestamp': timestamp,
            'frame_number': self.frame_count
        }
        
        # Guardar para grabación si está activa
        if self.recording:
            self.recorded_frames.append(frame_data.copy())
        
        return frame_data
    
    def _generate_depth_from_rgb(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Genera depth map simulado basado en RGB"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        # Invertir intensidad (objetos claros = cerca, oscuros = lejos)
        inverted = 255 - gray
        
        # Escalar a rango de depth típico (500-4000mm)
        depth = (inverted.astype(np.float32) / 255.0 * 3500 + 500).astype(np.uint16)
        
        # Agregar algo de ruido realista
        noise = np.random.normal(0, 50, depth.shape).astype(np.int16)
        depth = np.clip(depth.astype(np.int32) + noise, 500, 4000).astype(np.uint16)
        
        return depth
    
    def _extract_landmarks(self, mp_landmarks):
        """Extrae landmarks de MediaPipe"""
        landmarks = {}
        for idx, landmark in enumerate(mp_landmarks.landmark):
            landmarks[idx] = (
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            )
        return landmarks
    
    def _calculate_confidence(self, mp_landmarks):
        """Calcula confianza promedio"""
        visibilities = [lm.visibility for lm in mp_landmarks.landmark]
        return np.mean(visibilities)
    
    def visualize_frame(self, frame_data: Dict[str, Any]) -> np.ndarray:
        """Visualiza frame con overlays"""
        rgb_frame = frame_data['rgb'].copy()
        depth_frame = frame_data['depth']
        landmarks = frame_data['landmarks']
        
        # Dibujar landmarks
        if landmarks:
            h, w = rgb_frame.shape[:2]
            for lm_id, (x, y, z, vis) in landmarks.items():
                if vis > 0.5:
                    px, py = int(x * w), int(y * h)
                    # Color basado en profundidad
                    depth_val = depth_frame[py, px] if 0 <= py < h and 0 <= px < w else 1000
                    depth_norm = max(0, min(1, (4000 - depth_val) / 3500))
                    color = (0, int(255 * depth_norm), int(255 * (1 - depth_norm)))
                    cv2.circle(rgb_frame, (px, py), 4, color, -1)
        
        # Depth colormap
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_frame, alpha=0.08), cv2.COLORMAP_JET
        )
        
        # Combinar RGB + Depth
        combined = np.hstack((rgb_frame, depth_colormap))
        
        # Info overlay
        info = [
            f"Hybrid Kinect - Frame {frame_data['frame_number']}",
            f"FPS: {self.stats['avg_fps']:.1f}",
            f"Pose: {frame_data['pose_confidence']:.2f}",
            f"Landmarks: {len(landmarks)}"
        ]
        
        y_pos = 25
        for text in info:
            cv2.putText(combined, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 25
        
        return combined
    
    def start_recording(self, output_dir: str = "hybrid_recordings"):
        """Inicia grabación"""
        self.recording = True
        self.recorded_frames = []
        self.recording_dir = Path(output_dir)
        self.recording_dir.mkdir(exist_ok=True)
        print(f"🔴 Iniciando grabación híbrida: {output_dir}")
    
    def stop_recording(self):
        """Detiene grabación y guarda"""
        if not self.recording or not self.recorded_frames:
            return None
        
        self.recording = False
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Guardar video RGB
        rgb_video = self.recording_dir / f"hybrid_{timestamp}_rgb.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(rgb_video), fourcc, self.fps, 
                                (self.width, self.height))
        
        landmarks_data = []
        
        for frame_data in self.recorded_frames:
            writer.write(frame_data['rgb'])
            
            if frame_data['landmarks']:
                landmarks_data.append({
                    'frame': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'landmarks': frame_data['landmarks'],
                    'confidence': frame_data['pose_confidence']
                })
        
        writer.release()
        
        # Guardar landmarks
        landmarks_file = self.recording_dir / f"hybrid_{timestamp}_landmarks.json"
        with open(landmarks_file, 'w') as f:
            json.dump(landmarks_data, f, indent=2, default=str)
        
        print(f"✅ Grabación guardada: {rgb_video}")
        print(f"📊 Landmarks: {landmarks_file}")
        
        return self.recording_dir
    
    def get_stats(self):
        """Retorna estadísticas"""
        return self.stats.copy()
    
    def cleanup(self):
        """Limpia recursos"""
        if self.cap:
            self.cap.release()
        if hasattr(self.pose, 'close'):
            self.pose.close()
        print("🔄 Webcam híbrida limpiada")

# Funciones de compatibilidad
def detect_kinect_device() -> bool:
    """Simula detección de Kinect"""
    print("🎥 Usando webcam como alternativa a Kinect")
    return True

def calibrate_poomsae_area(capture_device):
    """Calibración simplificada"""
    print("🎯 Calibración híbrida (webcam)")
    return {
        'center_position': (0.5, 0.4, 1.0),
        'area_bounds': {'width': 0.8, 'height': 0.9}
    }
