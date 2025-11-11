#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend de Pose para Múltiples Personas
========================================

Detecta y rastrea múltiples personas en el mismo frame usando MediaPipe.
Permite evaluar a 2+ personas ejecutando Poomsae simultáneamente.

Características:
- Detección de múltiples esqueletos por frame
- Tracking persistente de cada persona
- Separación de landmarks por persona
- Filtrado de ruido y falsos positivos
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MultiPersonPoseDetector:
    """Detecta y rastrea múltiples personas en video usando MediaPipe"""
    
    def __init__(
        self,
        max_num_persons: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Inicializa el detector de múltiples personas
        
        Args:
            max_num_persons: Número máximo de personas a detectar
            min_detection_confidence: Confianza mínima para detección
            min_tracking_confidence: Confianza mínima para tracking
            model_complexity: Complejidad del modelo (0=lite, 1=full, 2=heavy)
        """
        self.max_num_persons = max_num_persons
        self.mp_pose = mp.solutions.pose
        
        # Crear un detector por persona
        self.pose_detectors = []
        for _ in range(max_num_persons):
            detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.pose_detectors.append(detector)
        
        # Tracking de personas
        self.person_tracks = {}  # {person_id: {'center': (x, y), 'landmarks': [], 'last_seen': frame_idx}}
        self.next_person_id = 0
        self.current_frame_idx = 0
        
        # Parámetros de tracking
        self.max_distance_threshold = 0.3  # Distancia máxima para asociar detecciones
        self.max_frames_lost = 10  # Frames sin detección antes de perder track
        
    def process_frame(self, frame: np.ndarray) -> Dict[int, Dict]:
        """
        Procesa un frame y detecta múltiples personas
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            Dict con {person_id: {'landmarks': landmarks, 'bbox': bbox, 'confidence': conf}}
        """
        self.current_frame_idx += 1
        
        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Detectar personas usando segmentación espacial
        detections = self._detect_multiple_persons(frame_rgb, w, h)
        
        # Asociar detecciones con tracks existentes
        tracked_persons = self._associate_detections(detections)
        
        # Limpiar tracks perdidos
        self._cleanup_lost_tracks()
        
        return tracked_persons
    
    def _detect_multiple_persons(self, frame_rgb: np.ndarray, w: int, h: int) -> List[Dict]:
        """
        Detecta múltiples personas dividiendo el frame o usando múltiples detecciones
        
        Returns:
            Lista de detecciones: [{'landmarks': landmarks, 'center': (x, y), 'bbox': bbox}]
        """
        detections = []
        
        # Estrategia 1: Dividir frame verticalmente si se esperan 2 personas lado a lado
        if self.max_num_persons == 2:
            # Dividir frame en mitad izquierda y derecha
            mid_x = w // 2
            
            # Detectar en mitad izquierda
            left_half = frame_rgb[:, :mid_x, :]
            left_result = self.pose_detectors[0].process(left_half)
            
            if left_result.pose_landmarks:
                landmarks = self._mediapipe_to_dict(left_result.pose_landmarks, offset_x=0)
                center = self._calculate_center(landmarks)
                bbox = self._calculate_bbox(landmarks, w, h)
                
                detections.append({
                    'landmarks': landmarks,
                    'center': center,
                    'bbox': bbox,
                    'confidence': self._calculate_confidence(left_result.pose_landmarks)
                })
            
            # Detectar en mitad derecha
            right_half = frame_rgb[:, mid_x:, :]
            right_result = self.pose_detectors[1].process(right_half)
            
            if right_result.pose_landmarks:
                landmarks = self._mediapipe_to_dict(right_result.pose_landmarks, offset_x=mid_x/w)
                center = self._calculate_center(landmarks)
                bbox = self._calculate_bbox(landmarks, w, h)
                
                detections.append({
                    'landmarks': landmarks,
                    'center': center,
                    'bbox': bbox,
                    'confidence': self._calculate_confidence(right_result.pose_landmarks)
                })
        
        else:
            # Para más de 2 personas, intentar detección completa
            result = self.pose_detectors[0].process(frame_rgb)
            
            if result.pose_landmarks:
                landmarks = self._mediapipe_to_dict(result.pose_landmarks)
                center = self._calculate_center(landmarks)
                bbox = self._calculate_bbox(landmarks, w, h)
                
                detections.append({
                    'landmarks': landmarks,
                    'center': center,
                    'bbox': bbox,
                    'confidence': self._calculate_confidence(result.pose_landmarks)
                })
        
        return detections
    
    def _mediapipe_to_dict(self, mp_landmarks, offset_x: float = 0) -> Dict[int, Tuple[float, float, float]]:
        """
        Convierte landmarks de MediaPipe a diccionario
        
        Args:
            mp_landmarks: Landmarks de MediaPipe
            offset_x: Offset horizontal (para frames divididos)
            
        Returns:
            Dict con {landmark_id: (x, y, visibility)}
        """
        landmarks = {}
        for idx, landmark in enumerate(mp_landmarks.landmark):
            landmarks[idx] = (
                landmark.x + offset_x,
                landmark.y,
                landmark.visibility
            )
        return landmarks
    
    def _calculate_center(self, landmarks: Dict) -> Tuple[float, float]:
        """Calcula el centro de masa de los landmarks"""
        if not landmarks:
            return (0.5, 0.5)
        
        x_coords = [lm[0] for lm in landmarks.values()]
        y_coords = [lm[1] for lm in landmarks.values()]
        
        return (np.mean(x_coords), np.mean(y_coords))
    
    def _calculate_bbox(self, landmarks: Dict, w: int, h: int) -> Tuple[int, int, int, int]:
        """Calcula bounding box (x, y, width, height)"""
        if not landmarks:
            return (0, 0, w, h)
        
        x_coords = [lm[0] * w for lm in landmarks.values()]
        y_coords = [lm[1] * h for lm in landmarks.values()]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _calculate_confidence(self, mp_landmarks) -> float:
        """Calcula confianza promedio de los landmarks"""
        if not mp_landmarks:
            return 0.0
        
        visibilities = [lm.visibility for lm in mp_landmarks.landmark]
        return np.mean(visibilities)
    
    def _associate_detections(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Asocia detecciones con tracks existentes usando distancia espacial
        
        Returns:
            Dict con {person_id: detection_data}
        """
        tracked_persons = {}
        used_detections = set()
        
        # Intentar asociar con tracks existentes
        for person_id, track in list(self.person_tracks.items()):
            if self.current_frame_idx - track['last_seen'] > self.max_frames_lost:
                continue
            
            best_match_idx = None
            min_distance = self.max_distance_threshold
            
            for idx, detection in enumerate(detections):
                if idx in used_detections:
                    continue
                
                # Calcular distancia euclidiana entre centros
                dist = np.sqrt(
                    (detection['center'][0] - track['center'][0]) ** 2 +
                    (detection['center'][1] - track['center'][1]) ** 2
                )
                
                if dist < min_distance:
                    min_distance = dist
                    best_match_idx = idx
            
            # Si encontramos match, actualizar track
            if best_match_idx is not None:
                detection = detections[best_match_idx]
                self.person_tracks[person_id] = {
                    'center': detection['center'],
                    'landmarks': detection['landmarks'],
                    'bbox': detection['bbox'],
                    'last_seen': self.current_frame_idx,
                    'confidence': detection['confidence']
                }
                tracked_persons[person_id] = detection
                used_detections.add(best_match_idx)
        
        # Crear nuevos tracks para detecciones no asociadas
        for idx, detection in enumerate(detections):
            if idx not in used_detections:
                person_id = self.next_person_id
                self.next_person_id += 1
                
                self.person_tracks[person_id] = {
                    'center': detection['center'],
                    'landmarks': detection['landmarks'],
                    'bbox': detection['bbox'],
                    'last_seen': self.current_frame_idx,
                    'confidence': detection['confidence']
                }
                tracked_persons[person_id] = detection
        
        return tracked_persons
    
    def _cleanup_lost_tracks(self):
        """Elimina tracks que no se han visto en mucho tiempo"""
        to_remove = []
        for person_id, track in self.person_tracks.items():
            if self.current_frame_idx - track['last_seen'] > self.max_frames_lost:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_tracks[person_id]
            logger.info(f"Track perdido: Persona {person_id}")
    
    def get_active_persons(self) -> List[int]:
        """Retorna IDs de personas actualmente trackeadas"""
        return [
            person_id for person_id, track in self.person_tracks.items()
            if self.current_frame_idx - track['last_seen'] <= 1
        ]
    
    def release(self):
        """Libera recursos de MediaPipe"""
        for detector in self.pose_detectors:
            detector.close()


def visualize_multi_person(frame: np.ndarray, tracked_persons: Dict[int, Dict]) -> np.ndarray:
    """
    Dibuja landmarks y bounding boxes de múltiples personas
    
    Args:
        frame: Frame BGR
        tracked_persons: Dict con {person_id: detection_data}
        
    Returns:
        Frame con visualizaciones
    """
    h, w = frame.shape[:2]
    vis_frame = frame.copy()
    
    # Colores diferentes por persona
    colors = [
        (0, 255, 0),    # Verde - Persona 0
        (255, 0, 0),    # Azul - Persona 1
        (0, 255, 255),  # Amarillo - Persona 2
        (255, 0, 255),  # Magenta - Persona 3
    ]
    
    for person_id, person_data in tracked_persons.items():
        color = colors[person_id % len(colors)]
        landmarks = person_data['landmarks']
        bbox = person_data['bbox']
        
        # Dibujar bounding box
        x, y, bw, bh = bbox
        cv2.rectangle(vis_frame, (x, y), (x + bw, y + bh), color, 2)
        
        # Etiqueta de persona
        label = f"Persona {person_id}"
        cv2.putText(vis_frame, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Dibujar landmarks
        for lm_id, (lm_x, lm_y, visibility) in landmarks.items():
            if visibility > 0.5:
                px, py = int(lm_x * w), int(lm_y * h)
                cv2.circle(vis_frame, (px, py), 3, color, -1)
        
        # Dibujar conexiones (esqueleto simplificado)
        connections = [
            (11, 12), (11, 13), (13, 15),  # Brazo izquierdo
            (12, 14), (14, 16),  # Brazo derecho
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27), (27, 29),  # Pierna izquierda
            (24, 26), (26, 28), (28, 30),  # Pierna derecha
        ]
        
        for start_idx, end_idx in connections:
            if start_idx in landmarks and end_idx in landmarks:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                if start[2] > 0.5 and end[2] > 0.5:
                    pt1 = (int(start[0] * w), int(start[1] * h))
                    pt2 = (int(end[0] * w), int(end[1] * h))
                    cv2.line(vis_frame, pt1, pt2, color, 2)
    
    return vis_frame
