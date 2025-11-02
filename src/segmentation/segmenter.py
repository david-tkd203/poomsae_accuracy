import numpy as np
from typing import Dict, List, Tuple, Any
import scipy.signal as signal

class EnhancedSegmenter:
    """
    Segmentador mejorado para Taekwondo Poomsae que combina:
    - Energía de efectores (manos/pies)
    - Movimiento de tronco y caderas  
    - Cambios angulares (giros)
    - Detección de posturas específicas
    """
    
    def __init__(self, 
                 min_segment_frames=8,      # Más corto para movimientos rápidos
                 max_pause_frames=5,
                 activity_threshold=0.15,   # Más bajo para sensibilidad
                 peak_threshold=0.3,
                 min_peak_distance=10,      # Evitar detecciones muy cercanas
                 smooth_window=5):
        
        self.minlen = min_segment_frames
        self.maxpause = max_pause_frames
        self.activity_thresh = activity_threshold
        self.peak_thresh = peak_threshold
        self.min_peak_dist = min_peak_distance
        self.smooth_win = smooth_window

    def _compute_comprehensive_energy(self, angles_dict: Dict[str, np.ndarray], 
                                    landmarks_dict: Dict[str, np.ndarray], 
                                    fps: float) -> np.ndarray:
        """
        Calcula energía combinando múltiples fuentes:
        1. Cambios angulares (articulaciones)
        2. Velocidad de efectores
        3. Movimiento de tronco
        4. Cambios de orientación
        """
        energies = []
        
        # 1. Energía angular (existente)
        angular_energy = self._angular_energy(angles_dict, fps)
        energies.append(angular_energy)
        
        # 2. Energía de efectores (manos y pies)
        effector_energy = self._effector_energy(landmarks_dict, fps)
        energies.append(effector_energy)
        
        # 3. Energía de tronco (hombros y caderas)
        trunk_energy = self._trunk_energy(landmarks_dict, fps)
        energies.append(trunk_energy)
        
        # 4. Energía de orientación (giros)
        orientation_energy = self._orientation_energy(landmarks_dict, fps)
        energies.append(orientation_energy)
        
        # Combinar todas las energías con pesos
        weights = [0.3, 0.4, 0.2, 0.1]  # Los efectores son más importantes
        total_energy = np.zeros_like(angular_energy)
        
        for energy, weight in zip(energies, weights):
            if len(energy) == len(total_energy):
                # Normalizar cada componente antes de combinar
                if np.max(energy) > 0:
                    energy_norm = energy / np.max(energy)
                    total_energy += weight * energy_norm
        
        return total_energy

    def _angular_energy(self, angles_dict: Dict[str, np.ndarray], fps: float) -> np.ndarray:
        """Energía basada en cambios angulares"""
        if not angles_dict:
            return np.zeros(1000)  # Fallback
            
        grads = []
        for angle_name, angle_series in angles_dict.items():
            if len(angle_series) > 1:
                grad = np.abs(np.gradient(angle_series)) * fps
                grads.append(grad)
        
        if not grads:
            return np.zeros(1000)
            
        # Promedio de gradientes angulares
        return np.mean(grads, axis=0)

    def _effector_energy(self, landmarks_dict: Dict[str, np.ndarray], fps: float) -> np.ndarray:
        """Energía de movimiento de manos y pies"""
        effectors = ['L_WRIST', 'R_WRIST', 'L_ANKLE', 'R_ANKLE']
        energies = []
        
        for effector in effectors:
            if effector in landmarks_dict and len(landmarks_dict[effector]) > 1:
                pos = landmarks_dict[effector]
                velocity = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * fps
                energies.append(velocity)
        
        if not energies:
            return np.zeros(1000)
            
        return np.max(energies, axis=0)  # Usar el máximo para sensibilidad

    def _trunk_energy(self, landmarks_dict: Dict[str, np.ndarray], fps: float) -> np.ndarray:
        """Energía de movimiento del tronco (hombros y caderas)"""
        trunk_points = ['L_SHOULDER', 'R_SHOULDER', 'L_HIP', 'R_HIP']
        energies = []
        
        for point in trunk_points:
            if point in landmarks_dict and len(landmarks_dict[point]) > 1:
                pos = landmarks_dict[point]
                velocity = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * fps
                energies.append(velocity)
        
        if not energies:
            return np.zeros(1000)
            
        return np.mean(energies, axis=0)  # Promedio para tronco

    def _orientation_energy(self, landmarks_dict: Dict[str, np.ndarray], fps: float) -> np.ndarray:
        """Energía basada en cambios de orientación (giros)"""
        if 'L_SHOULDER' not in landmarks_dict or 'R_SHOULDER' not in landmarks_dict:
            return np.zeros(1000)
            
        l_shoulder = landmarks_dict['L_SHOULDER']
        r_shoulder = landmarks_dict['R_SHOULDER']
        
        if len(l_shoulder) < 2:
            return np.zeros(1000)
        
        # Vector de hombros para orientación
        shoulder_vec = r_shoulder - l_shoulder
        orientations = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
        
        # Cambio de orientación
        orientation_change = np.abs(np.gradient(orientations)) * fps
        return orientation_change

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Suavizar señal para reducir ruido"""
        if len(signal) < self.smooth_win:
            return signal
            
        window = np.ones(self.smooth_win) / self.smooth_win
        return np.convolve(signal, window, mode='same')

    def _find_peaks_robust(self, energy_signal: np.ndarray) -> List[int]:
        """Encontrar picos robustos en la señal de energía"""
        if len(energy_signal) < 3:
            return []
        
        # Suavizar señal
        smoothed = self._smooth_signal(energy_signal)
        
        # ✅ MEJORADO: Usar percentiles adaptativos para mayor robustez
        q75 = np.percentile(smoothed, 75)
        q25 = np.percentile(smoothed, 25)
        # Usar rango intercuartil (IQR) para mejor adaptación a ruido
        dynamic_threshold = q25 + 0.35 * (q75 - q25)
        
        # Usar el máximo de threshold estático y dinámico
        effective_threshold = max(self.peak_thresh, dynamic_threshold * 0.5)
        
        # ✅ MEJORADO: Parámetros menos restrictivos para capturar más picos
        peaks, properties = signal.find_peaks(
            smoothed, 
            height=effective_threshold,      # Usar threshold adaptativo
            distance=self.min_peak_distance,
            prominence=max(0.05, effective_threshold * 0.3)  # Prominencia adaptativa
        )
        
        return peaks.tolist()

    def _expand_segment_around_peak(self, energy_signal: np.ndarray, peak_idx: int, 
                                  total_frames: int) -> Tuple[int, int]:
        """Expandir segmento alrededor de un pico"""
        
        # ✅ MEJORADO: Usar percentiles para threshold local más robusto
        q10 = np.percentile(energy_signal, 10)
        q50 = np.percentile(energy_signal, 50)
        local_thresh = q10 + 0.3 * (q50 - q10)
        
        # Buscar inicio hacia atrás
        start = peak_idx
        while start > 0 and energy_signal[start] > local_thresh:
            start -= 1
        start = max(0, start)
        
        # Buscar fin hacia adelante  
        end = peak_idx
        while end < total_frames - 1 and energy_signal[end] > local_thresh:
            end += 1
        end = min(total_frames - 1, end)
        
        # Asegurar longitud mínima
        if (end - start) < self.minlen:
            expansion = self.minlen - (end - start)
            start = max(0, start - expansion // 2)
            end = min(total_frames - 1, end + expansion - (start - max(0, start - expansion // 2)))
        
        return start, end

    def find_segments(self, angles_dict: Dict[str, np.ndarray], 
                     landmarks_dict: Dict[str, np.ndarray], 
                     fps: float) -> List[Tuple[int, int]]:
        """
        Encuentra segmentos usando energía comprehensiva
        """
        if not angles_dict and not landmarks_dict:
            return []
        
        # Calcular energía comprehensiva
        energy = self._compute_comprehensive_energy(angles_dict, landmarks_dict, fps)
        
        if len(energy) == 0:
            return []
        
        # Encontrar picos de actividad
        peaks = self._find_peaks_robust(energy)
        
        # Crear segmentos alrededor de cada pico
        segments = []
        used_frames = set()
        
        for peak in peaks:
            start, end = self._expand_segment_around_peak(energy, peak, len(energy))
            
            # Verificar si se superpone con segmentos existentes
            overlap = False
            for seg_start, seg_end in segments:
                if not (end < seg_start or start > seg_end):
                    overlap = True
                    break
            
            if not overlap and (end - start) >= self.minlen:
                segments.append((start, end))
                # Marcar frames como usados
                used_frames.update(range(start, end + 1))
        
        # Ordenar segmentos por tiempo
        segments.sort(key=lambda x: x[0])
        
        # Fusionar segmentos cercanos
        merged_segments = []
        for seg in segments:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                last_seg = merged_segments[-1]
                # Fusionar si están muy cerca
                if seg[0] - last_seg[1] <= self.maxpause:
                    merged_segments[-1] = (last_seg[0], seg[1])
                else:
                    merged_segments.append(seg)
        
        print(f"[SEGMENTER] Detectados {len(merged_segments)} segmentos")
        return merged_segments

    def debug_energy(self, angles_dict: Dict[str, np.ndarray],
                    landmarks_dict: Dict[str, np.ndarray], 
                    fps: float) -> Dict[str, Any]:
        """
        Función de debug para analizar componentes de energía
        """
        energy_components = {}
        
        # Calcular componentes individuales
        energy_components['angular'] = self._angular_energy(angles_dict, fps)
        energy_components['effector'] = self._effector_energy(landmarks_dict, fps) 
        energy_components['trunk'] = self._trunk_energy(landmarks_dict, fps)
        energy_components['orientation'] = self._orientation_energy(landmarks_dict, fps)
        
        # Energía total
        energy_components['total'] = self._compute_comprehensive_energy(
            angles_dict, landmarks_dict, fps)
        
        # Picos detectados
        energy_components['peaks'] = self._find_peaks_robust(energy_components['total'])
        
        return energy_components