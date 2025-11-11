#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación de Múltiples Personas en el Mismo Video
===================================================

Detecta, rastrea y evalúa a múltiples personas ejecutando Poomsae
simultáneamente en el mismo video.

Genera:
- Landmarks separados por persona (CSV)
- Reportes de evaluación individuales (Excel)
- Video con visualización de ambas personas
- Estadísticas comparativas

Uso:
    python evaluar_multiples_personas.py <video_path> [--num-persons 2] [--output-dir results]

Ejemplo:
    python evaluar_multiples_personas.py data/raw_videos/8yang/duo_001.mp4 --num-persons 2
"""

import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List
import logging

# Importar módulos del proyecto
sys.path.append(str(Path(__file__).parent))
from src.pose.multi_person_backend import MultiPersonPoseDetector, visualize_multi_person
from src.eval.patterns import load_spec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiPersonEvaluator:
    """Evalúa múltiples personas ejecutando Poomsae simultáneamente"""
    
    def __init__(
        self,
        video_path: str,
        num_persons: int = 2,
        output_dir: str = "results_multi_person",
        config_path: str = "config/default.yaml"
    ):
        """
        Inicializa el evaluador de múltiples personas
        
        Args:
            video_path: Ruta al video a procesar
            num_persons: Número de personas a detectar
            output_dir: Directorio de salida
            config_path: Ruta al archivo de configuración
        """
        self.video_path = Path(video_path)
        self.num_persons = num_persons
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Cargar configuración (simplificado para evitar dependencias)
        self.config = {}
        
        # Inicializar detector
        self.detector = MultiPersonPoseDetector(
            max_num_persons=num_persons,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Cargar especificación del Poomsae (simplificado)
        self.spec = {}
        
        # Almacenar landmarks por persona
        self.person_landmarks = defaultdict(list)  # {person_id: [frame_data]}
        
        # Estadísticas
        self.stats = {
            'total_frames': 0,
            'persons_detected': defaultdict(int),
            'simultaneous_detections': 0
        }
    
    def process_video(self) -> Dict[int, Path]:
        """
        Procesa el video completo y genera reportes por persona
        
        Returns:
            Dict con {person_id: report_path}
        """
        logger.info(f"🎬 Procesando video: {self.video_path}")
        logger.info(f"   Detectando hasta {self.num_persons} personas")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {self.video_path}")
        
        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"   📹 {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Preparar video de salida con visualización
        output_video_path = self.output_dir / f"{self.video_path.stem}_multi_person.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        
        # Procesar cada frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar personas
            tracked_persons = self.detector.process_frame(frame)
            
            # Guardar landmarks de cada persona
            for person_id, person_data in tracked_persons.items():
                self.person_landmarks[person_id].append({
                    'frame': frame_idx,
                    'time_s': frame_idx / fps,
                    'landmarks': person_data['landmarks'],
                    'bbox': person_data['bbox'],
                    'confidence': person_data['confidence']
                })
                self.stats['persons_detected'][person_id] += 1
            
            # Actualizar estadísticas
            if len(tracked_persons) >= 2:
                self.stats['simultaneous_detections'] += 1
            
            # Visualizar
            vis_frame = visualize_multi_person(frame, tracked_persons)
            
            # Agregar información en pantalla
            info_text = f"Frame: {frame_idx}/{total_frames} | Personas: {len(tracked_persons)}"
            cv2.putText(vis_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out_video.write(vis_frame)
            
            frame_idx += 1
            
            # Mostrar progreso
            if frame_idx % 100 == 0:
                logger.info(f"   Procesado {frame_idx}/{total_frames} frames...")
        
        cap.release()
        out_video.release()
        
        self.stats['total_frames'] = frame_idx
        
        logger.info(f"✅ Video procesado: {frame_idx} frames")
        logger.info(f"   📊 Personas detectadas: {len(self.person_landmarks)}")
        for person_id, count in self.stats['persons_detected'].items():
            pct = (count / frame_idx) * 100
            logger.info(f"      • Persona {person_id}: {count} frames ({pct:.1f}%)")
        logger.info(f"   👥 Detecciones simultáneas: {self.stats['simultaneous_detections']} frames")
        logger.info(f"   🎥 Video guardado: {output_video_path}")
        
        # Guardar landmarks de cada persona
        report_paths = {}
        for person_id in self.person_landmarks.keys():
            logger.info(f"\n📝 Generando reporte para Persona {person_id}...")
            report_path = self._generate_person_report(person_id, fps)
            report_paths[person_id] = report_path
        
        # Generar reporte comparativo
        self._generate_comparative_report(report_paths)
        
        return report_paths
    
    def _generate_person_report(self, person_id: int, fps: float) -> Path:
        """
        Genera reporte individual de una persona
        
        Args:
            person_id: ID de la persona
            fps: FPS del video
            
        Returns:
            Path al reporte generado
        """
        landmarks_data = self.person_landmarks[person_id]
        
        if len(landmarks_data) < 30:
            logger.warning(f"⚠️  Persona {person_id} tiene muy pocos frames ({len(landmarks_data)}), omitiendo reporte")
            return None
        
        # Guardar landmarks a CSV
        csv_path = self.output_dir / f"persona_{person_id}_landmarks.csv"
        self._save_landmarks_csv(landmarks_data, csv_path, fps)
        logger.info(f"   ✅ Landmarks guardados: {csv_path}")
        
        # Evaluar movimientos (usando la lógica existente del proyecto)
        # NOTA: Aquí deberías integrar con tu pipeline de evaluación existente
        # Por ahora solo creo un reporte básico
        
        report_path = self.output_dir / f"persona_{person_id}_reporte.xlsx"
        self._create_basic_report(person_id, landmarks_data, report_path)
        logger.info(f"   ✅ Reporte generado: {report_path}")
        
        return report_path
    
    def _save_landmarks_csv(self, landmarks_data: List[Dict], csv_path: Path, fps: float):
        """Guarda landmarks en formato CSV compatible con el pipeline existente"""
        rows = []
        
        for frame_data in landmarks_data:
            frame_idx = frame_data['frame']
            time_s = frame_data['time_s']
            landmarks = frame_data['landmarks']
            
            row = {
                'frame': frame_idx,
                'time_s': time_s,
            }
            
            # Agregar cada landmark
            for lm_id in range(33):  # MediaPipe tiene 33 landmarks
                if lm_id in landmarks:
                    x, y, visibility = landmarks[lm_id]
                    row[f'lm{lm_id}_x'] = x
                    row[f'lm{lm_id}_y'] = y
                    row[f'lm{lm_id}_vis'] = visibility
                else:
                    row[f'lm{lm_id}_x'] = 0.0
                    row[f'lm{lm_id}_y'] = 0.0
                    row[f'lm{lm_id}_vis'] = 0.0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    
    def _create_basic_report(self, person_id: int, landmarks_data: List[Dict], report_path: Path):
        """Crea un reporte Excel básico"""
        
        # Resumen básico
        resumen_data = {
            'persona_id': [person_id],
            'frames_detectados': [len(landmarks_data)],
            'tiempo_total_s': [landmarks_data[-1]['time_s'] if landmarks_data else 0],
            'confianza_promedio': [np.mean([d['confidence'] for d in landmarks_data])],
            'video_origen': [str(self.video_path)],
        }
        df_resumen = pd.DataFrame(resumen_data)
        
        # Detalle por frame
        detalle_data = []
        for frame_data in landmarks_data:
            detalle_data.append({
                'frame': frame_data['frame'],
                'time_s': frame_data['time_s'],
                'confidence': frame_data['confidence'],
                'bbox_x': frame_data['bbox'][0],
                'bbox_y': frame_data['bbox'][1],
                'bbox_w': frame_data['bbox'][2],
                'bbox_h': frame_data['bbox'][3],
            })
        df_detalle = pd.DataFrame(detalle_data)
        
        # Guardar Excel
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            df_resumen.to_excel(writer, sheet_name='resumen', index=False)
            df_detalle.to_excel(writer, sheet_name='detalle', index=False)
    
    def _generate_comparative_report(self, report_paths: Dict[int, Path]):
        """Genera reporte comparativo entre todas las personas"""
        logger.info("\n📊 Generando reporte comparativo...")
        
        comp_data = []
        for person_id, landmarks_data in self.person_landmarks.items():
            if len(landmarks_data) < 30:
                continue
            
            comp_data.append({
                'persona_id': person_id,
                'frames_detectados': len(landmarks_data),
                'porcentaje_presencia': (len(landmarks_data) / self.stats['total_frames']) * 100,
                'confianza_promedio': np.mean([d['confidence'] for d in landmarks_data]),
                'confianza_min': np.min([d['confidence'] for d in landmarks_data]),
                'confianza_max': np.max([d['confidence'] for d in landmarks_data]),
            })
        
        if not comp_data:
            logger.warning("⚠️  No hay suficientes datos para reporte comparativo")
            return
        
        df_comp = pd.DataFrame(comp_data)
        
        comp_path = self.output_dir / "reporte_comparativo.xlsx"
        with pd.ExcelWriter(comp_path, engine='openpyxl') as writer:
            df_comp.to_excel(writer, sheet_name='comparacion', index=False)
            
            # Agregar estadísticas generales
            stats_data = {
                'metrica': ['Total Frames', 'Detecciones Simultáneas', 'Porcentaje Simultáneo'],
                'valor': [
                    self.stats['total_frames'],
                    self.stats['simultaneous_detections'],
                    (self.stats['simultaneous_detections'] / self.stats['total_frames'] * 100) if self.stats['total_frames'] > 0 else 0
                ]
            }
            df_stats = pd.DataFrame(stats_data)
            df_stats.to_excel(writer, sheet_name='estadisticas', index=False)
        
        logger.info(f"   ✅ Reporte comparativo: {comp_path}")
    
    def cleanup(self):
        """Libera recursos"""
        self.detector.release()


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa múltiples personas ejecutando Poomsae en el mismo video"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Ruta al video a procesar"
    )
    parser.add_argument(
        "--num-persons",
        type=int,
        default=2,
        help="Número máximo de personas a detectar (default: 2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_multi_person",
        help="Directorio de salida (default: results_multi_person)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Archivo de configuración (default: config/default.yaml)"
    )
    
    args = parser.parse_args()
    
    # Verificar que el video existe
    if not Path(args.video_path).exists():
        print(f"❌ Error: No se encontró el video {args.video_path}")
        sys.exit(1)
    
    print("="*100)
    print(" 👥 EVALUACIÓN DE MÚLTIPLES PERSONAS EN EL MISMO VIDEO")
    print("="*100)
    print()
    
    # Crear evaluador
    evaluator = MultiPersonEvaluator(
        video_path=args.video_path,
        num_persons=args.num_persons,
        output_dir=args.output_dir,
        config_path=args.config
    )
    
    try:
        # Procesar video
        report_paths = evaluator.process_video()
        
        print()
        print("="*100)
        print(" ✅ PROCESAMIENTO COMPLETADO")
        print("="*100)
        print()
        print(f"📁 Resultados guardados en: {Path(args.output_dir).absolute()}")
        print()
        print("📄 Reportes generados:")
        for person_id, report_path in report_paths.items():
            if report_path:
                print(f"   • Persona {person_id}: {report_path.name}")
        print()
        print("💡 Próximos pasos:")
        print("   1. Revisa el video de salida para verificar la detección")
        print("   2. Usa los archivos CSV de landmarks para análisis detallado")
        print("   3. Integra con el pipeline de evaluación existente para scoring completo")
        print("="*100)
        
    except KeyboardInterrupt:
        print("\n⚠️  Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error durante el procesamiento: {e}", exc_info=True)
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
