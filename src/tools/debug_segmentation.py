# src/tools/debug_segmentation.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from segmentation.move_capture import capture_moves_enhanced, load_landmarks_csv, series_xy, LMK

def debug_segmentation(csv_path, video_path=None, output_dir="debug_plots", sensitivity=0.8):
    """Genera gráficos detallados para diagnosticar problemas de segmentación"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"=== DEBUG SEGMENTACIÓN ===")
    print(f"CSV: {csv_path}")
    print(f"Video: {video_path}")
    print(f"Sensibilidad: {sensitivity}")
    
    # Cargar datos directamente para análisis
    df = load_landmarks_csv(Path(csv_path))
    nframes = int(df["frame"].max()) + 1
    
    # Extraer series clave para análisis
    landmarks_dict = {}
    key_points = ['L_WRIST', 'R_WRIST', 'L_ANK', 'R_ANK', 'L_SH', 'R_SH', 'L_HIP', 'R_HIP']
    
    for point in key_points:
        if point in LMK:
            landmarks_dict[point] = series_xy(df, LMK[point], nframes)
    
    # Calcular energías manualmente para debug
    energies = calculate_energy_components(landmarks_dict, fps=30.0)
    
    # Ejecutar captura mejorada
    result = capture_moves_enhanced(
        Path(csv_path), 
        video_path=Path(video_path) if video_path else None,
        sensitivity=sensitivity
    )
    
    # Generar reporte detallado
    generate_energy_plots(energies, result, output_dir)
    generate_segmentation_report(result, output_dir)
    generate_movement_analysis(result, output_dir)
    
    return result, energies

def calculate_energy_components(landmarks_dict, fps=30.0):
    """Calcula componentes individuales de energía para análisis"""
    
    def gradient_energy(positions, fps):
        if len(positions) < 2:
            return np.zeros(1)
        velocity = np.linalg.norm(np.gradient(positions, axis=0), axis=1) * fps
        return velocity
    
    energies = {}
    
    # Energía de efectores individuales
    for effector in ['L_WRIST', 'R_WRIST', 'L_ANK', 'R_ANK']:
        if effector in landmarks_dict:
            energies[effector] = gradient_energy(landmarks_dict[effector], fps)
    
    # Energía máxima de efectores
    if all(e in energies for e in ['L_WRIST', 'R_WRIST', 'L_ANK', 'R_ANK']):
        effector_energies = np.stack([energies['L_WRIST'], energies['R_WRIST'], 
                                    energies['L_ANK'], energies['R_ANK']], axis=1)
        energies['EFFECTORS_MAX'] = np.max(effector_energies, axis=1)
    
    # Energía de tronco
    trunk_points = ['L_SH', 'R_SH', 'L_HIP', 'R_HIP']
    trunk_energies = []
    for point in trunk_points:
        if point in landmarks_dict:
            trunk_energies.append(gradient_energy(landmarks_dict[point], fps))
    
    if trunk_energies:
        energies['TRUNK_MEAN'] = np.mean(trunk_energies, axis=0)
    
    # Energía de orientación (giros)
    if 'L_SH' in landmarks_dict and 'R_SH' in landmarks_dict:
        l_shoulder = landmarks_dict['L_SH']
        r_shoulder = landmarks_dict['R_SH']
        
        if len(l_shoulder) > 1:
            shoulder_vec = r_shoulder - l_shoulder
            orientations = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
            orientation_change = np.abs(np.gradient(orientations)) * fps
            energies['ORIENTATION'] = orientation_change
    
    # Energía combinada (similar a la del segmentador)
    if 'EFFECTORS_MAX' in energies and 'TRUNK_MEAN' in energies:
        # Normalizar componentes
        eff_norm = energies['EFFECTORS_MAX'] / (np.max(energies['EFFECTORS_MAX']) + 1e-9)
        trunk_norm = energies['TRUNK_MEAN'] / (np.max(energies['TRUNK_MEAN']) + 1e-9)
        
        orient_norm = np.zeros_like(eff_norm)
        if 'ORIENTATION' in energies:
            orient_norm = energies['ORIENTATION'] / (np.max(energies['ORIENTATION']) + 1e-9)
        
        # Combinar con pesos
        energies['COMBINED'] = 0.4 * eff_norm + 0.3 * trunk_norm + 0.1 * orient_norm
    
    return energies

def generate_energy_plots(energies, result, output_dir):
    """Genera gráficos de las señales de energía"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    time = np.arange(len(next(iter(energies.values())))) / 30.0  # Asumiendo 30 FPS
    
    # Plot 1: Energías individuales
    ax1 = axes[0]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (name, energy) in enumerate(energies.items()):
        if name not in ['COMBINED', 'EFFECTORS_MAX'] and len(energy) == len(time):
            ax1.plot(time, energy, label=name, color=colors[i % len(colors)], alpha=0.7)
    
    # Marcar segmentos detectados
    for move in result.moves:
        ax1.axvspan(move.t_start, move.t_end, alpha=0.3, color='red', label='Segmento' if move.idx == 1 else "")
    
    ax1.set_title('Energías Individuales por Componente')
    ax1.set_ylabel('Energía')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energía combinada y umbrales
    ax2 = axes[1]
    if 'COMBINED' in energies:
        ax2.plot(time, energies['COMBINED'], label='Energía Combinada', color='black', linewidth=2)
        
        # Calcular umbrales aproximados
        combined = energies['COMBINED']
        threshold_low = np.percentile(combined, 40)
        threshold_high = np.percentile(combined, 75)
        
        ax2.axhline(y=threshold_low, color='orange', linestyle='--', label=f'Umbral Bajo ({threshold_low:.3f})')
        ax2.axhline(y=threshold_high, color='red', linestyle='--', label=f'Umbral Alto ({threshold_high:.3f})')
        
        # Marcar picos
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(combined, height=threshold_high, distance=10)
        ax2.plot(time[peaks], combined[peaks], 'ro', markersize=6, label='Picos Detectados')
    
    # Marcar segmentos
    for move in result.moves:
        ax2.axvspan(move.t_start, move.t_end, alpha=0.3, color='green')
    
    ax2.set_title('Energía Combinada y Detección de Segmentos')
    ax2.set_ylabel('Energía Normalizada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Análisis de segmentos detectados
    ax3 = axes[2]
    
    # Duración de segmentos
    durations = [move.duration for move in result.moves]
    segments_x = [i+1 for i in range(len(result.moves))]
    
    ax3.bar(segments_x, durations, color='skyblue', alpha=0.7)
    ax3.axhline(y=np.mean(durations) if durations else 0, color='red', linestyle='--', 
                label=f'Duración Promedio: {np.mean(durations):.2f}s' if durations else 'Sin segmentos')
    
    ax3.set_title('Duración de Segmentos Detectados')
    ax3.set_xlabel('Número de Segmento')
    ax3.set_ylabel('Duración (segundos)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "energy_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de energía guardado: {plot_path}")

def generate_segmentation_report(result, output_dir):
    """Genera reporte detallado de la segmentación"""
    
    report = {
        "video_id": result.video_id,
        "fps": result.fps,
        "total_frames": result.nframes,
        "total_segments": len(result.moves),
        "segments": []
    }
    
    for move in result.moves:
        segment_info = {
            "id": move.idx,
            "frames": f"{move.a}-{move.b}",
            "time": f"{move.t_start:.2f}-{move.t_end:.2f}s",
            "duration": f"{move.duration:.2f}s",
            "active_limb": move.active_limb,
            "speed_peak": f"{move.speed_peak:.3f}",
            "rotation": f"{move.rotation_deg:.1f}° ({move.rotation_bucket})",
            "stance": move.stance_pred,
            "kick": move.kick_pred,
            "arm_direction": move.arm_dir
        }
        report["segments"].append(segment_info)
    
    # Guardar reporte JSON
    report_path = output_dir / "segmentation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Guardar reporte texto plano
    txt_report_path = output_dir / "segmentation_report.txt"
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("=== REPORTE DE SEGMENTACIÓN ===\n")
        f.write(f"Video: {report['video_id']}\n")
        f.write(f"FPS: {report['fps']}\n")
        f.write(f"Total frames: {report['total_frames']}\n")
        f.write(f"Segmentos detectados: {report['total_segments']}\n\n")
        
        f.write("DETALLE DE SEGMENTOS:\n")
        f.write("-" * 80 + "\n")
        for seg in report["segments"]:
            f.write(f"Segmento {seg['id']}:\n")
            f.write(f"  Frames: {seg['frames']}\n")
            f.write(f"  Tiempo: {seg['time']} (duración: {seg['duration']})\n")
            f.write(f"  Extremidad activa: {seg['active_limb']}\n")
            f.write(f"  Velocidad pico: {seg['speed_peak']}\n")
            f.write(f"  Rotación: {seg['rotation']}\n")
            f.write(f"  Postura: {seg['stance']}\n")
            f.write(f"  Patada: {seg['kick']}\n")
            f.write(f"  Dirección brazo: {seg['arm_direction']}\n")
            f.write("-" * 80 + "\n")
    
    print(f"Reporte de segmentación guardado: {txt_report_path}")

def generate_movement_analysis(result, output_dir):
    """Análisis estadístico de los movimientos detectados"""
    
    if not result.moves:
        print("No hay movimientos para analizar")
        return
    
    # Estadísticas básicas
    durations = [move.duration for move in result.moves]
    speeds = [move.speed_peak for move in result.moves]
    
    stats = {
        "total_movements": len(result.moves),
        "duration_avg": np.mean(durations),
        "duration_std": np.std(durations),
        "duration_min": np.min(durations),
        "duration_max": np.max(durations),
        "speed_avg": np.mean(speeds),
        "speed_std": np.std(speeds),
        "speed_min": np.min(speeds),
        "speed_max": np.max(speeds),
    }
    
    # Distribución por extremidad
    limb_dist = {}
    for move in result.moves:
        limb = move.active_limb
        limb_dist[limb] = limb_dist.get(limb, 0) + 1
    
    stats["limb_distribution"] = limb_dist
    
    # Gráfico de distribución temporal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribución de duraciones
    ax1.hist(durations, bins=min(10, len(durations)), alpha=0.7, color='lightblue', edgecolor='black')
    ax1.axvline(stats["duration_avg"], color='red', linestyle='--', label=f'Promedio: {stats["duration_avg"]:.2f}s')
    ax1.set_xlabel('Duración (segundos)')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Duración de Movimientos')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribución por extremidad
    if limb_dist:
        limbs = list(limb_dist.keys())
        counts = list(limb_dist.values())
        ax2.bar(limbs, counts, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Extremidad Activa')
        ax2.set_ylabel('Cantidad de Movimientos')
        ax2.set_title('Distribución por Extremidad')
        
        # Añadir valores en las barras
        for i, v in enumerate(counts):
            ax2.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    stats_plot_path = output_dir / "movement_statistics.png"
    plt.savefig(stats_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Guardar estadísticas
    stats_path = output_dir / "movement_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        # Convertir numpy types a Python nativos para JSON
        stats_serializable = {}
        for key, value in stats.items():
            if isinstance(value, (np.float32, np.float64)):
                stats_serializable[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                stats_serializable[key] = int(value)
            else:
                stats_serializable[key] = value
        
        json.dump(stats_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"Análisis estadístico guardado: {stats_plot_path}")

def main():
    """Función principal con interfaz de línea de comandos mejorada"""
    if len(sys.argv) < 2:
        print("Uso: python debug_segmentation.py <csv_path> [video_path] [sensitivity]")
        print("Ejemplos:")
        print("  python debug_segmentation.py data/landmarks/8yang_001.csv")
        print("  python debug_segmentation.py data/landmarks/8yang_001.csv data/raw_videos/8yang_001.mp4 0.8")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    video_path = sys.argv[2] if len(sys.argv) > 2 else None
    sensitivity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    
    if not Path(csv_path).exists():
        print(f"Error: El archivo CSV no existe: {csv_path}")
        sys.exit(1)
    
    if video_path and not Path(video_path).exists():
        print(f"Error: El archivo de video no existe: {video_path}")
        sys.exit(1)
    
    try:
        result, energies = debug_segmentation(csv_path, video_path, sensitivity=sensitivity)
        
        print(f"\n=== RESUMEN FINAL ===")
        print(f"Segmentos detectados: {len(result.moves)}")
        print(f"Duración promedio: {np.mean([m.duration for m in result.moves]):.2f}s" if result.moves else "N/A")
        print(f"Extremidades utilizadas: {', '.join(set(m.active_limb for m in result.moves))}")
        
        if len(result.moves) < 20:  # Para 8yang esperamos ~24 movimientos
            print(f"⚠️  ADVERTENCIA: Pocos segmentos detectados. Considera aumentar la sensibilidad.")
        
    except Exception as e:
        print(f"Error durante el debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()