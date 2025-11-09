"""
Herramienta interactiva para etiquetar posturas manualmente.
Asiste el proceso mostrando features calculadas y permitiendo navegación.
"""
from __future__ import annotations
import argparse, sys, json, csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.segmentation.move_capture import load_landmarks_csv

# ==================== FEATURE EXTRACTION ====================

def _xy_mean(df: pd.DataFrame, lmk_id: int, a: int, b: int) -> Tuple[float, float]:
    """Media de coordenadas x,y para un landmark en rango de frames"""
    sub = df[(df["lmk_id"] == lmk_id) & (df["frame"] >= a) & (df["frame"] <= b)][["x", "y"]]
    if sub.empty:
        return (np.nan, np.nan)
    m = sub.mean().to_numpy(np.float32)
    return float(m[0]), float(m[1])

def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ángulo en grados entre 3 puntos (b es el vértice)"""
    ba = a - b
    bc = c - b
    num = float(np.dot(ba, bc))
    den = float(np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
    return float(ang)

def extract_features_for_labeling(df: pd.DataFrame, a: int, b: int) -> Dict[str, float]:
    """
    Extrae features relevantes para ayudar en el etiquetado manual.
    Retorna diccionario con todas las medidas útiles.
    """
    # Landmarks clave
    LANK = np.array(_xy_mean(df, 27, a, b), np.float32)
    RANK = np.array(_xy_mean(df, 28, a, b), np.float32)
    LKNE = np.array(_xy_mean(df, 25, a, b), np.float32)
    RKNE = np.array(_xy_mean(df, 26, a, b), np.float32)
    LHIP = np.array(_xy_mean(df, 23, a, b), np.float32)
    RHIP = np.array(_xy_mean(df, 24, a, b), np.float32)
    LSH = np.array(_xy_mean(df, 11, a, b), np.float32)
    RSH = np.array(_xy_mean(df, 12, a, b), np.float32)
    LHEEL = np.array(_xy_mean(df, 29, a, b), np.float32)
    RHEEL = np.array(_xy_mean(df, 30, a, b), np.float32)
    LFOOT = np.array(_xy_mean(df, 31, a, b), np.float32)
    RFOOT = np.array(_xy_mean(df, 32, a, b), np.float32)
    
    # Validar que tenemos datos
    if any(np.isnan(v).any() for v in [LANK, RANK, LSH, RSH, LHIP, RHIP]):
        return {k: np.nan for k in [
            "ankle_dist_sw", "hip_offset_x", "hip_offset_y",
            "knee_angle_left", "knee_angle_right",
            "foot_angle_left", "foot_angle_right",
            "hip_behind_feet", "ankle_dist_abs"
        ]}
    
    # 1. Distancia entre tobillos (normalizada)
    sh_w = float(np.linalg.norm(RSH - LSH) + 1e-6)
    ankle_dist = float(np.linalg.norm(RANK - LANK))
    ankle_dist_sw = ankle_dist / sh_w
    
    # 2. Offset de cadera respecto a centro de pies
    feet_center = 0.5 * (LANK + RANK)
    hip_center = 0.5 * (LHIP + RHIP)
    hip_offset_x = float(hip_center[0] - feet_center[0])
    hip_offset_y = float(hip_center[1] - feet_center[1])  # Y positivo = cadera atrás
    
    # 3. Ángulos de rodillas
    knee_angle_left = _angle3(LHIP, LKNE, LANK) if not any(np.isnan(v).any() for v in [LHIP, LKNE, LANK]) else np.nan
    knee_angle_right = _angle3(RHIP, RKNE, RANK) if not any(np.isnan(v).any() for v in [RHIP, RKNE, RANK]) else np.nan
    
    # 4. Orientación de pies (ángulo del vector talón→punta)
    def foot_angle(heel, foot):
        if np.isnan(heel).any() or np.isnan(foot).any():
            return np.nan
        vec = foot - heel
        if np.linalg.norm(vec) < 1e-6:
            return np.nan
        return float(np.degrees(np.arctan2(vec[1], vec[0])))
    
    foot_angle_left = foot_angle(LHEEL, LFOOT)
    foot_angle_right = foot_angle(RHEEL, RFOOT)
    
    # 5. Indicador de cadera atrás (para dwit_kubi/beom_seogi)
    hip_behind_feet = 1.0 if hip_offset_y > 0 else 0.0
    
    return {
        "ankle_dist_sw": ankle_dist_sw,
        "ankle_dist_abs": ankle_dist,
        "hip_offset_x": hip_offset_x,
        "hip_offset_y": hip_offset_y,
        "knee_angle_left": knee_angle_left,
        "knee_angle_right": knee_angle_right,
        "foot_angle_left": foot_angle_left,
        "foot_angle_right": foot_angle_right,
        "hip_behind_feet": hip_behind_feet
    }

# ==================== LABELING INTERFACE ====================

def load_sample_movements(
    moves_dir: Path, 
    landmarks_root: Path, 
    alias: str, 
    subset: str,
    sample_size: int = 200
) -> List[Dict]:
    """
    Carga muestra aleatoria de movimientos de todos los videos.
    Retorna lista de dicts con: video_id, move_idx, a, b, expected_stance, features
    """
    all_jsons = sorted(moves_dir.glob(f"{alias}_{alias}_*_moves.json"))
    # Filtrar archivos principales
    main_jsons = [
        j for j in all_jsons 
        if not any(x in j.stem for x in ["preview", "review", "scored", "clip", "NEW", "FINAL"])
    ]
    
    print(f"📂 Cargando movimientos de {len(main_jsons)} videos...")
    
    all_movements = []
    for json_path in main_jsons:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        video_id = data["video_id"]
        
        # Cargar CSV de landmarks
        csv_path = landmarks_root / alias / subset / f"{video_id}.csv"
        if not csv_path.exists():
            print(f"⚠️  CSV no encontrado: {csv_path}")
            continue
        
        df = load_landmarks_csv(csv_path)
        
        # Procesar cada movimiento
        for i, move in enumerate(data.get("moves", [])):
            a = int(move.get("a", 0))
            b = int(move.get("b", 0))
            
            # Extraer features
            features = extract_features_for_labeling(df, a, b)
            
            # Info del movimiento
            all_movements.append({
                "video_id": video_id,
                "move_idx": i + 1,
                "a": a,
                "b": b,
                "expected_stance": move.get("expected_stance", ""),
                "detected_stance": move.get("stance_pred", ""),  # Lo que detectó el sistema
                "expected_tech": move.get("expected_tech", ""),
                **features
            })
    
    print(f"✅ Total movimientos cargados: {len(all_movements)}")
    
    # Muestreo estratificado por postura esperada
    df_all = pd.DataFrame(all_movements)
    
    # Filtrar movimientos sin expected_stance
    df_all = df_all[df_all["expected_stance"].notna() & (df_all["expected_stance"] != "")]
    print(f"   Movimientos con expected_stance válida: {len(df_all)}")
    
    # Distribuir la muestra: 40% ap_kubi, 30% dwit_kubi, 15% beom_seogi, 15% otros
    samples = []
    
    for stance, target_pct in [("ap_kubi", 0.40), ("dwit_kubi", 0.30), ("beom_seogi", 0.15)]:
        subset = df_all[df_all["expected_stance"] == stance]
        n_sample = min(len(subset), int(sample_size * target_pct))
        samples.append(subset.sample(n=n_sample, random_state=42))
    
    # Resto de posturas (moa_seogi, otros)
    other_stances = df_all[~df_all["expected_stance"].isin(["ap_kubi", "dwit_kubi", "beom_seogi"])]
    n_other = sample_size - sum(len(s) for s in samples)
    if n_other > 0 and len(other_stances) > 0:
        samples.append(other_stances.sample(n=min(len(other_stances), n_other), random_state=42))
    
    sampled = pd.concat(samples, ignore_index=True)
    
    print(f"\n📊 Distribución de muestra ({len(sampled)} movimientos):")
    print(sampled["expected_stance"].value_counts())
    
    return sampled.to_dict('records')

def interactive_labeling(movements: List[Dict], output_csv: Path, resume: bool = True):
    """
    Interfaz interactiva para etiquetar posturas.
    Muestra features y permite navegar con comandos.
    """
    # Cargar etiquetas existentes si resume=True
    labeled = {}
    if resume and output_csv.exists():
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['video_id']}_{row['move_idx']}"
                labeled[key] = row['stance_label']
        print(f"📋 Cargadas {len(labeled)} etiquetas existentes")
    
    idx = 0
    
    # Comandos disponibles
    commands_help = """
    COMANDOS:
      1-4: Etiquetar como (1=ap_kubi, 2=dwit_kubi, 3=beom_seogi, 4=moa_seogi)
      u: Unknown/incierto
      n: Siguiente (sin etiquetar)
      p: Anterior
      s: Skip (siguiente sin etiquetar)
      j N: Saltar a movimiento N
      q: Guardar y salir
      h: Mostrar ayuda
    """
    
    print("\n" + "="*80)
    print("🏷️  HERRAMIENTA DE ETIQUETADO DE POSTURAS")
    print("="*80)
    print(commands_help)
    
    stance_map = {
        "1": "ap_kubi",
        "2": "dwit_kubi",
        "3": "beom_seogi",
        "4": "moa_seogi",
        "u": "unknown"
    }
    
    while idx < len(movements):
        move = movements[idx]
        key = f"{move['video_id']}_{move['move_idx']}"
        
        # Mostrar info del movimiento
        print("\n" + "-"*80)
        print(f"Movimiento {idx+1}/{len(movements)}")
        print(f"Video: {move['video_id']} | Move: {move['move_idx']}")
        print(f"Técnica: {move['expected_tech']}")
        print(f"Frames: {move['a']}-{move['b']}")
        print()
        print("POSTURA ESPERADA:", move['expected_stance'])
        print("POSTURA DETECTADA:", move['detected_stance'])
        
        if key in labeled:
            print(f"✓ ETIQUETA ACTUAL: {labeled[key]}")
        
        print()
        print("FEATURES:")
        print(f"  ankle_dist_sw:     {move['ankle_dist_sw']:.3f}  (normalizada)")
        print(f"  ankle_dist_abs:    {move['ankle_dist_abs']:.3f}  (píxeles)")
        print(f"  hip_offset_y:      {move['hip_offset_y']:.3f}  (+atrás, -adelante)")
        print(f"  hip_offset_x:      {move['hip_offset_x']:.3f}")
        print(f"  knee_angle_left:   {move['knee_angle_left']:.1f}°")
        print(f"  knee_angle_right:  {move['knee_angle_right']:.1f}°")
        print(f"  foot_angle_left:   {move['foot_angle_left']:.1f}°")
        print(f"  foot_angle_right:  {move['foot_angle_right']:.1f}°")
        print(f"  hip_behind_feet:   {move['hip_behind_feet']:.0f}")
        
        print()
        print("GUÍA DE CLASIFICACIÓN:")
        print("  ap_kubi (1):    ankle_dist ~2.5-5.0, cadera adelante, rodillas ~160-170°")
        print("  dwit_kubi (2):  ankle_dist ~0.8-2.5, cadera atrás, pierna delantera recta")
        print("  beom_seogi (3): ankle_dist ~0.3-0.8, cadera atrás, ambas rodillas flexionadas")
        print("  moa_seogi (4):  ankle_dist ~0.0-0.5, pies juntos")
        
        # Input
        cmd = input("\n>>> ").strip().lower()
        
        if cmd in stance_map:
            labeled[key] = stance_map[cmd]
            print(f"✓ Etiquetado como: {stance_map[cmd]}")
            idx += 1
        elif cmd == 'n':
            idx += 1
        elif cmd == 'p':
            idx = max(0, idx - 1)
        elif cmd == 's':
            # Skip a siguiente sin etiquetar
            found = False
            for i in range(idx + 1, len(movements)):
                k = f"{movements[i]['video_id']}_{movements[i]['move_idx']}"
                if k not in labeled:
                    idx = i
                    found = True
                    break
            if not found:
                print("⚠️  No hay más movimientos sin etiquetar")
        elif cmd.startswith('j '):
            try:
                target = int(cmd.split()[1])
                idx = max(0, min(len(movements) - 1, target - 1))
            except:
                print("❌ Formato incorrecto. Uso: j N")
        elif cmd == 'q':
            print("\n💾 Guardando etiquetas...")
            break
        elif cmd == 'h':
            print(commands_help)
        else:
            print("❌ Comando no reconocido. Usa 'h' para ayuda")
    
    # Guardar etiquetas
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Crear CSV con todas las etiquetas
    labeled_data = []
    for move in movements:
        key = f"{move['video_id']}_{move['move_idx']}"
        if key in labeled:
            labeled_data.append({
                "video_id": move["video_id"],
                "move_idx": move["move_idx"],
                "a": move["a"],
                "b": move["b"],
                "expected_stance": move["expected_stance"],
                "detected_stance": move["detected_stance"],
                "stance_label": labeled[key],
                **{k: move[k] for k in [
                    "ankle_dist_sw", "hip_offset_x", "hip_offset_y",
                    "knee_angle_left", "knee_angle_right",
                    "foot_angle_left", "foot_angle_right", "hip_behind_feet"
                ]}
            })
    
    df_out = pd.DataFrame(labeled_data)
    df_out.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"\n✅ Guardadas {len(labeled_data)} etiquetas en: {output_csv}")
    print(f"\n📊 Distribución de etiquetas:")
    print(df_out["stance_label"].value_counts())
    
    return len(labeled_data)

# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser(description="Herramienta interactiva para etiquetar posturas")
    ap.add_argument("--moves-dir", default="data/annotations/moves", help="Directorio con JSONs de movimientos")
    ap.add_argument("--landmarks-root", default="data/landmarks", help="Raíz de landmarks CSV")
    ap.add_argument("--alias", default="8yang", help="Alias del poomsae")
    ap.add_argument("--subset", default="train", help="Subset (train/val/test)")
    ap.add_argument("--sample-size", type=int, default=200, help="Tamaño de muestra a etiquetar")
    ap.add_argument("--output", default="data/labels/stance_labels.csv", help="CSV de salida")
    ap.add_argument("--no-resume", action="store_true", help="No cargar etiquetas existentes")
    args = ap.parse_args()
    
    moves_dir = Path(args.moves_dir)
    landmarks_root = Path(args.landmarks_root)
    output_csv = Path(args.output)
    
    # Cargar muestra
    movements = load_sample_movements(
        moves_dir, landmarks_root, args.alias, args.subset, args.sample_size
    )
    
    if not movements:
        print("❌ No se encontraron movimientos para etiquetar")
        return
    
    # Etiquetado interactivo
    n_labeled = interactive_labeling(movements, output_csv, resume=not args.no_resume)
    
    print(f"\n🎉 Proceso completado. {n_labeled} movimientos etiquetados.")

if __name__ == "__main__":
    main()
