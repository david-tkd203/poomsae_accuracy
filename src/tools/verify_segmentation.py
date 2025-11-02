#!/usr/bin/env python3
"""
Verificar segmentación - ver cuántos movimientos se detectan
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.segmentation.move_capture import SegmentationCaptor, load_landmarks_csv

# Cargar configuración
config_path = Path("config/default.yaml")
spec_path = Path("config/patterns/8yang_spec.json")
landmarks_path = Path("data/landmarks/8yang/8yang_001.csv")

# Cargar landmarks
df_landmarks = load_landmarks_csv(landmarks_path)
nframes = int(df_landmarks["frame"].max()) + 1
fps = 30.0

# Inicializar captor
captor = SegmentationCaptor(spec_path=spec_path, config_path=config_path)

# Capturar segmentos
print("=" * 80)
print("DEBUG SEGMENTACIÓN")
print("=" * 80)
print(f"Frames totales: {nframes}")
print(f"FPS: {fps}")

moves = captor.capture(df_landmarks, nframes, fps)
print(f"\nMovimientos detectados: {len(moves)}/36")
print("\nDetalles de cada segmento:")
for i, m in enumerate(moves, 1):
    start_f = int(m.get("a", 0) if "a" in m else m.get("t_start", 0) * fps)
    end_f = int(m.get("b", 0) if "b" in m else m.get("t_end", 0) * fps)
    active_limb = m.get("active_limb", "?")
    rotation = m.get("rotation_deg", 0)
    print(f"  {i:2d}. frames {start_f:4d}-{end_f:4d} ({end_f-start_f+1:3d}f) | limb={active_limb:15s} | rot={rotation:6.1f}°")

print(f"\n✓ Segmentación detectó {len(moves)}/36 movimientos")
if len(moves) == 36:
    print("  ✅ CORRECTO: 100% de movimientos detectados")
else:
    print(f"  ❌ INCORRECTO: Faltan {36-len(moves)} movimientos")
    print("\n  Movimientos faltantes:")
    # Cargar spec para ver cuáles se esperan
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    expected_moves = [m.get("_key", f"{m['idx']}{m.get('sub','')}") for m in spec["moves"]]
    for m in spec["moves"]:
        m["_key"] = f"{m.get('idx','?')}{m.get('sub','')}"
    
    # Si tenemos moves.json, comparar
    moves_json_path = Path("data/annotations/moves/8yang_001_moves_NEW.json")
    if moves_json_path.exists():
        moves_json = json.loads(moves_json_path.read_text(encoding="utf-8"))
        detected_segs = moves_json.get("moves", [])
        print(f"  Detectados en JSON: {len(detected_segs)}")
