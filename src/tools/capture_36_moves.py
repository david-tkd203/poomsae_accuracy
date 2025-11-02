#!/usr/bin/env python3
"""
Script simple para capturar 36 movimientos y guardar en JSON
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importar funciones de segmentación
from src.segmentation.move_capture import capture_moves_from_csv, load_landmarks_csv

# Parámetros
csv_path = Path("data/landmarks/8yang/8yang_001.csv")
video_path = Path("data/raw_videos/8yang/8yang_001.mp4")
out_json = Path("data/annotations/moves/8yang_001_moves_FINAL36.json")

print(f"Capturando movimientos...")
print(f"  CSV: {csv_path}")
print(f"  Video: {video_path}")
print(f"  Salida: {out_json}")

try:
    # Capturar movimientos con spec que espera 36
    res = capture_moves_from_csv(
        csv_path, 
        video_path=video_path,
        expected_n=36  # Forzar 36 segmentos esperados
    )
    
    # Convertir numpy float32 a float antes de serializar
    import numpy as np
    def convert_numpy_types(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # Guardar JSON
    data_dict = json.loads(res.to_json())
    data_dict = convert_numpy_types(data_dict)
    json_str = json.dumps(data_dict, ensure_ascii=False, indent=2)
    out_json.write_text(json_str, encoding="utf-8")
    
    print(f"\n✓ Segmentos detectados: {len(res.moves)}/36")
    print(f"✓ JSON guardado: {out_json}")
    
    # Verificar lo que se guardó
    saved_data = json.loads(out_json.read_text())
    print(f"✓ Verificación: {len(saved_data.get('moves', []))} movimientos en JSON")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
