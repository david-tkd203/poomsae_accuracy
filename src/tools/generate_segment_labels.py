import os
import json
import pandas as pd
from pathlib import Path
from src.features.feature_extractor import FeatureExtractor, angles_from_landmarks
import numpy as np

# Configuración de paths
SEGMENTS_DIR = Path("data/annotations/moves")
LANDMARKS_DIR = Path("data/landmarks/8yang/train/8yang")
OUT_CSV = Path("data/labels/segment_labels.csv")

# Tripletas de ángulos (ajusta según tu pipeline)
ANGLE_TRIPLETS = {
    "hip": [23, 24, 12],
    "shoulder": [11, 12, 24],
    "elbow_left": [13, 11, 23],
    "elbow_right": [14, 12, 24],
    "knee_left": [25, 23, 11],
    "knee_right": [26, 24, 12],
}

extractor = FeatureExtractor(ANGLE_TRIPLETS)

rows = []
for seg_file in SEGMENTS_DIR.glob("*_moves.json"):
    with open(seg_file, "r", encoding="utf-8") as f:
        seg_data = json.load(f)
    video_id = seg_data["video_id"]
    lmks_path = LANDMARKS_DIR / f"{video_id}.csv"
    if not lmks_path.exists():
        print(f"[WARN] No landmarks: {lmks_path}")
        continue
    lmks_df = pd.read_csv(lmks_path)
    for move in seg_data["moves"]:
        a, b = move["a"], move["b"]
        frames_idx = list(range(a, b + 1))
        # Extraer landmarks para el segmento
        lmks_seq = []
        for idx in frames_idx:
            row = lmks_df[lmks_df["frame"] == idx]
            if row.empty:
                # Si no hay fila para este frame, usar NaN para todos los landmarks
                lmks = [[np.nan, np.nan] for _ in range(33)]
            else:
                lmks = []
                for i in range(33):
                    x = row[f"x{i}"] if f"x{i}" in row else np.nan
                    y = row[f"y{i}"] if f"y{i}" in row else np.nan
                    # Si es una serie, tomar el primer valor; si es float, usar directamente
                    if hasattr(x, "values"): x = x.values[0]
                    if hasattr(y, "values"): y = y.values[0]
                    lmks.append([x, y])
            lmks_seq.append(lmks)
        if not lmks_seq:
            continue
        meta = {
            "video_id": video_id,
            "segment_idx": move["idx"],
            "active_limb": move["active_limb"],
            "expected_tech": move.get("expected_tech", ""),
            "expected_stance": move.get("expected_stance", ""),
        }
        feats = extractor.segment_features(lmks_seq, frames_idx, meta=meta)
        # Etiqueta: puedes ajustar aquí según tu preferencia
        feats["label"] = move.get("expected_tech", move.get("expected_stance", ""))
        rows.append(feats)

if not rows:
    print("[ERROR] No se generaron filas para el dataset de segmentos.")
else:
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] Dataset generado: {OUT_CSV} ({len(df)} filas)")
