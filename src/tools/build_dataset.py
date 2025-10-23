from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.features.feature_extractor import extract_features_for_segment

def main():
    ap = argparse.ArgumentParser(description="Construye dataset (X,y) desde labels + landmarks CSV.")
    ap.add_argument("--labels-csv", required=True, help="data/annotations/8yang_labels.csv")
    ap.add_argument("--landmarks-root", required=True, help="data/landmarks")
    ap.add_argument("--out-csv", required=True, help="data/annotations/8yang_dataset.csv")
    ap.add_argument("--alias", required=True, help="8yang")
    ap.add_argument("--subset", default="train", help="train/val/test")
    args = ap.parse_args()

    lbl = pd.read_csv(args.labels_csv)
    rows = []
    for _, r in lbl.iterrows():
        vpath = Path(str(r["video"]))
        csv_path = Path(args.landmarks_root) / args.alias / args.subset / (vpath.stem + ".csv")
        if not csv_path.exists():
            print(f"[WARN] falta landmarks: {csv_path}"); continue
        feats = extract_features_for_segment(str(csv_path), float(r["start_s"]), float(r["end_s"]))
        feats["label"] = r["label"]
        feats["move_id"] = r.get("move_id","")
        feats["video_id"] = vpath.stem
        rows.append(feats)
    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] dataset -> {args.out_csv} ({len(df)} filas)")
if __name__ == "__main__":
    main()
