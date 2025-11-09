# src/tools/capture_moves_batch.py
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
from typing import Optional, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.segmentation.move_capture import capture_moves_with_spec
# from src.segmentation.move_capture import render_preview  # TODO: agregar cuando esté disponible

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".mpg", ".mpeg")

def _load_spec(spec_path: Path) -> dict:
    d = json.loads(spec_path.read_text(encoding="utf-8"))
    moves = d.get("moves", [])
    d["_expected_n"] = len(moves)
    return d

def _find_video(videos_root: Path, alias: str, subset: Optional[str], stem: str) -> Optional[Path]:
    base = videos_root / alias
    if subset:
        base = base / subset
    for ext in VIDEO_EXTS:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    for p in base.glob(f"{stem}.*"):
        if p.suffix.lower() in VIDEO_EXTS:
            return p
    return None

def main():
    ap = argparse.ArgumentParser(description="Batch: captura de movimientos para todos los CSV de un subset.")
    ap.add_argument("--landmarks-root", required=True)
    ap.add_argument("--videos-root",    required=True)
    ap.add_argument("--alias",          required=True)
    ap.add_argument("--subset",         default="")
    ap.add_argument("--spec",           required=True)
    ap.add_argument("--out-dir",        required=True)

    # segmentación
    ap.add_argument("--vstart", type=float, default=0.55)
    ap.add_argument("--vstop",  type=float, default=0.18)
    ap.add_argument("--min-gap", type=float, default=0.22)
    ap.add_argument("--min-dur", type=float, default=0.22)
    ap.add_argument("--min-path-norm", type=float, default=0.015)
    ap.add_argument("--smooth-win", type=int, default=5)
    ap.add_argument("--poly-n", type=int, default=20)
    ap.add_argument("--no-force-expected", action="store_true",
                    help="No forzar N esperado; solo si la segmentación difiere mucho.")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview-dir", default="previews")
    ap.add_argument("--force-expected", action="store_true",
                help="Si se indica, fuerza a N=spec usando cortes por cuantiles.")
    
    # ML classifier
    ap.add_argument("--use-ml-classifier", action="store_true",
                   help="Usar clasificador ML en lugar del heurístico")
    ap.add_argument("--ml-model", type=str, default="data/models/stance_classifier_final.pkl",
                   help="Ruta al modelo ML entrenado")
    
    args = ap.parse_args()

    landmarks_root = Path(args.landmarks_root)
    videos_root    = Path(args.videos_root)
    alias          = args.alias.strip()
    subset         = args.subset.strip() or None
    spec_path      = Path(args.spec)
    out_dir        = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        Path(args.preview_dir).mkdir(parents=True, exist_ok=True)

    spec = _load_spec(spec_path)
    expected_n = int(spec.get("_expected_n", 0)) or None
    if not args.force_expected:
        expected_n = None  # NO forzar si no se pide
    force_expected = not args.no_force_expected

    csv_dir = landmarks_root / alias / (subset or "")
    csvs: List[Path] = sorted([p for p in csv_dir.glob("*.csv")])
    if not csvs:
        sys.exit(f"No hay CSV en {csv_dir}")

    print(f"[INFO] CSV: {len(csvs)}  expected_n={expected_n}  force_expected={force_expected}")

    for cpath in csvs:
        stem = cpath.stem
        vpath = _find_video(videos_root, alias, subset, stem)
        if vpath is None:
            print(f"[WARN] Sin video para {stem}; se usará fps=30.")
        out_json = out_dir / f"{alias}_{stem}_moves.json"
        try:
            # Preparar parámetros para ML si está habilitado
            ml_model_path = Path(args.ml_model) if args.use_ml_classifier else None
            
            res = capture_moves_with_spec(
                cpath, 
                video_path=vpath,
                use_ml_classifier=args.use_ml_classifier,
                ml_model_path=ml_model_path
            )
            # Escribir JSON con encoding correcto
            json_str = res.to_json()
            with open(out_json, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"[OK] {out_json.name}  (moves={len(res.moves)}  fps={res.fps:.2f})")

            # TODO: render_preview no disponible aún
            # if args.preview and vpath and vpath.exists():
            #     prev_path = Path(args.preview_dir) / f"{stem}_preview.mp4"
            #     try:
            #         render_preview(vpath, res, prev_path, csv_path=cpath, draw_pose=True)
            #         print(f"      Preview: {prev_path}")
            #     except Exception as e:
            #         print(f"      [WARN] Preview falló: {e}")
        except Exception as e:
            print(f"[ERR] {cpath.name}: {e}")

if __name__ == "__main__":
    main()
