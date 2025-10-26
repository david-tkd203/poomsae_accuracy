# src/tools/capture_moves_batch.py
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
from typing import Optional, List

# permitir "python -m src.tools.capture_moves_batch"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.segmentation.move_capture import capture_moves_from_csv  # usa las mejoras (a,b,pose_f)
from src.segmentation.move_capture import render_preview

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".mpg", ".mpeg")

def _load_spec(spec_path: Path) -> dict:
    d = json.loads(spec_path.read_text(encoding="utf-8"))
    moves = d.get("moves", [])
    d["_expected_n"] = len(moves)
    return d

def _find_video(videos_root: Path, alias: str, subset: Optional[str], stem: str) -> Optional[Path]:
    """
    Busca un archivo de video cuyo nombre base sea 'stem' en:
      videos_root/alias/(subset?)/stem.<ext>
    """
    base = videos_root / alias
    if subset:
        base = base / subset
    for ext in VIDEO_EXTS:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    # búsqueda más laxa: por glob stem.*
    cand = list(base.glob(f"{stem}.*"))
    for p in cand:
        if p.suffix.lower() in VIDEO_EXTS:
            return p
    return None

def main():
    ap = argparse.ArgumentParser(description="Batch: captura de movimientos para todos los CSV de un subset.")
    ap.add_argument("--landmarks-root", required=True, help="Raíz de landmarks (CSV normalizados)")
    ap.add_argument("--videos-root",    required=True, help="Raíz de videos crudos")
    ap.add_argument("--alias",          required=True, help="Alias (ej. 8yang)")
    ap.add_argument("--subset",         default="",    help="train/val/test (opcional)")
    ap.add_argument("--spec",           required=True, help="Spec del poomsae (para expected_n)")
    ap.add_argument("--out-dir",        required=True, help="Carpeta de salida para *_moves.json")
    # parámetros de segmentación (idénticos a tu CLI por video)
    ap.add_argument("--vstart", type=float, default=0.55)
    ap.add_argument("--vstop",  type=float, default=0.18)
    ap.add_argument("--min-gap", type=float, default=0.22)
    ap.add_argument("--min-dur", type=float, default=0.22)
    ap.add_argument("--min-path-norm", type=float, default=0.015)
    ap.add_argument("--smooth-win", type=int, default=5)
    ap.add_argument("--poly-n", type=int, default=20)
    ap.add_argument("--preview", action="store_true", help="Opcional: genera mp4 de preview por video")
    ap.add_argument("--preview-dir", default="previews", help="Carpeta de previews (si --preview)")
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

    # carpeta de CSV: data/landmarks/<alias>/(subset?)/ *.csv
    csv_dir = landmarks_root / alias
    if subset:
        csv_dir = csv_dir / subset
    csvs: List[Path] = sorted(csv_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"No hay CSV en {csv_dir}")

    print(f"[INFO] CSV encontrados: {len(csvs)} en {csv_dir}  | expected_n={expected_n}")

    for cpath in csvs:
        stem = cpath.stem  # video_id
        vpath = _find_video(videos_root, alias, subset, stem)
        if vpath is None:
            print(f"[WARN] No se encontró video para {stem}; se usará fps=30 (t0/t1 menos precisos).")
        out_json = out_dir / f"{alias}_{stem}_moves.json"
        try:
            res = capture_moves_from_csv(
                    cpath, video_path=vpath,
                    vstart=args.vstart, vstop=args.vstop,
                    min_dur=args.min_dur, min_gap=args.min_gap,   # ← así
                    smooth_win=args.smooth_win, poly_n=args.poly_n,
                    min_path_norm=args.min_path_norm,
                    expected_n=expected_n
                )
            out_json.write_text(res.to_json(), encoding="utf-8")
            print(f"[OK] {out_json.name}  (moves={len(res.moves)}  fps={res.fps:.2f})")

            if args.preview and vpath and vpath.exists():
                prev_path = Path(args.preview_dir) / f"{stem}_preview.mp4"
                try:
                    render_preview(vpath, res, prev_path, csv_path=cpath, draw_pose=True)
                    print(f"      Preview: {prev_path}")
                except Exception as e:
                    print(f"      [WARN] Preview falló: {e}")
        except Exception as e:
            print(f"[ERR] {cpath.name}: {e}")

if __name__ == "__main__":
    main()
