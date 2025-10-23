# src/tools/batch_all.py
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.segmentation.move_capture import capture_moves_from_csv, render_preview
from src.tools.score_pal_yang import score_many_to_excel, load_spec

VIDEO_EXTS = (".mp4",".mkv",".webm",".mov",".m4v",".avi",".mpg",".mpeg")

def iter_videos(d: Path) -> List[Path]:
    vids: List[Path] = []
    for ext in VIDEO_EXTS:
        vids.extend(sorted(d.glob(f"*{ext}")))
    return vids

def csv_for(video: Path, landmarks_root: Path, alias: str, subset: str|None) -> Path:
    rel = video.stem + ".csv"
    out = landmarks_root / alias
    if subset:
        out = out / subset
    return out / rel

def moves_json_for(video: Path, ann_dir: Path, alias: str) -> Path:
    ann_dir.mkdir(parents=True, exist_ok=True)
    return ann_dir / f"{alias}_{video.stem}_moves.json"

def preview_mp4_for(video: Path, ann_dir: Path, alias: str) -> Path:
    ann_dir.mkdir(parents=True, exist_ok=True)
    return ann_dir / f"{alias}_{video.stem}_moves_preview.mp4"

def main():
    ap = argparse.ArgumentParser(
        description="Pipeline por carpeta: usa landmarks existentes, captura movimientos, previews (con pose) y Excel (Pal Jang)."
    )
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--subset", default="")
    ap.add_argument("--landmarks-root", default="data/landmarks")
    ap.add_argument("--annotations-dir", default="data/annotations")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-xlsx", default="")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--force-capture", action="store_true")
    ap.add_argument("--vstart", type=float, default=0.60)
    ap.add_argument("--vstop",  type=float, default=0.20)
    ap.add_argument("--min-dur", type=float, default=0.25)
    ap.add_argument("--min-gap", type=float, default=0.18)
    ap.add_argument("--poly-n",  type=int,   default=20)
    ap.add_argument("--min-path-norm", type=float, default=0.02)
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    alias = args.alias.strip()
    subset = args.subset.strip() or None
    landmarks_root = Path(args.landmarks_root)
    ann_dir = Path(args.annotations_dir)
    spec_path = Path(args.spec)

    if not videos_dir.exists():
        sys.exit(f"No existe --videos-dir: {videos_dir}")
    vids = iter_videos(videos_dir)
    if not vids:
        sys.exit(f"No se encontraron videos en: {videos_dir}")

    _ = load_spec(spec_path)

    missing_csv = []
    for v in vids:
        csvp = csv_for(v, landmarks_root, alias, subset)
        if not csvp.exists():
            missing_csv.append(str(csvp))
    if missing_csv:
        print("[WARN] Faltan CSV de landmarks para algunos videos:")
        for p in missing_csv[:10]:
            print("  -", p)
        print("Ejecuta antes:")
        print(f'  python -m src.tools.extract_landmarks --in "{videos_dir}" --out-root "{landmarks_root}" --alias {alias}' + (f" --subset {subset}" if subset else ""))
        sys.exit(2)

    produced_jsons: List[Path] = []
    for v in vids:
        csvp = csv_for(v, landmarks_root, alias, subset)
        out_json = moves_json_for(v, ann_dir, alias)
        preview  = preview_mp4_for(v, ann_dir, alias)

        if out_json.exists() and not args.force_capture:
            print(f"[SKIP] Ya existe: {out_json.name}")
            produced_jsons.append(out_json)
            if args.render:
                try:
                    # Re-render (o render por primera vez) con pose
                    res = capture_moves_from_csv(
                        csvp, video_path=v,
                        vstart=args.vstart, vstop=args.vstop,
                        min_dur=args.min_dur, min_gap=args.min_gap,
                        poly_n=args.poly_n, min_path_norm=args.min_path_norm
                    )
                    render_preview(v, res, preview, csv_path=csvp, draw_pose=True)
                    print(f"[OK] Preview -> {preview}")
                except Exception as e:
                    print(f"[WARN] Render falló en {v.name}: {e}")
            continue

        res = capture_moves_from_csv(
            csvp, video_path=v,
            vstart=args.vstart, vstop=args.vstop,
            min_dur=args.min_dur, min_gap=args.min_gap,
            poly_n=args.poly_n, min_path_norm=args.min_path_norm
        )
        out_json.write_text(res.to_json(), encoding="utf-8")
        print(f"[OK] JSON -> {out_json}")
        produced_jsons.append(out_json)

        if args.render:
            try:
                render_preview(v, res, preview, csv_path=csvp, draw_pose=True)
                print(f"[OK] Preview -> {preview}")
            except Exception as e:
                print(f"[WARN] Render falló en {v.name}: {e}")

    if not args.out_xlsx:
        tag = videos_dir.name.replace(" ", "_")
        out_xlsx = ann_dir / f"{alias}_{tag}_report.xlsx"
    else:
        out_xlsx = Path(args.out_xlsx)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scoring {len(produced_jsons)} JSON(s) -> {out_xlsx}")
    score_many_to_excel(
        moves_jsons=produced_jsons,
        spec_json=spec_path,
        out_xlsx=out_xlsx,
        landmarks_root=landmarks_root,
        alias=alias,
        subset=subset
    )
    print(f"[OK] Reporte -> {out_xlsx}")

if __name__ == "__main__":
    main()
