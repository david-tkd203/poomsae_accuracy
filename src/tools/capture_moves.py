# src/tools/capture_moves.py
from __future__ import annotations
import argparse, sys, json
from pathlib import Path

# CORRECCIÓN COMPLETA DE IMPORTS
ROOT_DIR = Path(__file__).resolve().parents[2]  # Subir 2 niveles desde src/tools/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Importar desde el path correcto
try:
    from src.segmentation.move_capture import (
        capture_moves_from_csv, render_preview
    )
except ImportError as e:
    print(f"Error de importación: {e}")
    print("Buscando módulos alternativos...")
    # Fallback: intentar importar directamente
    import importlib.util
    spec = importlib.util.spec_from_file_location("move_capture", ROOT_DIR / "src" / "segmentation" / "move_capture.py")
    move_capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(move_capture)
    capture_moves_from_csv = move_capture.capture_moves_from_csv
    render_preview = move_capture.render_preview

def _expected_from_spec(spec_path: Path) -> int:
    d = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    mv = d.get("moves", [])
    return int(len(mv))

def main():
    ap = argparse.ArgumentParser(description="Captura movimientos desde landmarks CSV; opcional preview con esqueleto.")
    ap.add_argument("--csv", required=True, help="CSV de landmarks (data/landmarks/.../*.csv)")
    ap.add_argument("--video", required=True, help="Video original (para FPS exacto y preview)")
    ap.add_argument("--out-json", required=True, help="Salida JSON con movimientos detectados")
    ap.add_argument("--render-out", help="MP4 con overlay de trayectorias/segmentos + pose (si hay CSV)")
    ap.add_argument("--vstart", type=float, default=0.40, help="Umbral arranque velocidad (u/s)")
    ap.add_argument("--vstop", type=float, default=0.15, help="Umbral parada velocidad (u/s)")
    ap.add_argument("--min-dur", type=float, default=0.20, help="Duración mínima segmento (s)")
    ap.add_argument("--min-gap", type=float, default=0.15, help="Fusión de huecos < (s)")
    ap.add_argument("--poly-n", type=int, default=20, help="Puntos polilínea trayectoria")
    ap.add_argument("--min-path-norm", type=float, default=0.015, help="Longitud mínima de trayectoria (0..1) para aceptar el segmento")
    ap.add_argument("--expected-n", type=int, default=0, help="Forzar este número de segmentos (p. ej., 24)")
    ap.add_argument("--spec", default="", help="Spec JSON (si se indica, se fuerza expected-n = len(moves))")
    ap.add_argument("--no-pose", action="store_true", help="No dibujar esqueleto de pose en el preview")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    video_path = Path(args.video)
    out_json = Path(args.out_json)

    # Crear directorios de salida si no existen
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if args.render_out:
        Path(args.render_out).parent.mkdir(parents=True, exist_ok=True)

    expected_n = int(args.expected_n or 0)
    if args.spec:
        spec_p = Path(args.spec)
        if not spec_p.exists():
            sys.exit(f"No existe spec: {spec_p}")
        expected_n = _expected_from_spec(spec_p)

    print(f"Procesando: {csv_path}")
    print(f"Video: {video_path}")
    print(f"Segmentos esperados: {expected_n}")

    try:
        res = capture_moves_from_csv(
            csv_path, video_path=video_path,
            vstart=args.vstart, vstop=args.vstop,
            min_dur=args.min_dur, min_gap=args.min_gap,
            poly_n=args.poly_n, min_path_norm=args.min_path_norm,
            expected_n=expected_n if expected_n > 0 else None
        )

        out_json.write_text(res.to_json(), encoding="utf-8")
        print(f"[OK] JSON -> {out_json}")
        print(f"[OK] Segmentos detectados: {len(res.moves)}")

        if args.render_out:
            render_p = Path(args.render_out)
            render_preview(video_path, res, render_p, csv_path=csv_path, draw_pose=(not args.no_pose))
            print(f"[OK] Preview -> {render_p}")

    except Exception as e:
        print(f"[ERROR] Falló la captura: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()