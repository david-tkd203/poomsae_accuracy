from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import pandas as pd
import numpy as np

def load_spec_keys(spec_path: Path) -> list[str]:
    d = json.loads(spec_path.read_text(encoding="utf-8"))
    flat = []
    for m in d.get("moves", []):
        key = f"{m.get('idx','?')}{m.get('sub','')}"
        flat.append(key)
    return flat

def pct_ok_excl_ns(series: pd.Series) -> float:
    m = series.astype(str).values
    mask = (m != "NS")
    n = int(mask.sum())
    return float((m[mask] == "OK").sum())/n if n>0 else np.nan

def main():
    ap = argparse.ArgumentParser(description="Audita reportes 8yang: consolida métricas por movimiento y valida cobertura del spec.")
    ap.add_argument("--report-xlsx", required=True, help="Reporte por video generado por score_pal_yang.py (detalle/resumen)")
    ap.add_argument("--by-move-xlsx", default="", help="(Opcional) Consolidado previo por movimiento")
    ap.add_argument("--spec", required=True, help="8yang_spec.json para verificar cobertura (24 movimientos)")
    ap.add_argument("--out-xlsx", required=True, help="Salida Excel con análisis")
    args = ap.parse_args()

    # --- carga
    det = pd.read_excel(args.report_xlsx, sheet_name="detalle")
    if {"video_id","M","comp_arms","comp_legs","comp_kick","ded_total_move","is_correct","t0","t1"}.issubset(det.columns) is False:
        sys.exit("El sheet 'detalle' no tiene columnas mínimas requeridas")

    # normalizaciones
    det["video_id"] = det["video_id"].astype(str)
    det["M"] = det["M"].astype(str)
    det["is_correct"] = det["is_correct"].astype(str).str.lower()

    # --- métricas globales
    total_rows = len(det)
    ns_rows = det[(det["comp_arms"]=="NS") | (det["comp_legs"]=="NS") | (det["comp_kick"]=="NS")]
    scored_rows = det.drop(ns_rows.index)

    overview = pd.DataFrame([dict(
        videos=len(det["video_id"].unique()),
        rows_total=total_rows,
        rows_NS=len(ns_rows),
        rows_scored=len(scored_rows),
        pct_rows_scored=round(100.0*len(scored_rows)/max(1,total_rows),2),
        moves_correct_90p=int((det["is_correct"]=="yes").sum()),
        pct_correct_90p=round(100.0*(det["is_correct"]=="yes").mean(),2),
        comp_arms_ok_pct=round(100.0*pct_ok_excl_ns(det["comp_arms"]),2),
        comp_legs_ok_pct=round(100.0*pct_ok_excl_ns(det["comp_legs"]),2),
        comp_kick_ok_pct=round(100.0*pct_ok_excl_ns(det["comp_kick"]),2),
        ded_total_sum=round(float(det["ded_total_move"].fillna(0).sum()),3),
    )])

    # --- cobertura de spec
    spec_keys = load_spec_keys(Path(args.spec))
    present_keys = sorted(det["M"].dropna().unique().tolist())
    missing = sorted(list(set(spec_keys) - set(present_keys)))
    extra   = sorted(list(set(present_keys) - set(spec_keys)))
    coverage = pd.DataFrame(dict(
        spec_moves=[len(spec_keys)],
        present=[len(present_keys)],
        missing=[len(missing)],
        extra=[len(extra)]
    ))
    missing_df = pd.DataFrame({"missing_M": missing})
    extra_df   = pd.DataFrame({"extra_M": extra})

    # --- agregados por movimiento (a partir de “detalle” de todos los videos)
    def sev(x):
        # severidad agregada por fila: OK/LEVE/GRAVE tomando el peor componente
        order = {"OK":0,"LEVE":1,"GRAVE":2}
        vals = []
        for c in ("comp_arms","comp_legs","comp_kick"):
            v = str(x.get(c,""))
            if v in order: vals.append(order[v])
        return ["OK","LEVE","GRAVE"][max(vals) if vals else 0]

    det["_sev"] = det.apply(sev, axis=1)

    agg = det.groupby("M").agg(
        n=("M","size"),
        n_scored=("comp_arms", lambda s: int((s!="NS").sum())),
        n_ok90=("is_correct", lambda s: int((s=="yes").sum())),
        pct_ok90=("is_correct", lambda s: 100.0*float((s=="yes").mean())),
        mean_ded=("ded_total_move","mean"),
        med_ded=("ded_total_move","median"),
        arms_ok=("comp_arms", pct_ok_excl_ns),
        legs_ok=("comp_legs", pct_ok_excl_ns),
        kick_ok=("comp_kick", pct_ok_excl_ns)
    ).reset_index()

    sev_tab = det.pivot_table(index="M", columns="_sev", values="video_id", aggfunc="count", fill_value=0)
    for col in ["OK","LEVE","GRAVE"]:
        if col not in sev_tab.columns: sev_tab[col]=0
    sev_tab = sev_tab[["OK","LEVE","GRAVE"]].reset_index()

    # --- por-video rápido
    by_video = det.groupby("video_id").agg(
        rows=("M","size"),
        ns=("comp_arms", lambda s: int((s=="NS").sum())),
        scored=("comp_arms", lambda s: int((s!="NS").sum())),
        ok90=("is_correct", lambda s: int((s=="yes").sum())),
        pct_ok90=("is_correct", lambda s: 100.0*float((s=="yes").mean())),
        ded_sum=("ded_total_move","sum")
    ).reset_index()

    # --- NS diagnostic
    ns_diag = ns_rows[["video_id","M","tech_es","comp_arms","comp_legs","comp_kick","t0","t1"]].copy()

    # --- integra “by-move-xlsx” si existe (solo para referencia)
    by_move_in = {}
    if args.by_move_xlsx:
        try:
            by_move_in = pd.read_excel(args.by_move_xlsx, sheet_name=None)
        except Exception as e:
            print(f"[WARN] No se pudo leer by-move-xlsx: {e}")

    # --- export
    out = Path(args.out_xlsx)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        overview.to_excel(xw, index=False, sheet_name="overview")
        coverage.to_excel(xw, index=False, sheet_name="spec_coverage")
        missing_df.to_excel(xw, index=False, sheet_name="spec_missing")
        extra_df.to_excel(xw, index=False, sheet_name="spec_extra")
        agg.sort_values("M").to_excel(xw, index=False, sheet_name="per_move")
        sev_tab.sort_values("M").to_excel(xw, index=False, sheet_name="per_move_severity")
        by_video.sort_values("video_id").to_excel(xw, index=False, sheet_name="per_video")
        ns_diag.to_excel(xw, index=False, sheet_name="NS_rows")
        # vuelca sheets originales opcionales
        if by_move_in:
            for sh, df in by_move_in.items():
                df.to_excel(xw, index=False, sheet_name=f"by_move_in__{sh[:20]}")
    print(f"[OK] {out}")

if __name__ == "__main__":
    main()
