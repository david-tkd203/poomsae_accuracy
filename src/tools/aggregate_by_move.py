# src/tools/aggregate_by_move.py
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
from typing import List
import pandas as pd

# Usa el Excel generado por score_pal_yang.py para TODOS los videos y agrega vistas por movimiento

def main():
    ap = argparse.ArgumentParser(description="Agregación global por movimiento (M=idx/sub) en todo el dataset.")
    ap.add_argument("--in-xlsx", required=True, help="Excel de score_pal_yang (puede ser fusionado de varios)")
    ap.add_argument("--out-xlsx", required=True, help="Salida Excel con agregados por movimiento")
    args = ap.parse_args()

    x = Path(args.in_xlsx)
    if not x.exists():
        sys.exit(f"No existe: {x}")

    det = pd.read_excel(x, sheet_name="detalle")
    if det.empty:
        sys.exit("Hoja 'detalle' vacía")

    # limpiar NS
    d = det.copy()
    d = d[d["comp_arms"] != "NS"].copy()

    # etiquetas target por movimiento
    d["sev_arms"] = d["comp_arms"].map({"OK":0,"LEVE":1,"GRAVE":2}).fillna(0).astype(int)
    d["sev_legs"] = d["comp_legs"].map({"OK":0,"LEVE":1,"GRAVE":2}).fillna(0).astype(int)
    d["sev_kick"] = d["comp_kick"].map({"OK":0,"LEVE":1,"GRAVE":2}).fillna(0).astype(int)

    # resumen por M
    def pct_ok(col):
        return (col== "OK").mean()*100.0

    grp = d.groupby("M", as_index=False).agg(
        n_instances=("M","size"),
        pct_ok_arms=("comp_arms", pct_ok),
        pct_ok_legs=("comp_legs", pct_ok),
        pct_ok_kick=("comp_kick", pct_ok),
        ankle_dist_sw_mean=("ankle_dist_sw","mean"),
        ankle_dist_sw_std=("ankle_dist_sw","std"),
        rear_foot_turn_deg_mean=("rear_foot_turn_deg","mean"),
        kick_amp_mean=("kick_amp","mean"),
        kick_peak_mean=("kick_peak","mean"),
        kick_plantar_deg_mean=("kick_plantar_deg","mean"),
        kick_gaze_deg_mean=("kick_gaze_deg","mean"),
    ).sort_values("M")

    # confusión de postura esperada vs medida
    legs_conf = d.pivot_table(index="stance(exp)", columns="stance(meas)", values="M", aggfunc="size", fill_value=0).reset_index()

    # export
    out = Path(args.out_xlsx)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        d.to_excel(xw, index=False, sheet_name="detalle_all")
        grp.to_excel(xw, index=False, sheet_name="by_move")
        legs_conf.to_excel(xw, index=False, sheet_name="stance_confusion")

    print(f"[OK] {out}")

if __name__ == "__main__":
    main()
