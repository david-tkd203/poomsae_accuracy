# src/obs/analysis_bridge.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import json
import statistics as stats
from typing import Dict, Any, List

class LivePoomsaeEngine:
    """
    Consume el dict `fd` que entrega KinectPoomsaeCapture.get_frame()
    (usa claves 'biomechanics' y 'bodies') y expone un resumen listo
    para la UI del panel Poomsae (en vivo).
    """
    def __init__(self):
        self.last_summary: Dict[str, Any] = {}

    def update_from_fd(self, fd: Dict[str, Any]):
        bios = fd.get("biomechanics") or []
        # Tomamos el primer cuerpo trackeado como referencia rápida
        out = {}
        if bios:
            ang = bios[0].get("angles", {})
            met = bios[0].get("metrics", {})
            out["angles_deg"] = {
                "L_elbow": ang.get("left_elbow"),
                "R_elbow": ang.get("right_elbow"),
                "L_shoulder": ang.get("left_shoulder"),
                "R_shoulder": ang.get("right_shoulder"),
                "L_hip": ang.get("left_hip"),
                "R_hip": ang.get("right_hip"),
                "L_knee": ang.get("left_knee"),
                "R_knee": ang.get("right_knee"),
            }
            out["metrics"] = {
                "shoulder_balance": met.get("shoulder_balance"),
                "hip_balance": met.get("hip_balance"),
                "torso_inclination_deg": met.get("torso_inclination_deg"),
                "grave_L_knee": met.get("grave_left_knee"),
                "grave_R_knee": met.get("grave_right_knee"),
                "grave_L_hip":  met.get("grave_left_hip"),
                "grave_R_hip":  met.get("grave_right_hip"),
            }
        self.last_summary = out

    def as_table_rows(self) -> List[tuple]:
        """
        Devuelve lista [("Métrica","Valor"), ...] para poblar QTableWidget.
        """
        rows = []
        ang = (self.last_summary.get("angles_deg") or {})
        met = (self.last_summary.get("metrics") or {})
        for k, v in ang.items():
            rows.append((f"Ángulo {k}", f"{v:.1f}°" if isinstance(v, (int,float)) else "—"))
        for k, v in met.items():
            rows.append((k, f"{v:.2f}" if isinstance(v, (int,float)) else ("✔" if v else ("✖" if v is False else "—"))))
        return rows


def _find_backend_json(session_dir: Path) -> Path | None:
    """
    Busca el JSON de métricas generado por KinectPoomsaeCapture.stop_recording()
    dentro de la carpeta 'kinect_backend' de la sesión.
    """
    kb = session_dir / "kinect_backend"
    if not kb.exists():
        return None
    cands = list(kb.glob("*_bodies_metrics.json"))
    return cands[0] if cands else None


def generate_simple_report(session_dir: str | Path) -> Path:
    """
    Lee el *_bodies_metrics.json de la sesión y genera report_simple.html
    con estadísticas (media, p5, p95) de ángulos y métricas básicas.
    """
    session_dir = Path(session_dir)
    js = _find_backend_json(session_dir)
    if js is None:
        raise FileNotFoundError("No se encontró JSON de métricas en kinect_backend/")

    data = json.loads(Path(js).read_text(encoding="utf-8"))
    # data es lista de frames [{'frame':..,'biomechanics':[{'angles':..,'metrics':..}], ...}, ...]

    agg: Dict[str, list] = {}
    for item in data:
        bios = item.get("biomechanics") or []
        if not bios:
            continue
        a = bios[0].get("angles", {}) or {}
        m = bios[0].get("metrics", {}) or {}
        for k, v in a.items():
            if isinstance(v, (int,float)):
                agg.setdefault(f"angle.{k}", []).append(float(v))
        for k, v in m.items():
            if isinstance(v, (int,float)):
                agg.setdefault(f"metric.{k}", []).append(float(v))

    def _stat(xs):
        xs = sorted(xs)
        if not xs: return ("—","—","—")
        mean = stats.fmean(xs)
        p5  = xs[max(0, int(0.05*len(xs))-1)]
        p95 = xs[min(len(xs)-1, int(0.95*len(xs))-1)]
        return (mean, p5, p95)

    rows = []
    for k, xs in sorted(agg.items()):
        mean, p5, p95 = _stat(xs)
        def _fmt(x): return f"{x:.2f}" if isinstance(x, float) else x
        rows.append((k, _fmt(mean), _fmt(p5), _fmt(p95)))

    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<style>body{font-family:Inter,Segoe UI,Arial;padding:24px;background:#111;color:#eee}",
        "table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #333;padding:8px 10px;font-size:14px}",
        "th{background:#222}</style>",
        f"<h2>Reporte Poomsae – Sesión: {session_dir.name}</h2>",
        "<p>Estadísticas por ángulo/métrica (media, p5, p95) calculadas desde los frames capturados.</p>",
        "<table><tr><th>Variable</th><th>Media</th><th>P5</th><th>P95</th></tr>"
    ]
    html += [f"<tr><td>{k}</td><td>{m}</td><td>{p5}</td><td>{p95}</td></tr>" for (k,m,p5,p95) in rows]
    html += ["</table>"]
    out = session_dir / "report_simple.html"
    out.write_text("\n".join(html), encoding="utf-8")
    return out
