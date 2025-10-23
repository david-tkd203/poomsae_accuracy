# src/eval/patterns.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

def load_spec(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # normaliza y ordena por idx y sub
    moves = data.get("moves", [])
    def key(m):
        idx = int(m.get("idx", 0))
        sub = m.get("sub", "")
        # orden a < b < c < "" (sin sub al final del mismo idx)
        sorder = {"a":0,"b":1,"c":2,"":3}
        return (idx, sorder.get(str(sub), 9))
    data["moves"] = sorted(moves, key=key)
    return data

def is_kick_move(move: Dict[str, Any]) -> bool:
    tkor = (move.get("tech_kor") or "").lower()
    tes  = (move.get("tech_es") or "").lower()
    return ("chagi" in tkor) or ("chagi" in tes) or ("patada" in tes)

def label_from_score(s: float) -> str:
    if s is None:
        return "no_eval"
    if s >= 0.90:
        return "correcto"
    if s >= 0.50:
        return "error_leve"
    return "error_grave"

def turn_affinity(expected: str, detected: str) -> float:
    # afinidad simple por giro
    if expected == "NONE":
        return 1.0 if detected == "STRAIGHT" else 0.7  # pequeño movimiento tolerado
    if expected == detected:
        return 1.0
    # mismo lado distinto ángulo
    if expected.startswith("LEFT") and detected.startswith("LEFT"):
        return 0.7
    if expected.startswith("RIGHT") and detected.startswith("RIGHT"):
        return 0.7
    if "180" in expected and detected == "TURN_180":
        return 0.9
    if expected == "TURN_180" and ("180" in detected):
        return 0.9
    return 0.0
