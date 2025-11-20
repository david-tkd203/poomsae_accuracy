# src/obs/config.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict
import json

CFG_PATH = Path("obs_settings.json")


@dataclass
class ObsSettings:
    # ---------- Layout / Mosaico ----------
    layout: str = "studio"               # "studio" | "grid"
    kinect_mode: str = "streams"         # "streams" | "composite"
    include_webcams: bool = True
    target_height: int = 540

    # Grilla/mosaico responsivo
    prefer_aspect: float = 16/9          # Relación preferida de cada tile (W/H)
    max_cols: int = 6                    # Límite superior de columnas al auto-ajustar
    max_tiles: int = 6                   # Máximo de fuentes visibles en mosaico
    grid_override: Optional[Tuple[int, int]] = None  # (rows, cols) o None (Auto)
    show_labels: bool = True             # Mostrar etiquetas sobre cada fuente

    # ---------- Grabación ----------
    record_fps: int = 30
    record_codec: str = "mp4v"
    record_scale: float = 0.6667
    record_mosaic: bool = True
    out_root: str = "recordings"

    # ---------- Webcams / Descubrimiento ----------
    max_probe: int = 4
    consecutive_fail_stop: int = 2
    exclude_indices: List[int] = None           # se normaliza a list en load()
    exclude_name_contains: List[str] = None     # se normaliza a list en load()

    # ---------- 3D / Point Cloud ----------
    pc_every_n: int = 10                 # Guardar un .ply cada N frames (si está activo)
    pc_decimate: int = 2                 # Salto de muestreo para la nube de puntos
    pc_colorize: bool = True             # Colorear puntos
    pc_prefer_mapper: bool = True        # Usar CoordinateMapper si está disponible

    # Visor 3D (PointCloudWidget)
    viewer_fov: float = 90.0
    viewer_clip_near: float = 0.01
    viewer_clip_far: float = 12000.0
    viewer_point_size: float = 3.0
    viewer_accumulate: bool = True
    viewer_accum_budget: int = 1_800_000
    viewer_voxel_size: float = 0.02
    viewer_autofit: str = "once"         # "off" | "once" | "always"

    # ---------- GPU / OpenGL (placeholder para futuras optimizaciones) ----------
    use_opengl: bool = True

    # --------- Métodos ---------
    def save(self, path: Path = CFG_PATH):
        """
        Guarda el JSON de configuración actual. Convierte tuplas a listas cuando corresponde.
        """
        data = asdict(self)
        # Normaliza grid_override para JSON (lista o None)
        if isinstance(data.get("grid_override"), tuple):
            data["grid_override"] = list(data["grid_override"])

        # Asegura listas no nulas
        if data.get("exclude_indices") is None:
            data["exclude_indices"] = [1]   # webcam integrada típica
        if data.get("exclude_name_contains") is None:
            data["exclude_name_contains"] = []

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path = CFG_PATH) -> "ObsSettings":
        """
        Carga robusta:
        - Ignora claves desconocidas (compatibilidad hacia delante/atrás).
        - Corrige tipos comunes (p.ej., tupla -> lista, None -> lista vacía).
        - Mantiene defaults si faltan claves.
        """
        inst = ObsSettings()
        # Defaults seguros para listas mutables
        if inst.exclude_indices is None:
            inst.exclude_indices = [1]
        if inst.exclude_name_contains is None:
            inst.exclude_name_contains = []

        if not path.exists():
            return inst

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return inst

        # Mapa de campos válidos
        valid: Dict[str, Any] = {f.name: f for f in fields(ObsSettings)}

        for k, v in (data.items() if isinstance(data, dict) else []):
            if k not in valid:
                # Campo desconocido en versiones anteriores/posteriores
                continue

            try:
                if k == "grid_override":
                    # Acepta None o secuencia de 2 enteros
                    if v in (None, [], [None, None]):
                        setattr(inst, k, None)
                    elif isinstance(v, (list, tuple)) and len(v) >= 2:
                        r = int(v[0]) if v[0] is not None else 0
                        c = int(v[1]) if v[1] is not None else 0
                        setattr(inst, k, (r, c) if (r > 0 and c > 0) else None)
                    else:
                        setattr(inst, k, None)

                elif k in ("exclude_indices", "exclude_name_contains"):
                    if isinstance(v, (list, tuple)):
                        setattr(inst, k, list(v))
                    else:
                        setattr(inst, k, [])
                else:
                    # Asignación directa: ints, floats, bools, str…
                    setattr(inst, k, v)
            except Exception:
                # Si algo falla, conserva el valor por defecto del campo
                pass

        # Post-normalización por si venía None
        if inst.exclude_indices is None:
            inst.exclude_indices = [1]
        if inst.exclude_name_contains is None:
            inst.exclude_name_contains = []

        return inst
