# src/obs/config.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

CFG_PATH = Path("obs_settings.json")

@dataclass
class ObsSettings:
    layout: str = "studio"                 # studio | grid
    kinect_mode: str = "streams"           # streams | composite
    include_webcams: bool = True
    target_height: int = 540

    # Grabación
    record_fps: int = 30
    record_codec: str = "mp4v"
    record_scale: float = 0.6667
    record_mosaic: bool = True
    out_root: str = "recordings"

    # Webcams
    max_probe: int = 4
    consecutive_fail_stop: int = 2
    exclude_indices: list[int] = (1,)      # típica webcam integrada
    exclude_name_contains: list[str] = ()

    def save(self, path: Path = CFG_PATH):
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path = CFG_PATH) -> "ObsSettings":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return ObsSettings(**data)
        return ObsSettings()
