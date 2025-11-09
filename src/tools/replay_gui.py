# src/tools/replay_gui.py
"""
GUI de depuración para poomsae_accuracy
---------------------------------------
- Muestra el video fotograma a fotograma.
- Corre la segmentación usando src.segmentation.move_capture.capture_moves_from_csv
  (el mismo pipeline que estás usando en CLI).
- Carga el 8yang_spec.json y lo aplana (idx + sub).
- Alinea "lo que el JSON dice" vs "lo que el segmentador detectó".
- Permite fijar un "inicio real" (por ej. 5.00 s) para ver el desfase.
- Exporta un reporte JSON.

Requisitos:
    pip install pillow opencv-python

Ejecución:
    python -m src.tools.replay_gui --video ... --csv ... --spec ...
"""

from __future__ import annotations
import argparse
import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# IMPORTA TU SEGMENTADOR
# (asumimos que 'src' es paquete; si no, ajusta sys.path)
from src.segmentation import move_capture as mc


# ---------------------------------------------------------------------
# utilidades de especificación
# ---------------------------------------------------------------------

def load_spec(spec_path: Path) -> Dict[str, Any]:
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_moves(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Toma el 8yang_spec.json y lo pasa a una lista plana:
    [
      {"order":1, "idx":1, "sub":"a", "label":"Bloqueo...", ...},
      {"order":2, "idx":1, "sub":"b", ...},
      {"order":3, "idx":2, "sub":"a", ...},
      ...
    ]
    Esto es lo que pintamos en la interfaz.
    """
    flat = []
    order = 1
    for m in spec.get("moves", []):
        item = {
            "order": order,
            "idx": m.get("idx"),
            "sub": m.get("sub", ""),
            "tech_kor": m.get("tech_kor", ""),
            "tech_es": m.get("tech_es", ""),
            "stance_code": m.get("stance_code", ""),
            "level": m.get("level", ""),
            "turn": m.get("turn", ""),
            "travel": m.get("travel", ""),
        }
        # nombre presentable
        label = m.get("tech_es") or m.get("tech_kor") or f"Paso {m.get('idx')}"
        item["label"] = label
        flat.append(item)
        order += 1
    return flat


# ---------------------------------------------------------------------
# GUI principal
# ---------------------------------------------------------------------
class PoomsaeReplayApp(tk.Tk):
    def __init__(self, video_path: Path, csv_path: Path, spec_path: Path):
        super().__init__()
        self.title("Poomsae Accuracy · Video Replay / Debug")
        self.geometry("1250x720")

        self.video_path = video_path
        self.csv_path = csv_path
        self.spec_path = spec_path

        # ---------------- datos/fuentes ----------------
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.nframes: int = 0
        self.current_frame_idx: int = 0
        self.playing: bool = False
        self.play_thread: Optional[threading.Thread] = None
        self.stop_flag: bool = False

        # resultados de segmentación
        self.capture_result: Optional[mc.CaptureResult] = None
        # spec
        self.spec_data: Dict[str, Any] = {}
        self.spec_moves: List[Dict[str, Any]] = []
        # para desfase manual
        self.real_start_s: float = 0.0

        # cargamos al inicio
        self._load_video()
        self._load_spec()
        self._run_segmentation()

        # ---------------- UI ----------------
        self._build_ui()

        # pinta el primer frame
        self._show_frame(0)

    # -------------------------------------------------
    # carga de datos
    # -------------------------------------------------
    def _load_video(self):
        if not self.video_path.exists():
            messagebox.showerror("Error", f"No se encontró el video: {self.video_path}")
            return
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            messagebox.showerror("Error", f"No se pudo abrir el video: {self.video_path}")
            return
        self.cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def _load_spec(self):
        if not self.spec_path.exists():
            messagebox.showerror("Error", f"No se encontró el spec: {self.spec_path}")
            self.spec_data = {}
            self.spec_moves = []
            return
        self.spec_data = load_spec(self.spec_path)
        self.spec_moves = flatten_moves(self.spec_data)

    def _run_segmentation(self):
        """
        Llama a la misma función que usas en CLI para que NO haya diferencias.
        """
        if not self.csv_path.exists():
            messagebox.showerror("Error", f"No se encontró el CSV de landmarks: {self.csv_path}")
            return

        # ojo: aquí puedes ajustar los parámetros a lo que te está funcionando
        result = mc.capture_moves_from_csv(
            self.csv_path,
            video_path=self.video_path,
            vstart=0.55,
            vstop=0.18,
            min_dur=0.22,
            min_gap=0.22,
            smooth_win=5,
            poly_n=20,
            min_path_norm=0.015,
            expected_n=len(self.spec_moves)  # ← forzamos el mismo N que el spec
        )
        self.capture_result = result

    # -------------------------------------------------
    # construcción de la interfaz
    # -------------------------------------------------
    def _build_ui(self):
        # ---- layout principal ----
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        # izquierda: video
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        # derecha: info
        right = ttk.Frame(main, width=380)
        right.pack(side="right", fill="y")

        # video canvas
        self.video_label = tk.Label(left, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # barra inferior de controles
        controls = ttk.Frame(left)
        controls.pack(fill="x")

        self.play_btn = ttk.Button(controls, text="▶ Play", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=4, pady=4)

        ttk.Button(controls, text="⏸ Pause", command=self.pause).pack(side="left", padx=2)
        ttk.Button(controls, text="⏮ Prev mov", command=self.goto_prev_move).pack(side="left", padx=2)
        ttk.Button(controls, text="⏭ Next mov", command=self.goto_next_move).pack(side="left", padx=2)
        ttk.Button(controls, text="Set inicio real", command=self.set_real_start).pack(side="left", padx=5)
        ttk.Button(controls, text="Exportar reporte", command=self.export_report).pack(side="left", padx=5)

        # slider de tiempo
        self.time_scale = ttk.Scale(controls, from_=0, to=self.nframes-1, orient="horizontal", command=self._on_slider)
        self.time_scale.pack(side="right", fill="x", expand=True, padx=4, pady=4)

        # ---- Panel derecho ----
        # info general
        info_frame = ttk.LabelFrame(right, text="Info video / captura")
        info_frame.pack(fill="x", padx=6, pady=6)

        self.info_lbl = tk.Label(info_frame, justify="left", anchor="w")
        self.info_lbl.pack(fill="x", padx=4, pady=4)

        self._update_info_label()

        # expected moves
        spec_frame = ttk.LabelFrame(right, text="Secuencia esperada (spec)")
        spec_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.spec_tree = ttk.Treeview(spec_frame, columns=("idx", "label"), show="headings", height=10)
        self.spec_tree.heading("idx", text="#")
        self.spec_tree.heading("label", text="Movimiento")
        self.spec_tree.column("idx", width=40, anchor="center")
        self.spec_tree.column("label", width=240, anchor="w")
        self.spec_tree.pack(fill="both", expand=True)

        for m in self.spec_moves:
            text = m["label"]
            if m.get("sub"):
                text = f"{text} ({m['sub']})"
            self.spec_tree.insert("", "end", values=(m["order"], text))

        # detected moves
        det_frame = ttk.LabelFrame(right, text="Movimientos detectados")
        det_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.det_tree = ttk.Treeview(det_frame, columns=("i", "start", "dur", "limb"), show="headings", height=10)
        self.det_tree.heading("i", text="#")
        self.det_tree.heading("start", text="t_ini (s)")
        self.det_tree.heading("dur", text="dur (s)")
        self.det_tree.heading("limb", text="efector")
        self.det_tree.column("i", width=35, anchor="center")
        self.det_tree.column("start", width=70, anchor="e")
        self.det_tree.column("dur", width=70, anchor="e")
        self.det_tree.column("limb", width=80, anchor="center")
        self.det_tree.pack(fill="both", expand=True)

        self._fill_detected_tree()

    # -------------------------------------------------
    # acciones de UI
    # -------------------------------------------------
    def _on_slider(self, event=None):
        val = int(float(self.time_scale.get()))
        self.current_frame_idx = val
        self._show_frame(val, force=True)

    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def play(self):
        if self.playing:
            return
        self.playing = True
        self.play_btn.config(text="⏹ Stop")
        self.stop_flag = False
        self._play_loop()

    def pause(self):
        self.playing = False
        self.play_btn.config(text="▶ Play")

    def _play_loop(self):
        if not self.playing:
            return
        start = time.time()
        self._show_frame(self.current_frame_idx)
        self.current_frame_idx += 1
        if self.current_frame_idx >= self.nframes:
            self.current_frame_idx = self.nframes - 1
            self.pause()
            return
        self.time_scale.set(self.current_frame_idx)

        # cálculo de delay
        delay = max(1, int(1000 / self.fps))
        # reposicionar
        self.after(delay, self._play_loop)

    def goto_prev_move(self):
        if not self.capture_result or not self.capture_result.moves:
            return
        t = self.current_frame_idx / self.fps
        prev = None
        for m in self.capture_result.moves:
            if m.t_start < t:
                prev = m
            else:
                break
        if prev:
            self.current_frame_idx = int(prev.a)
            self.time_scale.set(self.current_frame_idx)
            self._show_frame(self.current_frame_idx, force=True)

    def goto_next_move(self):
        if not self.capture_result or not self.capture_result.moves:
            return
        t = self.current_frame_idx / self.fps
        for m in self.capture_result.moves:
            if m.t_start > t:
                self.current_frame_idx = int(m.a)
                self.time_scale.set(self.current_frame_idx)
                self._show_frame(self.current_frame_idx, force=True)
                break

    def set_real_start(self):
        """
        Marca "aquí empezó de verdad el poomsae".
        Esto te va a mostrar el desfase respecto a la primera detección.
        """
        self.real_start_s = self.current_frame_idx / self.fps
        self._update_info_label()
        messagebox.showinfo("Inicio real fijado",
                            f"Inicio real fijado en t = {self.real_start_s:.2f} s.\n"
                            f"Ahora el reporte mostrará el desfase de cada movimiento.")

    def export_report(self):
        if not self.capture_result:
            messagebox.showerror("Error", "No hay resultado de segmentación")
            return

        # alineación básica por orden
        rows = []
        det_moves = list(self.capture_result.moves)
        for i, exp in enumerate(self.spec_moves):
            det = det_moves[i] if i < len(det_moves) else None
            det_start = det.t_start if det else None
            det_end = det.t_end if det else None
            lag = None
            if det_start is not None:
                lag = det_start - self.real_start_s
            rows.append({
                "order": exp["order"],
                "expected_idx": exp["idx"],
                "expected_sub": exp["sub"],
                "expected_label": exp["label"],
                "expected_stance": exp["stance_code"],
                "detected_start_s": det_start,
                "detected_end_s": det_end,
                "detected_limb": det.active_limb if det else None,
                "detected_stance": det.stance_pred if det else None,
                "detected_rotation": det.rotation_bucket if det else None,
                "lag_vs_real_start_s": lag
            })

        report = {
            "video": str(self.video_path),
            "csv": str(self.csv_path),
            "spec": str(self.spec_path),
            "fps": self.fps,
            "nframes": self.nframes,
            "real_start_s": self.real_start_s,
            "detected_moves_raw": [m.__dict__ for m in self.capture_result.moves],
            "alignment": rows,
        }

        out_dir = Path("reports")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"replay_{self.video_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        messagebox.showinfo("Reporte exportado", f"Reporte guardado en:\n{out_path}")

    # -------------------------------------------------
    # dibujo de video
    # -------------------------------------------------
    def _show_frame(self, fidx: int, force: bool = False):
        if not self.cap:
            return

        # OpenCV no reposiciona solo, hay que setear el frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = self.cap.read()
        if not ok:
            return

        # dibujar overlay de segmentos
        frame = self._draw_overlays(frame, fidx)

        # convertir a tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        # redimensionar a la mitad si es muy grande
        im = im.resize((900, int(900 * im.size[1] / im.size[0])), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=im)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _draw_overlays(self, frame: np.ndarray, fidx: int) -> np.ndarray:
        if not self.capture_result:
            return frame

        H, W = frame.shape[:2]
        # que movimientos están activos en este frame
        active = [m for m in self.capture_result.moves if m.a <= fidx <= m.b]

        # línea de tiempo arriba
        t = fidx / self.fps
        cv2.putText(frame, f"{t:6.2f}s ({fidx}/{self.nframes})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"{t:6.2f}s ({fidx}/{self.nframes})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        y = 55
        for m in active[:3]:
            txt = f"det #{m.idx}  {m.active_limb}  {m.stance_pred}  {m.rotation_bucket}"
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            y += 20

        # resaltar el esperado en ese orden (aprox por tiempo)
        # buscamos el movimiento detectado "actual" → el primero que lo incluya
        if active:
            det_idx = active[0].idx
            # en el tree de la derecha, seleccionar la fila correspondiente
            for iid in self.spec_tree.get_children():
                vals = self.spec_tree.item(iid, "values")
                if vals and int(vals[0]) == det_idx:
                    self.spec_tree.selection_set(iid)
                    self.spec_tree.see(iid)
                    break

        return frame

    # -------------------------------------------------
    def _update_info_label(self):
        txt = [
            f"Video: {self.video_path.name}",
            f"FPS: {self.fps:.2f}",
            f"Frames: {self.nframes}",
            f"CSV: {self.csv_path.name}",
            f"Spec: {self.spec_path.name}",
            f"Inicio real (marcado): {self.real_start_s:.2f} s",
        ]
        if self.capture_result:
            txt.append(f"Movimientos detectados: {len(self.capture_result.moves)}")
        else:
            txt.append("Movimientos detectados: (no hay)")
        self.info_lbl.configure(text="\n".join(txt))

    def _fill_detected_tree(self):
        for iid in self.det_tree.get_children():
            self.det_tree.delete(iid)
        if not self.capture_result:
            return
        for m in self.capture_result.moves:
            self.det_tree.insert(
                "",
                "end",
                values=(
                    m.idx,
                    f"{m.t_start:.2f}",
                    f"{m.duration:.2f}",
                    m.active_limb,
                ),
            )


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="GUI de depuración de poomsae_accuracy")
    ap.add_argument("--video", required=True, help="Ruta del video .mp4 / .mov")
    ap.add_argument("--csv", required=True, help="Ruta del CSV de landmarks (salida de extract)")
    ap.add_argument("--spec", required=True, help="Ruta del archivo de especificación (8yang_spec.json)")
    args = ap.parse_args()

    app = PoomsaeReplayApp(Path(args.video), Path(args.csv), Path(args.spec))
    app.mainloop()


if __name__ == "__main__":
    main()
