#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, glob, re, time
import numpy as np

import open3d as o3d

# Opcional para exportar video (si no está, el script sigue funcionando)
try:
    import cv2
except Exception:
    cv2 = None

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(s))]

def load_pcd(path):
    p = o3d.io.read_point_cloud(path)
    if p.is_empty():
        # evitar crasheo si un archivo viene vacío
        p.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
        p.colors = o3d.utility.Vector3dVector(np.ones((1,3))*0.7)
    return p

def main():
    ap = argparse.ArgumentParser(description="Reproductor de secuencias PLY")
    ap.add_argument("folder", help="Carpeta con .ply (ej: recordings\\20251115_160422\\pointcloud)")
    ap.add_argument("--fps", type=float, default=10.0, help="FPS de reproducción (default 10)")
    ap.add_argument("--loop", action="store_true", help="Repetir al final")
    ap.add_argument("--point-size", type=float, default=2.0, help="Tamaño del punto (default 2)")
    ap.add_argument("--width", type=int, default=1280, help="Ancho ventana")
    ap.add_argument("--height", type=int, default=720, help="Alto ventana")
    ap.add_argument("--out", type=str, default=None, help="MP4 de salida (opcional). Requiere OpenCV.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, "*.ply")), key=natural_key)
    if not paths:
        print("No encontré .ply en:", args.folder)
        return

    print(f"Encontrados {len(paths)} archivos PLY.")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("PLY Player (Espacio=Play/Pause, ←/→, +/-, R, H, Esc)", width=args.width, height=args.height)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = float(args.point_size)

    # Estado
    idx = 0
    playing = True
    period = 1.0 / max(1e-6, args.fps)
    t_last = time.time()
    cache = {}

    # Cargar primero
    cache[idx] = load_pcd(paths[idx])
    pcd = cache[idx]
    vis.add_geometry(pcd)

    # Writer (opcional)
    writer = None
    if args.out and cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, args.fps, (args.width, args.height))
        if not writer.isOpened():
            print("[WARN] No se pudo abrir el MP4 de salida; continuo sin exportar.")
            writer = None
    elif args.out and cv2 is None:
        print("[WARN] Para exportar MP4 se requiere OpenCV (pip install opencv-python).")

    def show(i):
        nonlocal pcd
        if i not in cache:
            cache[i] = load_pcd(paths[i])
        new = cache[i]
        pcd.points = new.points
        pcd.colors = new.colors
        vis.update_geometry(pcd)
        vis.poll_events(); vis.update_renderer()
        print(f"[{i+1}/{len(paths)}] {os.path.basename(paths[i])}")

    # Animación (se llama cada frame de render)
    def anim_cb(_):
        nonlocal t_last, idx, playing
        now = time.time()
        if playing and (now - t_last) >= period:
            if idx < len(paths) - 1:
                idx += 1
            elif args.loop:
                idx = 0
            show(idx)
            t_last = now

            # captura para video (opcional)
            if writer is not None:
                img = np.asarray(vis.capture_screen_float_buffer(do_render=False))
                img = (img * 255).astype(np.uint8)
                # Open3D entrega RGB; convertimos a BGR para OpenCV
                img = img[:, :, ::-1]
                # redimensionamos exacto al tamaño de ventana destino
                img = cv2.resize(img, (args.width, args.height))
                writer.write(img)
        return False

    # Controles
    def toggle_play(_): 
        nonlocal playing; playing = not playing; return False
    def next_frame(_): 
        nonlocal idx, playing; playing = False; idx = min(idx+1, len(paths)-1); show(idx); return False
    def prev_frame(_): 
        nonlocal idx, playing; playing = False; idx = max(idx-1, 0); show(idx); return False
    def bigger(_): 
        opt.point_size = min(opt.point_size + 1.0, 20.0); return False
    def smaller(_): 
        opt.point_size = max(opt.point_size - 1.0, 1.0); return False
    def reload(_): 
        cache.clear(); show(idx); return False
    def home(_):
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1]); ctr.set_up([0, 1, 0]); ctr.set_lookat([0, 0, 0]); ctr.set_zoom(0.8)
        return False
    def quit_app(_):
        vis.close()
        return False

    # Keycodes: → 262, ← 263 (Open3D)
    vis.register_key_callback(ord(' '), toggle_play)
    vis.register_key_callback(262, next_frame)
    vis.register_key_callback(263, prev_frame)
    vis.register_key_callback(ord('+'), bigger)
    vis.register_key_callback(ord('-'), smaller)
    vis.register_key_callback(ord('R'), reload)
    vis.register_key_callback(ord('H'), home)
    vis.register_key_callback(256, quit_app)  # Esc

    vis.register_animation_callback(anim_cb)
    show(idx)
    vis.run()
    vis.destroy_window()
    if writer is not None:
        writer.release()
        print(f"MP4 guardado en: {args.out}")

if __name__ == "__main__":
    main()
