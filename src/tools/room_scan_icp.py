# src/tools/room_scan_icp.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time, argparse
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except Exception as e:
    raise SystemExit("Instala open3d:  pip install open3d\n" + str(e))

from src.kinect.kinect_backend import KinectPoomsaeCapture

def to_o3d_cloud(points: np.ndarray, voxel=0.03):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=30, help="duración de escaneo en segundos")
    ap.add_argument("--decimate", type=int, default=2, help="submuestreo desde depth (1=full)")
    ap.add_argument("--voxel", type=float, default=0.03, help="voxel (m) para downsample")
    ap.add_argument("--thresh", type=float, default=0.07, help="umbral ICP (m)")
    ap.add_argument("--maxpairs", type=int, default=800, help="máx. iteraciones ICP (pares)")
    ap.add_argument("--out", default=None, help="carpeta salida (por defecto recordings/scan_YYYYMMDD_HHMMSS)")
    args = ap.parse_args()

    kb = KinectPoomsaeCapture(enable_color=True, enable_depth=True, enable_ir=False,
                              enable_audio=False, enable_body=False, enable_body_index=False)
    if not kb.initialize():
        raise SystemExit("No se pudo inicializar Kinect.")

    ts = time.strftime("scan_%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("recordings")/ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scan] guardando en {out_dir}")

    pose = np.eye(4)  # transformación global
    global_pcd = None
    prev_pcd = None

    t0 = time.time()
    frames = 0
    while time.time() - t0 < args.secs:
        pc = kb.get_point_cloud(decimate=args.decimate, colorize=False, prefer_mapper=False)
        if not pc or pc["points"] is None or pc["points"].size == 0:
            continue
        pts = pc["points"].astype(np.float32)
        m = np.isfinite(pts).all(axis=1)
        pts = pts[m]
        if pts.size == 0:
            continue
        cur_pcd = to_o3d_cloud(pts, voxel=args.voxel)

        if prev_pcd is None:
            global_pcd = cur_pcd
            prev_pcd = cur_pcd
            continue

        # ICP point-to-plane
        reg = o3d.pipelines.registration.registration_icp(
            cur_pcd, prev_pcd, args.thresh,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=args.maxpairs)
        )
        pose = pose @ reg.transformation  # acumula
        cur_pcd_global = cur_pcd.transform(pose.copy())

        # fusiona
        global_pcd += cur_pcd_global
        global_pcd = global_pcd.voxel_down_sample(args.voxel)  # controla tamaño
        prev_pcd = cur_pcd

        frames += 1
        if frames % 20 == 0:
            print(f"[scan] frames={frames}, puntos={np.asarray(global_pcd.points).shape[0]}")

    # guarda
    out_ply = out_dir / "fused_cloud.ply"
    o3d.io.write_point_cloud(str(out_ply), global_pcd)
    print(f"[scan] listo: {out_ply}")

    # visor opcional
    try:
        o3d.visualization.draw_geometries([global_pcd])
    except Exception:
        pass

if __name__ == "__main__":
    main()
