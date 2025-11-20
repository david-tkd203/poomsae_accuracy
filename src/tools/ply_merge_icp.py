# src/tools/ply_merge_icp.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, glob
from pathlib import Path
import numpy as np
import open3d as o3d

def prep(pcd, voxel):
    if voxel and voxel>0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="carpeta con pc_*.ply")
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--thresh", type=float, default=0.07)
    ap.add_argument("--maxit", type=int, default=600)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.folder)/"pc_*.ply")))
    if not files:
        raise SystemExit("No hay pc_*.ply en la carpeta.")

    tgt = prep(o3d.io.read_point_cloud(files[0]), args.voxel)
    T = np.eye(4)
    global_pcd = tgt

    for f in files[1:]:
        src = prep(o3d.io.read_point_cloud(f), args.voxel)
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, args.thresh, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=args.maxit)
        )
        T = T @ reg.transformation
        src_global = src.transform(T.copy())
        global_pcd += src_global
        global_pcd = global_pcd.voxel_down_sample(args.voxel)
        tgt = src  # encadena

    out = Path(args.folder)/"merged_fused_cloud.ply"
    o3d.io.write_point_cloud(str(out), global_pcd)
    print(f"[merge] listo: {out}")
    try:
        o3d.visualization.draw_geometries([global_pcd])
    except Exception:
        pass

if __name__ == "__main__":
    main()
