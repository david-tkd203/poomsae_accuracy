# bg_remove.py
# Extracción de persona y eliminación de fondo con OpenCV (MOG2 o GrabCut+HOG)

import argparse, os
from collections import deque
import cv2
import numpy as np

# ------------------- Utilidades -------------------

def gray_world_balance(img_bgr):
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1,3).mean(axis=0)
    scale = means.mean() / (means + 1e-6)
    img *= scale
    return np.clip(img, 0, 255).astype(np.uint8)

def refine_mask(mask, k_close=5, k_open=3, blur=7):
    mask = (mask > 0).astype(np.uint8) * 255
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close,k_close))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,k_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2, iterations=1)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (blur|1, blur|1), 0)
    return mask

def composite(frame, mask, bg_mode="black", alpha_out=False):
    h, w = frame.shape[:2]
    if bg_mode == "black":
        bg = np.zeros_like(frame)
    elif bg_mode == "green":
        bg = np.zeros_like(frame); bg[:] = (0,255,0)
    elif bg_mode == "blur":
        bg = cv2.GaussianBlur(frame, (31,31), 0)
    elif bg_mode == "gray":
        bg = np.full_like(frame, 128)
    else:
        bg = np.zeros_like(frame)

    mask3 = cv2.merge([mask, mask, mask]) / 255.0
    fg = (frame * mask3 + bg * (1.0 - mask3)).astype(np.uint8)

    if alpha_out:
        return np.dstack([fg, mask])
    return fg

# ------------------- MOG2 (fondo estático) -------------------

class MOG2Remover:
    def __init__(self, history=500, var_threshold=25, detect_shadows=True):
        self.backsub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )

    def warmup(self, cap, frames=120, resize_w=960):
        for _ in range(frames):
            ok, f = cap.read()
            if not ok: break
            f = resize_to_width(f, resize_w)
            self.backsub.apply(f, learningRate=0.5)

    def infer_mask(self, frame_bgr):
        fg = self.backsub.apply(frame_bgr, learningRate=-1)
        # 0: fondo, 127: sombra, 255: primer plano
        mask = (fg == 255).astype(np.uint8) * 255
        return refine_mask(mask)

# ------------------- GrabCut + HOG (cámara móvil) -------------------

def detect_person_bbox_hog(frame_bgr):
    if not hasattr(detect_person_bbox_hog, "hog"):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        detect_person_bbox_hog.hog = hog
    hog = detect_person_bbox_hog.hog
    h, w = frame_bgr.shape[:2]
    scale_w = 960
    if w != scale_w:
        rs = cv2.resize(frame_bgr, (scale_w, int(h * scale_w / w)))
        scx, scy = w/scale_w, h/rs.shape[0]
    else:
        rs = frame_bgr; scx = scy = 1.0

    rects, _ = hog.detectMultiScale(rs, winStride=(8,8), padding=(8,8), scale=1.05)
    if len(rects) == 0:
        return None
    x,y,ww,hh = max(rects, key=lambda r: r[2]*r[3])
    x,y,ww,hh = int(x*scx), int(y*scy), int(ww*scx), int(hh*scy)
    x1,y1,x2,y2 = x, y, x+ww, y+hh
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(w-1,x2), min(h-1,y2)
    if x2<=x1 or y2<=y1: return None
    return (x1,y1,x2,y2)

def grabcut_person_mask(frame_bgr, init_box=None, iters=3):
    h, w = frame_bgr.shape[:2]
    if init_box is None:
        init_box = detect_person_bbox_hog(frame_bgr)
        if init_box is None:
            # ROI central como *fallback*
            cw, ch = int(w*0.6), int(h*0.85)
            x1 = (w-cw)//2; y1 = int(h*0.08)
            init_box = (x1, y1, x1+cw, y1+ch)

    x1,y1,x2,y2 = init_box
    mask = np.zeros((h,w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (x1,y1,x2-x1,y2-y1)
    cv2.grabCut(frame_bgr, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    # 0:BG, 2:prob BG -> fondo; 1:FG, 3:prob FG -> frente
    mask_bin = np.where((mask==1)|(mask==3), 255, 0).astype('uint8')
    return refine_mask(mask_bin)

# ------------------- I/O helpers -------------------

def resize_to_width(frame, w_target=960):
    h, w = frame.shape[:2]
    if w == w_target: return frame
    return cv2.resize(frame, (w_target, int(h * w_target / w)))

def write_video_writer(path, w, h, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

# ------------------- Main -------------------

def run_image(args):
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("No se pudo leer la imagen.")
    img = resize_to_width(img, 960)
    img_bal = gray_world_balance(img)

    if args.method == "mog2":
        # Para imagen, MOG2 no tiene *background model*; usar GrabCut
        mask = grabcut_person_mask(img_bal, None, iters=4)
    else:
        mask = grabcut_person_mask(img_bal, None, iters=4)

    out = composite(img_bal, mask, bg_mode=args.bg, alpha_out=args.alpha)

    cv2.imshow("Entrada", img)
    cv2.imshow("Mascara", mask)
    cv2.imshow("Salida", out if not args.alpha else out[:,:,:3])

    if args.save:
        ext = os.path.splitext(args.save.lower())[1]
        if args.alpha and ext in [".png", ".tif", ".tiff", ".webp"]:
            cv2.imwrite(args.save, out)                 # BGRA
        else:
            cv2.imwrite(args.save, out if not args.alpha else out[:,:,:3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_stream(args):
    cap = cv2.VideoCapture(0 if (args.source is None) else args.source)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la fuente de video.")
    writer = None
    history = deque(maxlen=8)

    mog = None
    if args.method == "mog2":
        mog = MOG2Remover(history=600, var_threshold=25, detect_shadows=True)
        if args.warmup > 0 and isinstance(args.source, str):
            mog.warmup(cap, frames=args.warmup)

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = resize_to_width(frame, 960)
        frame_bal = gray_world_balance(frame)

        if args.method == "mog2":
            mask = mog.infer_mask(frame_bal)
        else:
            # Opción: usar HOG una vez cada N frames y luego rectángulo previo (recorte rápido)
            mask = grabcut_person_mask(frame_bal, None, iters=2)

        out = composite(frame_bal, mask, bg_mode=args.bg, alpha_out=False)

        if args.save and writer is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            writer = write_video_writer(args.save, out.shape[1], out.shape[0], fps=fps)
        if writer is not None:
            writer.write(out)

        cv2.imshow("Entrada", frame)
        cv2.imshow("Mascara", mask)
        cv2.imshow("Salida", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break

    if writer is not None: writer.release()
    cap.release()
    cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, help="Ruta a imagen")
    ap.add_argument("--source", type=str, help="Ruta a video (MP4, etc.)")
    ap.add_argument("--camera", type=int, help="Índice de cámara (si no se usa --source)")
    ap.add_argument("--method", choices=["mog2","grabcut"], default="grabcut",
                    help="mog2: fondo estático; grabcut: fondo cambiante")
    ap.add_argument("--bg", choices=["black","green","blur","gray"], default="black",
                    help="Fondo de salida")
    ap.add_argument("--alpha", action="store_true", help="Sólo imagen: exportar con canal alfa")
    ap.add_argument("--save", type=str, help="Ruta de salida (imagen o video)")
    ap.add_argument("--warmup", type=int, default=0, help="MOG2: frames de calentamiento (sólo video)")
    args = ap.parse_args()

    if args.image:
        run_image(args)
    else:
        if args.source is None and args.camera is not None:
            args.source = args.camera
        run_stream(args)

if __name__ == "__main__":
    main()
