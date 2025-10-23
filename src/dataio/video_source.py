import cv2

class VideoSource:
    def __init__(self, path_or_index=0, resize_w=960):
        self.cap = cv2.VideoCapture(path_or_index)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir el video")
        self.resize_w = resize_w

    def frames(self):
        idx = 0
        while True:
            ok, img = self.cap.read()
            if not ok: break
            h,w = img.shape[:2]
            if w != self.resize_w:
                img = cv2.resize(img, (self.resize_w, int(h*self.resize_w/w)))
            yield idx, img
            idx += 1

    def release(self):
        self.cap.release()
