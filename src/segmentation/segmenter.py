import numpy as np

class Segmenter:
    """
    Segmentación por actividad: usa cambios angulares agregados.
    Retorna segmentos [start,end] de “unidad de movimiento”.
    """
    def __init__(self, speed_thresh_deg_s=45.0, min_segment_frames=10, max_pause_frames=8):
        self.vth = speed_thresh_deg_s
        self.minlen = min_segment_frames
        self.maxpause = max_pause_frames

    def _activity(self, angle_series, fps):
        # energía de cambio: suma de |grad| de varios ángulos
        grads = [np.abs(np.gradient(a)) for a in angle_series if len(a)>0]
        if not grads: return np.zeros(1)
        g = np.sum(grads, axis=0) * fps
        return g

    def find_segments(self, angles_dict, fps):
        """
        angles_dict: dict nombre->serie de ángulos (deg) para todo el video.
        fps: frames por segundo.
        """
        act = self._activity(list(angles_dict.values()), fps)
        active = act > self.vth
        segs = []
        start = None; silence = 0
        for i, on in enumerate(active):
            if on and start is None:
                start = i; silence = 0
            elif on and start is not None:
                silence = 0
            elif not on and start is not None:
                silence += 1
                if silence >= self.maxpause:
                    end = i - silence
                    if end - start + 1 >= self.minlen:
                        segs.append((start, end))
                    start = None; silence = 0
        if start is not None:
            end = len(active)-1
            if end - start + 1 >= self.minlen:
                segs.append((start, end))
        return segs
