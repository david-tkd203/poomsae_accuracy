from abc import ABC, abstractmethod

class PoseBackend(ABC):
    @abstractmethod
    def iter_frames(self):
        """Yield dict: {'frame_idx':int,'image':ndarray}"""
        ...

    @abstractmethod
    def landmarks_for_frame(self, frame_idx):
        """Return list[(x,y[,z,vis])...] for ese frame_idx. Debe ser estable."""
        ...
