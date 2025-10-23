import joblib
import numpy as np

class SegmentInfer:
    def __init__(self, model_path):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.cols  = bundle["cols"]

    def predict_row(self, feat_row_dict):
        x = np.array([feat_row_dict.get(c, 0.0) for c in self.cols], float).reshape(1,-1)
        y = self.model.predict(x)[0]
        proba = getattr(self.model, "predict_proba", lambda X: None)(x)
        return int(y), (proba[0] if proba is not None else None)
