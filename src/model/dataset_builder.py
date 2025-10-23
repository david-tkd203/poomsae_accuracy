import pandas as pd, glob, json
from pathlib import Path

class DatasetBuilder:
    """
    Une: features por segmento + anotaciones humanas (leve/grave/correcto).
    Anotaciones CSV (data/annotations/*.csv):
      video_id, segment_id, label  # label in {correcto, leve, grave}
    Features por segmento en CSV (uno por video_id) o unificado.
    """
    MAP = {"correcto":0, "leve":1, "grave":2}

    def build(self, features_glob, annotations_glob):
        feats = []
        for fp in glob.glob(features_glob):
            df = pd.read_csv(fp)
            feats.append(df)
        F = pd.concat(feats, ignore_index=True)

        anns = []
        for ap in glob.glob(annotations_glob):
            A = pd.read_csv(ap)
            anns.append(A)
        A = pd.concat(anns, ignore_index=True)

        df = F.merge(A, on=["video_id","segment_id"], how="inner")
        df["y"] = df["label"].map(self.MAP)
        # separar X/Y
        drop_cols = ["video_id","segment_id","label","y"]
        meta_cols = [c for c in df.columns if c.startswith("meta_")]
        X = df.drop(columns=drop_cols+meta_cols, errors="ignore")
        y = df["y"].values
        return X, y, df
