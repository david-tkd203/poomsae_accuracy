# src/model/train.py
from __future__ import annotations
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    elif path.suffix.lower() in [".csv"]:
        return pd.read_csv(path)
    else:
        raise SystemExit(f"Formato de dataset no soportado: {path}")

def _detect_target_col(df: pd.DataFrame) -> str:
    for c in ["y", "label", "target", "gt_label"]:
        if c in df.columns:
            return c
    raise SystemExit("No se encontró la columna de etiqueta (y/label/target).")

def _select_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    # Tomar solo columnas numéricas y descartar la etiqueta y típicos metadatos textuales
    drop_like = ["video", "video_id", "start_s", "end_s", "move_id", "segment_id"]
    bad = set([target_col])
    for c in df.columns:
        lc = c.lower()
        if any(k == c or lc.startswith(k) for k in drop_like) and (df[c].dtype == "O"):
            bad.add(c)
    # Solo numéricas
    num_cols = [c for c in df.columns if c not in bad and np.issubdtype(df[c].dtype, np.number)]
    if not num_cols:
        raise SystemExit("No se hallaron columnas numéricas de features.")
    return num_cols

def _make_model(model_name: str, args, n_classes: int):
    # Modelos + preprocesamiento: RF/HGB no requieren escalado; SVM/LogReg sí.
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight=(args.class_weight if args.class_weight != "none" else None),
            n_jobs=-1,
            random_state=args.seed,
        )
        pre = None  # imputación mínima (RF tolera NaN? No, imputamos)
    elif model_name == "hgb":
        clf = HistGradientBoostingClassifier(
            max_depth=args.hgb_max_depth if args.hgb_max_depth > 0 else None,
            learning_rate=args.hgb_lr,
            max_iter=args.hgb_max_iter,
            l2_regularization=args.hgb_l2,
            early_stopping=True,
            random_state=args.seed,
        )
        pre = None
    elif model_name == "svm":
        clf = SVC(
            kernel="rbf",
            C=args.svm_C,
            gamma="scale",
            probability=True,
            class_weight=(args.class_weight if args.class_weight != "none" else None),
            random_state=args.seed,
        )
        pre = ("scale", Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("sc", RobustScaler())]))
    elif model_name == "logreg":
        clf = LogisticRegression(
            C=args.lr_C,
            max_iter=2000,
            class_weight=(args.class_weight if args.class_weight != "none" else None),
            n_jobs=-1,
            random_state=args.seed,
        )
        pre = ("scale", Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("sc", StandardScaler())]))
    else:
        raise SystemExit(f"Modelo no soportado: {model_name}")

    return clf, pre

def _build_pipeline(num_cols: list[str], model, scaler_block):
    # Siempre imputamos median para seguridad
    if scaler_block is None:
        preproc = ColumnTransformer(
            transformers=[("num", SimpleImputer(strategy="median"), num_cols)],
            remainder="drop",
        )
    else:
        # scaler_block = ("scale", Pipeline([...]))
        preproc = ColumnTransformer(
            transformers=[("num", scaler_block[1], num_cols)],
            remainder="drop",
        )
    pipe = Pipeline([("pre", preproc), ("clf", model)])
    return pipe

def _kfold_eval(pipe, X: pd.DataFrame, y_enc: np.ndarray, cv_splits: int, seed: int, sample_weight=None):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    f1s, bals = [], []
    for tr, va in skf.split(X, y_enc):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y_enc[tr], y_enc[va]

        fit_kwargs = {}
        # sample_weight para HGB u otros
        if sample_weight is not None:
            fit_kwargs["clf__sample_weight"] = sample_weight[tr]

        pipe.fit(Xtr, ytr, **fit_kwargs)
        yhat = pipe.predict(Xva)
        f1s.append(f1_score(yva, yhat, average="macro"))
        bals.append(balanced_accuracy_score(yva, yhat))
    return float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(bals)), float(np.std(bals))

def main():
    ap = argparse.ArgumentParser(description="Entrena y guarda un clasificador de calidad de movimiento.")
    ap.add_argument("--data", required=True, help="Parquet/CSV de entrenamiento (características + y/label).")
    ap.add_argument("--out", required=True, help="Ruta .joblib de salida.")
    ap.add_argument("--model", default="rf", choices=["rf","hgb","svm","logreg"], help="Tipo de modelo.")
    ap.add_argument("--cv", type=int, default=5, help="Folds de cross-val.")

    # General
    ap.add_argument("--class-weight", default="none", choices=["none","balanced"], help="Pesos por clase.")
    ap.add_argument("--seed", type=int, default=42)

    # RF
    ap.add_argument("--rf-n-estimators", type=int, default=400)
    ap.add_argument("--rf-max-depth", type=int, default=20)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)

    # HGB
    ap.add_argument("--hgb-max-depth", type=int, default=8)
    ap.add_argument("--hgb-max-iter", type=int, default=500)
    ap.add_argument("--hgb-lr", type=float, default=0.05)
    ap.add_argument("--hgb-l2", type=float, default=0.0)

    # SVM
    ap.add_argument("--svm-C", type=float, default=4.0)

    # LogReg
    ap.add_argument("--lr-C", type=float, default=2.0)

    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    df = _read_dataset(data_path)
    target_col = _detect_target_col(df)

    # Codificar etiquetas
    y_raw = df[target_col].astype(str).values
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)

    # Selección de features
    feat_cols = _select_feature_cols(df, target_col)
    X = df[feat_cols].copy()

    # Info de clases
    classes = list(le.classes_)
    print("[INFO] Clases:", classes)
    cls_counts = pd.Series(y).value_counts().sort_index()
    print("[INFO] Distribución:", {classes[i]: int(cls_counts.get(i,0)) for i in range(len(classes))})

    # Modelo + preprocesamiento
    clf, scaler_block = _make_model(args.model, args, n_classes=len(classes))
    pipe = _build_pipeline(feat_cols, clf, scaler_block)

    # sample_weight si class_weight = balanced y el modelo no lo soporta nativo (HGB no tiene class_weight)
    sample_weight = None
    if args.model == "hgb" and args.class_weight == "balanced":
        weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(classes)), y=y)
        wmap = {i: weights[i] for i in range(len(classes))}
        sample_weight = np.array([wmap[yi] for yi in y], dtype=np.float32)

    # CV
    f1m, f1s, bam, bas = _kfold_eval(pipe, X, y, args.cv, args.seed, sample_weight=sample_weight)
    print(f"[CV] F1_macro: {f1m:.4f} ± {f1s:.4f} | BalancedAcc: {bam:.4f} ± {bas:.4f}")

    # Entrena en todo el conjunto
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight
    pipe.fit(X, y, **fit_kwargs)

    # Reporte sobre entrenamiento (referencial)
    yhat = pipe.predict(X)
    print("\n[TRAIN] Clasification report (referencial):")
    print(classification_report(y, yhat, target_names=classes, digits=3))
    print("[TRAIN] Matriz de confusión:\n", confusion_matrix(y, yhat))

    # Guardar
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": pipe,
        "feature_cols": feat_cols,
        "label_encoder": le,
        "meta": {
            "model_name": args.model,
            "cv_f1_macro_mean": f1m,
            "cv_f1_macro_std": f1s,
            "cv_balanced_acc_mean": bam,
            "cv_balanced_acc_std": bas,
            "classes": classes,
            "data_path": str(data_path),
        },
    }
    joblib.dump(payload, out_path)
    # Métricas a JSON
    metrics_json = out_path.with_suffix(".metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload["meta"], f, ensure_ascii=False, indent=2)

    print(f"[OK] Modelo -> {out_path}")
    print(f"[OK] Métricas -> {metrics_json}")


if __name__ == "__main__":
    main()
