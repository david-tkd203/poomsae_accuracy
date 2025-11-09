"""
Clasificador ML para posturas de Taekwondo.
Entrena Random Forest para clasificar ap_kubi, dwit_kubi, beom_seogi.
"""
from __future__ import annotations
from typing import Tuple, Optional, List
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class StanceClassifier:
    """
    Clasificador de posturas basado en Random Forest.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_split: int = 5,
        random_state: int = 42,
        class_weight: str = 'balanced'
    ):
        """
        Args:
            n_estimators: Número de árboles en el bosque
            max_depth: Profundidad máxima de cada árbol
            min_samples_split: Mínimo de muestras para split
            random_state: Semilla para reproducibilidad
            class_weight: Peso de clases ('balanced', 'balanced_subsample', None)
        """
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            class_weight=class_weight  # Manejar desbalance de clases
        )
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5
    ) -> dict:
        """
        Entrena el clasificador con cross-validation.
        
        Args:
            X_train: Features de entrenamiento (n_samples, n_features)
            y_train: Etiquetas (n_samples,)
            feature_names: Nombres de las features
            cv_folds: Número de folds para cross-validation
        
        Returns:
            Dict con métricas de entrenamiento
        """
        self.feature_names = feature_names
        
        # Codificar etiquetas
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        print(f"🎓 Entrenando Random Forest...")
        print(f"   Muestras: {len(X_train)}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Clases: {list(self.label_encoder.classes_)}")
        print(f"   Distribución: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        # Cross-validation
        print(f"\n🔄 Cross-validation ({cv_folds}-fold)...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.clf, X_train, y_encoded, cv=cv, scoring='accuracy')
        
        print(f"   Accuracy por fold: {[f'{s:.3f}' for s in cv_scores]}")
        print(f"   Media: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Entrenamiento final
        self.clf.fit(X_train, y_encoded)
        self.is_fitted = True
        
        # Feature importance
        if feature_names:
            importances = self.clf.feature_importances_
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            print(f"\n📊 Feature Importances:")
            for name, imp in feat_imp:
                print(f"   {name:20s}: {imp:.4f}")
        
        return {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
            "importances": self.clf.feature_importances_.tolist()
        }
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> dict:
        """
        Evalúa el clasificador en conjunto de prueba.
        
        Args:
            X_test: Features de prueba
            y_test: Etiquetas verdaderas
            verbose: Si imprimir resultados
        
        Returns:
            Dict con métricas detalladas
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado. Llamar train() primero.")
        
        # Predicciones
        y_pred = self.predict(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Por clase
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.label_encoder.classes_)
        
        if verbose:
            print(f"\n📊 EVALUACIÓN EN TEST SET")
            print(f"   Accuracy:  {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            print(f"   F1-score:  {f1:.3f}")
            print(f"\n📋 Reporte por clase:")
            print(classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_
            ))
            print(f"\n🔲 Confusion Matrix:")
            print(cm)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "class_report": class_report
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clases para nuevas muestras"""
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado")
        y_encoded = self.clf.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades de cada clase"""
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado")
        return self.clf.predict_proba(X)
    
    def save(self, path: Path):
        """Guarda modelo entrenado"""
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'clf': self.clf,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }, f)
        print(f"✅ Modelo guardado en: {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'StanceClassifier':
        """Carga modelo desde archivo"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.clf = data['clf']
        instance.label_encoder = data['label_encoder']
        instance.feature_names = data['feature_names']
        instance.is_fitted = True
        
        return instance
    
    def plot_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """Plotea y guarda confusion matrix"""
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=self.label_encoder.classes_)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix - Stance Classification')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Confusion matrix guardada en: {save_path}")
        plt.close()

# ==================== TRAINING SCRIPT ====================

def train_from_csv(
    csv_path: Path,
    model_output: Path,
    test_size: float = 0.2,
    cv_folds: int = 5
) -> StanceClassifier:
    """
    Entrena clasificador desde CSV de labels.
    
    Args:
        csv_path: Ruta al CSV con features y labels
        model_output: Ruta donde guardar modelo
        test_size: Fracción para test set
        cv_folds: Folds para cross-validation
    
    Returns:
        Clasificador entrenado
    """
    print(f"📂 Cargando dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Features y target
    feature_cols = [
        "ankle_dist_sw", "hip_offset_x", "hip_offset_y",
        "knee_angle_left", "knee_angle_right",
        "foot_angle_left", "foot_angle_right",
        "hip_behind_feet"
    ]
    
    # Agregar knee_angle_diff si existe
    if "knee_angle_diff" in df.columns:
        feature_cols.append("knee_angle_diff")
    
    X = df[feature_cols].to_numpy(np.float32)
    y = df["stance_label"].to_numpy()
    
    print(f"\n📊 Dataset:")
    print(f"   Total: {len(df)} muestras")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Clases: {np.unique(y)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n📐 Train/Test split ({100*(1-test_size):.0f}/{100*test_size:.0f}):")
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test:  {len(X_test)} muestras")
    
    # Entrenar
    clf = StanceClassifier(n_estimators=100, max_depth=15, random_state=42)
    metrics = clf.train(X_train, y_train, feature_names=feature_cols, cv_folds=cv_folds)
    
    # Evaluar
    test_metrics = clf.evaluate(X_test, y_test)
    
    # Guardar
    clf.save(model_output)
    
    # Confusion matrix
    cm_path = model_output.parent / "confusion_matrix.png"
    clf.plot_confusion_matrix(X_test, y_test, save_path=cm_path)
    
    return clf

if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Entrenar clasificador de posturas")
    ap.add_argument("--csv", default="data/labels/stance_labels_auto.csv", help="CSV con labels")
    ap.add_argument("--output", default="data/models/stance_classifier.pkl", help="Archivo de salida")
    ap.add_argument("--test-size", type=float, default=0.2, help="Fracción para test")
    ap.add_argument("--cv-folds", type=int, default=5, help="Folds para CV")
    args = ap.parse_args()
    
    clf = train_from_csv(
        Path(args.csv),
        Path(args.output),
        test_size=args.test_size,
        cv_folds=args.cv_folds
    )
    
    print("\n🎉 Entrenamiento completado!")
