"""
ml_model.py
============
Ensemble-based machine–learning pipeline for the SOL/USDT trading dashboard.

•  Builds robust features via feature_engineering.generate_features
•  Trains an ensemble (RandomForest + GradientBoosting + LogisticRegression)
•  Persists model, scalers and metadata to model/latest_model.joblib
•  Provides convenience wrappers  load_model()  and  predict_signal()
"""
"""
ml_model.py
============
Ensemble-based machine–learning pipeline for SOL/USDT trading.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler

from config import MODEL_DIR, ML_FEATURES
from feature_engineering import generate_features, validate_features, get_feature_importance

LOGGER = logging.getLogger(__name__)
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)


class MLTradingModel:
    """Wrapper around an ensemble of ML classifiers."""

    def __init__(self):
        self.ensemble = None
        self.scaler = None
        self.feature_names = []
        self.metadata = {}
        self.is_trained = False

    @staticmethod
    def _make_ensemble():
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        return VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("lr", lr)], voting="soft"
        )

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        LOGGER.info("ML model – training started")
        X = generate_features(df)
        if not validate_features(X):
            raise ValueError("Feature validation failed")

        df_aligned = df.loc[X.index].copy()
        if "target" not in df_aligned:
            df_aligned["target"] = (df_aligned["close"].shift(-1) > df_aligned["close"]).astype(int)

        mask = ~df_aligned["target"].isna()
        X, y = X[mask], df_aligned["target"][mask]

        if len(X) < 300:
            raise ValueError("Insufficient training data")

        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.scaler = RobustScaler().fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.ensemble = self._make_ensemble()
        self.ensemble.fit(X_train_s, y_train)
        self.is_trained = True

        train_pred = self.ensemble.predict(X_train_s)
        test_pred = self.ensemble.predict(X_test_s)
        cv_scores = cross_val_score(self.ensemble, X_train_s, y_train, cv=5, scoring="accuracy")

        self.metadata = {
            "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_acc": accuracy_score(y_train, train_pred),
            "test_acc": accuracy_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred, average="weighted"),
            "recall": recall_score(y_test, test_pred, average="weighted"),
            "f1": f1_score(y_test, test_pred, average="weighted"),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "feature_count": len(self.feature_names),
        }
        self.metadata["feature_importance"] = get_feature_importance(
            self.ensemble, self.feature_names
        )

        self._save_latest()
        LOGGER.info("Model training completed – test_acc %.3f", self.metadata["test_acc"])
        return self.metadata

    def predict_proba(self, feature_df: pd.DataFrame) -> float:
        if not (self.is_trained and self.ensemble and self.scaler):
            LOGGER.warning("ML model not trained, returning 0.5")
            return 0.5

        for col in self.feature_names:
            if col not in feature_df:
                feature_df[col] = 0.0

        X = feature_df[self.feature_names].tail(1).fillna(0)
        X_s = self.scaler.transform(X)
        return float(self.ensemble.predict_proba(X_s)[0, 1])

    def _save_latest(self):
        bundle = {
            "model": self.ensemble,
            "scaler": self.scaler,
            "features": self.feature_names,
            "metadata": self.metadata,
        }
        path = Path(MODEL_DIR) / "latest_model.joblib"
        joblib.dump(bundle, path)
        LOGGER.info("Model bundle loaded from %s", path)

    def load(self, path: str = None) -> bool:
        file = Path(path) if path else Path(MODEL_DIR) / "latest_model.joblib"
        if not file.exists():
            LOGGER.warning("Model file not found: %s", file)
            return False

        bundle = joblib.load(file)
        self.ensemble = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle["features"]
        self.metadata = bundle["metadata"]
        self.is_trained = True
        LOGGER.info("Model bundle loaded from %s", file)
        return True

_ml = MLTradingModel()

def train_model(data_path: str) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
    return _ml.train(df)

def load_model(path: str = None) -> bool:
    return _ml.load(path)

def predict_signal(model, features: pd.DataFrame) -> float:
    return _ml.predict_proba(features)
