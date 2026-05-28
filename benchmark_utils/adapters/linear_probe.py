"""Linear probe adaptation strategy.

A foundation model encoder extracts embeddings; a linear classifier (or
regressor) is trained on top.  Suitable for classification and — with a
threshold on reconstruction error — anomaly detection.

Usage
-----
    adapter = LinearProbeAdapter(encoder, task="classification", n_classes=5)
    adapter.fit(X_train, y_train)        # called inside Solver.run()
    label = adapter.predict(x)           # called by objective
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base import BaseTSFMAdapter


class LinearProbeAdapter(BaseTSFMAdapter):
    """Frozen encoder + linear head.

    Parameters
    ----------
    encoder : object with ``encode(x: np.ndarray (T, C)) -> np.ndarray (D,)``
    task : {"classification", "anomaly_detection"}
    n_classes : int, required when task == "classification"
    max_iter : int
        Maximum iterations for the logistic regression solver.
    """

    def __init__(
        self,
        encoder,
        task="classification",
        n_classes=None,
        classifier="logistic_regression",
        penalty="l2",
        max_iter=1000,
        n_estimators=100,
    ):
        self.encoder = encoder
        self.task = task
        self.n_classes = n_classes
        self.classifier = classifier
        self.penalty = penalty
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self._label_enc = LabelEncoder()

    def fit(self, X_train, y_train, **kwargs):
        embeddings = self.encoder.encode(X_train)

        if self.task == "classification":
            y_enc = self._label_enc.fit_transform(y_train)

            # Define classifier
            match self.classifier.lower():
                case "logistic_regression":
                    self._head = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty=self.penalty,
                            max_iter=self.max_iter,
                        ),
                    )
                case "ridge_regression":
                    self._head = make_pipeline(
                        StandardScaler(),
                        RidgeClassifier(
                            max_iter=self.max_iter,
                            random_state=42,
                        ),
                    )
                case "random_forest":
                    self._head = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        n_jobs=-1,
                        random_state=42,
                        verbose=0,
                    )
                case "_":
                    raise ValueError(
                        f"Unknown classifier '{self.classifier}'. Choose between 'logistic_regression', 'ridge_regression', and 'random_forest'."
                    )
            self._head.fit(embeddings, y_enc)

        elif self.task == "anomaly_detection":
            # Train a reconstruction baseline: predict embedding from itself
            # (identity ridge) then use residual norm as anomaly score.
            # Participants can replace with a more principled approach.
            self._train_embeddings = embeddings
            self._train_mean = embeddings.mean(axis=0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        emb = self.encoder.encode(X)

        if self.task == "classification":
            label_enc = self._head.predict(emb)
            return self._label_enc.inverse_transform(label_enc)

        elif self.task == "anomaly_detection":
            # Score: L2 distance from the training mean embedding,
            # broadcast to every timestep (uniform window score).
            score = float(np.linalg.norm(emb - self._train_mean))
            return np.full(X.shape[0], score, dtype=np.float32)

        raise ValueError(f"Unknown task: {self.task}")
