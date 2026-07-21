"""Model training, serialization, and evaluation for ML agent (T504, T505, T506).

Supports:
- Majority class baseline
- Decision Tree classifier
- Random Forest classifier (main model)
- Action masking during prediction to guarantee zero illegal moves
- Model serialization via joblib
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from wumpus.domain import Action
from wumpus.encoder import FEATURE_VERSION, label_to_action


class MajorityBaseline:
    """Baseline model that always predicts the majority class from training data."""

    def __init__(self) -> None:
        self.majority_class: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaseline":
        values, counts = np.unique(y, return_counts=True)
        self.majority_class = int(values[np.argmax(counts)])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        probs = np.zeros((n_samples, 4), dtype=np.float32)
        probs[:, self.majority_class] = 1.0
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.majority_class, dtype=np.int64)


def train_models(
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    seed: int = 42,
) -> dict[str, Any]:
    """Train Majority, DecisionTree, and RandomForest models and return validation metrics."""
    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    models: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}

    # 1. Majority baseline
    maj = MajorityBaseline().fit(X_train, y_train)
    y_pred_maj = maj.predict(X_val)
    models["majority"] = maj
    metrics["majority"] = {
        "accuracy": float(accuracy_score(y_val, y_pred_maj)),
        "macro_f1": float(f1_score(y_val, y_pred_maj, average="macro", zero_division=0)),
    }

    # 2. Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, random_state=seed, class_weight="balanced")
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_val)
    models["decision_tree"] = dt
    metrics["decision_tree"] = {
        "accuracy": float(accuracy_score(y_val, y_pred_dt)),
        "macro_f1": float(f1_score(y_val, y_pred_dt, average="macro", zero_division=0)),
    }

    # 3. Random Forest (main model)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    models["random_forest"] = rf
    metrics["random_forest"] = {
        "accuracy": float(accuracy_score(y_val, y_pred_rf)),
        "macro_f1": float(f1_score(y_val, y_pred_rf, average="macro", zero_division=0)),
    }

    return {
        "models": models,
        "metrics": metrics,
    }


def predict_masked_action(
    model: Any,
    x_vec: np.ndarray,
    legal_actions: tuple[Action, ...],
) -> Action:
    """Predict action using model probabilities masked by legal actions.

    Guarantees that an illegal action is NEVER chosen if any legal action exists.
    """
    action_order = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    # Get class probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_vec.reshape(1, -1))[0]
        # Handle cases where model didn't see all 4 classes during training
        if len(probs) < 4 and hasattr(model, "classes_"):
            full_probs = np.zeros(4, dtype=np.float32)
            for cls_idx, cls_lbl in enumerate(model.classes_):
                full_probs[cls_lbl] = probs[cls_idx]
            probs = full_probs
    else:
        pred_label = model.predict(x_vec.reshape(1, -1))[0]
        probs = np.zeros(4, dtype=np.float32)
        probs[pred_label] = 1.0

    # Mask illegal actions with 0 probability
    masked_probs = np.copy(probs)
    for idx, act in enumerate(action_order):
        if act not in legal_actions:
            masked_probs[idx] = -1.0  # mask out

    # Pick highest probability among legal actions
    best_idx = int(np.argmax(masked_probs))
    best_action = action_order[best_idx]

    # Fallback if all masked out
    if best_action not in legal_actions:
        return legal_actions[0]

    return best_action


def save_model(model: Any, filepath: Path) -> None:
    """Serialize model to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_version": FEATURE_VERSION,
    }
    joblib.dump(payload, filepath)


def load_model(filepath: Path) -> Any:
    """Deserialize model from disk."""
    payload = joblib.load(filepath)
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"Feature version mismatch: model is {payload.get('feature_version')}, expected {FEATURE_VERSION}"
        )
    return payload["model"]
