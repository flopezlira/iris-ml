"""
This module contains the ModelManager class which is responsible for loading the model from the specified path and
"""
import logging
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


class ModelManager:
    """Class to manage the model loading and prediction."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        """Load the model from the specified path."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = joblib.load(f)
        except FileNotFoundError as error:
            logging.error("Model file not found: %s", self.model_path)
            raise FileNotFoundError from error

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict the class labels for the input features."""
        if self.model is None:
            self.load_model()
        return self.model.predict(features)

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model on the test data and return the metrics."""
        if self.model is None:
            self.load_model()
        predictions = self.model.predict(x_test)
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }
        return metrics
