"""
Handles file operations such as loading pickle files, loading test data, listing model files, etc.
"""
import os
from typing import Any, List, Tuple

import joblib
import numpy as np


class FileManager:
    """Utility class to manage file operations."""

    @staticmethod
    def load_pickle_file(file_path: str) -> Any:
        """Load a pickle file from the specified path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'rb') as f:
            return joblib.load(f)

    @staticmethod
    def load_test_data(features_path: str = 'data/test-data.p', labels_path: str = 'data/test-labels.p') -> \
            Tuple[np.ndarray, np.ndarray]:
        """Load test data (features and labels) from specified paths."""
        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Test data files not found")
        x_test = np.load(features_path, allow_pickle=True)
        y_test = np.load(labels_path, allow_pickle=True)
        return x_test, y_test

    @staticmethod
    def list_model_files(directory: str = 'models') -> List[str]:
        """List all model files in the specified directory."""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        return [f for f in os.listdir(directory) if f.endswith('.p')]

    @staticmethod
    def get_first_model_file(directory: str = 'models') -> str:
        """Get the first model file in the specified directory."""
        model_files = FileManager.list_model_files(directory)
        if not model_files:
            raise FileNotFoundError("No model files found in the directory")
        return model_files[0]
