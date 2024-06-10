import joblib
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model_manager import ModelManager


@pytest.fixture
def setup_model(tmp_path):
    model = RandomForestClassifier(n_estimators=10)
    X_train = np.random.rand(50, 4)
    y_train = np.random.randint(0, 3, 50)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.p"
    with open(model_path, 'wb') as f:
        joblib.dump(model, f)

    return model_path


def test_load_model(setup_model):
    model_path = setup_model
    model_manager = ModelManager(model_path)
    model_manager.load_model()
    assert model_manager.model is not None


def test_predict(setup_model):
    model_path = setup_model
    model_manager = ModelManager(model_path)
    model_manager.load_model()
    features = np.random.rand(1, 4)
    prediction = model_manager.predict(features)
    assert prediction.shape == (1,)


def test_evaluate(setup_model):
    model_path = setup_model
    model_manager = ModelManager(model_path)
    model_manager.load_model()
    X_test = np.random.rand(10, 4)
    y_test = np.random.randint(0, 3, 10)
    metrics = model_manager.evaluate(X_test, y_test)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
