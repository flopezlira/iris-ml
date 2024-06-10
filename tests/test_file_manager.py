import pytest
import os
import numpy as np
from file_manager import FileManager


@pytest.fixture
def setup_files(tmp_path):
    # Crear archivos temporales de prueba
    model_path = tmp_path / "models"
    model_path.mkdir()
    (model_path / "model1.p").write_text("dummy model 1 content")
    (model_path / "model2.p").write_text("dummy model 2 content")

    data_path = tmp_path / "data"
    data_path.mkdir()
    np.save(data_path / "test_features.npy", np.random.rand(10, 4))
    np.save(data_path / "test_labels.npy", np.random.randint(0, 3, 10))
    return model_path, data_path


def test_list_model_files(setup_files):
    model_path, _ = setup_files
    model_files = FileManager.list_model_files(model_path)
    assert len(model_files) == 2
    assert "model1.p" in model_files
    assert "model2.p" in model_files


def test_get_first_model_file(setup_files):
    model_path, _ = setup_files
    first_model_file = FileManager.get_first_model_file(model_path)
    assert first_model_file == "model1.p"


def test_load_pickle_file(setup_files):
    model_path, _ = setup_files
    with pytest.raises(FileNotFoundError):
        FileManager.load_pickle_file("non_existent_file.p")


def test_load_test_data(setup_files):
    _, data_path = setup_files
    X_test, y_test = FileManager.load_test_data(data_path / "test_features.npy", data_path / "test_labels.npy")
    assert X_test.shape == (10, 4)
    assert y_test.shape == (10,)
