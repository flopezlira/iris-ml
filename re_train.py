"""
Selects a different model from the 'models' directory, evaluates it with the test data and stores the metrics in the
MongoDB database. The model in use is updated to the new model.
"""
import random

from file_manager import FileManager
from model_manager import ModelManager
from mongodb_manager import MongoDBManager


def retrain_model() -> None:
    """Select a different model, evaluate it and store the metrics in the database. Update the current model."""
    db_manager = MongoDBManager()
    file_manager = FileManager()

    # Obtener el archivo del modelo actual
    try:
        current_model_file = db_manager.get_current_model()
    except Exception:
        current_model_file = None

    # Seleccionar un modelo diferente aleatoriamente
    model_files = file_manager.list_model_files()
    if not model_files:
        raise FileNotFoundError("No models found in the 'models' directory")

    if current_model_file:
        model_files.remove(current_model_file)
    if not model_files:
        raise FileNotFoundError("No different models available")

    new_model_file = random.choice(model_files)
    model_manager = ModelManager(f'models/{new_model_file}')

    # Cargar los datos de prueba
    x_test, y_test = file_manager.load_test_data()

    # Evaluar el modelo
    metrics = model_manager.evaluate(x_test, y_test)
    db_manager.store_metrics(new_model_file, metrics)

    # Actualizar el modelo en uso
    db_manager.update_current_model(new_model_file)


if __name__ == "__main__":
    retrain_model()
