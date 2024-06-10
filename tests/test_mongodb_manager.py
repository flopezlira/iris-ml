import pytest
import mongomock
from mongodb_manager import MongoDBManager


@pytest.fixture
def mongo_client():
    mongo_client = mongomock.MongoClient()
    db = mongo_client['ml_prototype']
    db['current_model'].insert_one({'model_file': 'model1.p'})
    db['metrics_history'].insert_one({'model_file': 'model1.p', 'accuracy': 0.9})
    return mongo_client


def test_get_current_model(mongo_client, monkeypatch):
    def mock_mongo_client(*args, **kwargs):
        return mongo_client

    monkeypatch.setattr('mongodb_manager.MongoClient', mock_mongo_client)

    db_manager = MongoDBManager()
    current_model = db_manager.get_current_model()
    assert current_model == 'model1.p'


def test_update_current_model(mongo_client, monkeypatch):
    def mock_mongo_client(*args, **kwargs):
        return mongo_client

    monkeypatch.setattr('mongodb_manager.MongoClient', mock_mongo_client)

    db_manager = MongoDBManager()
    db_manager.update_current_model('model2.p')
    current_model = db_manager.get_current_model()
    assert current_model == 'model2.p'


def test_store_metrics(mongo_client, monkeypatch):
    def mock_mongo_client(*args, **kwargs):
        return mongo_client

    monkeypatch.setattr('mongodb_manager.MongoClient', mock_mongo_client)

    db_manager = MongoDBManager()
    db_manager.store_metrics('model2.p', {'accuracy': 0.85})
    metrics_history = db_manager.db['metrics_history'].find_one({'model_file': 'model2.p'})
    assert metrics_history['accuracy'] == 0.85
