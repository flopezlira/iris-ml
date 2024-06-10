import logging
import os
from datetime import datetime
from typing import Dict

from pymongo import MongoClient, errors
from dotenv import load_dotenv
from file_manager import FileManager

load_dotenv()


class MongoDBManager:
    """
    Singleton class to manage the connection to a MongoDB database.
    """
    _instance = None

    def __new__(cls, uri: str = None, db_name: str = None):
        if cls._instance is None:
            cls._instance = super(MongoDBManager, cls).__new__(cls)
            uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            db_name = os.getenv('DB_NAME', 'ml_prototype')
            try:
                cls._instance.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                cls._instance.client.server_info()  # Forzar conexión para verificar que está funcionando
                cls._instance.db = cls._instance.client[db_name]
            except errors.ServerSelectionTimeoutError as err:
                raise Exception(f"Could not connect to MongoDB: {err}")
        return cls._instance

    def get_current_model(self) -> str:
        """Retrieve the current model file from the database."""
        collection = self.db['current_model']
        result = collection.find_one()
        if result is None:
            model_file = FileManager.get_first_model_file()
            return model_file
        return result['model_file']

    def update_current_model(self, model_file: str) -> None:
        """Update the current model file in the database."""
        collection = self.db['current_model']
        collection.update_one({}, {'$set': {'model_file': model_file}}, upsert=True)

    def store_metrics(self, model_file: str, metrics: Dict) -> None:
        """Store evaluation metrics of a model in the database."""
        collection = self.db['metrics_history']
        metrics['model_file'] = model_file
        metrics['timestamp'] = datetime.now().isoformat()
        collection.insert_one(metrics)
