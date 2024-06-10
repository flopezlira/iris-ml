from pymongo import MongoClient
import json


def print_collection_content(collection_name: str) -> None:
    """
    Imprime el contenido de una colección de MongoDB.

    Args:
        collection_name (str): El nombre de la colección a imprimir.
    """
    client = MongoClient('localhost', 27017)
    db = client['ml_prototype']
    collection = db[collection_name]
    documents = collection.find()

    print(f"\nContenido de la colección '{collection_name}':")
    for doc in documents:
        print(json.dumps(doc, indent=4, default=str))


if __name__ == "__main__":
    print_collection_content('current_model')
    print_collection_content('metrics_history')
