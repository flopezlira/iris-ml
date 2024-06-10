"""
Flask API for making predictions with a trained model and updating the current model. The API expects the features of
the event as query parameters for the '/predict' endpoint and a JSON payload with the key 'model_file' for the
'/update_model' endpoint. The API also provides a '/metrics' endpoint for retrieving the metrics history.
"""
import json
import logging

import numpy as np
from flask import Flask, jsonify, request, make_response
from flask_restx import Api, Resource, fields
from mongodb_manager import MongoDBManager
from bson import json_util, ObjectId
from model_manager import ModelManager

app = Flask(__name__)
api = Api(app, version='1.0', title='ML API', description='Iris test ML API')

ns = api.namespace('api', description='Operations')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

model_input = api.model('ModelInput', {
    'sepallength': fields.Float(required=True, description='Sepal length'),
    'sepalwidth': fields.Float(required=True, description='Sepal width'),
    'petallength': fields.Float(required=True, description='Petal length'),
    'petalwidth': fields.Float(required=True, description='Petal width')
})

model_update = api.model('ModelUpdate', {
    'model_file': fields.String(required=True, description='The model file to update to')
})


@ns.route('/predict')
class Predict(Resource):
    """
    Endpoint for making predictions with the current model. Expects query parameters for the features of the event.
    """
    @api.expect(model_input)
    @api.response(200, 'Success')
    @api.response(500, 'Internal Server Error')
    def get(self):
        """Predict the species of an iris flower"""
        db_manager = MongoDBManager()
        try:
            model_file = db_manager.get_current_model()
            model_manager = ModelManager(f'models/{model_file}')
            model_manager.load_model()

            # Extraer características del evento
            try:
                sepallength = float(request.args.get('sepallength'))
                sepalwidth = float(request.args.get('sepalwidth'))
                petallength = float(request.args.get('petallength'))
                petalwidth = float(request.args.get('petalwidth'))
            except (TypeError, ValueError) as error:
                logging.error("Invalid input: %s", error)
                return make_response(jsonify({'error': 'Invalid input'}), 400)

            features = np.array([[sepallength, sepalwidth, petallength, petalwidth]])
            species = model_manager.predict(features)[0]
            species_name = SPECIES_MAP[species]

            return jsonify({'species': int(species), 'species_name': species_name, 'model': model_file})
        except Exception as err:
            logging.error("Error during prediction: %s", err)
            return make_response(jsonify({'error': str(err)}), 500)


@ns.route('/update_model')
class UpdateModel(Resource):
    """Endpoint for updating the current model. Expects a JSON payload with the key 'model_file' containing the name
    of the new model file."""

    @api.expect(model_update)
    @api.response(200, 'Success')
    @api.response(500, 'Internal Server Error')
    def post(self):
        """Update the current model"""
        db_manager = MongoDBManager()
        try:
            data = request.get_json()
            model_file = data.get('model_file')

            if not model_file:
                raise ValueError("No model file provided in the request")

            db_manager.update_current_model(model_file)
            logging.info("Updated current_model to %s", model_file)

            return jsonify({'status': 'model updated', 'model': model_file})
        except Exception as error:
            logging.error("Error updating model: %s", error)
            return jsonify({'error': str(error)}), 500


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


@ns.route('/metrics')
class GetMetrics(Resource):
    """Endpoint for retrieving the metrics history. Returns a JSON response with the metrics for each model."""

    @api.response(200, 'Success')
    @api.response(500, 'Internal Server Error')
    def get(self):
        """Get evaluation metrics of the models"""
        db_manager = MongoDBManager()
        try:
            collection = db_manager.db['metrics_history']
            documents = collection.find()
            metrics_list = []
            for doc in documents:
                metrics_list.append(json.loads(json_util.dumps(doc)))
            logging.info("Retrieved metrics from MongoDB")
            return make_response(jsonify(metrics_list), 200)
        except Exception as error:
            logging.error("Error retrieving metrics: %s", error)
            return jsonify({'error': str(error)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
