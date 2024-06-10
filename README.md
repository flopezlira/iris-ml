# Iris ML Project

## Author
Francisco López-Lira Hinojo
flopezlira@gmail.com

## Description
This application implements a simple online ML model serving system.

### Applications
There are two main applications:
- The API. Serves the ML Model with three endpoints:
  - /predict: Makes predictions using the current model
  - /update_model: Updates the current model
  - /metrics: Retrieves the metrics history
- Scheduled task. Randomly choose a model and makes a test using test files in /data. After that, updates the current model

There is also an auxiliary application for testing purposes:
- print_db.py Auxiliar app to see the contents of the DB


## Design Criteria

### MongoDB for Metrics Storage
I chose MongoDB to store the metrics instead of using the file system or another mechanism because MongoDB offers better scalability and flexibility. Storing metrics in MongoDB allows for easy querying and integration with dashboard generation tools in the future.

### Singleton Pattern for MongoDB
I used the Singleton pattern for MongoDBManager to ensure that there is only one instance of the MongoDB connection throughout the application. This helps in managing resources efficiently and avoiding multiple connections to the database, which can be resource-intensive.

### Use of Classes
The use of classes in the project helps in organizing code better, making it more modular and easier to maintain. Each class has a single responsibility:
- `ModelManager` handles the loading, prediction, and evaluation of models.
- `FileManager` manages file operations such as loading test data and listing model files.
- `MongoDBManager` manages the interactions with the MongoDB database.


### API Documentation with Swagger and Flask-RESTX
The API is documented using Swagger and Flask-RESTX. This provides a user-friendly interface to explore and test the API endpoints. The documentation can be accessed by navigating to `http://localhost:5000/` when the application is running.

## Prerequisites
- Python 3.10 or later
- MongoDB instance running locally or on a server
- Flask and related dependencies
- Virtual environment


## Usage
Once installed and running, the application will be accessible at 
`http://localhost:5000/`.
Example:
````
curl "http://127.0.0.1:5000/api/predict?sepallength=5.1&sepalwidth=3.5&petallength=1.4&petalwidth=3.2"
````

## Installation (Non-Docker)
1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd iris_ml
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running with Docker

### Docker Setup
1. **Install Docker**: Make sure Docker is installed on your system. Follow the instructions [here](https://docs.docker.com/get-docker/).

2. **Change DB connection**: Change MONGODB_URI in .env to mongodb://mongo:27017/

3**Build the Docker Images**:
    ```sh
    docker-compose build
    ```

4**Run the Docker Containers**:
    ```sh
    docker-compose up
    ```

This will start both the application and a MongoDB instance. 

## Testing
To test, cd to tests and run 
````
pytest
````

## Example of collections
````
Example of collection 'current_model':
{
    "_id": "66660a1aeedd381ef829c1b4",
    "model_file": "rf-5.p"
}

Example of collection 'metrics_history':
{
    "_id": "66660a1beedd381ef829c1b5",
    "model_file": "rf-10.p",
    "timestamp": "2024-06-09T22:01:31.005676",
    "accuracy": 0.9555555555555556,
    "precision": 0.9611111111111111,
    "recall": 0.9555555555555556,
    "f1_score": 0.955648148148148,
    "confusion_matrix": [
        [
            14,
            0,
            0
        ],
        [
            0,
            15,
            2
        ],
        [
            0,
            0,
            14
        ]
    ]
}

````


