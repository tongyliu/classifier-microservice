from flask import Flask, request

import handlers
from db import DatabaseManager


application = Flask(__name__)
dbm = DatabaseManager()

# Creates tables if they don't already exist
dbm.setup_tables()


@application.route('/health', methods=['GET'])
def health():
    return {'message': 'Ok'}


@application.route('/models', methods=['POST'])
def create_model():
    """Create an untrained classifier model."""
    try:
        return handlers.create_model(dbm, request.json)
    except handlers.HandlerError as e:
        return e.to_response()


@application.route('/models/<model_id>', methods=['GET'])
def get_model(model_id: int):
    """Get a trained or untrained model, identified by its id."""
    try:
        return handlers.get_model(dbm, model_id)
    except handlers.HandlerError as e:
        return e.to_response()


@application.route('/models/<model_id>/train', methods=['POST'])
def train_model(model_id: int):
    """Do one step of partial training of a model."""
    try:
        return handlers.train_model(dbm, model_id, request.json)
    except handlers.HandlerError as e:
        return e.to_response()


@application.route('/models/<model_id>/predict', methods=['GET'])
def predict(model_id: int):
    """Get a features vector and a model and use it to predict."""
    try:
        return handlers.predict(dbm, model_id, request.args)
    except handlers.HandlerError as e:
        return e.to_response()


@application.route('/models', methods=['GET'])
def get_models():
    """Get model training statistics."""
    try:
        return handlers.get_models(dbm)
    except handlers.HandlerError as e:
        return e.to_response()
