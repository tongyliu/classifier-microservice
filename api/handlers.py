import base64
import pickle
import json
from typing import Any, Dict, Tuple, Type, Union

import numpy as np

from db import DatabaseManager
from models import MODEL_CLASSES, run_predict, run_train_step


# Type alias for either a JSON dict or JSON dict + status code
ResponseType = Union[Dict[str, Any], Tuple[Dict[str, Any], int]]


class HandlerError(Exception):
    def __init__(self, message: str, status: int = 400):
        self.message = message
        self.status = status

    def to_response(self):
        return {'error': self.message}, self.status


def _validate_params(expected: Dict[str, Type], actual: Dict[str, Any]) -> bool:
    for param_name, param_type in expected.items():
        if param_name not in actual:
            raise HandlerError(f'Missing parameter {param_name}')

        if not isinstance(actual[param_name], param_type):
            raise HandlerError(f'Expected {param_name} to be an instance of {param_type.__name__}')


def create_model(dbm: DatabaseManager, body: Dict[str, Any]) -> ResponseType:
    _validate_params({'model': str, 'params': dict, 'd': int, 'n_classes': int}, body)

    model_cls = MODEL_CLASSES.get(body['model'])

    if not model_cls:
        raise HandlerError(f'Unrecognized model type: {body["model"]}')

    model = model_cls(**body['params'])
    model_pkl = pickle.dumps(model)
    params = json.dumps(body['params'])
    model_id = dbm.create_model(body['model'], params, body['d'], body['n_classes'], model_pkl)
    return {'id': model_id}


def get_model(dbm: DatabaseManager, model_id: int) -> ResponseType:
    model_data = dbm.get_model(model_id)

    if not model_data:
        raise HandlerError(f'Model with id {model_id} not found', 404)

    model_data['params'] = json.loads(model_data['params'])
    del model_data['model_pkl']

    return model_data


def train_model(dbm: DatabaseManager, model_id: int, body: Dict[str, any]) -> ResponseType:
    # Note that this isn't safe if there are concurrent callers for the same model.
    # In practice if there are enough concurrent updates for this to be a concern
    # we'd want to batch the updates in a queue e.g. Kafka and have this operate
    # over mini-batches instead of single examples.
    _validate_params({'x': list, 'y': int}, body)
    model_data = dbm.get_model(model_id)

    if not model_data:
        raise HandlerError(f'Model with id {model_id} not found', 404)

    model = pickle.loads(model_data.pop('model_pkl'))

    try:
        run_train_step(model, model_data, body['x'], body['y'])
    except ValueError as e:
        raise HandlerError(str(e))

    updated_pkl = pickle.dumps(model)

    n_trained = model_data['n_trained'] + 1
    dbm.update_model(model_id, updated_pkl, n_trained)

    return {'id': model_data['id'], 'n_trained': n_trained}


def predict(dbm: DatabaseManager, model_id: int, args: Dict[str, any]) -> ResponseType:
    _validate_params({'x': str}, args)
    model_data = dbm.get_model(model_id)

    if not model_data:
        raise HandlerError(f'Model with id {model_id} not found', 404)

    x_str = base64.b64decode(args['x'])

    try:
        x = json.loads(x_str)
    except json.JSONDecodeError:
        raise HandlerError(f'Expected x to be a valid JSON array')

    model = pickle.loads(model_data.pop('model_pkl'))

    try:
        prediction = run_predict(model, model_data, x)
    except ValueError as e:
        raise HandlerError(str(e))

    return {'id': model_data['id'], 'x': x, 'y': prediction}


def get_models(dbm: DatabaseManager) -> ResponseType:
    models = dbm.get_models()

    types = np.array([x['model'] for x in models])
    n_trained = np.array([x['n_trained'] for x in models])
    unique_types = set(types)
    normalized_n_trained = {}

    # Get unique n_trained counts for each model type
    # Then map to normalized scores
    for model_type in unique_types:
        unique_n_trained = np.unique(n_trained[types == model_type])

        if len(unique_n_trained) == 1:
            distribution = [1.0]
        else:
            distribution = np.linspace(0, 1, len(unique_n_trained))

        normalized_n_trained[model_type] = {
            k: v for k, v in zip(unique_n_trained, distribution)
        }

    for model in models:
        model['training_score'] = normalized_n_trained[model['model']][model['n_trained']]

    return {'models': models}
