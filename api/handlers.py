import base64
import pickle
import json
from typing import Any, Dict, Tuple, Type, Union

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
    return {"model_id": model_id}


def get_model(dbm: DatabaseManager, model_id: int) -> ResponseType:
    model_data = dbm.get_model(model_id)

    if not model_data:
        raise HandlerError(f'Model with id {model_id} not found', 404)

    model_data['params'] = json.loads(model_data['params'])
    del model_data['model_pkl']

    return model_data


def train_model(dbm: DatabaseManager, model_id: str, body: Dict[str, any]) -> ResponseType:
    model_id = int(model_id)
    _validate_params({'X': list, 'y': int}, body)
    model_data = dbm.get_model(model_id)

    if not model_data:
        raise HandlerError(f'Model with id {model_id} not found', 404)

    model = pickle.loads(model_data.pop('model_pkl'))
    run_train_step(model, model_data, body['X'], body['y'])
    updated_pkl = pickle.dumps(model)

    n_trained = model_data['n_trained'] + 1
    dbm.update_model(model_id, updated_pkl, n_trained)

    return {'n_trained': n_trained}



def predict(dbm: DatabaseManager, model_id: int, args: Dict[str, any]) -> ResponseType:
    _validate_params({'x': str}, args)
    model_data = dbm.get_model(model_id)

    if not model_data:
        raise HandlerError(f'Model with id {model_id} not found', 404)

    x_str = base64.b64decode(args['x'])

    try:
        X = json.loads(x_str)
    except json.JSONDecodeError:
        raise HandlerError(f'Expected x to be a valid JSON array')

    model = pickle.loads(model_data.pop('model_pkl'))
    prediction = run_predict(model, model_data, X)

    return {'X': X, 'y': prediction}


def get_models(dbm: DatabaseManager) -> ResponseType:
    return {"models": []}
