import base64
import pickle
import json
from typing import Any, Dict, Tuple, Type, Union

from db import DatabaseManager
from models import MODEL_CLASSES


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
    model = dbm.get_model(model_id)

    if not model:
        raise HandlerError(f'Model with id {model_id} not found', 404)

    model['params'] = json.loads(model['params'])
    del model['model_pkl']

    return model


def train_model(dbm: DatabaseManager, model_id: int, body: Dict[str, any]) -> ResponseType:
    print(body)
    return body


def predict(dbm: DatabaseManager, model_id: int, args: Dict[str, any]) -> ResponseType:
    print(args)
    return args


def get_models(dbm: DatabaseManager) -> ResponseType:
    return {"models": []}
