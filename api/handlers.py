from typing import Any, Dict, Tuple, Union

from db import DatabaseManager


# Type alias for either a JSON dict or JSON dict + status code
ResponseType = Union[Dict[str, Any], Tuple[Dict[str, Any], int]]


def create_model(dbm: DatabaseManager, body: Dict[str, Any]) -> ResponseType:
    print(body)
    return body


def get_model(dbm: DatabaseManager, model_id: int) -> ResponseType:
    print(model_id)
    return {'model_id': model_id}


def train_model(dbm: DatabaseManager, model_id: int, body: Dict[str, any]) -> ResponseType:
    print(body)
    return body


def predict(dbm: DatabaseManager, model_id: int, args: Dict[str, any]) -> ResponseType:
    print(args)
    return args


def get_models(dbm: DatabaseManager) -> ResponseType:
    return {"models": []}
