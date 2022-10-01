from typing import Any, Dict, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier


MODEL_CLASSES = {
    'SGDClassifier': SGDClassifier,
    'CategoricalNB': CategoricalNB,
    'MLPClassifier': MLPClassifier
}


def run_train_step(
    model: BaseEstimator,
    model_data: Dict[str, Any],
    X: List[float],
    y: int
) -> None:
    expected_dim = model_data['d']
    expected_n = model_data['n_classes']

    if len(X) != expected_dim:
        raise ValueError(f'Expected X to be dimension {expected_dim}')

    if y >= expected_n:
        raise ValueError(f'Expected y to be in range [0, {expected_n})')

    X = np.array([X])
    y = np.array([y])
    model.partial_fit(X, y, classes=list(range(expected_n)))


def run_predict(
    model: BaseEstimator,
    model_data: Dict[str, Any],
    X: List[float]
) -> int:
    expected_dim = model_data['d']

    if len(X) != expected_dim:
        raise ValueError(f'Expected X to be dimension {expected_dim}')

    X = np.array([X])
    return int(model.predict(X)[0])
