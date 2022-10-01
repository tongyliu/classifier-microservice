from enum import Enum

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier


class ClassifierType(Enum):
    SGDClassifier = SGDClassifier
    CategoricalNB = CategoricalNB
    MLPClassifier = MLPClassifier
