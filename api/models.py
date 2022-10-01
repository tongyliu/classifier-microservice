from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier


MODEL_CLASSES = {
    'SGDClassifier': SGDClassifier,
    'CategoricalNB': CategoricalNB,
    'MLPClassifier': MLPClassifier
}
