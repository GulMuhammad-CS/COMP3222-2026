import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from solution.pool import generate_oof_probas


class OutOfFoldVerifierClassifier(BaseEstimator, ClassifierMixin):
    #Custom classifier that raises an AssertionError if predict_proba is called
    #on any sample that was present in its training fold.
    def __init__(self):
        self.seen_x_ = None

    def fit(self, X, y):
        self.seen_x_ = np.asarray(X).copy()
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        for val_case in X_arr:
            for train_case in self.seen_x_:
                if np.array_equal(val_case, train_case):
                    raise AssertionError(
                        "Data leak detected: predict_proba was called on a training case!"
                    )

        n_classes = len(self.classes_)
        return np.full((len(X_arr), n_classes), 1.0 / n_classes)


def test_oof_predictions_are_genuinely_out_of_fold():
    #Verifies that predict_proba is strictly called on out-of-fold validation data.
    X = np.arange(100).reshape(50, 2)
    y = np.array([0, 1] * 25)

    estimators = [OutOfFoldVerifierClassifier()]

    # Will raise AssertionError if any validation sample was present in the model's training fold
    oof_probas, scores = generate_oof_probas(estimators, X, y, cv=5, random_state=42)

    assert oof_probas.shape == (1, 50, 2)
    assert scores.shape == (1,)


def test_generate_oof_probas_output_shapes():
    #Verifies that oof_probas and scores match the required output dimensions.
    n_cases = 40
    n_features = 4
    n_classes = 3

    X = np.random.randn(n_cases, n_features)
    y = np.array([0, 1, 2] * 13 + [0])  # Exactly 40 cases across 3 classes

    estimators = [
        DummyClassifier(strategy="prior"),
        DecisionTreeClassifier(max_depth=2, random_state=0),
        DecisionTreeClassifier(max_depth=4, random_state=0),
    ]
    n_estimators = len(estimators)

    oof_probas, scores = generate_oof_probas(estimators, X, y, cv=4, random_state=42)

    # Verify exact tensor shapes
    assert oof_probas.shape == (n_estimators, n_cases, n_classes)
    assert scores.shape == (n_estimators,)