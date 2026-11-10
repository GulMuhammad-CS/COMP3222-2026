import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

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