import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def _safe_indexing(data, indices):
    #Safely index arrays, lists, pandas DataFrames, or Series by positional indices
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return np.asarray(data)[indices]


def generate_oof_probas(
    estimators,
    X,
    y,
    cv=5,
    random_state=None,
):
    # making y a numpy array
    y_arr = np.asarray(y)
    # getting an array of each unique class label
    classes_all = np.unique(y_arr)
    # getting number of different classes, cases, and estimators
    n_classes = len(classes_all)
    n_cases = len(y_arr)
    n_estimators = len(estimators)

    # Initialize container for out-of-fold class probabilities as an array of zeroes
    oof_probas = np.zeros((n_estimators, n_cases, n_classes), dtype=np.float64)

    # Classification CV strategy: StratifiedKFold maintains class ratio across folds for testing and training
    #uses cv as seed for fold size, and sets random state
    if isinstance(cv, int):
        cv_splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )
    else:
        cv_splitter = cv

    # Compute OOF probabilities fold-by-fold for each model
    for est_idx, estimator in enumerate(estimators):
        for train_idx, val_idx in cv_splitter.split(X, y_arr):
            X_train = _safe_indexing(X, train_idx)
            y_train = y_arr[train_idx]
            X_val = _safe_indexing(X, val_idx)

            # Fit isolated clone to avoid state leakage across folds
            clf = clone(estimator)
            clf.fit(X_train, y_train)

            fold_probas = clf.predict_proba(X_val)

            # Map model class indices to global np.unique(y) column ordering
            class_indices = np.searchsorted(classes_all, clf.classes_)

            val_probas_full = np.zeros((len(val_idx), n_classes), dtype=np.float64)
            val_probas_full[:, class_indices] = fold_probas

            oof_probas[est_idx, val_idx] = val_probas_full

    # Calculate out-of-fold accuracy score for each classifier
    scores = np.zeros(n_estimators, dtype=np.float64)
    for est_idx in range(n_estimators):
        oof_pred_indices = np.argmax(oof_probas[est_idx], axis=1)
        oof_preds = classes_all[oof_pred_indices]
        scores[est_idx] = accuracy_score(y_arr, oof_preds)

    return oof_probas, scores