import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def _safe_indexing(data, indices):
    """Safely index arrays, lists, pandas DataFrames, or Series by positional indices."""
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
    """Generates out-of-fold class probabilities and accuracy scores for a list of classifiers.

    Parameters
    ----------
    estimators : sequence of estimator objects
        List or sequence of scikit-learn compatible classifier models.
    X : array-like or pandas DataFrame of shape (n_cases, n_features)
        Training feature matrix.
    y : array-like or pandas Series of shape (n_cases,)
        Target class labels.
    cv : int or CV generator, default=5
        Number of folds for StratifiedKFold CV, or a custom CV splitter instance.
    random_state : int, RandomState instance, or None, default=None
        Controls fold assignment shuffling. Pass an int for reproducible folds, or
        None to leave assignment unseeded.

    Returns
    -------
    oof_probas : ndarray of shape (n_estimators, n_cases, n_classes)
        Out-of-fold predicted class probabilities for each classifier. Columns follow
        the order of np.unique(y).
    scores : ndarray of shape (n_estimators,)
        Out-of-fold accuracy score of each classifier.
    """
    y_arr = np.asarray(y)
    classes_all = np.unique(y_arr)
    n_classes = len(classes_all)
    n_cases = len(y_arr)
    n_estimators = len(estimators)

    # Initialize container for out-of-fold class probabilities
    oof_probas = np.zeros((n_estimators, n_cases, n_classes), dtype=np.float64)

    # Classification CV strategy: StratifiedKFold maintains class ratio across folds
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