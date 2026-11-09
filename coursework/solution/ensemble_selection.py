import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, log_loss

from solution.pool import generate_oof_probas


class GreedyEnsembleSelector(ClassifierMixin, BaseEstimator):
    """Greedy forward ensemble selection algorithm.

    Selects a subset of classifiers from a pool by iteratively adding the classifier
    that yields the greatest improvement in out-of-fold performance.

    Parameters
    ----------
    estimators : list of tuple (str, estimator)
        List of (name, estimator) pairs.
    cv : int, default=5
        Number of cross-validation folds. Must be an integer >= 2.
    max_ensemble_size : int or None, default=None
        Maximum number of classifiers to include in the ensemble.
    metric : str, default="accuracy"
        Metric to optimize. Supported values: "accuracy", "log_loss".
    random_state : int, RandomState instance, or None, default=None
        Controls CV fold assignment.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels sorted as np.unique(y).
    selected_indices : list of int
        Indices in `estimators` of selected classifiers in order of selection.
    selected_names : list of str
        Names of selected classifiers in order of selection.
    selection_scores : list of float
        Out-of-fold score after each classifier addition.
    estimators_ : list of estimator
        Fitted clones of selected classifiers, in selection order.
    """

    def __init__(
        self,
        estimators,
        cv=5,
        max_ensemble_size=None,
        metric="accuracy",
        random_state=None,
    ):
        self.estimators = estimators
        self.cv = cv
        self.max_ensemble_size = max_ensemble_size
        self.metric = metric
        self.random_state = random_state

    def _validate_inputs(self, X, y):
        # Validate cv
        if not isinstance(self.cv, (int, np.integer)):
            raise TypeError("cv must be an integer.")
        if self.cv < 2:
            raise ValueError("cv must be an integer of at least 2.")

        # Validate max_ensemble_size
        if self.max_ensemble_size is not None:
            if not isinstance(self.max_ensemble_size, (int, np.integer)):
                raise TypeError("max_ensemble_size must be None or an integer.")
            if self.max_ensemble_size <= 0:
                raise ValueError("max_ensemble_size must be a positive integer.")

        # Validate metric
        if self.metric not in ("accuracy", "log_loss"):
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Validate estimators list
        if not isinstance(self.estimators, (list, tuple)):
            raise TypeError("estimators must be a list or tuple of (name, estimator) pairs.")
        if len(self.estimators) == 0:
            raise ValueError("estimators pool cannot be empty.")

        names = []
        for item in self.estimators:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("Each element in estimators must be a (name, estimator) pair.")
            name, est = item
            if not isinstance(name, str):
                raise TypeError("Estimator names must be strings.")
            if not hasattr(est, "predict_proba"):
                raise ValueError(f"Estimator '{name}' does not implement predict_proba.")
            names.append(name)

        if len(names) != len(set(names)):
            raise ValueError("Duplicate estimator names found in pool.")

        # Validate X and y length
        if len(X) != len(y):
            raise ValueError("X and y must have equal lengths.")

    def _compute_score(self, y_true, probas):
        if self.metric == "accuracy":
            # Class with highest probability; tie goes to lower column index (np.argmax)
            #np.argmax used here to make array of indices of max value at each row
            pred_indices = np.argmax(probas, axis=1)
            preds = self.classes_[pred_indices]
            return accuracy_score(y_true, preds)
        elif self.metric == "log_loss":
            return log_loss(y_true, probas, labels=self.classes_)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _is_strictly_better(self, new_score, current_score):
        #using strict inequalities ensures a tiebreaker to favour the classifier occurring earlier
        if self.metric == "accuracy":
            return new_score > current_score
        else:  # log_loss
            return new_score < current_score

    def fit(self, X, y):
        self._validate_inputs(X, y)

        #making an array of y, making it into an array of class labels based on unique values in y_arr
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        #storing number of estimators
        n_estimators = len(self.estimators)

        # 1. Obtain Out of Fold probability matrix for all estimators in pool
        #unpacking estimators by removing the names and only storing the estimator objects
        raw_estimators = [est for _, est in self.estimators]
        #uses function in pool.py to create aforementioned OOF predictions, based on given estimators
        oof_probas, _ = generate_oof_probas(
            raw_estimators,
            X,
            y_arr,
            cv=self.cv,
            random_state=self.random_state,
        )
        #below lists used to record indices, names, and the cumulative ensemble scores everytime a classifier is added using the greedy selection
        selected_indices = []
        selected_names = []
        selection_scores = []
        #generates the list of remaining classifiers that haven't yet been chosen
        remaining_indices = list(range(n_estimators))

        # 2. Select initial best single classifier
        best_initial_idx = None
        best_initial_score = None
        #iterates through every model's index in the list of remaining classifier's
        for idx in remaining_indices:
            #evaluates te OOF predictions for each classifier, against target in y_arr using the specified metric
            score = self._compute_score(y_arr, oof_probas[idx])
            #setting the initial scores
            if best_initial_score is None:
                best_initial_score = score
                best_initial_idx = idx
            #setting new best  initial score and index if new score is better than the existing values
            elif self._is_strictly_better(score, best_initial_score):
                best_initial_score = score
                best_initial_idx = idx
        #adds details of the winning classifiers to the lists generated above
        selected_indices.append(best_initial_idx)
        selected_names.append(self.estimators[best_initial_idx][0])
        selection_scores.append(best_initial_score)
        #removing the winner's index so that it isn't selected again
        remaining_indices.remove(best_initial_idx)

        # Sum of OOF probabilities for selected ensemble
        #stores the sum of OOF probs for all currently selected models to avoid constantly recalculating probability averages in the future
        current_probas_sum = oof_probas[best_initial_idx].copy()
        #sets the new benchmark to be beaten by any new candidate model in subsequent loops
        current_score = best_initial_score

        # 3. Iterative greedy forward selection loop
        while remaining_indices:
            #breaks the loop if the no. of selected indices outnumbers the max size of the ensemble given the ensemble has at least 1 model in it?
            if (
                self.max_ensemble_size is not None
                and len(selected_indices) >= self.max_ensemble_size
            ):
                break
            #initializes the values for the best candidate details
            best_candidate_idx = None
            best_candidate_score = None
            #loop to calculate avg OOF probabilities, and initialize and change the values of
            # the best candidate score and index as needed
            for cand_idx in remaining_indices:
                # Calculate average OOF probabilities if cand_idx were added
                cand_probas_sum = current_probas_sum + oof_probas[cand_idx]
                cand_avg_probas = cand_probas_sum / (len(selected_indices) + 1)
                cand_score = self._compute_score(y_arr, cand_avg_probas)

                if best_candidate_score is None:
                    best_candidate_score = cand_score
                    best_candidate_idx = cand_idx
                elif self._is_strictly_better(cand_score, best_candidate_score):
                    best_candidate_score = cand_score
                    best_candidate_idx = cand_idx

            # Check if best candidate yields a strict improvement over current ensemble
            if self._is_strictly_better(best_candidate_score, current_score):
                #adds the details of the best candidate to the indices, names, and scores lists
                selected_indices.append(best_candidate_idx)
                selected_names.append(self.estimators[best_candidate_idx][0])
                selection_scores.append(best_candidate_score)
                # stores the sum of OOF probabilities for all currently selected models to avoid recalculating averages
                current_probas_sum += oof_probas[best_candidate_idx]
                current_score = best_candidate_score
                remaining_indices.remove(best_candidate_idx)
            else:
                # breaks as no remaining classifier can improve the ensemble
                break

        # 4. Store fitted clones of selected estimators on complete training dataset
        # stores a list of the original positional indices from the estimators pool in the exact  order models were selected
        self.selected_indices = selected_indices
        # stores the corresponding string names in selection order
        self.selected_names = selected_names
        # stores the OOF socre achieved after each classifier was added
        self.selection_scores = selection_scores

        self.estimators_ = []
        for idx in selected_indices:
            _, est_template = self.estimators[idx]
            # clones the original estimators objects as those are templates that shouldn't directly be changed, so clone creates an unfitted copyof each chosen estimator template
            fitted_clf = clone(est_template)
            # fits each cloned estimator on the entire dataset X and y
            fitted_clf.fit(X, y)
            # appends fully trained model into self.estimators in the order they were chosen
            self.estimators_.append(fitted_clf)

        return self

    def predict_proba(self, X):
        if not hasattr(self, "estimators_"):
            raise NotFittedError("This GreedyEnsembleSelector instance is not fitted yet.")

        # Compute predicted probabilities from all selected models
        probas_list = [clf.predict_proba(X) for clf in self.estimators_]
        return np.mean(probas_list, axis=0)

    def predict(self, X):
        probas = self.predict_proba(X)
        pred_indices = np.argmax(probas, axis=1)
        return self.classes_[pred_indices]