import csv
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from part3_setup import (
    CV,
    DATASETS,
    RANDOM_STATE,
    load_dataset,
    make_pool,
    make_stacking,
)
from solution.ensemble_selection import GreedyEnsembleSelector
from solution.pool import generate_oof_probas


def evaluate_best_single(X_train, y_train, X_test, y_test):
    """Evaluates the single best classifier based on out-of-fold accuracy."""
    pool = make_pool()
    raw_estimators = [est for _, est in pool]

    # Compute OOF accuracies on training data
    _, scores = generate_oof_probas(
        raw_estimators,
        X_train,
        y_train,
        cv=CV,
        random_state=RANDOM_STATE,
    )

    # Pick index of max score; np.argmax breaks ties by returning the first index
    best_idx = np.argmax(scores)
    _, best_estimator_template = pool[best_idx]

    # Refit fresh clone on complete training data and evaluate on test set
    clf = clone(best_estimator_template)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


def evaluate_full_pool(X_train, y_train, X_test, y_test):
    """Evaluates equal-weight probability average of all pool members."""
    pool = make_pool()
    classes = np.unique(y_train)

    probas_list = []
    for _, est_template in pool:
        clf = clone(est_template)
        clf.fit(X_train, y_train)
        probas_list.append(clf.predict_proba(X_test))

    # Compute mean probabilities across full pool
    mean_probas = np.mean(probas_list, axis=0)

    # Argmax tie-breaker defaults to lowest column index (matching spec)
    pred_indices = np.argmax(mean_probas, axis=1)
    y_pred = classes[pred_indices]
    return accuracy_score(y_test, y_pred)


def evaluate_stacking(X_train, y_train, X_test, y_test):
    """Evaluates scikit-learn StackingClassifier baseline."""
    stacking = make_stacking()
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    return accuracy_score(y_test, y_pred)


def run_comparison():
    """Runs comparison across all datasets and outputs results.csv."""
    results = []

    for name in DATASETS:
        print(f"Evaluating dataset: {name}...")
        X_train, y_train, X_test, y_test = load_dataset(name)

        # Strategy 1: Best Single Classifier
        best_single_acc = evaluate_best_single(X_train, y_train, X_test, y_test)

        # Strategy 2: Full Pool Ensemble
        full_pool_acc = evaluate_full_pool(X_train, y_train, X_test, y_test)

        # Strategy 3: Stacking Classifier
        stacking_acc = evaluate_stacking(X_train, y_train, X_test, y_test)

        # Strategy 4: Greedy Selection by Accuracy
        selector_acc = GreedyEnsembleSelector(
            estimators=make_pool(),
            cv=CV,
            metric="accuracy",
            random_state=RANDOM_STATE,
        )
        selector_acc.fit(X_train, y_train)
        greedy_acc_pred = selector_acc.predict(X_test)
        greedy_acc = accuracy_score(y_test, greedy_acc_pred)
        n_selected_acc = len(selector_acc.selected_indices)
        oof_acc = selector_acc.selection_scores[-1]

        # Strategy 5: Greedy Selection by Log Loss
        selector_log = GreedyEnsembleSelector(
            estimators=make_pool(),
            cv=CV,
            metric="log_loss",
            random_state=RANDOM_STATE,
        )
        selector_log.fit(X_train, y_train)
        greedy_log_pred = selector_log.predict(X_test)
        greedy_log_loss = accuracy_score(y_test, greedy_log_pred)
        n_selected_log = len(selector_log.selected_indices)

        # Store formatted row matching required structure
        results.append({
            "dataset": name,
            "best_single": f"{best_single_acc:.4f}",
            "full_pool": f"{full_pool_acc:.4f}",
            "stacking": f"{stacking_acc:.4f}",
            "greedy_accuracy": f"{greedy_acc:.4f}",
            "greedy_log_loss": f"{greedy_log_loss:.4f}",
            "n_selected_accuracy": n_selected_acc,
            "n_selected_log_loss": n_selected_log,
            "oof_accuracy": f"{oof_acc:.4f}",
        })

    # Write formatted output to results.csv in current directory
    fieldnames = [
        "dataset",
        "best_single",
        "full_pool",
        "stacking",
        "greedy_accuracy",
        "greedy_log_loss",
        "n_selected_accuracy",
        "n_selected_log_loss",
        "oof_accuracy",
    ]

    with open("results.csv", mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Successfully generated results.csv")


if __name__ == "__main__":
    run_comparison()