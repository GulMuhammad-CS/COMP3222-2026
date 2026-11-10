import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

from solution.ensemble_selection import GreedyEnsembleSelector


@pytest.fixture
def sample_pool():
    return [
        ("dummy_prior", DummyClassifier(strategy="prior")),
        ("dt_depth2", DecisionTreeClassifier(max_depth=2, random_state=0)),
        ("dt_depth5", DecisionTreeClassifier(max_depth=5, random_state=0)),
    ]


def test_not_fitted_error_raised_before_fit(sample_pool):
    """Calling predict or predict_proba before fit must raise NotFittedError."""
    selector = GreedyEnsembleSelector(estimators=sample_pool)
    X = np.random.randn(10, 2)

    with pytest.raises(NotFittedError):
        selector.predict(X)

    with pytest.raises(NotFittedError):
        selector.predict_proba(X)


def test_fit_returns_self_and_populates_fitted_attributes(sample_pool):
    """Verifies fit returns self and populates required fitted attributes[cite: 3, 4]."""
    X = np.random.randn(40, 4)
    y = np.array([0, 1] * 20)

    selector = GreedyEnsembleSelector(estimators=sample_pool, cv=3, random_state=42)
    fitted_selector = selector.fit(X, y)

    # Must follow scikit-learn conventions and return self[cite: 3]
    assert fitted_selector is selector

    # Check fitted attributes exist and are lists of equal length
    assert hasattr(selector, "classes_")
    assert len(selector.classes_) == 2
    assert isinstance(selector.selected_indices, list)
    assert isinstance(selector.selected_names, list)
    assert isinstance(selector.selection_scores, list)
    assert isinstance(selector.estimators_, list)

    k = len(selector.selected_indices)
    assert len(selector.selected_names) == k
    assert len(selector.selection_scores) == k
    assert len(selector.estimators_) == k


def test_max_ensemble_size_limit(sample_pool):
    """Ensures selection stops when max_ensemble_size is reached."""
    X = np.random.randn(50, 4)
    y = np.array([0, 1] * 25)

    selector = GreedyEnsembleSelector(
        estimators=sample_pool, cv=3, max_ensemble_size=1, random_state=0
    )
    selector.fit(X, y)

    assert len(selector.selected_indices) == 1


def test_metric_support_accuracy_and_log_loss(sample_pool):
    """Verifies support for both 'accuracy' and 'log_loss' metrics[cite: 4]."""
    X = np.random.randn(40, 4)
    y = np.array([0, 1] * 20)

    for metric in ("accuracy", "log_loss"):
        selector = GreedyEnsembleSelector(
            estimators=sample_pool, cv=3, metric=metric, random_state=0
        )
        selector.fit(X, y)

        probs = selector.predict_proba(X)
        preds = selector.predict(X)
        assert probs.shape == (40, 2)
        assert preds.shape == (40,)


def test_tie_breaker_favors_earliest_estimator():
    pool = [
        ("first_dummy", DummyClassifier(strategy="prior")),
        ("second_dummy", DummyClassifier(strategy="prior")),
    ]
    X = np.random.randn(20, 2)
    y = np.array([0, 1] * 10)

    selector = GreedyEnsembleSelector(estimators=pool, cv=2, random_state=0)
    selector.fit(X, y)

    assert selector.selected_indices[0] == 0
    assert selector.selected_names[0] == "first_dummy"


def test_invalid_arguments_raise_errors(sample_pool):
    """Invalid arguments must raise ValueError or TypeError."""
    X = np.ones((20, 2))
    y = np.array([0, 1] * 10)

    # Invalid metric name[cite: 4]
    with pytest.raises(ValueError):
        GreedyEnsembleSelector(estimators=sample_pool, metric="invalid").fit(X, y)

    # Empty estimator pool[cite: 2]
    with pytest.raises(ValueError):
        GreedyEnsembleSelector(estimators=[]).fit(X, y)

    # cv < 2[cite: 2]
    with pytest.raises(ValueError):
        GreedyEnsembleSelector(estimators=sample_pool, cv=1).fit(X, y)

    # Mismatched X and y lengths[cite: 2]
    with pytest.raises(ValueError):
        GreedyEnsembleSelector(estimators=sample_pool).fit(X, y[:10])