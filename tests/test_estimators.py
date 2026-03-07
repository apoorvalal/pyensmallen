import numpy as np
from scipy.special import expit

import pyensmallen as pye


def test_linear_regression_estimator_recovers_coefficients():
    np.random.seed(42)
    n, k = 1500, 4
    X = np.random.randn(n, k)
    coef = np.array([1.5, -0.25, 0.75, 2.0])
    intercept = -0.4
    y = intercept + X @ coef + 0.05 * np.random.randn(n)

    model = pye.LinearRegression()
    model.fit(X, y)

    assert np.allclose(model.coef_, coef, atol=0.06)
    assert np.isclose(model.intercept_, intercept, atol=0.06)
    assert model.coef_std_errors_ is not None
    assert model.intercept_std_error_ is not None
    assert model.confidence_intervals().shape == (k + 1, 2)
    assert model.confidence_intervals(covariance_type="robust").shape == (k + 1, 2)
    summary = model.summary()
    assert summary.shape == (k + 1, 4)
    assert np.allclose(summary[:, 0], model.params_)
    assert np.allclose(summary[:, 1], model.std_errors_)
    robust_summary = model.summary(covariance_type="robust")
    assert robust_summary.shape == (k + 1, 4)
    assert np.allclose(robust_summary[:, 0], model.params_)
    assert np.allclose(robust_summary[:, 1], model.robust_std_errors_)
    assert model.score(X, y) > 0.99


def test_logistic_regression_estimator_predicts_probabilities():
    np.random.seed(42)
    n, k = 3000, 3
    X = np.random.randn(n, k)
    coef = np.array([1.1, -0.8, 0.6])
    intercept = -0.2
    probs = expit(intercept + X @ coef)
    y = np.random.binomial(1, probs)

    model = pye.LogisticRegression(alpha=0.05, l1_ratio=0.2)
    model.fit(X, y)
    fitted_probs = model.predict_proba(X)

    assert fitted_probs.shape == (n, 2)
    assert np.allclose(fitted_probs.sum(axis=1), 1.0, atol=1e-8)
    assert model.score(X, y) > 0.7
    assert model.coef_std_errors_ is None


def test_poisson_regression_estimator_recovers_mean_structure():
    np.random.seed(42)
    n, k = 3000, 3
    X = np.random.randn(n, k)
    coef = np.array([0.2, -0.15, 0.1])
    intercept = 0.3
    mean = np.exp(intercept + X @ coef)
    y = np.random.poisson(mean)

    model = pye.PoissonRegression(alpha=0.02, l1_ratio=0.0)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.all(preds > 0)
    assert np.allclose(model.coef_, coef, atol=0.12)
    assert np.isclose(model.intercept_, intercept, atol=0.12)


def test_regularization_shrinks_linear_coefficients():
    np.random.seed(42)
    n, k = 1200, 6
    X = np.random.randn(n, k)
    coef = np.array([2.0, -1.5, 0.0, 0.0, 0.75, 0.0])
    y = 0.5 + X @ coef + 0.2 * np.random.randn(n)

    baseline = pye.LinearRegression(alpha=0.0)
    baseline.fit(X, y)

    ridge = pye.LinearRegression(alpha=1.0, l1_ratio=0.0)
    ridge.fit(X, y)

    lasso_like = pye.LinearRegression(alpha=1.0, l1_ratio=1.0)
    lasso_like.fit(X, y)

    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(baseline.coef_)
    assert np.linalg.norm(lasso_like.coef_, ord=1) < np.linalg.norm(
        baseline.coef_, ord=1
    )
    assert ridge.coef_std_errors_ is None


def test_ols_robust_covariance_matches_sandwich_formula():
    np.random.seed(42)
    n, k = 1200, 3
    X = np.random.randn(n, k)
    sigma = 0.2 + 0.3 * np.abs(X[:, 0])
    y = 0.5 + X @ np.array([1.0, -0.5, 0.25]) + sigma * np.random.randn(n)

    model = pye.LinearRegression()
    model.fit(X, y)

    X_design = np.column_stack([np.ones(n), X])
    residuals = y - X_design @ model.params_
    xtx_inv = np.linalg.inv(X_design.T @ X_design)
    meat = X_design.T @ ((residuals**2)[:, np.newaxis] * X_design)
    expected = xtx_inv @ meat @ xtx_inv

    assert np.allclose(model.robust_covariance_, expected, atol=1e-8, rtol=1e-6)


def test_logit_robust_covariance_matches_qmle_sandwich():
    np.random.seed(42)
    n, k = 4000, 3
    X = np.random.randn(n, k)
    X_design = np.column_stack([np.ones(n), X])
    beta = np.array([-0.2, 0.8, -0.5, 0.4])
    probs = expit(X_design @ beta)
    y = np.random.binomial(1, probs)

    model = pye.LogisticRegression()
    model.fit(X, y)

    fitted_probs = expit(X_design @ model.params_)
    scores = X_design * (y - fitted_probs)[:, np.newaxis]
    weights = fitted_probs * (1.0 - fitted_probs)
    bread = (X_design.T @ (weights[:, np.newaxis] * X_design)) / n
    meat = (scores.T @ scores) / n
    expected = np.linalg.inv(bread) @ meat @ np.linalg.inv(bread) / n

    assert np.allclose(model.robust_covariance_, expected, atol=1e-8, rtol=1e-6)


def test_poisson_robust_covariance_matches_qmle_sandwich():
    np.random.seed(42)
    n, k = 4000, 2
    X = np.random.randn(n, k)
    X_design = np.column_stack([np.ones(n), X])
    beta = np.array([0.1, 0.2, -0.15])
    mean = np.exp(X_design @ beta)
    y = np.random.poisson(mean)

    model = pye.PoissonRegression()
    model.fit(X, y)

    fitted_mean = np.exp(np.clip(X_design @ model.params_, -30.0, 30.0))
    scores = X_design * (y - fitted_mean)[:, np.newaxis]
    bread = (X_design.T @ (fitted_mean[:, np.newaxis] * X_design)) / n
    meat = (scores.T @ scores) / n
    expected = np.linalg.inv(bread) @ meat @ np.linalg.inv(bread) / n

    assert np.allclose(model.robust_covariance_, expected, atol=1e-8, rtol=1e-6)
