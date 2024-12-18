# %%
import numpy as np
import scipy.optimize
from scipy.special import expit

import pyensmallen as pye

# %%


def test_ols_optimization():
    # Generate test data
    np.random.seed(42)
    n, k = 1000, 5
    X = np.random.randn(n, k)
    true_params = np.random.rand(k)
    y = X @ true_params
    start = np.random.rand(k)

    # pyensmallen solution
    optimizer = pye.L_BFGS()
    result_ens = optimizer.optimize(
        lambda params, gradient: pye.losses.linear_obj(params, gradient, X, y),
        start,
    )

    # scipy solution
    result_scipy = scipy.optimize.minimize(
        fun=lambda b: np.sum((X @ b - y) ** 2),
        x0=start,
        jac=lambda b: 2 * X.T @ (X @ b - y),
    ).x

    assert np.allclose(result_ens, result_scipy, rtol=1e-5)
    assert np.allclose(result_ens, true_params, rtol=1e-5)


# %%


def test_logit_optimization():
    # Generate test data
    np.random.seed(42)
    n, k = 1000, 5
    X = np.random.randn(n, k)
    true_params = np.random.rand(k)
    p = expit(X @ true_params)
    y = np.random.binomial(1, p)
    start = np.random.rand(k)

    # pyensmallen solution
    optimizer = pye.L_BFGS()
    result_ens = optimizer.optimize(
        lambda params, gradient: pye.losses.logistic_obj(params, gradient, X, y),
        start,
    )

    # scipy solution
    result_scipy = scipy.optimize.minimize(
        fun=lambda b: -np.sum(
            y * np.log(expit(X @ b)) + (1 - y) * np.log(1 - expit(X @ b))
        ),
        x0=start,
        jac=lambda b: X.T @ (expit(X @ b) - y),
    ).x

    assert np.allclose(result_ens, result_scipy, rtol=1e-5)


# %%


def test_poisson_optimization():
    # Generate test data
    np.random.seed(42)
    n, k = 1000, 5
    X = np.random.randn(n, k)
    true_params = np.random.rand(k)
    lambda_ = np.exp(X @ true_params)
    y = np.random.poisson(lambda_)
    start = np.random.rand(k)

    # pyensmallen solution
    optimizer = pye.L_BFGS()
    result_ens = optimizer.optimize(
        lambda params, gradient: pye.losses.poisson_obj(params, gradient, X, y),
        start,
    )

    # scipy solution
    result_scipy = scipy.optimize.minimize(
        fun=lambda b: np.sum(np.exp(X @ b) - y * (X @ b)),
        x0=start,
        jac=lambda b: X.T @ (np.exp(X @ b) - y),
    ).x

    assert np.allclose(result_ens, result_scipy, rtol=1e-5)


# %%
test_ols_optimization()
test_logit_optimization()
test_poisson_optimization()
# %%
