import numpy as np
import scipy as sp


def linear_obj(params, gradient, X, y):
    """Least Squares Loss for Linear Regression.

    Args:
        params (Coefficint vector): ndarray of coefficients
        gradient (_type_): pointer to the gradient vector
        X (ndarray): covariate matrix
        y (ndarray): response vector

    Returns:
        _type_: _description_
    """
    params = params.reshape(-1, 1)
    residuals = X @ params - y.reshape(-1, 1)
    objective = np.sum(residuals**2)
    grad = 2 * X.T @ residuals
    gradient[:] = grad.flatten()
    return objective


def logistic_obj(params, gradient, X, y):
    """Logistic likelihood for Logistic Regression. Already negated; to be minimize.

    Args:
        params (Coefficint vector): ndarray of coefficients
        gradient (_type_): pointer to the gradient vector
        X (ndarray): covariate matrix
        y (ndarray): response vector
    """
    z = X @ params
    h = sp.special.expit(z)
    objective = -np.sum(y * np.log(h) + (1 - y) * np.log1p(-h))
    if np.isnan(objective):
        objective = np.inf
    grad = X.T @ (h - y)
    gradient[:] = grad
    return objective


def poisson_obj(params, gradient, X, y):
    """Poisson Likelihood for Poisson Regression. Already negated; to be minimize.

    Args:
        params (Coefficint vector): ndarray of coefficients
        gradient (_type_): pointer to the gradient vector
        X (ndarray): covariate matrix
        y (ndarray): response vector
    """
    params = params.reshape(-1, 1)
    y = y.reshape(-1, 1)
    Xbeta = X @ params
    lambda_ = np.exp(Xbeta)
    objective = np.sum(lambda_ - np.multiply(y, np.log(lambda_)))
    # Compute the gradient
    grad = X.T @ (lambda_ - y)
    gradient[:] = grad.ravel()
    return objective
