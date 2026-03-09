"""Estimator-style APIs built on top of the low-level optimizers.

The classes in this module wrap common supervised-learning objectives in a
scikit-learn-like interface with fitted attributes and lightweight inference
helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

from ._pyensmallen import L_BFGS


@dataclass
class OptimizationResult:
    """Container for the raw optimizer output."""

    params: np.ndarray
    objective_value: float


class _BaseEstimator:
    """Shared implementation for optimizer-backed estimators.

    Parameters
    ----------
    alpha : float, default=0.0
        Overall penalty strength. Inference is only exposed when ``alpha == 0``.
    l1_ratio : float, default=0.0
        Mixing weight between smooth L1 and L2 penalties. ``0`` corresponds to
        pure L2 penalization.
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    max_iterations : int, default=1000
        Maximum number of optimizer iterations.
    tolerance : float, default=1e-8
        Minimum gradient norm passed to the underlying optimizer.
    l1_smoothing : float, default=1e-6
        Smoothing constant used in the differentiable approximation to the
        absolute value penalty.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        l1_smoothing: float = 1e-6,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1")
        if l1_smoothing <= 0:
            raise ValueError("l1_smoothing must be strictly positive")

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.l1_smoothing = l1_smoothing

        # Fitted attributes.
        self.n_features_in_ = None
        self.params_ = None
        self.coef_ = None
        self.intercept_ = None
        self.objective_value_ = None
        self.covariance_ = None
        self.std_errors_ = None
        self.coef_std_errors_ = None
        self.intercept_std_error_ = None
        self.robust_covariance_ = None
        self.robust_std_errors_ = None
        self.robust_coef_std_errors_ = None
        self.robust_intercept_std_error_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the estimator to a design matrix and response vector.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Response vector.

        Returns
        -------
        self
            The fitted estimator.
        """
        X_arr, y_arr = self._validate_inputs(X, y)
        X_design = self._design_matrix(X_arr)
        initial_params = self._initial_params(X_design, y_arr)
        result = self._optimize(X_design, y_arr, initial_params)

        self.n_features_in_ = X_arr.shape[1]
        self.params_ = result.params
        self.objective_value_ = result.objective_value

        if self.fit_intercept:
            self.intercept_ = float(self.params_[0])
            self.coef_ = self.params_[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self.params_.copy()

        self._compute_inference(X_design, y_arr)
        return self

    def _validate_inputs(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self._validate_target(y_arr)
        return X_arr, y_arr

    def _validate_target(self, y: np.ndarray) -> None:
        return None

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return np.ascontiguousarray(X)
        intercept = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.ascontiguousarray(np.hstack([intercept, X]))

    def _optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_params: np.ndarray,
    ) -> OptimizationResult:
        objective = self._objective(X, y)
        optimizer = L_BFGS()
        optimizer.maxIterations = self.max_iterations
        optimizer.minGradientNorm = self.tolerance
        params = optimizer.optimize(objective, np.ascontiguousarray(initial_params))

        grad = np.zeros_like(params)
        objective_value = objective(params, grad)
        return OptimizationResult(params=params, objective_value=objective_value)

    def _objective(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]

        def objective(params: np.ndarray, gradient: np.ndarray) -> float:
            loss_value, grad = self._loss_and_gradient(X, y, params)
            loss_value = loss_value / n_samples
            grad = grad / n_samples

            penalty_value, penalty_grad = self._penalty(params)
            gradient[:] = grad + penalty_grad
            return float(loss_value + penalty_value)

        return objective

    def _penalty(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        penalty_params = params[1:] if self.fit_intercept else params
        penalty_grad = np.zeros_like(params)

        if self.alpha == 0 or penalty_params.size == 0:
            return 0.0, penalty_grad

        l2_value = 0.5 * np.dot(penalty_params, penalty_params)
        l2_grad = penalty_params

        smooth_abs = np.sqrt(penalty_params**2 + self.l1_smoothing**2)
        l1_value = np.sum(smooth_abs - self.l1_smoothing)
        l1_grad = penalty_params / smooth_abs

        value = self.alpha * ((1 - self.l1_ratio) * l2_value + self.l1_ratio * l1_value)
        grad = self.alpha * ((1 - self.l1_ratio) * l2_grad + self.l1_ratio * l1_grad)

        if self.fit_intercept:
            penalty_grad[1:] = grad
        else:
            penalty_grad[:] = grad

        return float(value), penalty_grad

    def _initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[1], dtype=np.float64)

    def _loss_and_gradient(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> tuple[float, np.ndarray]:
        raise NotImplementedError

    def _compute_inference(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.alpha != 0:
            self.covariance_ = None
            self.std_errors_ = None
            self.coef_std_errors_ = None
            self.intercept_std_error_ = None
            self.robust_covariance_ = None
            self.robust_std_errors_ = None
            self.robust_coef_std_errors_ = None
            self.robust_intercept_std_error_ = None
            return

        try:
            covariance = self._covariance_matrix(X, y, self.params_)
        except np.linalg.LinAlgError:
            covariance = None

        self.covariance_ = covariance
        (
            self.std_errors_,
            self.coef_std_errors_,
            self.intercept_std_error_,
        ) = self._split_standard_errors(covariance)

        try:
            robust_covariance = self._robust_covariance_matrix(X, y, self.params_)
        except np.linalg.LinAlgError:
            robust_covariance = None

        self.robust_covariance_ = robust_covariance
        (
            self.robust_std_errors_,
            self.robust_coef_std_errors_,
            self.robust_intercept_std_error_,
        ) = self._split_standard_errors(robust_covariance)

    def _covariance_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray | None:
        raise NotImplementedError

    def _split_standard_errors(
        self, covariance: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        if covariance is None:
            return None, None, None

        std_errors = np.sqrt(np.diag(covariance))
        if self.fit_intercept:
            return std_errors, std_errors[1:].copy(), float(std_errors[0])
        return std_errors, std_errors.copy(), None

    def _score_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def _bread_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def _robust_covariance_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        n_obs = X.shape[0]
        scores = self._score_matrix(X, y, params)
        bread = self._bread_matrix(X, y, params)
        bread_inv = np.linalg.inv(bread)
        meat = (scores.T @ scores) / n_obs
        return (bread_inv @ meat @ bread_inv) / n_obs

    def _inference_arrays(
        self, covariance_type: str = "nonrobust"
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if covariance_type == "nonrobust":
            return self.covariance_, self.std_errors_
        if covariance_type == "robust":
            return self.robust_covariance_, self.robust_std_errors_
        raise ValueError("covariance_type must be either 'nonrobust' or 'robust'")

    def confidence_intervals(
        self, alpha: float = 0.05, covariance_type: str = "nonrobust"
    ) -> np.ndarray:
        """Return parameter confidence intervals.

        Parameters
        ----------
        alpha : float, default=0.05
            Two-sided significance level.
        covariance_type : {"nonrobust", "robust"}, default="nonrobust"
            Covariance estimator used to construct intervals.

        Returns
        -------
        ndarray of shape (n_params, 2)
            Lower and upper confidence bounds for each fitted parameter.
        """
        if self.params_ is None:
            raise ValueError("Call fit before requesting confidence intervals")
        covariance, std_errors = self._inference_arrays(covariance_type=covariance_type)
        if covariance is None or std_errors is None:
            raise ValueError(
                "Confidence intervals are only available for unregularized fitted models"
            )

        z_value = norm.ppf(1.0 - alpha / 2.0)
        lower = self.params_ - z_value * std_errors
        upper = self.params_ + z_value * std_errors
        return np.column_stack([lower, upper])

    def summary(
        self, alpha: float = 0.05, covariance_type: str = "nonrobust"
    ) -> np.ndarray:
        """Return a compact numeric parameter summary.

        Parameters
        ----------
        alpha : float, default=0.05
            Two-sided significance level used for the confidence interval.
        covariance_type : {"nonrobust", "robust"}, default="nonrobust"
            Covariance estimator used to compute standard errors and intervals.

        Returns
        -------
        ndarray of shape (n_params, 4)
            Array with columns ``[coef, se, ci_lb, ci_ub]``.
        """
        if self.params_ is None:
            raise ValueError("Call fit before requesting a summary")
        _, std_errors = self._inference_arrays(covariance_type=covariance_type)
        if std_errors is None:
            raise ValueError(
                "Summary output is only available for unregularized fitted models"
            )

        intervals = self.confidence_intervals(alpha=alpha, covariance_type=covariance_type)
        return np.column_stack(
            [self.params_, std_errors, intervals[:, 0], intervals[:, 1]]
        )

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the linear predictor for the provided feature matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            The linear predictor ``X @ coef_ + intercept_``.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        return X_arr @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError


class LinearRegression(_BaseEstimator):
    """Linear regression estimated by direct optimization of squared loss.

    The fitted model minimizes the average squared-error objective with optional
    smooth penalization. For unregularized fits, both classical and robust
    sandwich covariance estimators are available.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated slope coefficients.
    intercept_ : float
        Estimated intercept. Equals ``0.0`` when ``fit_intercept=False``.
    params_ : ndarray of shape (n_params,)
        Full parameter vector, including the intercept when present.
    std_errors_ : ndarray or None
        Classical standard errors for ``params_``.
    robust_std_errors_ : ndarray or None
        Heteroskedasticity-robust sandwich standard errors for ``params_``.
    """

    def _initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        params, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.asarray(params, dtype=np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the linear regression model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Response vector.

        Returns
        -------
        LinearRegression
            The fitted estimator.
        """
        return super().fit(X, y)

    def _loss_and_gradient(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> tuple[float, np.ndarray]:
        residuals = X @ params - y
        loss = 0.5 * np.dot(residuals, residuals)
        grad = X.T @ residuals
        return loss, grad

    def _covariance_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        n_obs, n_params = X.shape
        residuals = y - X @ params
        sigma2 = np.dot(residuals, residuals) / max(n_obs - n_params, 1)
        xtx_inv = np.linalg.inv(X.T @ X)
        return sigma2 * xtx_inv

    def _score_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        residuals = y - X @ params
        return X * residuals[:, np.newaxis]

    def _bread_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        n_obs = X.shape[0]
        return (X.T @ X) / n_obs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the conditional mean response.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted mean response.
        """
        return self.decision_function(X)

    def confidence_intervals(
        self, alpha: float = 0.05, covariance_type: str = "nonrobust"
    ) -> np.ndarray:
        """Return coefficient confidence intervals.

        Parameters
        ----------
        alpha : float, default=0.05
            Two-sided significance level.
        covariance_type : {"nonrobust", "robust"}, default="nonrobust"
            Covariance estimator used to construct intervals.

        Returns
        -------
        ndarray of shape (n_params, 2)
            Lower and upper confidence bounds for each fitted parameter.
        """
        return super().confidence_intervals(alpha=alpha, covariance_type=covariance_type)

    def summary(
        self, alpha: float = 0.05, covariance_type: str = "nonrobust"
    ) -> np.ndarray:
        """Return ``[coef, se, ci_lb, ci_ub]`` for each fitted parameter."""
        return super().summary(alpha=alpha, covariance_type=covariance_type)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination, ``R^2``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Observed response vector.

        Returns
        -------
        float
            The in-sample ``R^2`` statistic.
        """
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        residuals = y_arr - self.predict(X)
        total = y_arr - y_arr.mean()
        denom = np.dot(total, total)
        if np.isclose(denom, 0.0):
            return 1.0 if np.allclose(residuals, 0.0) else 0.0
        return float(1.0 - np.dot(residuals, residuals) / denom)


class LogisticRegression(_BaseEstimator):
    """Binary logistic regression estimated by maximum likelihood.

    The model minimizes the negative Bernoulli log-likelihood. For
    unregularized fits, the class exposes classical inverse-information
    covariance estimates and robust QMLE sandwich covariance estimates.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated slope coefficients.
    intercept_ : float
        Estimated intercept. Equals ``0.0`` when ``fit_intercept=False``.
    params_ : ndarray of shape (n_params,)
        Full parameter vector, including the intercept when present.
    std_errors_ : ndarray or None
        Classical standard errors for ``params_``.
    robust_std_errors_ : ndarray or None
        Robust QMLE standard errors for ``params_``.
    """

    def _validate_target(self, y: np.ndarray) -> None:
        unique_values = np.unique(y)
        if not np.all(np.isin(unique_values, [0.0, 1.0])):
            raise ValueError("LogisticRegression expects a binary target encoded as 0/1")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the logistic regression model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Binary response encoded as ``0`` or ``1``.

        Returns
        -------
        LogisticRegression
            The fitted estimator.
        """
        return super().fit(X, y)

    def _initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        params = np.zeros(X.shape[1], dtype=np.float64)
        if self.fit_intercept:
            p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            params[0] = float(logit(p))
        return params

    def _loss_and_gradient(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> tuple[float, np.ndarray]:
        linear_predictor = X @ params
        probs = expit(linear_predictor)
        loss = np.sum(np.logaddexp(0.0, linear_predictor) - y * linear_predictor)
        grad = X.T @ (probs - y)
        return loss, grad

    def _covariance_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        probs = expit(X @ params)
        weights = np.clip(probs * (1.0 - probs), 1e-10, None)
        fisher = X.T @ (weights[:, np.newaxis] * X)
        return np.linalg.inv(fisher)

    def _score_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        probs = expit(X @ params)
        return X * (y - probs)[:, np.newaxis]

    def _bread_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        n_obs = X.shape[0]
        probs = expit(X @ params)
        weights = np.clip(probs * (1.0 - probs), 1e-10, None)
        return (X.T @ (weights[:, np.newaxis] * X)) / n_obs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for the binary response.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Column-stacked probabilities for classes ``0`` and ``1``.
        """
        probs = expit(self.decision_function(X))
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels using a 0.5 threshold."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def confidence_intervals(
        self, alpha: float = 0.05, covariance_type: str = "nonrobust"
    ) -> np.ndarray:
        """Return coefficient confidence intervals."""
        return super().confidence_intervals(alpha=alpha, covariance_type=covariance_type)

    def summary(
        self, alpha: float = 0.05, covariance_type: str = "nonrobust"
    ) -> np.ndarray:
        """Return ``[coef, se, ci_lb, ci_ub]`` for each fitted parameter."""
        return super().summary(alpha=alpha, covariance_type=covariance_type)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean classification accuracy on the provided sample."""
        y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
        return float(np.mean(self.predict(X) == y_arr))


class PoissonRegression(_BaseEstimator):
    """Poisson regression estimated by maximum likelihood.

    The model minimizes the negative Poisson log-likelihood. For unregularized
    fits, the class exposes classical inverse-information covariance estimates
    and robust QMLE sandwich covariance estimates.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated slope coefficients.
    intercept_ : float
        Estimated intercept. Equals ``0.0`` when ``fit_intercept=False``.
    params_ : ndarray of shape (n_params,)
        Full parameter vector, including the intercept when present.
    std_errors_ : ndarray or None
        Classical standard errors for ``params_``.
    robust_std_errors_ : ndarray or None
        Robust QMLE standard errors for ``params_``.
    """

    def _validate_target(self, y: np.ndarray) -> None:
        if np.any(y < 0):
            raise ValueError("PoissonRegression expects a non-negative target")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Poisson regression model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Non-negative count response.

        Returns
        -------
        PoissonRegression
            The fitted estimator.
        """
        return super().fit(X, y)

    def _initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        params = np.zeros(X.shape[1], dtype=np.float64)
        if self.fit_intercept:
            params[0] = np.log(np.clip(y.mean(), 1e-6, None))
        return params

    def _loss_and_gradient(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> tuple[float, np.ndarray]:
        linear_predictor = np.clip(X @ params, -30.0, 30.0)
        mean = np.exp(linear_predictor)
        loss = np.sum(mean - y * linear_predictor)
        grad = X.T @ (mean - y)
        return loss, grad

    def _covariance_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        mean = np.exp(np.clip(X @ params, -30.0, 30.0))
        fisher = X.T @ (mean[:, np.newaxis] * X)
        return np.linalg.inv(fisher)

    def _score_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        mean = np.exp(np.clip(X @ params, -30.0, 30.0))
        return X * (y - mean)[:, np.newaxis]

    def _bread_matrix(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        n_obs = X.shape[0]
        mean = np.exp(np.clip(X @ params, -30.0, 30.0))
        return (X.T @ (mean[:, np.newaxis] * X)) / n_obs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the conditional mean count."""
        return np.exp(np.clip(self.decision_function(X), -30.0, 30.0))

    def confidence_intervals(
        self, alpha: float = 0.05, covariance_type: str = "nonrobust"
    ) -> np.ndarray:
        """Return coefficient confidence intervals."""
        return super().confidence_intervals(alpha=alpha, covariance_type=covariance_type)

    def summary(
        self, alpha: float = 0.05, covariance_type: str = "nonrobust"
    ) -> np.ndarray:
        """Return ``[coef, se, ci_lb, ci_ub]`` for each fitted parameter."""
        return super().summary(alpha=alpha, covariance_type=covariance_type)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the negative mean Poisson deviance."""
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        mean = np.clip(self.predict(X), 1e-10, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            deviance_terms = np.where(
                y_arr > 0,
                y_arr * np.log(y_arr / mean) - (y_arr - mean),
                mean,
            )
        return float(-2.0 * np.mean(deviance_terms))
