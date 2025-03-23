from typing import Callable, Optional, Union

import functools

import numpy as np
import pandas as pd
import scipy

from . import _pyensmallen as pe

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
except ImportError:
    raise ImportError("JAX is not installed. Please install JAX to use this module.")


class EnsmallenEstimator:
    """
    GMM Estimator using PyEnsmallen optimization backend with JAX autodifferentiation.
    """

    def __init__(
        self,
        moment_cond: Callable,
        weighting_matrix: Union[str, np.ndarray] = "optimal",
    ):
        """
        Initialize GMM estimator.

        Args:
            moment_cond: Function that computes moment conditions. Should be JAX-compatible.
            weighting_matrix: Either "optimal" for two-step GMM or a custom weighting matrix
        """
        self.moment_cond = moment_cond
        self.weighting_matrix = weighting_matrix

        # Initialize attributes that will be set during fit
        self.z_ = None
        self.y_ = None
        self.x_ = None
        self.n_ = None
        self.k_ = None
        self.W_ = None
        self.theta_ = None
        self.Gamma_ = None
        self.vtheta_ = None
        self.std_errors_ = None
        self.bootstrap_thetas_ = None
        self.bootstrap_std_errors_ = None

        # JAX-specific attributes
        self.z_jax_ = None
        self.y_jax_ = None
        self.x_jax_ = None
        self.W_jax_ = None

    def gmm_objective(self, beta: np.ndarray, gradient: np.ndarray) -> float:
        """
        Compute GMM objective function value and gradient for PyEnsmallen.
        Uses JAX for autodifferentiation.

        Args:
            beta: Parameter vector
            gradient: Will be filled with gradient values

        Returns:
            Objective function value
        """
        # Convert numpy arrays to JAX arrays if not done already
        if self.z_jax_ is None:
            self.z_jax_ = jnp.array(self.z_, dtype=jnp.float64)
            self.y_jax_ = jnp.array(self.y_, dtype=jnp.float64)
            self.x_jax_ = jnp.array(self.x_, dtype=jnp.float64)

        # Convert beta to JAX array
        beta_jax = jnp.array(beta, dtype=jnp.float64)

        # Compute moment conditions for weighting matrix update
        moments = self._jit_moment_cond(beta_jax, self.z_jax_, self.y_jax_, self.x_jax_)
        moments_np = np.array(moments)

        # Update weighting matrix if using optimal weighting
        if self.weighting_matrix == "optimal":
            self.W_ = self.optimal_weighting_matrix(moments_np)
            self.W_jax_ = jnp.array(self.W_, dtype=jnp.float64)
        elif isinstance(self.weighting_matrix, str):
            if self.W_jax_ is None:
                self.W_ = np.eye(moments_np.shape[1])
                self.W_jax_ = jnp.array(self.W_, dtype=jnp.float64)
        elif self.W_jax_ is None:
            self.W_ = np.asarray(self.weighting_matrix)
            self.W_jax_ = jnp.array(self.W_, dtype=jnp.float64)

        # Compute objective value using JIT-compiled function
        obj_value = float(
            self._jit_objective(
                beta_jax, self.z_jax_, self.y_jax_, self.x_jax_, self.W_jax_
            )
        )

        # Compute gradient using JAX autodiff
        grad_fn = jax.grad(
            lambda b: self._jit_objective(
                b, self.z_jax_, self.y_jax_, self.x_jax_, self.W_jax_
            )
        )
        grad_value = grad_fn(beta_jax)

        # Assign gradient (PyEnsmallen requires modifying gradient in-place)
        gradient[:] = np.array(grad_value)

        return obj_value

    def optimal_weighting_matrix(
        self,
        moments: np.ndarray,
        epsi: float = 1e-8,
    ) -> np.ndarray:
        """
        Calculate optimal weighting matrix: (E[g_i g_i'])^(-1)

        Args:
            moments: Matrix of moment conditions
            epsi: Regularization parameter for stability

        Returns:
            Optimal weighting matrix
        """
        # Compute the moment covariance matrix with regularization for stability
        S = (1 / self.n_) * (moments.T @ moments)
        epsilon = epsi * np.eye(S.shape[0])
        return np.linalg.inv(S + epsilon)

    def fit(
        self,
        z: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        verbose: bool = False,
    ) -> None:
        """
        Fit the GMM model using PyEnsmallen optimizer with JAX gradients.

        Args:
            z: Instrument matrix
            y: Outcome vector
            x: Covariate matrix (including intercept)
            verbose: Whether to print optimization details
        """
        # Store data
        self.z_, self.y_, self.x_ = z, y, x
        self.n_, self.k_ = x.shape

        # Initialize JAX arrays (done on first call to gmm_objective)
        self.z_jax_ = None
        self.y_jax_ = None
        self.x_jax_ = None
        self.W_jax_ = None

        # Initialize weighting matrix
        self.W_ = None

        # Initialize optimizer with specified parameters
        optimizer = pe.L_BFGS()

        # Run optimization
        try:
            # Try OLS for initialization if possible
            initial_point = np.linalg.lstsq(x, y, rcond=None)[0]
        except:  # noqa: E722
            # Fall back to zeros if OLS fails
            initial_point = np.zeros(self.k_)

        self.theta_ = optimizer.optimize(self.gmm_objective, initial_point)

        # Compute standard errors
        try:
            self.compute_asymptotic_variance()
        except Exception as e:
            if verbose:
                print(f"Error computing standard errors: {e}")
            self.std_errors_ = None

    def compute_asymptotic_variance(self) -> None:
        """
        Compute asymptotic variance and standard errors.
        """
        # Compute Jacobian of moment conditions
        self.Gamma_ = self.jacobian_moment_cond()

        # Asymptotic variance: (G'WG)^(-1)
        self.vtheta_ = np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)

        # Standard errors: sqrt(diag(vtheta) / n)
        self.std_errors_ = np.sqrt(np.diag(self.vtheta_) / self.n_)

    def jacobian_moment_cond(self) -> np.ndarray:
        """
        Compute Jacobian of moment conditions using JAX.

        Returns:
            Jacobian matrix
        """
        # Use JAX's jacfwd to compute the Jacobian of mean moment conditions
        beta_jax = jnp.array(self.theta_, dtype=jnp.float64)
        jac_fn = jax.jacfwd(
            lambda b: self._jit_mean_moment_fn(b, self.z_jax_, self.y_jax_, self.x_jax_)
        )
        jac = jac_fn(beta_jax)

        return np.array(jac)

    ######################################################################
    def bootstrap_scores(
        self,
        n_bootstrap: int = 1000,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Compute bootstrap standard errors using the fast score bootstrap method.

        This method is computationally more efficient than the full bootstrap.
        It bootstraps the score functions (moment conditions) rather than
        recomputing the GMM estimator for each bootstrap sample.

        Args:
            n_bootstrap: Number of bootstrap iterations
            seed: Random seed for reproducibility
            verbose: Whether to print progress

        Returns:
            Bootstrap standard errors
        """
        if not hasattr(self, "theta_"):
            raise ValueError("Model must be fitted first")

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Ensure JAX arrays are initialized
        if self.z_jax_ is None:
            self.z_jax_ = jnp.array(self.z_, dtype=jnp.float64)
            self.y_jax_ = jnp.array(self.y_, dtype=jnp.float64)
            self.x_jax_ = jnp.array(self.x_, dtype=jnp.float64)

        # Compute the jacobian of moment conditions (Gamma matrix)
        if self.Gamma_ is None:
            self.Gamma_ = self.jacobian_moment_cond()

        # Compute the inverse of Gamma'WGamma
        if not hasattr(self, "vtheta_") or self.vtheta_ is None:
            self.compute_asymptotic_variance()

        # Compute estimated M matrix (defined in the method description)
        # M = (Gamma'WGamma)^(-1) * Gamma'W
        M = self.vtheta_ @ self.Gamma_.T @ self.W_

        # Compute the score functions for each observation
        # Z_i = g(X_i, theta)
        Z = self.moment_cond(self.z_, self.y_, self.x_, self.theta_)
        Z_mean = np.mean(Z, axis=0)

        # Store bootstrap parameter estimates
        bootstrap_thetas = np.zeros((n_bootstrap, len(self.theta_)))

        # Create bootstrap indices for sampling scores
        n = self.n_

        for b in range(n_bootstrap):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{n_bootstrap}")

            # Generate bootstrap sample indices
            indices = np.random.choice(n, size=n, replace=True)

            # Bootstrap the scores
            Z_boot = Z[indices]
            Z_boot_mean = np.mean(Z_boot, axis=0)

            # Compute bootstrap estimate using fast method
            # theta^* = theta + M(Z^* - Z)
            delta = Z_boot_mean - Z_mean
            theta_boot = self.theta_ + M @ delta

            bootstrap_thetas[b] = theta_boot

        # Calculate bootstrap standard errors
        bootstrap_se = np.std(bootstrap_thetas, axis=0)

        # Store the bootstrap results
        self.bootstrap_thetas_ = bootstrap_thetas
        self.bootstrap_std_errors_ = bootstrap_se

        return bootstrap_se

    def bootstrap_full(
        self,
        n_bootstrap: int = 1000,
        seed: Optional[int] = None,
        batch_size: int = 50,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Compute bootstrap standard errors using batched processing.

        Args:
            n_bootstrap: Number of bootstrap iterations
            seed: Random seed for reproducibility
            batch_size: Number of bootstrap samples to process in each batch
            verbose: Whether to print progress

        Returns:
            Bootstrap standard errors
        """
        if not hasattr(self, "theta_"):
            raise ValueError("Model must be fitted first")

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Create bootstrap indices
        n = self.n_
        bootstrap_indices = np.random.choice(n, size=(n_bootstrap, n), replace=True)
        bootstrap_thetas = []

        # Process bootstraps in batches to improve memory efficiency
        for batch_start in range(0, n_bootstrap, batch_size):
            batch_end = min(batch_start + batch_size, n_bootstrap)
            batch_indices = bootstrap_indices[batch_start:batch_end]

            batch_results = []
            for i, indices in enumerate(batch_indices):
                iter_num = batch_start + i
                theta_boot = self._run_single_bootstrap(indices, iter_num, seed)
                batch_results.append(theta_boot)

                if verbose and (iter_num + 1) % 10 == 0:
                    print(
                        f"Completed {iter_num + 1}/{n_bootstrap} bootstrap iterations"
                    )

            bootstrap_thetas.extend(batch_results)

            if verbose:
                print(
                    f"Completed batch {batch_start // batch_size + 1}/{(n_bootstrap - 1) // batch_size + 1}"
                )

        # Convert results to array
        bootstrap_thetas = np.array(bootstrap_thetas)

        # Calculate bootstrap standard errors
        bootstrap_se = np.std(bootstrap_thetas, axis=0)

        # Store the bootstrap results
        self.bootstrap_thetas_ = bootstrap_thetas
        self.bootstrap_std_errors_ = bootstrap_se

        return bootstrap_se

    def _run_single_bootstrap(self, indices, iteration=0, seed=None):
        """
        Run a single bootstrap iteration.

        Args:
            indices: Indices for bootstrap sample
            iteration: Bootstrap iteration number (for seed)
            seed: Base random seed

        Returns:
            Bootstrap parameter estimates
        """
        # Set seed for this iteration if provided
        if seed is not None:
            np.random.seed(seed + iteration)

        # Sample data using the indices
        z_boot = self.z_[indices]
        y_boot = self.y_[indices]
        x_boot = self.x_[indices]

        # Create a new estimator instance
        estimator = EnsmallenEstimator(self.moment_cond, self.weighting_matrix)
        estimator.fit(z_boot, y_boot, x_boot)

        return estimator.theta_

    @staticmethod
    def iv_moment(
        z: Union[np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        x: Union[np.ndarray, jnp.ndarray],
        beta: Union[np.ndarray, jnp.ndarray],
    ) -> Union[np.ndarray, jnp.ndarray]:
        """
        Standard IV moment function: z_i * (y_i - x_i'β).
        Works with both NumPy and JAX arrays.

        Args:
            z: Instrument matrix
            y: Outcome vector
            x: Covariate matrix (including intercept)
            beta: Parameter vector

        Returns:
            Matrix of moment conditions
        """
        if isinstance(z, jnp.ndarray):
            # JAX implementation
            residuals = y - jnp.dot(x, beta)
            return z * residuals[:, jnp.newaxis]
        else:
            # NumPy implementation
            return z * (y - x @ beta)[:, np.newaxis]

    ######################################################################
    def summary(
        self,
        prec: int = 4,
        param_names: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Generate summary statistics for the fitted model.

        Args:
            prec: Precision for rounding results

        Returns:
            DataFrame with model summary statistics
        """
        if not hasattr(self, "theta_") or self.std_errors_ is None:
            raise ValueError("Model must be fitted with valid standard errors first")

        # Calculate t-statistics and p-values
        t_stats = self.theta_ / self.std_errors_
        p_values = 2 * (1 - scipy.stats.norm.cdf(np.abs(t_stats)))

        # 95% confidence intervals
        ci_lower = self.theta_ - scipy.stats.norm.ppf(0.975) * self.std_errors_
        ci_upper = self.theta_ + scipy.stats.norm.ppf(0.975) * self.std_errors_

        # Create parameter names
        if param_names is None:
            param_names = [f"θ_{i}" for i in range(len(self.theta_))]

        # Create summary DataFrame
        result = pd.DataFrame(
            {
                "parameter": param_names,
                "coef": np.round(self.theta_, prec),
                "std err": np.round(self.std_errors_, prec),
                "t": np.round(t_stats, prec),
                "p-value": np.round(p_values, prec),
                "[0.025": np.round(ci_lower, prec),
                "0.975]": np.round(ci_upper, prec),
            }
        )

        # Add bootstrap standard errors if available
        if hasattr(self, "bootstrap_std_errors_"):
            result["boot_se"] = np.round(self.bootstrap_std_errors_, prec)
            result["ratio"] = np.round(
                self.bootstrap_std_errors_ / self.std_errors_, prec
            )

        return result

    ######################################################################
    # internals, jitted functions
    def _moment_cond_jax(self, z, y, x, beta):
        """
        JAX-compatible wrapper for moment conditions function.

        Args:
            z: Instrument matrix (JAX array)
            y: Outcome vector (JAX array)
            x: Covariate matrix (JAX array)
            beta: Parameter vector (JAX array)

        Returns:
            Matrix of moment conditions (JAX array)
        """
        return self.moment_cond(z, y, x, beta)

    def _gmm_objective_jax(self, beta, z, y, x, W):
        """
        JAX implementation of GMM objective function.

        Args:
            beta: Parameter vector (JAX array)
            z: Instrument matrix (JAX array)
            y: Outcome vector (JAX array)
            x: Covariate matrix (JAX array)
            W: Weighting matrix (JAX array)

        Returns:
            Objective function value (scalar)
        """
        # Compute moment conditions
        moments = self._moment_cond_jax(z, y, x, beta)

        # Calculate objective function: (1/n * sum(g_i))' W (1/n * sum(g_i))
        mavg = jnp.mean(moments, axis=0)
        obj_value = mavg.T @ W @ mavg

        return obj_value

    # JIT-compiled functions
    @functools.partial(jax.jit, static_argnums=(0,))
    def _jit_moment_cond(self, beta, z, y, x):
        """JIT-compiled moment conditions"""
        return self._moment_cond_jax(z, y, x, beta)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _jit_objective(self, beta, z, y, x, W):
        """JIT-compiled objective function"""
        moments = self._jit_moment_cond(beta, z, y, x)
        mavg = jnp.mean(moments, axis=0)
        return mavg.T @ W @ mavg

    @functools.partial(jax.jit, static_argnums=(0,))
    def _jit_mean_moment_fn(self, beta, z, y, x):
        """JIT-compiled mean of moment conditions"""
        moments = self._moment_cond_jax(z, y, x, beta)
        return jnp.mean(moments, axis=0)
