#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmarking Performance of pyensmallen Optimization Library

This script compares the performance of pyensmallen against other optimization
libraries across various model types and dataset sizes. It generates synthetic
data for each test case and reports execution times and parameter accuracy.

The script benchmarks:
- Linear Regression
- Logistic Regression
- Poisson Regression

Across libraries:
- pyensmallen
- scipy.optimize
- statsmodels
- JAX + optax (where applicable)
- CVXPY (where applicable)
"""

import numpy as np
import pandas as pd
import time
import argparse
import pyensmallen
import scipy.optimize
from scipy.special import expit
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import tqdm

# Check if JAX is available and import it
try:
    import jax
    import jax.numpy as jnp
    import optax
    HAS_JAX = True
    # Enable double precision
    jax.config.update("jax_enable_x64", True)
except ImportError:
    HAS_JAX = False
    print("JAX not found, skipping JAX benchmarks")

# Check if CVXPY is available and import it
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("CVXPY not found, skipping CVXPY benchmarks")


def linear_objective(params, gradient, X, y):
    """Linear regression objective function for pyensmallen"""
    params = params.reshape(-1, 1)
    residuals = X @ params - y.reshape(-1, 1)
    objective = np.sum(residuals**2)
    grad = 2 * X.T @ residuals
    gradient[:] = grad.flatten()
    return objective


def logistic_objective(params, gradient, X, y):
    """Logistic regression objective function for pyensmallen"""
    z = X @ params
    h = expit(z)
    objective = -np.sum(y * np.log(h) + (1 - y) * np.log1p(-h))
    if np.isnan(objective):
        objective = np.inf
    grad = X.T @ (h - y)
    gradient[:] = grad
    return objective


def poisson_objective(params, gradient, X, y):
    """Poisson regression objective function for pyensmallen"""
    params = params.reshape(-1, 1)
    y = y.reshape(-1, 1)
    Xbeta = X @ params
    lambda_ = np.exp(Xbeta)
    objective = np.sum(lambda_ - np.multiply(y, np.log(lambda_)))
    # Compute the gradient
    grad = X.T @ (lambda_ - y)
    gradient[:] = grad.ravel()
    return objective


def generate_data(model_type, n_samples, n_features, seed=42):
    """Generate synthetic data for benchmark tests

    Parameters
    ----------
    model_type : str
        Type of model ('linear', 'logistic', or 'poisson')
    n_samples : int
        Number of samples (observations)
    n_features : int
        Number of features (predictors)
    seed : int, optional
        Random seed for reproducibility, by default 42

    Returns
    -------
    tuple
        X, y, true_params
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    true_params = np.random.rand(n_features)

    if model_type == 'linear':
        # Linear regression: y = X @ beta + noise
        y = X @ true_params + np.random.normal(0, 0.1, n_samples)

    elif model_type == 'logistic':
        # Logistic regression: p(y=1) = sigmoid(X @ beta)
        p = expit(X @ true_params)
        y = np.random.binomial(1, p)

    elif model_type == 'poisson':
        # Poisson regression: y ~ Poisson(exp(X @ beta))
        lambda_ = np.exp(X @ true_params)
        y = np.random.poisson(lambda_)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return X, y, true_params


def benchmark_linear_regression(X, y, true_params, libraries=None):
    """Benchmark linear regression across different libraries

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target variable
    true_params : ndarray
        True parameters
    libraries : list, optional
        List of libraries to benchmark, by default all

    Returns
    -------
    dict
        Results including parameters and execution times
    """
    if libraries is None:
        libraries = ['pyensmallen', 'scipy', 'statsmodels']
        if HAS_JAX:
            libraries.append('jax')
        if HAS_CVXPY:
            libraries.append('cvxpy')

    k = X.shape[1]
    results = {}
    initial_params = np.random.rand(k)

    # pyensmallen
    if 'pyensmallen' in libraries:
        start_time = time.time()
        optimizer = pyensmallen.L_BFGS()
        result_ens = optimizer.optimize(
            lambda params, gradient: linear_objective(params, gradient, X, y),
            initial_params.copy(),
        )
        end_time = time.time()
        results['pyensmallen'] = {
            'params': result_ens,
            'time': end_time - start_time,
        }

    # scipy
    if 'scipy' in libraries:
        start_time = time.time()
        result_scipy = scipy.optimize.minimize(
            fun=lambda b: np.sum((X @ b - y) ** 2),
            x0=initial_params.copy(),
            jac=lambda b: 2 * X.T @ (X @ b - y),
        ).x
        end_time = time.time()
        results['scipy'] = {
            'params': result_scipy,
            'time': end_time - start_time,
        }

    # statsmodels
    if 'statsmodels' in libraries:
        start_time = time.time()
        sm_result = sm.OLS(y, X).fit().params
        end_time = time.time()
        results['statsmodels'] = {
            'params': sm_result,
            'time': end_time - start_time,
        }

    # JAX
    if 'jax' in libraries and HAS_JAX:
        start_time = time.time()
        X_jnp, y_jnp = jnp.array(X), jnp.array(y)

        def compute_loss(beta):
            y_pred = jnp.dot(X_jnp, beta)
            loss = jnp.mean((y_pred - y_jnp) ** 2)
            return loss

        params = jnp.array(initial_params.copy())
        solver = optax.lbfgs()
        opt_state = solver.init(params)
        value_and_grad = jax.value_and_grad(compute_loss)

        # Optimization loop
        for i in range(10):
            value, grad = value_and_grad(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

        end_time = time.time()
        results['jax'] = {
            'params': np.array(params),
            'time': end_time - start_time,
        }

    # CVXPY
    if 'cvxpy' in libraries and HAS_CVXPY:
        start_time = time.time()
        b = cp.Variable(k)
        cost = cp.norm(X @ b - y, p=2) ** 2
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve(solver=cp.SCS)
        end_time = time.time()
        results['cvxpy'] = {
            'params': b.value,
            'time': end_time - start_time,
        }

    # Add true parameters
    results['true'] = {
        'params': true_params,
        'time': np.nan,
    }

    return results


def benchmark_logistic_regression(X, y, true_params, libraries=None):
    """Benchmark logistic regression across different libraries

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target variable
    true_params : ndarray
        True parameters
    libraries : list, optional
        List of libraries to benchmark, by default all

    Returns
    -------
    dict
        Results including parameters and execution times
    """
    if libraries is None:
        libraries = ['pyensmallen', 'scipy', 'statsmodels']
        if HAS_JAX:
            libraries.append('jax')
        if HAS_CVXPY:
            libraries.append('cvxpy')

    k = X.shape[1]
    results = {}
    initial_params = np.random.rand(k)

    # pyensmallen
    if 'pyensmallen' in libraries:
        start_time = time.time()
        optimizer = pyensmallen.L_BFGS()
        result_ens = optimizer.optimize(
            lambda params, gradient: logistic_objective(params, gradient, X, y),
            initial_params.copy(),
        )
        end_time = time.time()
        results['pyensmallen'] = {
            'params': result_ens,
            'time': end_time - start_time,
        }

    # scipy
    if 'scipy' in libraries:
        start_time = time.time()
        try:
            result_scipy = scipy.optimize.minimize(
                fun=lambda b: -np.sum(
                    y * np.log(expit(X @ b) + 1e-15)
                    + (1 - y) * np.log(1 - expit(X @ b) + 1e-15)
                ),
                x0=initial_params.copy(),
                jac=lambda b: X.T @ (expit(X @ b) - y),
            ).x
            scipy_time = time.time() - start_time
            scipy_result = result_scipy
        except Exception as e:
            print(f"scipy optimization failed: {e}")
            scipy_time = np.nan
            scipy_result = np.full_like(initial_params, np.nan)

        end_time = time.time()
        results['scipy'] = {
            'params': scipy_result,
            'time': scipy_time,
        }

    # statsmodels
    if 'statsmodels' in libraries:
        start_time = time.time()
        try:
            sm_result = sm.Logit(y, X).fit(disp=0).params
            sm_time = time.time() - start_time
        except Exception as e:
            print(f"statsmodels optimization failed: {e}")
            sm_time = np.nan
            sm_result = np.full_like(initial_params, np.nan)

        end_time = time.time()
        results['statsmodels'] = {
            'params': sm_result,
            'time': sm_time,
        }

    # JAX
    if 'jax' in libraries and HAS_JAX:
        start_time = time.time()
        X_jnp, y_jnp = jnp.array(X), jnp.array(y)

        def logistic_likelihood(beta):
            z = jnp.dot(X_jnp, beta)
            h = jax.nn.sigmoid(z)
            loss = -jnp.sum(y_jnp * jnp.log(h + 1e-15) + (1 - y_jnp) * jnp.log(1 - h + 1e-15))
            return loss

        params = jnp.array(initial_params.copy())
        solver = optax.lbfgs()
        opt_state = solver.init(params)
        value_and_grad = jax.value_and_grad(logistic_likelihood)

        # Optimization loop
        for i in range(10):
            value, grad = value_and_grad(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

        end_time = time.time()
        results['jax'] = {
            'params': np.array(params),
            'time': end_time - start_time,
        }

    # CVXPY
    if 'cvxpy' in libraries and HAS_CVXPY:
        start_time = time.time()
        try:
            b = cp.Variable(k)
            log_likelihood = cp.sum(
                cp.multiply(y, X @ b)
                - cp.logistic(X @ b)
            )
            prob = cp.Problem(cp.Maximize(log_likelihood))
            prob.solve(solver=cp.SCS, verbose=False)
            cvxpy_time = time.time() - start_time
            cvxpy_result = b.value
        except Exception as e:
            print(f"CVXPY optimization failed: {e}")
            cvxpy_time = np.nan
            cvxpy_result = np.full_like(initial_params, np.nan)

        results['cvxpy'] = {
            'params': cvxpy_result,
            'time': cvxpy_time,
        }

    # Add true parameters
    results['true'] = {
        'params': true_params,
        'time': np.nan,
    }

    return results


def benchmark_poisson_regression(X, y, true_params, libraries=None):
    """Benchmark poisson regression across different libraries

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target variable
    true_params : ndarray
        True parameters
    libraries : list, optional
        List of libraries to benchmark, by default all

    Returns
    -------
    dict
        Results including parameters and execution times
    """
    if libraries is None:
        libraries = ['pyensmallen', 'scipy', 'statsmodels']
        if HAS_JAX:
            libraries.append('jax')

    k = X.shape[1]
    results = {}
    initial_params = np.random.rand(k)

    # pyensmallen
    if 'pyensmallen' in libraries:
        start_time = time.time()
        optimizer = pyensmallen.L_BFGS()
        result_ens = optimizer.optimize(
            lambda params, gradient: poisson_objective(params, gradient, X, y),
            initial_params.copy(),
        )
        end_time = time.time()
        results['pyensmallen'] = {
            'params': result_ens,
            'time': end_time - start_time,
        }

    # scipy
    if 'scipy' in libraries:
        start_time = time.time()
        try:
            result_scipy = scipy.optimize.minimize(
                fun=lambda b: np.sum(np.exp(X @ b) - y * (X @ b)),
                x0=initial_params.copy(),
                jac=lambda b: X.T @ (np.exp(X @ b) - y),
            ).x
            scipy_time = time.time() - start_time
            scipy_result = result_scipy
        except Exception as e:
            print(f"scipy optimization failed: {e}")
            scipy_time = np.nan
            scipy_result = np.full_like(initial_params, np.nan)

        results['scipy'] = {
            'params': scipy_result,
            'time': scipy_time,
        }

    # statsmodels
    if 'statsmodels' in libraries:
        start_time = time.time()
        try:
            sm_result = sm.Poisson(y, X).fit(disp=0, maxiter=100).params
            sm_time = time.time() - start_time
        except Exception as e:
            print(f"statsmodels optimization failed: {e}")
            sm_time = np.nan
            sm_result = np.full_like(initial_params, np.nan)

        results['statsmodels'] = {
            'params': sm_result,
            'time': sm_time,
        }

    # JAX
    if 'jax' in libraries and HAS_JAX:
        start_time = time.time()
        X_jnp, y_jnp = jnp.array(X), jnp.array(y)

        def poisson_likelihood(beta):
            z = jnp.dot(X_jnp, beta)
            lambda_ = jnp.exp(z)
            loss = jnp.sum(lambda_ - y_jnp * z)
            return loss

        params = jnp.array(initial_params.copy())

        # Use Adam instead of L-BFGS for better convergence with Poisson
        solver = optax.adam(1e-2)
        opt_state = solver.init(params)
        value_and_grad_fn = jax.value_and_grad(poisson_likelihood)

        # JIT the update function
        @jax.jit
        def update_step(params, opt_state):
            loss, grads = value_and_grad_fn(params)
            updates, new_opt_state = solver.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        # Optimization loop
        for i in range(500):  # More iterations for Adam
            params, opt_state, _ = update_step(params, opt_state)

        end_time = time.time()
        results['jax'] = {
            'params': np.array(params),
            'time': end_time - start_time,
        }

    # Add true parameters
    results['true'] = {
        'params': true_params,
        'time': np.nan,
    }

    return results


def calculate_metrics(results):
    """Calculate comparison metrics between estimated and true parameters

    Parameters
    ----------
    results : dict
        Results dictionary with parameters and times

    Returns
    -------
    dict
        Dictionary with added metrics
    """
    true_params = results['true']['params']

    for library, res in results.items():
        if library == 'true':
            continue

        params = res['params']

        # Calculate MSE
        if not np.any(np.isnan(params)):
            mse = np.mean((params - true_params) ** 2)
        else:
            mse = np.nan

        # Calculate correlation
        if not np.any(np.isnan(params)):
            corr = np.corrcoef(params, true_params)[0, 1]
        else:
            corr = np.nan

        results[library]['mse'] = mse
        results[library]['correlation'] = corr

    return results


def run_benchmarks(sizes, model_types=None, libraries=None, n_trials=1, output_dir='.'):
    """Run benchmarks for different model types and data sizes

    Parameters
    ----------
    sizes : list of tuples
        List of (n_samples, n_features) tuples to test
    model_types : list, optional
        List of model types to test, by default ['linear', 'logistic', 'poisson']
    libraries : list, optional
        List of libraries to test, by default all available
    n_trials : int, optional
        Number of trials to run for each configuration, by default 1
    output_dir : str, optional
        Directory to save results, by default '.'

    Returns
    -------
    dict
        Dictionary with all benchmark results
    """
    if model_types is None:
        model_types = ['linear', 'logistic', 'poisson']

    if libraries is None:
        libraries = ['pyensmallen', 'scipy', 'statsmodels']
        if HAS_JAX:
            libraries.append('jax')
        if HAS_CVXPY:
            libraries.append('cvxpy')

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_type in model_types:
        all_results[model_type] = {}

        for n_samples, n_features in sizes:
            size_key = f"n{n_samples}_k{n_features}"
            all_results[model_type][size_key] = []

            # tqdm progress bar
            for trial in tqdm.tqdm(range(n_trials), desc=f"Running {model_type} regression with n={n_samples}, k={n_features}"):
                print(f"Running {model_type} regression with n={n_samples}, k={n_features}, trial {trial+1}/{n_trials}")

                # Generate data
                X, y, true_params = generate_data(model_type, n_samples, n_features, seed=42+trial)

                # Run benchmarks for the appropriate model type
                if model_type == 'linear':
                    results = benchmark_linear_regression(X, y, true_params, libraries)
                elif model_type == 'logistic':
                    results = benchmark_logistic_regression(X, y, true_params, libraries)
                elif model_type == 'poisson':
                    results = benchmark_poisson_regression(X, y, true_params, libraries)

                # Calculate metrics
                results = calculate_metrics(results)
                all_results[model_type][size_key].append(results)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"benchmark_results_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    for model_type in all_results:
        for size_key in all_results[model_type]:
            for trial_idx, trial in enumerate(all_results[model_type][size_key]):
                for lib in trial:
                    if 'params' in trial[lib] and isinstance(trial[lib]['params'], np.ndarray):
                        all_results[model_type][size_key][trial_idx][lib]['params'] = trial[lib]['params'].tolist()

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


def main():
    """Main function to run benchmarks"""
    parser = argparse.ArgumentParser(description='Run performance benchmarks for pyensmallen')

    parser.add_argument('--sizes', type=str, default="1_000,5;10_000,5;100_000,5; 1_000_000,5; 10_000_000,5; 1_000,20;10_000,20;100_000,20; 1_000_000,20; 10_000_000,20",
                      help='Semicolon-separated list of comma-separated n_samples,n_features pairs')

    parser.add_argument('--models', type=str, default="linear,logistic,poisson",
                      help='Comma-separated list of model types to benchmark')

    parser.add_argument('--libraries', type=str, default="pyensmallen,scipy,statsmodels",
                      help='Comma-separated list of libraries to benchmark')

    parser.add_argument('--trials', type=int, default=20,
                      help='Number of trials to run for each configuration')

    parser.add_argument('--output', type=str, default='.',
                      help='Directory to save benchmark results')

    args = parser.parse_args()

    # Parse sizes
    size_pairs = [tuple(map(int, pair.split(','))) for pair in args.sizes.split(';')]

    # Parse model types
    model_types = args.models.split(',')

    # Parse libraries
    libraries = args.libraries.split(',')

    # Run benchmarks
    results = run_benchmarks(
        sizes=size_pairs,
        model_types=model_types,
        libraries=libraries,
        n_trials=args.trials,
        output_dir=args.output
    )

    print(f"Benchmarks complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
