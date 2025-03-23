import numpy as np
import pyensmallen as pye


def test_iv_gmm():
    """
    Test GMM estimation with IV moment conditions.
    Compare results with the known true parameters.
    """
    # Generate synthetic data for IV estimation
    def generate_test_data(n=5000, seed=42):
        np.random.seed(seed)

        # Generate instruments
        z1 = np.random.normal(0, 1, n)
        z2 = np.random.normal(0, 1, n)
        Z = np.column_stack([np.ones(n), z1, z2])

        # Generate error terms with correlation
        error = np.random.normal(0, 1, n)
        v = 0.7 * error + np.random.normal(0, 0.5, n)

        # Generate endogenous variable
        x = 0.5 * z1 - 0.2 * z2 + v
        X = np.column_stack([np.ones(n), x])

        # Generate outcome
        true_beta = np.array([-0.5, 1.2])
        y = X @ true_beta + error

        return y, X, Z, true_beta

    # Generate test data
    y, X, Z, true_beta = generate_test_data()

    # Create and fit GMM estimator
    gmm = pye.EnsmallenEstimator(pye.EnsmallenEstimator.iv_moment, "optimal")
    gmm.fit(Z, y, X)

    # Verify results are close to true parameters
    assert np.allclose(gmm.theta_, true_beta, rtol=1e-1)

    # Verify standard errors are computed
    assert gmm.std_errors_ is not None
    assert len(gmm.std_errors_) == len(true_beta)
    
    # Test bootstrap score method
    bootstrap_se = gmm.bootstrap_scores(n_bootstrap=100, seed=42)
    assert len(bootstrap_se) == len(true_beta)
    
    # Test that bootstrap standard errors are similar to asymptotic ones
    assert np.allclose(bootstrap_se, gmm.std_errors_, rtol=0.15)


def test_logit_gmm():
    """
    Test GMM estimation with logistic regression moment conditions.
    Compare results with the known true parameters.
    """
    # Import required for logit moment conditions
    import jax.numpy as jnp
    import jax.scipy.special as jsp
    
    # Define logit moment condition function
    def logit_moment(z, y, x, beta):
        # Use jax.scipy.special.expit instead of scipy.special.expit
        resid = y - jsp.expit(x @ beta)
        return z * resid[:, jnp.newaxis]
    
    # Generate test data
    np.random.seed(42)
    n = 1000
    p = 2
    X = np.random.normal(size=(n, p))
    X = np.column_stack([np.ones(n), X])
    true_beta = np.array([0.5, -0.5, 0.5])
    
    # Generate binary outcome using logistic function
    p = 1 / (1 + np.exp(-X @ true_beta))
    y = np.random.binomial(1, p)
    Z = X.copy()  # Using X as instruments for this test
    
    # Create and fit GMM estimator
    gmm = pye.EnsmallenEstimator(logit_moment, "optimal")
    gmm.fit(Z, y, X)
    
    # Verify results are close to true parameters
    assert np.allclose(gmm.theta_, true_beta, rtol=0.15)
    
    # Verify standard errors are computed
    assert gmm.std_errors_ is not None
    assert len(gmm.std_errors_) == len(true_beta)
    
    # Test bootstrap score method
    bootstrap_se = gmm.bootstrap_scores(n_bootstrap=100, seed=42)
    assert len(bootstrap_se) == len(true_beta)


if __name__ == "__main__":
    test_iv_gmm()
    test_logit_gmm()