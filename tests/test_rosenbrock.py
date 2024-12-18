# %%
import numpy as np

import pyensmallen as pye


# %%
def rosenbrock(x, a=2):
    return np.sum((a - x[:-1]) ** 2 + 100.0 * (x[1:] - x[:-1] ** 2) ** 2)


def rosenbrock_gradient(x, a=2):
    grad = np.zeros_like(x)
    # Gradient for the first element
    grad[0] = -2 * (a - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    # Gradient for the middle elements
    grad[1:-1] = (
        -2 * (a - x[1:-1])
        + 200 * (x[1:-1] - x[:-2] ** 2)
        - 400 * x[1:-1] * (x[2:] - x[1:-1] ** 2)
    )
    # Gradient for the last element
    grad[-1] = 200 * (x[-1] - x[-2] ** 2)
    return grad


def objective_function(x, grad):
    grad[:] = rosenbrock_gradient(x)
    return rosenbrock(x)


# %%


def test_rosenbrock():
    # Initial guess
    initial_x = np.array([-1.2, 1.0])

    # pyensmallen solution
    optimizer = pye.L_BFGS()
    result_ens = optimizer.optimize(objective_function, initial_x)

    # true value (a, a^2)
    true_value = np.array([2, 4])

    assert np.allclose(result_ens, true_value, rtol=1e-5)


# %%
