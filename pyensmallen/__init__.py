"""
PyEnsmallen: Python bindings for the Ensmallen optimization library

This package provides Python bindings for the Ensmallen C++ optimization library,
offering various optimization algorithms including:
- L-BFGS
- Frank-Wolfe
- Adam and variants (AdaMax, AMSGrad, OptimisticAdam, Nadam)

It also includes common loss functions for machine learning:
- Linear regression (least squares)
- Logistic regression
- Poisson regression
"""

import os

from ._pyensmallen import *
from .losses import linear_obj, logistic_obj, poisson_obj

# Core optimizers and loss functions (always available)
__all__ = [
    "L_BFGS",
    "FrankWolfe",
    "SimplexFrankWolfe",
    "Adam",
    "AdaMax",
    "AMSGrad",
    "OptimisticAdam",
    "Nadam",
    "linear_obj",
    "logistic_obj",
    "poisson_obj",
]

# Conditionally import GMM functionality (requires JAX)
try:
    from .gmm import EnsmallenEstimator
    __all__.append("EnsmallenEstimator")
except ImportError:
    # JAX not available - install with: pip install pyensmallen[gmm]
    pass
