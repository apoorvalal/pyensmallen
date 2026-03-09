# Econometrics and Supervised Learning Roadmap

This document collects proposed functionality expansions for `pyensmallen`, based on the existing notebooks and current API surface.

## First Tranche

The first set of items to prioritize:

1. Estimator classes for common supervised models
2. First-class regularization support
3. Proper stochastic / mini-batch training support

These are the highest-leverage additions for making `pyensmallen` useful beyond optimizer demos and low-level objective wrappers.

## Full Proposal List

### 1. Estimator classes for common supervised models

Add estimator APIs for standard econometrics and ML models:

- `LinearRegression`
- `LogisticRegression`
- `PoissonRegression`
- `MultinomialLogit`
- `Probit`
- `NegativeBinomial`
- optionally `CoxPH`

Each estimator should expose a workflow-level API:

- `fit`
- `predict`
- `predict_proba` where applicable
- `score`
- fitted coefficients and intercept
- convergence diagnostics
- optional standard errors and summaries

Rationale:
The current API is objective-first. Real workflows usually want model objects, not raw closures.

### 2. First-class regularization support

Add penalized estimation support across core models:

- L1
- L2
- elastic net
- regularization paths
- cross-validated penalty selection

This should work naturally with existing constrained optimization ideas already present in the package.

Rationale:
This is central to both supervised learning and modern econometrics, especially in high-dimensional settings.

### 3. Productized JAX bridge

Turn the current notebook pattern into a supported API:

- `JaxObjective`
- `AutoDiffObjective`
- or `AutoDiffEstimator`

The wrapper should accept a JAX loss function and automatically provide:

- objective evaluation
- gradients
- shape handling
- low-boilerplate integration with ensmallen optimizers

Rationale:
The multinomial logit notebook already shows this is useful. It should be library functionality, not notebook glue code.

### 4. Proper stochastic / mini-batch training support

Expose true separable-objective support for first-order optimizers:

- mini-batch iteration
- batch indexing
- data shuffling
- epoch-level callbacks
- objective tracking
- early stopping hooks

This is especially important for:

- large supervised-learning problems
- neural-style differentiable objectives
- scalable generalized linear models

Rationale:
The Adam-family bindings exist, but the current wrapper behaves like full-batch optimization. That limits the ML use case substantially.

### 5. Inference utilities beyond point estimation

Expand the econometrics side with reusable inference tools:

- sandwich covariance
- HC0-HC3 robust standard errors
- clustered standard errors
- HAC / Newey-West
- Wald, likelihood-ratio, and score tests
- delta method
- marginal effects
- bootstrap helpers for MLE models

Rationale:
The package already goes in this direction for GMM. Extending it to MLE models would make it much more useful for empirical work.

### 6. Model selection and evaluation tools

Add workflow-level evaluation and tuning utilities:

- train / validation splitting
- K-fold cross-validation
- time-series cross-validation
- standard supervised metrics
- calibration diagnostics
- hyperparameter search
- early stopping support

Metrics should include at least:

- RMSE
- MAE
- log loss
- AUC

Rationale:
Several notebooks currently hand-roll comparison and tuning logic that should live in the library.

### 7. Higher-level causal and panel estimators

Potential estimator layer additions include:

- `SyntheticControl`
- balancing weights estimators
- ridge-augmented synthetic control
- matrix-completion synthetic control
- DiD and event-study estimators
- IV / 2SLS / LIML
- doubly robust or orthogonal-score estimators

Rationale:
This is a natural applied econometrics extension, though a substantial part of this already exists in the sibling `synthlearners` repository.

### 8. Formula and DataFrame ergonomics

Improve usability for empirical workflows:

- formula interface
- automatic intercept handling
- categorical encoding
- missing-data policy
- sample weights
- grouped / clustered identifiers
- pandas-friendly summaries

Rationale:
Econometrics users often work from tabular data first, not prebuilt dense matrices.

## Suggested Implementation Order

1. Estimator classes for core GLMs
2. Regularization support
3. True separable-objective and mini-batch support
4. Inference utilities for MLE models
5. Productized JAX autodiff bridge
6. Evaluation and model-selection utilities
7. Selective integration points with `synthlearners`
8. Additional causal and panel estimators only where they belong in this repo

## Repo Boundary

Current working assumption:

- `pyensmallen` should focus on optimization primitives, reusable objectives, supervised estimators, autodiff integration, and inference utilities.
- `synthlearners` should remain the home for most panel and synthetic-control estimators, while depending on `pyensmallen` where useful.

