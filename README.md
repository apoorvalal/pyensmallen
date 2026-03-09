# `pyensmallen`: python bindings for the [`ensmallen`](https://ensmallen.org/) library for numerical optimization


Lightweight python bindings for `ensmallen` library. Currently supports
+ L-BFGS, with intended use for optimisation of smooth objectives for m-estimation
+ ADAM (and variants with different step-size routines) - makes use of ensmallen's templatization.
+ Frank-Wolfe, with intended use for constrained optimization of smooth losses
  - constraints are either lp-ball (lasso, ridge, elastic-net) or simplex
+ (Generalized) Method of Moments estimation with ensmallen optimizers.
  - this uses ensmallen for optimization [and relies on `jax` for automatic differentiation to get gradients and jacobians]. This is the main use case for `pyensmallen` and is the reason for the bindings.
+ Estimator classes for linear, logistic, and Poisson regression with classical and robust inference for unregularized fits

See [ensmallen docs](https://ensmallen.org/docs.html) for details. The `notebooks/` directory walks through several statistical examples.

## speed
`pyensmallen` is very fast. A comprehensive set of benchmarks is available in the `benchmarks` directory. The benchmarks are run on an intel 12th gen framework laptop. Benchmarks vary data size (sample size and number of covariates) and parametric family (linear, logistic, poisson) and compare `pyensmallen` with `scipy` and `statsmodels` (I initially also tried to keep `cvxpy` in the comparison set but it was far too slow to be in the running). At large data sizes, pyensmallen is roughly an order of magnitude faster than scipy, which in turn is an order of magnitude faster than statsmodels. So, a single statsmodels run takes around as long as a pyensmallen run that naively uses the nonparametric bootstrap for inference. This makes the bootstrap a viable option for inference in large data settings.

![](paper/benchmark_time.png)

## Installation:

Make sure your system has `blas` installed. On macos, this can be done via brew. Linux systems should have it installed by default. If you are using conda, you can install `blas` via conda-forge.

Then,

__from pypi__

```
uv pip install pyensmallen
```

__from source__
1. Install `armadillo` and `ensmallen` for your system (build from source, or via conda-forge; I went with the latter)
2. git clone this repository
3. If you are using `uv`:
   - `uv pip install --python .venv/bin/python meson meson-python ninja pybind11`
   - `uv pip install --python .venv/bin/python --no-build-isolation -e .`
4. If you are using vanilla `pip` in an activated environment:
   - `python -m pip install meson meson-python ninja pybind11`
   - `python -m pip install --no-build-isolation -e .`
5. Profit? Or at least minimize loss?

__full development environment__

To install everything required to run tests and notebooks:

```bash
uv pip install --python .venv/bin/python meson meson-python ninja pybind11
uv pip install --python .venv/bin/python --no-build-isolation -e ".[full]"
```

Vanilla `pip` equivalent:

```bash
python -m pip install meson meson-python ninja pybind11
python -m pip install --no-build-isolation -e ".[full]"
```

The `full` extra includes the Python dependencies used by:

- the test suite
- GMM and autodiff examples
- benchmark notebooks
- plotting and notebook tooling

__documentation__

### doc-generation

The repository includes a Quarto documentation site in `docs/`. The docs are built from three sources:

- hand-written Quarto pages in `docs/*.qmd`
- generated API reference pages in `docs/reference/*.qmd`, built from Python and pybind11 docstrings with `quartodoc`
- executed notebook pages in `docs/notebooks/*.ipynb`

Use the render script instead of calling `quarto render` directly:

```bash
scripts/render_docs.sh
```

The script does the following:

- uses the repository `.venv` as the Quarto Python runtime
- forces JAX onto CPU so notebook execution is stable during docs builds
- copies the tracked notebooks from `notebooks/` into `docs/notebooks/`
- runs `quartodoc` to regenerate the API reference pages from docstrings
- runs `quarto render docs` to execute the notebooks and build the site

If you need the full docs toolchain first:

```bash
uv pip install --python .venv/bin/python meson meson-python ninja pybind11
uv pip install --python .venv/bin/python --no-build-isolation -e ".[full]"
```

The rendered site lands in `docs/_site/`. The generated API source pages land in `docs/reference/`.

__from wheel__
- download the appropriate `.whl` for your system from the more recent release listed in `Releases` and run `uv pip install ./pyensmallen...` OR
- copy the download url and run `uv pip install https://github.com/apoorvalal/pyensmallen/releases/download/<version>/pyensmallen-<version>-<pyversion>-linux_x86_64.whl`

## Documentation

GitHub Pages is served from the `gh-pages` branch. This repository does not
auto-publish the website when `master` changes, so updating the site is a
separate step.

After rendering the documentation site into `docs/_site/`, publish it with:

```bash
scripts/publish_docs.sh
```

The script:

- creates a temporary worktree for `gh-pages`
- syncs `docs/_site/` into that worktree
- commits the site update if anything changed
- pushes `gh-pages` to `origin`
