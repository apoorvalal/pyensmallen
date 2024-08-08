# `pyensmallen`: python bindings for the [`ensmallen`](https://ensmallen.org/) library for numerical optimization

Very minimal python bindings for `ensmallen` library. Currently supports 
+ L-BFGS, with intended use for optimisation of smooth objectives for m-estimation
+ ADAM (and variants with different step-size routines) - makes use of ensmallen's templatization.

See [ensmallen docs](https://ensmallen.org/docs.html) for details.

Installation:
1. Install `armadillo` and `ensmallen` for your system (build from source, or via conda-forge; I went with the latter)
2. git clone this repository
3. `pip install -e .`
4. Profit? Or at least minimize loss?


