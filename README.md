# `pyensmallen`: python bindings for the [`ensmallen`](https://ensmallen.org/) library for numerical optimization

Very minimal python bindings for `ensmallen` library. Currently supports 
+ L-BFGS, with intended use for optimisation of smooth objectives for m-estimation
+ ADAM (and variants with different step-size routines) - makes use of ensmallen's templatization.
+ Frank-Wolfe, with intended use for constrained optimization of smooth losses 

See [ensmallen docs](https://ensmallen.org/docs.html) for details.

Installation:

__from source__ 
1. Install `armadillo` and `ensmallen` for your system (build from source, or via conda-forge; I went with the latter)
2. git clone this repository
3. `pip install -e .`
4. Profit? Or at least minimize loss?

__from wheel__
- download the appropriate `.whl` for your system from the more recent release listed in `Releases` and run `pip install ./pyensmallen...`
- copy the download url and run `pip install https://github.com/apoorvalal/pyensmallen/releases/download/<version>/pyensmallen-<version>-<pyversion>-linux_x86_64.whl`

Will likely be uploaded to pypi once we've ironed out a few rough edges. 
