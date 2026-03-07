project('pyensmallen',
        ['cpp'],
        version: '0.3.2',
        default_options: ['cpp_std=c++14'])

py = import('python').find_installation(pure: false)
ensmallen_dep = dependency('ensmallen')
armadillo_dep = dependency('Armadillo')
pybind11_include = run_command(
  py,
  ['-c', 'import pybind11; print(pybind11.get_include())'],
  check: true
).stdout().strip()

# Simplify to bare minimum
py.extension_module('_pyensmallen',
    'pyensmallen/module.cpp',
    dependencies: [armadillo_dep, ensmallen_dep],
    cpp_args: ['-I' + pybind11_include],
    install: true,
    install_dir: py.get_install_dir() / 'pyensmallen'
)

# Install Python sources separately
py.install_sources(
    [
        'pyensmallen/__init__.py',
        'pyensmallen/estimators.py',
        'pyensmallen/losses.py',
        'pyensmallen/gmm.py',
        'pyensmallen/utils.hpp',
        'pyensmallen/first_order.hpp',
        'pyensmallen/newton_type.hpp',
        'pyensmallen/constrained.hpp'
    ],
    pure: false,
    subdir: 'pyensmallen'
)
