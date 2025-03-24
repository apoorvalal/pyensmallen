project('pyensmallen',
        ['cpp'],
        version: '0.2.5',
        default_options: ['cpp_std=c++14'])

py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11')
ensmallen_dep = dependency('ensmallen')
armadillo_dep = dependency('Armadillo')

# Simplify to bare minimum
py.extension_module('_pyensmallen',
    'pyensmallen/module.cpp',
    dependencies: [pybind11_dep, armadillo_dep, ensmallen_dep],
    install: true,
    install_dir: py.get_install_dir() / 'pyensmallen'
)

# Install Python sources separately
py.install_sources(
    [
        'pyensmallen/__init__.py',
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