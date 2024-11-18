project('pyensmallen',
        ['cpp'],
        version: '0.0.3',
        default_options: ['cpp_std=c++14'])

py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11')
ensmallen_dep = dependency('ensmallen')
armadillo_dep = dependency('Armadillo')

# Simplify to bare minimum
py.extension_module('_pyensmallen',
    'pyensmallen/_pyensmallen.cpp',
    dependencies: [pybind11_dep, armadillo_dep, ensmallen_dep],
    install: true,
    install_dir: py.get_install_dir() / 'pyensmallen'
)

# Install Python sources separately
py.install_sources(
    ['pyensmallen/__init__.py', 'pyensmallen/losses.py'],
    pure: false,
    subdir: 'pyensmallen'
)
