py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11')
ensmallen_dep = dependency('ensmallen')
armadillo_dep = dependency('Armadillo')

py.install_sources(
  [
    '__init__.py',
  ],
  pure: false,
  subdir: 'pyensmallen',
)

py.extension_module('_pyensmallen',
    '_pyensmallen.cpp',
    install: true,
    dependencies : [pybind11_dep, armadillo_dep, ensmallen_dep],
    subdir: 'pyensmallen',
)
