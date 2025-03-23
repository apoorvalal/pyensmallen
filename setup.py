import os
import sys
import platform

import pybind11
import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

# Define possible ensmallen include locations
def get_ensmallen_include():
    """Get possible include paths for ensmallen library."""
    # Check environment variables first
    if 'ENSMALLEN_INCLUDE' in os.environ:
        return [os.environ['ENSMALLEN_INCLUDE']]
    
    # Common locations on various platforms
    paths = []
    
    # Check micromamba/conda paths if using CI
    if 'MAMBA_ROOT_PREFIX' in os.environ:
        if platform.system() == 'Darwin':
            paths.append(os.path.join(os.environ['MAMBA_ROOT_PREFIX'], 'envs', 'pyensmallen', 'include'))
        else:
            paths.append('/host' + os.path.join(os.environ['MAMBA_ROOT_PREFIX'], 'envs', 'pyensmallen', 'include'))
    
    # Add other common system locations
    if platform.system() == 'Linux':
        paths.extend([
            '/usr/include',
            '/usr/local/include',
            '/opt/homebrew/include',
        ])
    elif platform.system() == 'Darwin':
        paths.extend([
            '/usr/local/include',
            '/opt/homebrew/include',
        ])
    
    # Add CI-specific paths from environment variables
    if 'CMAKE_PREFIX_PATH' in os.environ:
        cmake_prefix = os.environ['CMAKE_PREFIX_PATH']
        paths.append(os.path.join(cmake_prefix, 'include'))


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


    # Return all the possible include paths
    return paths

def get_include_dirs():
    """Get all include directories needed for compilation."""
    include_dirs = [
        get_pybind_include(),
        get_pybind_include(user=True),
    ]
    
    # Add ensmallen include locations
    include_dirs.extend(get_ensmallen_include())
    
    # Print the include paths for debugging in CI
    print("Include paths being searched:")
    for path in include_dirs:
        print(f"  - {path}")
    
    return include_dirs

def get_library_dirs():
    """Get library directories for linking."""
    # Default library directories
    library_dirs = []
    
    # Add platform specific defaults
    if platform.system() == 'Linux':
        library_dirs.append("/usr/lib/x86_64-linux-gnu/")
    
    # Add micromamba lib dirs if available
    if 'MAMBA_ROOT_PREFIX' in os.environ:
        if platform.system() == 'Darwin':
            library_dirs.append(os.path.join(os.environ['MAMBA_ROOT_PREFIX'], 'envs', 'pyensmallen', 'lib'))
        else:
            library_dirs.append('/host' + os.path.join(os.environ['MAMBA_ROOT_PREFIX'], 'envs', 'pyensmallen', 'lib'))
    
    # Add paths from CMAKE_PREFIX_PATH if available
    if 'CMAKE_PREFIX_PATH' in os.environ:
        cmake_prefix = os.environ['CMAKE_PREFIX_PATH']
        library_dirs.append(os.path.join(cmake_prefix, 'lib'))
    
    # Print the library paths for debugging
    print("Library paths being searched:")
    for path in library_dirs:
        print(f"  - {path}")
    
    return library_dirs

ext_modules = [
    Extension(
        "pyensmallen._pyensmallen",
        ["pyensmallen/module.cpp"],
        include_dirs=get_include_dirs(),
        libraries=["armadillo", "ensmallen", "lapack", "blas"],
        library_dirs=get_library_dirs(),
        language="c++",
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
]


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on the specified compiler."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag."""
    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]
    for flag in flags:
        if has_flag(compiler, flag):
            return flag
    raise RuntimeError("Unsupported compiler -- at least C++11 support is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    # Minimal setup using metadata from pyproject.toml
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.4"],  # Needed for build but not in final package
    cmdclass={"build_ext": BuildExt},
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
)
