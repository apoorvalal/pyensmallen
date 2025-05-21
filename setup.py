import sys

import pybind11
import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

__version__ = "0.2.7"


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


ext_modules = [
    Extension(
        "pyensmallen._pyensmallen",
        ["pyensmallen/module.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        libraries=["armadillo", "lapack", "blas"],  # Add LAPACK and BLAS
        library_dirs=["/usr/lib/x86_64-linux-gnu/"],  # Add LAPACK and BLAS
        language="c++",
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
    name="pyensmallen",
    version=__version__,
    author="Apoorva Lal",
    author_email="lal.apoorva@gmail.com",
    description="Python bindings for the ensmallen optimization library",
    long_description="",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.4"],
    setup_requires=["pybind11>=2.4"],
    cmdclass={"build_ext": BuildExt},
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.10",
    distclass=BinaryDistribution,
)
