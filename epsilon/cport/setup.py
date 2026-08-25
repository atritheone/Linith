from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "linith_selfplay_cpp",
        ["linith_selfplay_cpp.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="linith_selfplay_cpp",
    version="0.1.0",
    description="C++ Hard-vs-Hard self-play generator for Linith",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
