.. simpleopt documentation master file, created by
   sphinx-quickstart on Sun Apr 16 17:24:17 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to simpleopt's documentation!
=====================================

What is simpleopt and where is the code?
----------------------------------------
simpleopt is a set of simple optimization routines written in Python.
It also contains some test functions implemented in C++ with bindings
using pybind11 to make them available in Python.

The code is available on this `GitHub repository <https://github.com/luisgarciar/simpleopt>`_.


Contents
--------

.. toctree::
   :maxdepth: 3
   :caption: Contents:

Requirements
------------
To build the C++ test functions, simpleopt requires a C++ compiler, the CMake build system, and the Eigen library.

To install the C++ requirements on Ubuntu using the ```apt-get``` package manager, run the following commands in a
terminal window:

```bash
    sudo apt-get install gcc g++ cmake make libeigen3-dev lib
   ```
In macOS you can install the C++ requirements using the package manager
Homebrew, by typing the following commands in a terminal window:

```bash
    brew install gcc cmake make eigen
   ```

The C++ libraries Catch2 (for testing the C++ functions) and pybind11 are fetch during the build process,
so you do not need to install them prior to building the package.


For the optimization routines, Python 3.6 or higher is required. We recommend that
you use a virtual environment. Download the source code from the repository and navigate in your terminal
to the root directory of the repository. The command

```bash
    pip install .
   ```
runs the setup.py script which takes care of compiling the C++ functions and installing the package in your
virtual environment.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
