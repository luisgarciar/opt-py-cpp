.. simpleopt documentation master file, created by
   sphinx-quickstart on Sun Apr 16 17:24:17 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to simpleopt's documentation!
=====================================

What is simpleopt and where is the code?
----------------------------------------
simpleopt is a set of simple optimization routines written in Python along with a set of test functions.
The module ``simpleopt.opt`` includes implementations of the steepest descent and the conjugate gradient
methods, as well as a class for representing optimization problems.

The separate module ``quad`` allows to construct and call functions of the form
.. math::
f(x) = x^T A x + b^T x

where ..math:: `A` is an, .. math:: `n \times n` real   matrix and .. math:: `b` is an .. math:: `n \times 1` real
vector. The module ``quad`` is implemented in C++ using the Eigen library for linear algebra computations, and is
wrapped in Python using pybind11.

The code is available on this `GitHub repository <https://github.com/luisgarciar/simpleopt>`_.


Contents
--------

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   simpleopt
.. quad

Requirements and Installation
-----------------------------
To build the C++ test functions simpleopt requires a C++ compiler, the CMake and Make build systems, and the Eigen
library for linear algebra computations. For detailed installation instructions of these components see the  README file
in the `repository <https://github.com/luisgarciar/simpleopt)>`.

For the optimization routines, Python 3.6 or higher is required. We recommend that you use a virtual environment. Download the source code from the repository and navigate in your terminal
to the root directory of the repository. The command ``pip install .`` runs the installation script, installs the
python requirements, and compiles the C++ functions and makes the package available in your virtual environment.

To test the installation, run the following command in your terminal:

.. code-block:: bash

   pytest tests

If the installation was successful, all tests should pass (but warnings are ok).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
