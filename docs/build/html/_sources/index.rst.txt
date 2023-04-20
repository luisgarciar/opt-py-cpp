.. simpleopt documentation master file, created by
   sphinx-quickstart on Sun Apr 16 17:24:17 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of simpleopt!
==========================================

What is simpleopt and where is the code?
----------------------------------------
simpleopt is a set of simple optimization routines written in Python along with a set of test functions.
The module ``simpleopt.opt`` includes implementations of the steepest descent and the conjugate gradient
methods, as well as a class for representing optimization problems.

The separate module ``quad`` allows to construct and call functions of the form

.. math::
          f(x)=x^{T}Ax+b^{T}x

where :math:`A \in \mathbb{R}^{n \times n}` and :math:`b,x \in \mathbb{R}^{n}`. The module ``quad`` is implemented
in C++ using the Eigen library for linear algebra computations, and is wrapped in Python using pybind11.


The code is available at this `GitHub repository <https://github.com/luisgarciar/simpleopt>`_.

Contents
--------

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   simpleopt
   quad

Requirements and Installation
-----------------------------
To build the C++ test functions simpleopt requires a C++ compiler, the CMake and Make build systems, and the Eigen
library for linear algebra computations. For the optimization routines, Python 3.9 or higher is required, as well as
a set of packages listed in the ``requirements.txt`` file in the root directory of the repository. For detailed
installation instructions of these components see the  README file in the `repository <https://github.com/luisgarciar/simpleopt)>`_.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
