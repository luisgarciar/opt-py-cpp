## Optimization tools with Python and C++

This repository contains a collection of simple optimization tools
written in Python along with test functions written in C++.

The package pybind11 is used to bind the C++ functions to Python.

### Requirements and installation

To compile and build the test functions, you will need
a C++ compiler and the CMake and Make build systems. We also use the Eigen library for
linear algebra operations.

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

The C++ libraries Catch2 and pybind11 are fetch during in the repository,
so you do not need to install them prior to building the package.

To build the rest of the package, you will need Python 3.6 or later.
We recommend using a virtual environment, such as the one provided by the conda package manager
from the [Anaconda]((https://www.anaconda.com/distribution/)) distribution. If you are using conda, you can create a
virtual environment with the following command:

```bash
    conda create -n opt python=3.6
   ```

Note that the name of the virtual environment is `opt` in the above command.
To activate the virtual environment, run the following command:

```bash
    conda activate opt
   ```

Then navigate to the root directory of the repository and run the `setup.py` script using pip:

    ```bash
        pip install .
    ```

This will install the Python requirements, build the C++ functions, bind them and make them available in Python by
adding the module `simpleopt` to the Python path.

### Usage

