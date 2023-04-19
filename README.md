## Optimization tools with Python and C++

This repository contains a collection of simple optimization tools
written in Python along with test functions written in C++. 

The package pybind11 is used to bind the C++ functions to Python.

### Requirements and installation

To compile and build the test functions, you will need
a C++ compiler and the CMake and Make build systems. We also use 
the Eigen library for linear algebra operations.

To install the C++ requirements on Ubuntu Linux (and similar Debian-based distributions)
using the ```apt-get``` package manager, run the following commands in a terminal window:

```bash
    sudo apt-get install gcc g++ cmake make libeigen3-dev lib
   ```

In macOS you can install the C++ requirements using the package manager
[Homebrew](https://brew.sh), by typing the following commands in a terminal window:

```bash
    brew install gcc cmake make eigen
   ```

The C++ library pybind11 is fetch during the build process,
so you do not need to install them prior to building the package.

To build the rest of the package, you will need Python 3.6 or later. We recommend using a virtual environment,
such as the one provided by the conda package manager from the [Anaconda]((https://www.anaconda.com/distribution/)) 
distribution. If you are using conda, you can create a virtual environment using the `environment.yml` file provided.
After downloading the source code from this repository, navigate to the root directory of the repository
and run the following command:

```bash
    conda env create -f environment.yml
   ```

Note that the name of the virtual environment is `simpleopt` in the `environment.yml` file.  To activate the virtual
environment, run the following command:

```bash
    conda activate simpleopt
   ```

Alternatively, you can install the Python requirements using pip. In a terminal window, navigate to the root directory
of the repository and run the following command:

```bash
    pip install -r requirements.txt
   ```

Next run the `setup.py` script using pip:

``` bash
    pip install .
```

This will compile the C++ functions, bind them and make them available in Python as the module `quad`, and will also 
add the package `simpleopt` to the Python path. To test that the installation was successful, run the following
command:

```bash
    python -c "import simpleopt; import quad"
   ```
If the installation was successful, you should not see any error messages.
### Usage

