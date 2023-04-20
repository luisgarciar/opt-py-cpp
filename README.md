## Optimization tools with Python and C++

This repository contains a collection of simple optimization tools
written in Python along with test functions written in C++. The test functions
are written in C++ and bound to Python using the pybind11 library. The test functions
are of the form
    $$f(x) = x^{T} A x + b^{T} x$$
where $A\in \mathbb{R}^{n \times n}$ and $b\in \mathbb{R}^{n}$ are given parameters.  


### Requirements and installation

To compile and build the test functions, you will need
a C++ compiler and the CMake and Make build systems. We also use 
the Eigen library for linear algebra operations.

To install the C++ requirements on Ubuntu Linux (and similar Debian-based distributions)
using the ```apt-get``` package manager, run the following commands in a terminal window:

```bash
    sudo apt-get install gcc g++ cmake make libeigen3-dev
   ```

In macOS, you can install the C++ requirements using the package manager
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

``` bash
    pip install -r requirements.txt
   ```

Next, run the `setup.py` script using pip:

```bash
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

To create a quadratic function, you can use the `Function` class in the `quad` module.
Note that the numpy arrays passed to the constructor and to evaluate the function must be of `float64` type.
An example is shown below.

```python
    import quad
    import numpy as np

    A = np.array([[1, 0], [0, 1]], dtype=np.float64)
    b = np.array([1, 1], dtype=np.float64)
    f = quad.Function(A, b)
    x = np.array([1, 1], dtype=np.float64)
    
    print(f.eval(x))
    print(f.grad(x))
   ```

To use the optimization tools, you can use the `Problem` class in the `simpleopt` module. First, you need to create
an instance of the `Problem` class. To call the solver, the `solve` should be used. An example is shown below.
For more information, see the documentation.

```python
    import simpleopt
    import numpy as np
    import quad

    A = np.array([[6.0, 2.0], [2.0, 6.0]], dtype=np.float64)
    b = np.array([1, 1], dtype=np.float64)
    f = quad.Function(A, b)
    x0 = np.array([1, 1], dtype=np.float64)
    max_iter = 100
    problem = simpleopt.opt.Problem(f.eval, f.grad, dim=2, prob_type="min", method='sd')
    sol, info = problem.solve(x0, max_iter=max_iter)
   ```

## Demo
A simple example of how to use the package is shown in the file `demo.py`. To run the example, go to the root directory
of the repository and run the following command:

```bash
    python demo.py
   ```

### Tests
A test suite is provided in the `tests` directory. To run the tests, go to the root directory of the repository and run
the following command:

```bash
    python -m pytest tests
   ```

### Documentation
The documentation is provided in the `docs` directory. To build the documentation, navigate to the
``docs`` directory of the repository and run the following command:

```bash
    make clean html
   ```

Alternatively, the documentation can be found [online](https://luisgarciar-simpleopt.readthedocs.io/en/latest/simpleopt.html). 

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.