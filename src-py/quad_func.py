from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', 'src-cpp/bind'))
sys.path.append(CODE_DIR)

import py_quad_func as qf


