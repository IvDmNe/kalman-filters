from math import factorial

import numpy as np
from numpy.linalg import matrix_power as mpower


def make_discrete_matrix(matrix, dt, iterations=4):
    res = np.zeros_like(matrix)
    for i in range(iterations):
        res = res + mpower(matrix * dt, i) / factorial(i)
    return res