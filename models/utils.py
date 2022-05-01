from math import factorial

import numpy as np
from numpy.linalg import matrix_power as mpower


def make_discrete_matrix(matrix, dt, iterations=4):
    res = np.zeros_like(matrix)
    for i in range(iterations):
        res = res + mpower(matrix * dt, i) / factorial(i)
    return res


def covariance(A, B, coeffs):
    A_mean = np.sum(np.multiply(A, coeffs[:, np.newaxis]), axis=0)
    B_mean = np.sum(np.multiply(B, coeffs[:, np.newaxis]), axis=0)


    cov = np.zeros((A.shape[1], B.shape[1]))
    for a, b, weight in zip(A, B, coeffs):
        cov += weight * (a - A_mean)[:, np.newaxis] @ ((b - B_mean)[:, np.newaxis]).T
        
    return cov