from math import factorial
from typing import Union
import numpy as np
from numpy.linalg import matrix_power as mpower
import jax.numpy as jnp

ndarray = Union[jnp.ndarray, np.ndarray]


def make_discrete_matrix(matrix, dt, iterations=4):
    res = np.zeros_like(matrix)
    for i in range(iterations):
        res = res + mpower(matrix * dt, i) / factorial(i)
    return res


def covariance(A, B=None, coeffs=None):

    if coeffs is None:
        coeffs = np.ones(len(A), dtype=np.float32) / len(A)
    if B is None:
        B = A.copy()

    A = A[..., np.newaxis] if len(A.shape) == 1 else A
    B = B[..., np.newaxis] if len(B.shape) == 1 else B
    A_mean = np.sum(np.multiply(A, coeffs[:, np.newaxis]), axis=0)
    B_mean = np.sum(np.multiply(B, coeffs[:, np.newaxis]), axis=0)

    cov = np.zeros((A.shape[1], B.shape[1]))
    for a, b, weight in zip(A, B, coeffs):
        cov += weight * \
            (a - A_mean)[:, np.newaxis] @ ((b - B_mean)[:, np.newaxis]).T

    return cov


def vector2matrix(x: np.ndarray):
    return x if len(x.shape) > 1 else x[:, np.newaxis]
