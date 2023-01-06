from math import factorial
from typing import Union
import numpy as np
from numpy.linalg import matrix_power as mpower
import jax.numpy as jnp

Ndarray = Union[jnp.ndarray, np.ndarray]


def make_discrete_matrix(matrix: Ndarray, time_delta: Union[float, int], iterations=4):
    res = np.zeros_like(matrix)
    for i in range(iterations):
        res = res + mpower(matrix * time_delta, i) / factorial(i)
    return res


def vector2matrix(src_vector: np.ndarray):
    return src_vector if len(src_vector.shape) > 1 else src_vector[:, np.newaxis]
