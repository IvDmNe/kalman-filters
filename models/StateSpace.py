from typing import Callable, Union


import numpy as np
import jax.numpy as jnp

from models.utils import make_discrete_matrix, vector2matrix

ndarray = Union[jnp.ndarray, np.ndarray]


class DiscreteModel:
    # Discrete State Space Model
    def __init__(self,
                 state_f: Callable,
                 out_f: Callable,
                 init_state: ndarray,
                 model_noise_f: Callable = lambda: 0.0,
                 obs_noise_f: Callable = lambda: 0.0,
                 save_history: bool = False):

        self.state_f = state_f
        self.out_f = out_f

        self.state = init_state
        self.model_noise_f = model_noise_f
        self.obs_noise_f = obs_noise_f

        self.save_history = save_history
        self.history = []
        if self.save_history:
            self.history.append(init_state)

    def step(self,
             u: ndarray = None,
             dt: Union[int, float] = 0):

        if u is None:
            u = jnp.zeros_like(self.init_state)

        u = vector2matrix(u)

        self.state = self.state_f(self.state, u, dt) + self.model_noise_f()
        out = self.out_f(self.state) + self.obs_noise_f()
        if self.save_history:
            self.history.append(self.state)
        return out

    def get_history(self):
        return np.stack(self.history)


class LinearDiscreteModel(DiscreteModel):
    # Linear Discrete State Space Model
    def __init__(self,
                 A: ndarray,
                 B: ndarray,
                 C: ndarray,
                 init_state: ndarray,
                 dt: Union[int, float],
                 matrix_exp_iterations: int = 5):

        A_discrete = make_discrete_matrix(
            A, dt, iterations=matrix_exp_iterations)
        B_discrete = np.linalg.inv(A) @ (A_discrete - np.eye(A.shape[0])) @ B

        self.state_f = lambda x, u, dt: A_discrete @ x + B_discrete * u
        self.out_f = lambda x: C @ x
        super().__init__(self.state_f, self.out_f, init_state)
