from typing import Callable, Union


import numpy as np
import jax.numpy as jnp

from models.common import make_discrete_matrix, vector2matrix, Ndarray


class DiscreteModel:
    # Discrete State Space Model
    def __init__(self,
                 state_f: Callable,
                 observation_f: Callable,
                 init_state: Ndarray,
                 model_noise_f: Callable = lambda: 0.0,
                 obs_noise_f: Callable = lambda: 0.0,
                 save_history: bool = False):

        self.state_f = state_f
        self.observation_f = observation_f

        self.state = vector2matrix(init_state)
        self.model_noise_f = model_noise_f
        self.obs_noise_f = obs_noise_f

        self.save_history = save_history
        self.history = []
        if self.save_history:
            self.history.append(self.state)

    def step(self,
             input_vector: Ndarray = None,
             time_delta: Union[int, float] = 0):

        if input_vector is None:
            input_vector = jnp.zeros_like(self.state)

        input_vector = vector2matrix(input_vector)

        self.state = self.state_f(self.state, input_vector, time_delta) + self.model_noise_f()
        out = self.observation_f(self.state) + self.obs_noise_f()
        if self.save_history:
            self.history.append(self.state)
        return out

    def get_history(self, squeeze: bool = True):
        if not self.save_history:
            raise RuntimeError('History is not saved')

        return np.stack(self.history)[..., 0] if squeeze else np.stack(self.history)


class LinearDiscreteModel(DiscreteModel):
    # Linear Discrete State Space Model
    def __init__(self,
                 A: Ndarray,
                 B: Ndarray,
                 C: Ndarray,
                 init_state: Ndarray,
                 time_delta: Union[int, float],
                 matrix_exp_iterations: int = 5):

        A_discrete = make_discrete_matrix(
            A, time_delta, iterations=matrix_exp_iterations)
        B_discrete = np.linalg.inv(A) @ (A_discrete - np.eye(A.shape[0])) @ B

        self.state_f = lambda x, u, time_delta: A_discrete @ x + B_discrete * u
        self.out_f = lambda x: C @ x
        super().__init__(self.state_f, self.out_f, init_state)
