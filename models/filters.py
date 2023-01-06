from typing import Callable, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax import jit
from jax import vmap

from scipy.linalg import cholesky

from models.common import vector2matrix, ndarray


@jit
def calc_K(pred_cov: ndarray, H: ndarray, obs_noise_cov: ndarray):
    return pred_cov @ H.T @ inv(H @ pred_cov @ H.T + obs_noise_cov)


@jit
def calc_obs_noise(A, pred_cov, model_noise_cov):
    return A @ pred_cov @ A.T + model_noise_cov


class ExtendedKalmanFilter:
    def __init__(self,
                 model_state_function: Callable,
                 obs_function: Callable,
                 model_noise_cov: ndarray,
                 obs_noise_cov: ndarray,
                 init_state_mean: ndarray,
                 init_state_cov: ndarray,
                 save_history=True,
                 jit_enable=True) -> None:

        self.model_state_function = model_state_function
        self.obs_function = obs_function

        self.model_noise_cov = model_noise_cov
        self.obs_noise_cov = obs_noise_cov

        self.pred_state = vector2matrix(init_state_mean)
        self.pred_cov = init_state_cov

        # self.model_jacobian = jax.jacrev(model_state_function, (0, 1))
        self.model_jacobian = jax.jacrev(model_state_function, (0))
        self.obs_jacobian = jax.jacrev(obs_function)

        if jit_enable:
            self.model_state_function = jit(self.model_state_function)
            self.obs_function = jit(self.obs_function)
            self.model_jacobian = jit(self.model_jacobian)
            self.obs_jacobian = jit(self.obs_jacobian)

        self.save_history = save_history
        self.history = []
        if self.save_history:
            self.history.append(self.pred_state)

    def step(self,
             cur_input: Union[np.ndarray, jnp.ndarray],
             cur_obs: Union[np.ndarray, jnp.ndarray],
             dt: Union[int, float]) -> Union[jnp.ndarray, np.ndarray]:

        cur_input = vector2matrix(cur_input)

        # calculate prediction
        A = self.model_jacobian(
            self.pred_state[..., 0], cur_input[..., 0], dt)

        self.pred_state = self.model_state_function(
            self.pred_state, cur_input, dt)
        self.pred_cov = calc_obs_noise(
            A, self.pred_cov, self.model_noise_cov)

        # Calculate coefficient
        H = self.obs_jacobian(self.pred_state[..., 0])
        K = calc_K(self.pred_cov, H, self.obs_noise_cov)

        # Calculate correction for state and Covariance
        self.pred_state = self.pred_state + \
            K @ (cur_obs - self.obs_function(self.pred_state))

        self.pred_cov = (np.eye(len(self.pred_state)) - K @ H) @ self.pred_cov

        if self.save_history:
            self.history.append(self.pred_state)

        return self.pred_state

    def get_history(self) -> np.ndarray:
        return np.stack(self.history)


@jit
def joint_cov(x: ndarray, y: ndarray, weights):
    x_mean = jnp.average(x, axis=0, weights=weights)
    y_mean = jnp.average(y, axis=0, weights=weights)
    cov = jnp.stack([w * (x_obs - x_mean) @ (y_obs -
                    y_mean).T for x_obs, y_obs, w in zip(x, y, weights)])
    return jnp.sum(cov, axis=0)


class UnscentedKalmanFilter:
    def __init__(self,
                 model_state_function: Callable,
                 obs_function: Callable,
                 model_noise_cov: Union[np.ndarray, jnp.ndarray, int, float],
                 obs_noise_cov: Union[np.ndarray, jnp.ndarray, int, float],
                 init_state_mean: Union[np.ndarray, jnp.ndarray],
                 init_state_cov: Union[np.ndarray, jnp.ndarray, int, float],
                 save_history=True,
                 jit_enable=True) -> None:

        self.model_state_function = model_state_function
        self.obs_function = obs_function

        if isinstance(model_noise_cov, np.ndarray):
            self.model_noise_cov = model_noise_cov
        else:
            self.model_noise_cov = np.eye(
                len(init_state_mean)) * model_noise_cov

        if isinstance(obs_noise_cov, np.ndarray):
            self.obs_noise_cov = obs_noise_cov
        else:
            self.obs_noise_cov = np.eye(1) * obs_noise_cov

        self.pred_state = vector2matrix(init_state_mean)
        self.pred_cov = init_state_cov

        self.vector_model_state_function = jit(
            vmap(self.model_state_function, in_axes=[0, None, None]))
        self.vector_obs_function = jit(vmap(self.obs_function))

        if jit_enable:
            self.model_state_function = jit(self.model_state_function)
            self.obs_function = jit(self.obs_function)

        self.save_history = save_history
        self.history = []
        if self.save_history:
            self.history.append(self.pred_state)

        N = self.pred_state.shape[0]
        kappa = 3 - N

        self.sigma_sqrt = jnp.sqrt(N + kappa)
        self.sigma_weights = np.ones(2*N + 1) * (1 / 2 * 1 / (N + kappa))
        self.sigma_weights[0] = kappa / (N + kappa)
        self.sigma_weights = jnp.array(self.sigma_weights)

    def step(self,
             cur_input: ndarray,
             cur_obs: ndarray,
             dt: Union[int, float]) -> ndarray:

        cur_input = vector2matrix(cur_input)

        L = cholesky(self.pred_cov, check_finite=True)
        sigma_points = [self.pred_state]

        for i in range(self.pred_state.shape[0]):
            sigma_points.append(self.pred_state + self.sigma_sqrt * L[:, [i]])
            sigma_points.append(self.pred_state - self.sigma_sqrt * L[:, [i]])
        sigma_points = jnp.stack(sigma_points)

        responses = self.vector_model_state_function(
            sigma_points, cur_input, dt)
        responses_avg = jnp.average(
            responses, axis=0, weights=self.sigma_weights)
        responses_cov = joint_cov(
            responses, responses, weights=self.sigma_weights) + self.model_noise_cov

        outputs = self.vector_obs_function(responses)
        outputs_avg = jnp.average(outputs, axis=0, weights=self.sigma_weights)
        outputs_cov = joint_cov(
            outputs, outputs, weights=self.sigma_weights) + self.obs_noise_cov

        cov_xy = joint_cov(responses, outputs, self.sigma_weights)

        K = cov_xy @ inv(outputs_cov)
        self.pred_state = responses_avg + K @ (cur_obs - outputs_avg)
        self.pred_cov = responses_cov - K @ outputs_cov @ K.T

        if self.save_history:
            self.history.append(self.pred_state)

        return self.pred_state

    def get_history(self) -> ndarray:
        return np.stack(self.history)
