import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from numpy.linalg import cholesky
from jax import vmap

from models.utils import covariance
class KalmanFilter:
    def __init__(self, model_state_function, obs_function, model_noise_cov, obs_noise_cov, init_state_mean, init_state_cov):
        self.model_state_function = model_state_function
        self.obs_function = obs_function
        self.model_noise_cov = model_noise_cov if isinstance(model_noise_cov, np.ndarray) else np.eye(len(init_state_mean)) * model_noise_cov
        self.obs_noise_cov = obs_noise_cov if isinstance(obs_noise_cov, np.ndarray) else np.eye(1) * obs_noise_cov

        self.pred_state = init_state_mean
        self.pred_cov = init_state_cov

        self.model_jacobian = jax.jacfwd(model_state_function, (0, 1))
        self.obs_jacobian = jax.jacfwd(obs_function, 0)
    
    def step(self, cur_input, cur_obs, dt):
        
        # calculate Prediction   
        A, B = self.model_jacobian(self.pred_state, cur_input, dt)
        A = jnp.squeeze(A)
        L = np.eye(2)

        cur_input = np.array([[cur_input]])
        self.pred_state = A @ self.pred_state + B @ cur_input
        self.pred_cov = A @ self.pred_cov @ A.T + L @ self.model_noise_cov @ L.T

        # Calculate coefficient
        H = self.obs_jacobian(self.pred_state)
        H = jnp.squeeze(H)[np.newaxis, ...]
        M = np.eye(1)

        K = self.pred_cov @ H.T @ np.linalg.inv(H @ self.pred_cov @ H.T + M @ self.obs_noise_cov @ M.T)
        self.pred_state = self.pred_state + K @ np.array([(cur_obs - self.obs_function(self.pred_state))])
        self.pred_cov = (np.eye(len(self.pred_state)) - K @ H) @ self.pred_cov


        return self.pred_state


# class UnscentedKalmanFilter(KalmanFilter):
    
#     def __init__(self, model_state_function, obs_function, model_noise_cov, obs_noise_cov, init_state_mean, init_state_cov):
#         super().__init__(model_state_function, obs_function, model_noise_cov, obs_noise_cov, init_state_mean, init_state_cov)
#         self.vector_obs_function = vmap(self.obs_function)
#         self.vector_model_state_function = vmap(self.model_state_function, in_axes=[0, None, None])
#         self.obs_noise_cov = obs_noise_cov if isinstance(obs_noise_cov, np.ndarray) else np.eye(1) * obs_noise_cov
    
    
#     def step(self, cur_inputs, cur_obs, dt):
#         L = cholesky(self.pred_cov)

#         N = self.pred_state.shape[0]
#         k = 2 - N

#         sigma_points = np.empty((2*N + 1, N, 1))
#         sigma_points[0] = self.pred_state

#         mult = np.sqrt(N + k)

#         for i in range(1, N):
#             sigma_points[i] = self.pred_state + mult * L[:, i][..., np.newaxis]
#             sigma_points[i] = self.pred_state - mult * L[:, i][..., np.newaxis]

#         sigma_points = self.vector_model_state_function(sigma_points, cur_inputs, dt)

#         coeffs = np.ones((2*N + 1, 1), dtype=np.float32) / (2 * (N + k)) 
#         coeffs[0] = float(k) / (N + k)
#         ys = self.vector_obs_function(sigma_points) + np.random.normal([[0]], self.obs_noise_cov, size=(len(sigma_points), 1))

#         means = (ys * coeffs).mean(axis=0)

#         # v1 = np.sum(coeffs)
#         # v2 = np.sum(w * coeffs)
#         # m -= np.sum(m * w, axis=None, keepdims=True) / v1
#         # cov = np.dot(m * w, m.T) * v1 / (v1**2 - v2)


#         # Correction
#         cov = np.cov(ys.T, aweights=coeffs[:, 0]) + self.obs_noise_cov
#         uni_cov = np.cov(sigma_points[..., 0], ys, aweights=coeffs)
#         K = uni_cov @ np.inv(cov)

#         self.pred_state = self.pred_state + K @ (cur_inputs - means)
#         self.pred_cov = self.pred_cov - K @ cov @ K.T

#         return self.pred_state


class UnscentedKalmanFilter(KalmanFilter):
    
    def __init__(self, model_state_function, obs_function, model_noise_cov, obs_noise_cov, init_state_mean, init_state_cov):
        super().__init__(model_state_function, obs_function, model_noise_cov, obs_noise_cov, init_state_mean, init_state_cov)
        self.vector_obs_function = vmap(self.obs_function)
        self.vector_model_state_function = vmap(self.model_state_function, in_axes=[0, None, None])
        self.obs_noise_cov = obs_noise_cov if isinstance(obs_noise_cov, np.ndarray) else np.eye(1) * obs_noise_cov
    
    
    def step(self, cur_inputs, cur_obs, dt):
        L = cholesky(self.pred_cov)

        N = self.pred_state.shape[0]
        k = 2 - N 

        sigma_points = np.empty((N * 2 + 1, self.pred_state.shape[0]))
        sigma_points[0] = self.pred_state[:, 0]

        mult = np.sqrt(N + k)

        # print(sigma_points.shape)

        # print(mult)

        # print(L)
        for i in range(1, N + 1):
            sigma_points[i] = self.pred_state[:, 0] + mult * L[:, i - 1]
            sigma_points[i + N] = self.pred_state[:, 0] - mult * L[:, i - 1]

        sigma_points = self.vector_model_state_function(sigma_points, cur_inputs, dt)

        coeffs = np.ones((2*N + 1), dtype=np.float32) / (2 * (N + k)) 
        coeffs[0] = float(k) / (N + k)
        # print(sigma_points)
        # print(covariance(sigma_points, sigma_points, coeffs))
        self.pred_cov = covariance(sigma_points, sigma_points, coeffs) + self.model_noise_cov

        # print(sigma_points.shape)
        # print(self.vector_obs_function(sigma_points).shape, (np.random.normal([[0]], self.obs_noise_cov, size=(len(sigma_points), 1))).shape)
        ys = self.vector_obs_function(sigma_points) + np.random.normal([[0]], self.obs_noise_cov, size=(len(sigma_points), 1))

        # means = (ys * coeffs).mean(axis=0)

        # v1 = np.sum(coeffs)
        # v2 = np.sum(w * coeffs)
        # m -= np.sum(m * w, axis=None, keepdims=True) / v1
        # cov = np.dot(m * w, m.T) * v1 / (v1**2 - v2)


        # Correction
        # print(ys.shape, means.shape)

        # print(ys, means)
        # print(ys - means)

        # print(((ys - means) @ (ys - means).T).shape)

        cov = covariance(ys, ys, coeffs)

        # uni_cov = np.sum(coeffs * (sigma_points - sigma_points.mean(axis=0)) @ (ys - means).T, axis=0)
        # print(self.pred_state.shape, ((cur_inputs - means)).shape)

        # print(sigma_points.shape, ys.shape)
        joint_cov = covariance(sigma_points, ys, coeffs)
        # print('-'*30)
        K = joint_cov @ np.linalg.inv(cov)

        means = np.sum(np.multiply(ys, coeffs[:, np.newaxis]), axis=0)
        # print(means.shape)
        # print(self.pred_state.shape, K.shape, ((cur_inputs - means)).shape)

        self.pred_state = self.pred_state + K @ (cur_obs - means)
        # print(self.pred_state)
        self.pred_cov = self.pred_cov - K @ cov @ K.T
        # print(self.pred_cov)
        return self.pred_state

        


