import numpy as np
import jax
import jax.numpy as jnp
from jax import grad

class Kalman:
    def __init__(self, model_state_function, obs_function, model_noise_cov, obs_noise_cov, init_state_mean, init_state_cov):
        self.model_state_function = model_state_function
        self.obs_f = obs_function
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
        # B = self.model_inp_jacob(self.pred_state, cur_input, dt)
        L = np.eye(2)

        cur_input = np.array([[cur_input]])
        self.pred_state = A @ self.pred_state + B @ cur_input

        self.pred_cov = A @ self.pred_cov @ A.T + L @ self.model_noise_cov @ L.T

        # Calculate coefficient
        H = self.obs_jacobian(self.pred_state)
        H = jnp.squeeze(H)[np.newaxis, ...]
        M = np.eye(1)

        K = self.pred_cov @ H.T @ np.linalg.inv(H @ self.pred_cov @ H.T + M @ self.obs_noise_cov @ M.T)
        self.pred_state = self.pred_state + K @ np.array([(cur_obs - self.obs_f(self.pred_state))])
        self.pred_cov = (np.eye(len(self.pred_state)) - K @ H) @ self.pred_cov


        return self.pred_state
