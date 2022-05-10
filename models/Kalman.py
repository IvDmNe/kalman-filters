import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from numpy.linalg import cholesky
from jax import vmap

from models.utils import covariance
class KalmanFilter:
    def __init__(self, 
                model_state_function, 
                obs_function, 
                model_noise_cov, 
                obs_noise_cov, 
                init_state_mean, 
                init_state_cov,
                save_history=True):

        self.model_state_function = model_state_function
        self.obs_function = obs_function
        self.model_noise_cov = model_noise_cov if isinstance(model_noise_cov, np.ndarray) else np.eye(len(init_state_mean)) * model_noise_cov
        self.obs_noise_cov = obs_noise_cov if isinstance(obs_noise_cov, np.ndarray) else np.eye(1) * obs_noise_cov

        self.pred_state = init_state_mean
        self.pred_cov = init_state_cov

        self.model_jacobian = jax.jacfwd(model_state_function, (0, 1))
        self.obs_jacobian = jax.jacfwd(obs_function, 0)

        self.save_history = save_history
        self.history = []
        if self.save_history:
            self.history.append(init_state_mean)
    
    def step(self, cur_input, cur_obs, dt):
        
        # calculate Prediction   
        A, B = self.model_jacobian(self.pred_state, cur_input, dt)
        A = A[:, 0, :, 0]
        B = B[:, 0, :, 0]
        # print(A.shape, B.shape)
        A = jnp.squeeze(A)
        L = np.eye(2)

        # print(cur_input.shape, A.shape, B.shape, self.pred_state.shape)
        # cur_input = np.array([[cur_input]])
        self.pred_state = A @ self.pred_state + B @ cur_input
        self.pred_cov = A @ self.pred_cov @ A.T + L @ self.model_noise_cov @ L.T

        # Calculate coefficient
        H = self.obs_jacobian(self.pred_state)
        H = H[..., 0]
        H = jnp.squeeze(H)[np.newaxis, ...]
        M = np.eye(1)


        K = self.pred_cov @ H.T @ np.linalg.inv(H @ self.pred_cov @ H.T + M @ self.obs_noise_cov @ M.T)
        self.pred_state = self.pred_state + K @ np.array([(cur_obs - self.obs_function(self.pred_state))])
        self.pred_cov = (np.eye(len(self.pred_state)) - K @ H) @ self.pred_cov

        if self.save_history:
            self.history.append(self.pred_state)

        return self.pred_state

    def get_history(self):
        return np.stack(self.history)

class UnscentedKalmanFilter(KalmanFilter):
    
    def __init__(self, model_state_function, obs_function, model_noise_cov, obs_noise_cov, init_state_mean, init_state_cov):
        super().__init__(model_state_function, obs_function, model_noise_cov, obs_noise_cov, init_state_mean, init_state_cov)
        self.vector_model_state_function = vmap(self.model_state_function, in_axes=[0, None, None])
        self.vector_obs_function = vmap(self.obs_function)
        self.obs_noise_cov = obs_noise_cov if isinstance(obs_noise_cov, np.ndarray) else np.eye(1) * obs_noise_cov
    
    
    def step(self, cur_inputs, cur_obs, dt):
        if not isinstance(cur_obs, (np.ndarray, jnp.ndarray)):
            cur_obs = np.array((cur_obs)) 
        L = cholesky(self.pred_cov)

        N = self.pred_state.shape[0]
        k = 2 - N 

        sigma_points = np.empty((N * 2 + 1, self.pred_state.shape[0]))
        sigma_points[0] = self.pred_state[:, 0]

        mult = np.sqrt(N + k)
        # mult = 0.5

        for i in range(1, N + 1):

            addition = mult * L[:, i - 1]
            sigma_points[i] = self.pred_state[:, 0] + addition
            sigma_points[i + N] = self.pred_state[:, 0] - addition

        coeffs = np.ones((2*N + 1), dtype=np.float32) / (2 * (N + k))
        coeffs[0] = float(k) / (N + k)


        out_sigma_points = self.vector_model_state_function(sigma_points[..., np.newaxis], cur_inputs, dt)

        print(self.pred_cov)
        print(np.cov(sigma_points.T, aweights=coeffs, bias=True))

        # print(self.pred_cov)
        self.pred_cov = np.cov(out_sigma_points[..., 0].T, aweights=coeffs, bias=True) + self.model_noise_cov
        print(self.pred_cov)
        ys = self.vector_obs_function(out_sigma_points) + np.random.normal([[0]], self.obs_noise_cov, size=(len(out_sigma_points), 1))

        # Correction
        cov = np.cov(ys.T, aweights=coeffs, bias=True)

        if cov.shape == ():
            cov = cov[np.newaxis, np.newaxis]

        joint_cov = np.cov(out_sigma_points[..., 0].T, ys.T, aweights=coeffs, bias=True)[:-1, [-1]]
        # print(np.cov(out_sigma_points[..., 0].T, ys.T, aweights=coeffs, bias=True))
        # print(joint_cov)
        K = joint_cov @ np.linalg.inv(cov)
        # print(K)

        # print(covariance(out_sigma_points[..., 0], ys, coeffs))

        means = np.sum(np.multiply(ys, coeffs[..., np.newaxis]), axis=0)
        # print(means.shape)
        # print(self.pred_state.shape, K.shape, ((cur_inputs - means)).shape)

        self.pred_state = self.pred_state + K @ (cur_obs - means)[..., np.newaxis]
        # print(cur_obs - means)
        # print((K @ (cur_obs - means)[..., np.newaxis]).shape)
        # print(self.pred_state)
        # print(self.pred_cov)
        # print(joint_cov @ joint_cov.T @ np.linalg.inv(cov).T)
        # print(K)
        # print(cov)
        # print()
        self.pred_cov = self.pred_cov - K @ cov @ K.T
        # self.pred_cov = np.clip(self.pred_cov, 1e-6, 1e6)
        # print(self.pred_cov)
        # print()
        # print(self.pred_cov)

        if self.save_history:
            self.history.append(self.pred_state)

        return self.pred_state

        


