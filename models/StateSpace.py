import numpy as np

from models.utils import make_discrete_matrix


class DSSM:
    # Discrete State Space Model
    def __init__(self, state_f, out_f, init_state, model_noise_f=None, obs_noise_f=None, save_history=False):
        self.state_f = state_f
        self.out_f = out_f
        self.state = init_state
        self.model_noise_f = model_noise_f if model_noise_f else lambda : 0.0
        self.obs_noise_f = obs_noise_f if obs_noise_f else lambda : 0.0

        self.save_history = save_history
        self.history = []
        if self.save_history:
            self.history.append(init_state)


    def step(self, u=0, dt=0):
        self.state = self.state_f(self.state, u, dt) + self.model_noise_f()
        out = self.out_f(self.state) + self.obs_noise_f()
        if self.save_history:
            self.history.append(self.state)
        return out

    def get_history(self):
        return np.stack(self.history)


class LDSSM(DSSM):
    # Linear Discrete State Space Model
    def __init__(self, A, B, C, init_state, dt, matrix_exp_iterations=5):

        A_discrete = make_discrete_matrix(A, dt, iterations=matrix_exp_iterations)
        B_discrete = np.linalg.inv(A) @ (A_discrete - np.eye(A.shape[0])) @ B


        self.state_f = lambda x, u, dt: A_discrete @ x + B_discrete * u
        self.out_f = lambda x: C @ x
        super().__init__(self.state_f, self.out_f, init_state)
