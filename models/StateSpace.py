import numpy as np

from utils import make_discrete_matrix


class Plant:
    def __init__(self, state_f, out_f, init_state, model_noise_f=None, obs_noise_f=None):
        self.state_f = state_f
        self.out_f = out_f
        self.state = init_state
        self.model_noise_f = model_noise_f if model_noise_f else lambda : 0.0
        self.obs_noise_f = obs_noise_f if obs_noise_f else lambda : 0.0


    def step(self, u=0, dt=0):
        # print(self.state)
        self.state = self.state_f(self.state, u, dt) + self.model_noise_f()
        # print(self.state)
        out = self.out_f(self.state) + self.obs_noise_f()
        return out


class LinearDiscreteStateSpaceModel(Plant):
    def __init__(self, A, B, C, D, init_state, dt):

        A_discrete = make_discrete_matrix(A, dt, iterations=10)
        B_discrete = np.linalg.inv(A) @ (A_discrete - np.eye(A.shape[0])) @ B


        self.state_f = lambda x, u, dt: A_discrete @ x + B_discrete * u
        self.out_f = lambda x: C @ x
        super().__init__(self.state_f, self.out_f, init_state)
