import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import jax.numpy as jnp

from models import DiscreteModel, LinearDiscreteModel


def simple_state_f(x, u, dt=0):
    return x + u


def simple_obs_f(x):
    return 0.5 * x


class TestClass:

    def test_init(self):
        model = DiscreteModel(
            simple_state_f,
            simple_obs_f,
            np.array([[1], [2.]]),
        )
        model.step(np.array([0, 1.]))

    def test_jax(self):
        model = DiscreteModel(
            simple_state_f,
            simple_obs_f,
            jnp.array([[1], [2.]]),
        )
        model.step(jnp.array([0, 1.]))

    def test_dimensions(self):
        for i in range(1, 6):
            model = DiscreteModel(
                simple_state_f,
                simple_obs_f,
                np.random.rand(i),
                save_history=True
            )
            model.step(jnp.array([7]))
            history = model.get_history()
            assert history.shape[1] == i, f'Incorrect calculations for dimension {i}'

    def test_linear_model(self):
        freq = 1
        Magnitude = 2
        dt = 0.01

        A = np.array([[0, 1],
                      [-freq**2, 0]])

        B = np.zeros((2, 1))
        C = np.array([[1, 0]])

        init_state = np.array([[0], [Magnitude * freq]])

        model = LinearDiscreteModel(
            A, B, C, init_state, dt, matrix_exp_iterations=20)

        a = np.arange(0, 1, step=dt)
        ys = [0]

        for _ in (a[1:]):
            y = model.step(time_delta=dt)
            ys.append(y)
