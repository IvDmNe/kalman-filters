import jax.numpy as jnp
from jax import jit
import numpy as np

from models import DiscreteModel, ExtendedKalmanFilter, UnscentedKalmanFilter


def state_f(x, u, dt):
    return jnp.array([[1, dt], [0, 1]]) @ x + jnp.array([[0], [dt]]) @ u


def out_f(x):
    return 100 * jnp.arctan(1 / (0.0001 * x[:1]))


class TestFilters:

    def test_extended_kalman_filter_sanity(self):

        model_noise_mean, model_noise_cov = 0, 0.5
        obs_noise_mean, obs_noise_cov = 0., 2.0

        plant = DiscreteModel(jit(state_f),
                              jit(out_f),
                              jnp.array([[0], [5.]]),
                              model_noise_f=lambda: np.random.normal(
                                  model_noise_mean, 0),
                              obs_noise_f=lambda: np.random.normal(
            obs_noise_mean, obs_noise_cov),
            save_history=True)

        init_state_mean = jnp.array([0., 1])
        init_state_cov = jnp.array([[0.5, 0.], [0., 1]])

        ekf = ExtendedKalmanFilter(state_f,
                                   out_f,
                                   model_noise_cov,
                                   obs_noise_cov,
                                   init_state_mean,
                                   init_state_cov
                                   )

        dt = 2
        time_limit = 50
        a = np.linspace(0, time_limit, int(time_limit / dt), endpoint=True)

        us = np.cos(0.03*a)[..., np.newaxis]

        for u in us:
            out = plant.step(u, dt)
            ekf.step(u, out, dt)

    def test_unscented_kalman_filter_sanity(self):

        model_noise_mean, model_noise_cov = 0, 0.5
        obs_noise_mean, obs_noise_cov = 0., 2.0

        plant = DiscreteModel(jit(state_f),
                              jit(out_f),
                              jnp.array([[0], [5.]]),
                              model_noise_f=lambda: np.random.normal(
                                  model_noise_mean, 0),
                              obs_noise_f=lambda: np.random.normal(
            obs_noise_mean, obs_noise_cov),
            save_history=True)

        init_state_mean = jnp.array([0., 1])
        init_state_cov = jnp.array([[0.5, 0.], [0., 1]])

        filter = UnscentedKalmanFilter(state_f,
                                       out_f,
                                       model_noise_cov,
                                       obs_noise_cov,
                                       init_state_mean,
                                       init_state_cov
                                       )

        dt = 2
        time_limit = 50
        a = np.linspace(0, time_limit, int(time_limit / dt), endpoint=True)

        us = np.cos(0.03*a)[..., np.newaxis]

        for u in us:
            out = plant.step(u, dt)
            filter.step(u, out, dt)
