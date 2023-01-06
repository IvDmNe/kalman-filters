from tqdm.auto import tqdm
from models.Kalman import UnscentedKalmanFilter
import numpy as np
import jax.numpy as jnp
from jax import jit

from models import DiscreteModel


if __name__ == '__main__':

    # state_f = lambda x, u, dt: jnp.array([[1, dt], [0, 1]]) @ x + jnp.array([[0], [dt]]) @ u
    def state_f(x, u, dt): return jnp.array(
        [[1, 0], [0, 1]]) @ x + jnp.array([[0], [1]]) @ u
    # state_f = lambda x, u, dt: jnp.array([[1, 0], [0, 1]], dtype=np.float32) @ x \
    #                             + jnp.array([[0], [1]], dtype=np.float32) @ u
    # out_f = lambda x: jnp.sqrt(jnp.abs(x[:1]))
    # out_f = lambda x: jnp.sin(0.1 * x[:1])
    def out_f(x): return x
    # out_f = lambda x: x[:1]
    # out_f = lambda x: jnp.arctan(10 / (2 - x[:1]))
    # out_f = lambda x: jnp.sin(0.1 * x[:1])

    model_noise_mean, model_noise_cov = 0, 0.5
    obs_noise_mean, obs_noise_cov = 0., 0.1

    plant = DiscreteModel(jit(state_f),
                          jit(out_f),
                          jnp.array([[0], [5.]]),
                          model_noise_f=lambda: np.random.normal(
        model_noise_mean, model_noise_cov),
        obs_noise_f=lambda: np.random.normal(obs_noise_mean, obs_noise_cov),
        save_history=True)

    init_state_mean = jnp.array([0., 1])
    # init_state_cov = np.array([[0.42, 0.], [0., 0.42]])
    init_state_cov = jnp.array([[0.01, 0.], [0., 1]])

    # kalman = ExtendedKalmanFilter(state_f,
    kalman = UnscentedKalmanFilter(state_f,
                                   out_f,
                                   model_noise_cov,
                                   obs_noise_cov,
                                   init_state_mean,
                                   init_state_cov
                                   )

    dt = 0.5
    time_limit = 200
    a = np.linspace(0, time_limit, int(time_limit / dt), endpoint=True)

    us = np.cos(0.3*a)[..., np.newaxis]
    # us = np.ones_like(a)
    # us = np.zeros_like(a)
    # us = us[...]
    gt_states = []
    obs_states = []
    outs = []

    # add init states

    # outs.append()

    # for i, u in enumerate((us)):
    for i, u in enumerate(tqdm(us)):

        out = plant.step(u, dt)
        cur_state = kalman.step(u, out, dt)

        print('state_diff:', cur_state - plant.state)
        outs.append(out)

    # print(ukf.pred_cov)
    # gt_states = jnp.array(gt_states).squeeze()
    gt_states = plant.get_history().squeeze()
    obs_states_ekf = kalman.get_history().squeeze()

    outs = np.stack(outs)[..., 0]

    # print([a.shape for a in outs])
    # fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(17.8, 7.2))
    # axs[0].plot(a, np.squeeze(us), color='red')
    # axs[0].title.set_text('u')
    # axs[1].plot(a, gt_states[1:, 1], color='blue')
    # axs[1].title.set_text('dot x')
    # axs[2].plot(a, gt_states[1:, 0], color='blue', label='GT')
    # axs[2].plot(a, obs_states_ekf[1:, 0], color='purple', label='Kalman')
    # axs[2].legend()
    # axs[2].title.set_text('x')
    # axs[3].plot(a, outs, color='green')
    # axs[3].title.set_text('y')
    # axs[4].plot(a, gt_states[1:, 0] - obs_states_ekf[1:, 0], label='Kalman')
    # axs[4].legend()
    # axs[4].title.set_text('Error')
    # plt.show()
