from tqdm.auto import tqdm
import numpy as np
import jax.numpy as jnp
from jax import jit
from matplotlib import pyplot as plt

from models import DSSM, KalmanFilter, UnscentedKalmanFilter

if __name__ == '__main__':

    state_f = lambda x, u, dt: jnp.array([[1, dt], [0, 1]], dtype=np.float32) @ x \
                                + jnp.array([[0], [dt]], dtype=np.float32) @ u
    # state_f = lambda x, u, dt: jnp.array([[1, 0], [0, 1]], dtype=np.float32) @ x \
    #                             + jnp.array([[0], [1]], dtype=np.float32) @ u
    # out_f = lambda x: jnp.sqrt(jnp.abs(jnp.sin(0.1 * x[0])))
    out_f = lambda x: jnp.sin(x[0])
    # out_f = lambda x: x[0]




    model_noise_mean, model_noise_cov = 0, 0.5
    obs_noise_mean, obs_noise_cov = 0.00, 0.5

    plant = DSSM(state_f, out_f, jnp.array([[0], [0.]]), 
                model_noise_f=lambda : np.random.normal(model_noise_mean, model_noise_cov),
                obs_noise_f=lambda : np.random.normal(obs_noise_mean, obs_noise_cov),
                save_history=True)


    init_state_mean=np.array([[0.], [0.]])
    init_state_cov = np.array([[0.42, 0.], [0., 0.42]])

    ekf = KalmanFilter(state_f,
                    out_f,
                    model_noise_cov, 
                    obs_noise_cov, 
                    init_state_mean,
                    init_state_cov
    )

    ukf = UnscentedKalmanFilter(state_f,
                    out_f,
                    model_noise_cov, 
                    obs_noise_cov, 
                    init_state_mean,
                    init_state_cov
    )



    dt = 0.1
    time_limit = 20
    a = np.linspace(0, time_limit, int(time_limit / dt), endpoint=True)
    # a = np.arange(0, 1000)

    us = -np.sin(0.3*a)
    # us = np.ones_like(a)
    # us = np.zeros_like(a)
    us = us[..., np.newaxis, np.newaxis]
    gt_states = []
    obs_states = []
    outs = []

    for i, u in enumerate(tqdm(us)):
        if i == 0:
            outs.append(0)
            obs_states.append(jnp.array([[0], [0]]))
            gt_states.append(jnp.array([[0], [0.5]]))
            continue

        out = plant.step(u, dt)

        ekf.step(u, out, dt)
        ukf.step(u, out, dt)
        outs.append(out)

    # print(ukf.pred_cov)
    # gt_states = jnp.array(gt_states).squeeze()
    gt_states = plant.get_history().squeeze()
    obs_states_ekf = ekf.get_history().squeeze()
    obs_states_ukf = ukf.get_history().squeeze()

    # obs_states = jnp.array(obs_states).squeeze()

    # a = a[1:]
    # us = us[1:]


    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(17.8, 7.2))
    axs[0].plot(a, np.squeeze(us), color='red')
    axs[0].grid()
    axs[0].title.set_text('Inputs')
    axs[1].plot(a, gt_states[:, 1], color='blue')
    axs[1].grid()
    axs[1].title.set_text('GT speed')
    axs[2].plot(a, gt_states[:, 0], color='blue', label='GT pos')
    axs[2].plot(a, obs_states_ekf[:, 0], color='purple', label='EKF')
    axs[2].plot(a, obs_states_ukf[:, 0], color='purple', label='UKF')
    axs[2].legend()
    axs[2].grid()
    axs[2].title.set_text('GT position')
    axs[3].plot(a, outs, color='green')
    axs[3].grid()
    axs[3].title.set_text('Observations')
    axs[4].plot(a, gt_states[:, 0] - obs_states_ekf[:, 0], label='EKF')
    axs[4].plot(a, gt_states[:, 0] - obs_states_ukf[:, 0], label='UKF')
    axs[4].legend()
    axs[4].grid()
    axs[4].title.set_text('Error')
    plt.show()

    # gt_states[:, 1].min(), gt_states[:, 1].max(), gt_states[:, 1].min() - gt_states[:, 1].max()