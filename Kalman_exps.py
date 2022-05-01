from tqdm.auto import tqdm
import numpy as np
import jax.numpy as jnp
from jax import jit
from matplotlib import pyplot as plt

from models import DSSM, KalmanFilter, UnscentedKalmanFilter

if __name__ == '__main__':

    state_f = lambda x, u, dt: jnp.array([[1, dt], [0, 1]], dtype=np.float32) @ x + jnp.array([[0], [dt]], dtype=np.float32) * u

    out_f = lambda x: jnp.sqrt(jnp.abs(jnp.sin(0.1 * x[0])))



    model_noise_mean, model_noise_cov = 0, 0.1
    obs_noise_mean, obs_noise_cov = 0.01, 0.4

    plant = DSSM(state_f, out_f, jnp.array([[0], [5]]), 
                model_noise_f=lambda : np.random.normal(model_noise_mean, model_noise_cov),
                obs_noise_f=lambda : np.random.normal(obs_noise_mean, obs_noise_cov))


    init_state_mean=np.array([[0.], [5.]])
    init_state_cov = np.array([[0.01, 0.], [0., 1.]])
    observer = UnscentedKalmanFilter(state_f,
    # observer = KalmanFilter(state_f,
                    out_f,
                    model_noise_cov, 
                    obs_noise_cov, 
                    init_state_mean,
                    init_state_cov
    )


    dt = 0.05
    time_limit = 20
    a = np.linspace(0, time_limit, int(time_limit / dt), endpoint=True)
    # a = np.arange(0, 1000)

    us = -np.sin(0.01*a)
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

        state_obs = observer.step(u, out, dt)
        outs.append(out)
        gt_states.append(plant.state)
        obs_states.append(state_obs)


    gt_states = jnp.array(gt_states)
    obs_states = jnp.array(obs_states)
    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(17.8, 7.2))
    axs[0].plot(a, us, color='red')
    axs[0].grid()
    axs[0].title.set_text('Inputs')
    axs[1].plot(a, gt_states[:, 1], color='blue')
    axs[1].grid()
    axs[1].title.set_text('GT speed')
    axs[2].plot(a, gt_states[:, 0], color='blue')
    axs[2].grid()
    axs[2].title.set_text('GT position')
    axs[3].plot(a, outs, color='green')
    axs[3].grid()
    axs[3].title.set_text('Observations')
    axs[4].plot(a, obs_states[:, 0], color='purple')
    axs[4].grid()
    axs[4].title.set_text('Predicted position')
    axs[5].plot(a, gt_states[:, 0] - obs_states[:, 0])
    axs[5].grid()
    axs[5].title.set_text('Error')
    plt.show()

    gt_states[:, 1].min(), gt_states[:, 1].max(), gt_states[:, 1].min() - gt_states[:, 1].max()