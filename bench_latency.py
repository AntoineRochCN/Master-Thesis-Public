from SAC_delayed import SAC_delayed
from SAC_VC import SAC_VC
from DSAC import DSAC
from DADAC import DADAC
from utils_env import *
from utils_SAC import *
import multiprocessing as mp
import numpy as np
from itertools import product
import json

def run_sim(caso, seed, dist, latency_action, latency_obs):
    if latency_obs == 0 and latency_action == 0:
        ret = {"max_mean_10": 0, "algo": caso, "seed": seed, "dist": dist, "latency_action": latency_action, "latency_obs": latency_obs}
        return ret
    env = gym.make("Walker2d-v4")
    test_env_ = gym.make("Walker2d-v4")
    dist_action = dist
    kwargs_action = {"delta_max": latency_action}
    dist_obs = dist
    kwargs_obs = {"delta_max": latency_obs}
    delayed_env = LatencyEnv(env, dist_action, kwargs_action, dist_obs, kwargs_obs)
    delayed_test_env = LatencyEnv(test_env_, dist_action, kwargs_action, dist_obs, kwargs_obs)

    n_timesteps = 10**4+1000
    simulation_list = ["SAC", "SAC_VC", "DSAC", "DADAC"]
    simulation_type = simulation_list[caso]
    act_shape = env.action_space.shape[0]
    noise = Buffered_NormalActionNoise(np.zeros(act_shape), np.ones(act_shape)*0.1**2)
    train_freq = 1

    callback = CustomEvalCallback(delayed_test_env, n_eval_episodes=10, eval_freq=2500, verbose=0, learning_timesteps=n_timesteps)

    save_dir = "./latency_influence/{}/".format(dist_action) + simulation_list[caso]

    match caso:
        
        case 0:
            model = SAC_delayed("MlpPolicy", delayed_env, tensorboard_log=save_dir, replay_buffer_class=ReplayBuffer_DelayedSAC,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), learning_rate=1e-4, policy_delay=2, learning_starts=10000 // train_freq, buffer_size=1000000, tau = 0.005,
                train_freq=train_freq, seed=seed, gradient_steps=1, action_noise=noise, alpha_0=np.e)
        case 1:
            model = SAC_VC("MlpPolicy", delayed_env, delayed_env.merged_dist, tensorboard_log=save_dir, replay_buffer_class=CustomReplayBuffer,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), learning_rate=1e-4, policy_delay=2, learning_starts=10000 // train_freq, buffer_size=1000000, tau = 0.005,
                train_freq=train_freq, seed=seed, gradient_steps=1, action_noise=noise, alpha_0=np.e)
        case 2:
            train_freq = 20
            model = DSAC("CustomMlpPolicy", delayed_env, tensorboard_log=save_dir, replay_buffer_class=ReplayBuffer_DelayedSAC,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), learning_rate=1e-4, policy_delay=2, learning_starts=10000 // train_freq, buffer_size=1000000, tau = 0.005,
                train_freq=train_freq, seed=seed, gradient_steps=1, action_noise=noise, alpha_0=np.e)
        case 3:
            model = DADAC("CustomMlpPolicy", delayed_env, delayed_env.merged_dist, tensorboard_log=save_dir, replay_buffer_class=CustomReplayBuffer,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), learning_rate=1e-4, policy_delay=2, learning_starts=10000 // train_freq, buffer_size=1000000, tau = 0.005,
                train_freq=train_freq, seed=seed, gradient_steps=1, action_noise=noise, alpha_0=np.e)
    model.learn(n_timesteps, progress_bar=False,tb_log_name=simulation_type, callback=callback)
    ret = {"max_mean_10": 0, "algo": caso, "seed": seed, "dist": dist, "latency_action": latency_action, "latency_obs": latency_obs}
    return ret


if __name__ == '__main__':
    
    seed_list = [12345,22345,32345,42355,52345]
    dist_action = ["constant"]
    caso = [3]
    latency_action = [1,2,3,4,5]
    latency_obs = [1,2,3,4,5]

    args = list(product(caso, seed_list, dist_action, latency_action, latency_obs))
    n_thread = 10

    with mp.Pool(n_thread) as p:
        result = p.starmap(run_sim, args)

    with open("./latency_influence/results.json", 'w') as file:
        json.dump(result, file)
