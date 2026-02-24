from SAC_delayed import SAC_delayed
from SAC_VC import SAC_VC
from DSAC import DSAC
from DADAC import DADAC
from utils_env import *
from utils_SAC import *
import time
import multiprocessing as mp
import numpy as np
from itertools import product
import json

def run_sim(caso, seed, dist_obs, env_name, dist_obs_kwargs, dist_act, dist_act_kwargs, n_timesteps, case_number):
    env = gym.make(env_name)
    test_env_ = gym.make(env_name)
    time.sleep(2*int(seed // 10000))
    
    delayed_env = LatencyEnv(env, dist_act, dist_act_kwargs, dist_obs, dist_obs_kwargs)
    delayed_test_env = LatencyEnv(test_env_, dist_act, dist_act_kwargs, dist_obs, dist_obs_kwargs)

    simulation_list = ["SAC", "SAC_VC", "DSAC", "DADAC"]
    simulation_type = simulation_list[caso]
    act_shape = env.action_space.shape[0]
    noise = Buffered_NormalActionNoise(np.zeros(act_shape), np.ones(act_shape)*0.1**2)
    train_freq = 20

    callback = CustomEvalCallback(delayed_test_env, n_eval_episodes=10, eval_freq=2500, verbose=0, learning_timesteps=n_timesteps)

    save_dir = "./parallel_runs/sim_number_{}/".format(case_number)

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
            model = DSAC("CustomMlpPolicy", delayed_env, tensorboard_log=save_dir, replay_buffer_class=ReplayBuffer_DelayedSAC,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), learning_rate=1e-4, policy_delay=2, learning_starts=10000 // train_freq, buffer_size=1000000, tau = 0.005,
                train_freq=train_freq, seed=seed, gradient_steps=1, action_noise=noise, alpha_0=np.e)
        case 3:
            model = DADAC("CustomMlpPolicy", delayed_env, delayed_env.merged_dist, tensorboard_log=save_dir, replay_buffer_class=CustomReplayBuffer,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), learning_rate=1e-4, policy_delay=2, learning_starts=10000 // train_freq, buffer_size=1000000, tau = 0.005,
                train_freq=train_freq, seed=seed, gradient_steps=1, action_noise=noise, alpha_0=np.e)
            
    t1 = time.time()
    model.learn(n_timesteps, progress_bar=False,tb_log_name=simulation_type, callback=callback)
    t2 = time.time() - t1
    ret = {"max_mean_10": callback.last_10_mean_reward, "algo": caso, "seed": seed, "dist_act": dist_act, "duration": t2, "case_number": case_number, 
           "dist_act_kwargs": dist_act_kwargs, "dist_obs": dist_obs, "dist_obs_kwargs": dist_obs_kwargs, "env": env_name}
    return ret


def arg_maker(caso):
    seed_list = [12345,22345,32345,42355,52345]
    env_names = ["Walker2d-v4", "Hopper-v4"]
    n_timestep = [10**6]
    match caso:
        ### Debugging case
        case 0: 
            seed_list = [0]
            dist_obs = ["gamma"]
            dist_obs_kwargs = [{"delta_max": 6, "mean": 2}]
            dist_act = ["test"]
            dist_act_kwargs = [{}]
            algo_nums = [0,1,2,3]
            env = env_names
            n_timestep = [10**3]

        ### Example of a simulation
        case 1: 
            dist_obs = ["gamma"]
            dist_obs_kwargs = [{"delta_max": 6, "mean": 2}]
            dist_act = ["test"]
            dist_act_kwargs = [{}]
            algo_nums = [0,1,2,3]
            env = env_names
        
    args = list(product(algo_nums, seed_list, dist_obs, env, dist_obs_kwargs, dist_act, dist_act_kwargs, n_timestep, [caso]))

    return args

if __name__ == '__main__':
    caso = eval(input("Case: "))
    args = arg_maker(caso)
    print("Number of distinct simulations: ", len(args))
    print("Parametrizations: ")
    for arg in args:
        print(arg)
    use_cuda = eval(input("Use cuda (0/1): "))
    if use_cuda == 0:
        import os
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=true "
            "intra_op_parallelism_threads=1 "
            "inter_op_parallelism_threads=1"
            )
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
    elif use_cuda == 1:
        pass
    else:
        assert False, "Not a good cuda value, ending the program"

    n_thread = eval(input("Number of parallel simulations: "))
    
    mp.set_start_method('spawn')
    with mp.Pool(n_thread) as p:
        result = p.starmap(run_sim, args)

    with open("./parallel_runs/sim_number_{}/results.json".format(caso), 'w') as file:
        json.dump(result, file)

    print("End of case ", caso)
