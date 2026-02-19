import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from DADAC_JAX import DADAC_JAX
from SAC_delayed_JAX import SAC_delayed_JAX
from SAC_VC_JAX import SAC_VC_JAX
from DSAC_JAX import DSAC_JAX
import time
import multiprocessing as mp
import numpy as np
from itertools import product
import json
from env import *
from copy import deepcopy

def run_sim(caso, seed, dist_obs, dist_obs_kwargs, dist_act, dist_act_kwargs, n_timesteps, case_number, 
            stop_loss, stop_limit, ep_length, train_freq, alpha_0, gamma, target_entropy, batch_size, 
            policy_delay, leverage, fees):
    print("New sim", " stop loss: ", stop_loss, ' leverage: ', leverage, " fees: ", fees)
    path = "/home/aroch/Documents/Travail_Local/Uchile/Trabajo_Titulacion/Master-Thesis-Project/binance_scrapper_v0/transformed_data_bis.npy"

    env = EnvDataBinance.create(path, ep_length=ep_length, stop_loss=stop_loss, stop_limit=stop_limit, leverage=leverage, transaction_cost=fees)
    env_test = EnvDataBinance.create(path, ep_length=ep_length, stop_loss=stop_loss, stop_limit=stop_limit, seed=seed+1, leverage=leverage, transaction_cost=fees)

    obs_length = 20
    action_length = 1
    shape = 2*obs_length + 3
    eval_freq = 5000
    n_eval = 50
    log_interval = 100

    save_dir = "./algo_comparison/sim_number_{}/".format(case_number)

    simulation_list = ["SAC", "SAC_VC", "DSAC", "DADAC"]
    simulation_type = simulation_list[caso]

    match caso:
        
        case 0:
            buffer_size = int(10**6)
            pos = 0
        
            full = jnp.zeros(1, dtype=jnp.float32)

            latency_manager = JaxLatencyEnv.create(distribution_action=dist_act, dist_action_kwargs=dist_act_kwargs,
                                                    distribution_obs=dist_obs, dist_obs_kwargs=dist_obs_kwargs,
                                                    seed=seed, action_dim=action_length, obs_dim=obs_length, max_ep_length=ep_length)
            latency_manager_test = deepcopy(latency_manager)
            max_latency = int(latency_manager.merged_dist.max_latency + 2)
            
            buffer = jnp.empty((buffer_size,shape), dtype=jnp.float32)
            buffer = CustomBufferBis(buffer=buffer, pos = pos, buffer_size=buffer_size, full=full)
            DISRESPECT_THE_ENV = foolish_env(obs_length,3)
            model = SAC_delayed_JAX("DiscretePolicy", DISRESPECT_THE_ENV, env, env_test, latency_manager, latency_manager_test, buffer, tensorboard_log=save_dir, replay_buffer_class=None,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), learning_rate=1e-4, policy_delay=policy_delay, learning_starts= 10000//train_freq, buffer_size=1000000, tau = 0.005,
                train_freq=train_freq, seed=seed, gradient_steps=1, action_noise=None, alpha_0=alpha_0, gamma=gamma, target_entropy=target_entropy, learning_rate_alpha=1e-4, batch_size=batch_size)
        case 1:
            buffer_size = int(10**6)
            pos = 0
        
            full = jnp.zeros(1, dtype=jnp.float32)

            latency_manager = JaxLatencyEnv.create(distribution_action=dist_act, dist_action_kwargs=dist_act_kwargs,
                                                    distribution_obs=dist_obs, dist_obs_kwargs=dist_obs_kwargs,
                                                    seed=seed, action_dim=action_length, obs_dim=obs_length, max_ep_length=ep_length)
            latency_manager_test = deepcopy(latency_manager)
            max_latency = int(latency_manager.merged_dist.max_latency + 2)
            
            buffer = jnp.empty((buffer_size, max_latency,shape), dtype=jnp.float32)
            buffer = CustomBufferLatency(buffer=buffer, pos = pos, buffer_size=buffer_size, full=full, max_latency =max_latency)
            DISRESPECT_THE_ENV = foolish_env(obs_length,3)
            model = SAC_VC_JAX("DiscretePolicy", DISRESPECT_THE_ENV, env, env_test, latency_manager, latency_manager_test, buffer, tensorboard_log=save_dir, replay_buffer_class=None,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), learning_rate=1e-4, policy_delay=policy_delay, learning_starts= 10000//train_freq, buffer_size=1000000, tau = 0.005,
                train_freq=train_freq, seed=seed, gradient_steps=1, action_noise=None, alpha_0=alpha_0, gamma=gamma, target_entropy=target_entropy, learning_rate_alpha=1e-4, batch_size=batch_size)
        case 2:
            buffer_size = int(10**6)
            pos = 0
        
            full = jnp.zeros(1, dtype=jnp.float32)

            latency_manager = JaxLatencyEnv.create(distribution_action=dist_act, dist_action_kwargs=dist_act_kwargs,
                                                    distribution_obs=dist_obs, dist_obs_kwargs=dist_obs_kwargs,
                                                    seed=seed, action_dim=action_length, obs_dim=obs_length, max_ep_length=ep_length)
            latency_manager_test = deepcopy(latency_manager)
            max_latency = int(latency_manager.merged_dist.max_latency + 2)
            
            buffer = jnp.empty((buffer_size,shape), dtype=jnp.float32)
            buffer = CustomBufferBis(buffer=buffer, pos = pos, buffer_size=buffer_size, full=full)
            DISRESPECT_THE_ENV = foolish_env(obs_length,3)
            model = DSAC_JAX("DiscretePolicy", DISRESPECT_THE_ENV, env, env_test, latency_manager, latency_manager_test, buffer, tensorboard_log=save_dir, replay_buffer_class=None,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), learning_rate=1e-4, policy_delay=policy_delay, learning_starts= 10000//train_freq, buffer_size=1000000, tau = 0.005,
                train_freq=train_freq, seed=seed, gradient_steps=1, action_noise=None, alpha_0=alpha_0, gamma=gamma, target_entropy=target_entropy, learning_rate_alpha=1e-4, batch_size=batch_size)
        case 3:
            buffer_size = int(10**6)
            pos = 0
        
            full = False

            latency_manager = JaxLatencyEnv.create(distribution_action=dist_act, dist_action_kwargs=dist_act_kwargs,
                                                    distribution_obs=dist_obs, dist_obs_kwargs=dist_obs_kwargs,
                                                    seed=seed, action_dim=action_length, obs_dim=obs_length, max_ep_length=ep_length)
            latency_manager_test = deepcopy(latency_manager)
            max_latency = int(latency_manager.merged_dist.max_latency + 2)
            
            buffer = jnp.empty((buffer_size, max_latency,shape), dtype=jnp.float32)
            buffer = CustomBufferLatency(buffer=buffer, pos = pos, buffer_size=buffer_size, full=full, max_latency =max_latency)
            DISRESPECT_THE_ENV = foolish_env(obs_length,3)
            model = DADAC_JAX("DiscretePolicy", DISRESPECT_THE_ENV, env, env_test, latency_manager, latency_manager_test, buffer, tensorboard_log=save_dir, replay_buffer_class=None,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), learning_rate=1e-4, policy_delay=policy_delay, learning_starts= 10000//train_freq, buffer_size=1000000, tau = 0.005,
                train_freq=train_freq, seed=seed, gradient_steps=1, action_noise=None, alpha_0=alpha_0, gamma=gamma, target_entropy=target_entropy, learning_rate_alpha=1e-4, batch_size=batch_size)
            
    t1 = time.time()
    carry_out = model.learn_jax(n_timesteps,tb_log_name=simulation_type, eval_freq=eval_freq, n_eval=n_eval, log_interval=log_interval)
    t2 = time.time() - t1
    
    mat_pf = carry_out.test_rec.pf_mat
    mat_rew = carry_out.test_rec.reward_mat
    mat_len = carry_out.test_rec.length_mat
    idx = (np.arange(len(mat_pf))+1) * eval_freq >= 0.9 * n_timesteps

    max_mean_10 = float(np.max(np.mean(mat_rew[idx], axis=1)))
    max_pf_10 = float(np.max(np.mean(mat_pf[idx], axis=1)))
    max_len_10 = float(np.max(np.mean(mat_len[idx], axis=1)))
    max_pf_med_10 = float(np.max(np.median(mat_pf[idx], axis=1)))
    
    ret = {"max_mean_10": max_mean_10,"max_pf_10": max_pf_10, "max_len_10": max_len_10,"algo": caso, "seed": seed, "dist_act": dist_act, "duration": t2, "case_number": case_number, 
           "dist_act_kwargs": dist_act_kwargs, "dist_obs": dist_obs, "dist_obs_kwargs": dist_obs_kwargs, "algo": simulation_type, "max_pf_med_10": max_pf_med_10}
    return ret


def arg_maker(caso):
    seed_list = [12345,32345,52345]
    n_timestep = [10**6]
    ep_length = [3000]
    train_freq = [20]
    alpha_0 = [1.0]
    gamma = [0.99]
    target_entropy = [jnp.log(3)*0.05]
    batch_size = [256]
    stop_loss = [0.95]
    stop_limit = [1.05]
    policy_delay = [2]
    leverage = [1.0]
    fees = [0.075/100]

    match caso:
        ### CUSTOM
        case 0: 
            seed_list = [12345]
            n_timestep = [10**4]
            dist_obs = ["gamma"]
            dist_obs_kwargs = [{"delta_max": 6, "mean": 2}]
            dist_act = ["test"]
            dist_act_kwargs = [{}]
            algo_nums = [3]

        case 1: ### test target entropy
            seed_list = [12345]
            #target_entropy = [jnp.log(3)*0.2, jnp.log(3)*0.4, jnp.log(3)*0.6]
            target_entropy = [jnp.log(3)*0.2]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}]
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}]
            algo_nums = [3]

        case 2: ### test batch_size
            target_entropy = [jnp.log(3)*0.2]
            batch_size = [128,256,512]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}]
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}]
            algo_nums = [3]

        case 95: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            leverage = [1.0]
            stop_loss = [0.95,0.98]
            stop_limit = [1.02,1.05]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [0]

        case 96: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            leverage = [1.0]
            stop_loss = [0.95,0.98]
            stop_limit = [1.02,1.05]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [1]

        case 97: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            leverage = [1.0]
            stop_loss = [0.95,0.98]
            stop_limit = [1.02,1.05]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [2]

        #### REAL TESTS

        case 98: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            seed_list = [12345]
            leverage = [1.0, 3.0, 5.0]
            stop_loss = [0.95]
            stop_limit = [1.05]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [3]

        case 99: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            seed_list = [12345]
            leverage = [1.0, 3.0, 5.0]
            stop_loss = [0.98]
            stop_limit = [1.02]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [3]

        case 100: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            seed_list = [22345]
            leverage = [1.0, 3.0, 5.0]
            stop_loss = [0.95]
            stop_limit = [1.05]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [3]

        case 101: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            seed_list = [22345]
            leverage = [1.0, 3.0, 5.0]
            stop_loss = [0.98]
            stop_limit = [1.02]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [3]
        
        case 102: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            seed_list = [32345]
            leverage = [1.0, 3.0, 5.0]
            stop_loss = [0.95]
            stop_limit = [1.05]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [3]

        case 103: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            seed_list = [32345]
            leverage = [1.0, 3.0, 5.0]
            stop_loss = [0.98]
            stop_limit = [1.02]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [3]
        
        case 104: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            seed_list = [42345]
            leverage = [1.0, 3.0, 5.0]
            stop_loss = [0.95]
            stop_limit = [1.05]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [3]

        case 105: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            seed_list = [42345]
            leverage = [1.0, 3.0, 5.0]
            stop_loss = [0.98]
            stop_limit = [1.02]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [3]
        
        case 106: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            seed_list = [52345]
            leverage = [1.0, 3.0, 5.0]
            stop_loss = [0.95]
            stop_limit = [1.05]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [3]

        case 107: ### NEED TO BENCH LATENCY MEGA OVER BIG FINAL TEST
            ### Remember to take latest data
            seed_list = [52345]
            leverage = [1.0, 3.0, 5.0]
            stop_loss = [0.98]
            stop_limit = [1.02]
            fees = [0.0, 0.02/100, 0.075/100]
            dist_obs = ["measured_observation"]
            dist_obs_kwargs = [{}] ### real latency
            dist_act = ["measured_action"]
            dist_act_kwargs = [{}] ### real latency
            algo_nums = [3]

    args = list(product(algo_nums, seed_list, dist_obs, dist_obs_kwargs, dist_act, dist_act_kwargs, n_timestep, [caso], 
                        stop_loss, stop_limit, ep_length, train_freq, alpha_0, gamma, target_entropy, batch_size, 
                        policy_delay, leverage, fees))
    for arg in args:
        print(arg)
    return args

if __name__ == '__main__':
    caso_min = eval(input("Case min: "))
    caso_max = eval(input("Case max: "))
    use_cuda = eval(input("Use cuda (0/1): "))
    if use_cuda == 0:
        import os
        """
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=true "
            "intra_op_parallelism_threads=1 "
            "inter_op_parallelism_threads=1"
            )
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        """
        os.environ["JAX_PLATFORMS"] = "cpu"
        
        import jax
    elif use_cuda == 1:
        pass
    else:
        assert False, "Not a good cuda value, ending the program"
    n_max_sim = 25
    n_thread = eval(input("Number of threads: "))

    for caso in range(caso_min, caso_max + 1):
        args = arg_maker(caso)
        if caso == caso_min:
            mp.set_start_method('spawn')
        with mp.Pool(n_thread) as p:
            result = p.starmap(run_sim, args)

        print(result)
        print(caso)
        with open("./algo_comparison/sim_number_{}/results.json".format(caso), 'w') as file:
            json.dump(result, file)

        print("End of case ", caso)
