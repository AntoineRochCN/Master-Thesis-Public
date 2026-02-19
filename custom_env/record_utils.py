import jax.numpy as jnp
import os
from flax import struct
import jax
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

@struct.dataclass
class LossRecord:
    pos: int
    history: jax.Array

@struct.dataclass
class EnvRecord:
    buffer: jax.Array
    pos: int
    cumulated_reward: float = 0.0

@struct.dataclass
class MetricRecord:
    losses: LossRecord
    env_stats: EnvRecord
    n_timesteps: int
    pos : int

@struct.dataclass
class TestRecord:
    reward_mat: jax.Array
    pf_mat: jax.Array
    length_mat: jax.Array
    sharpe_mat: jax.Array
    opt_ratio_mat: jax.Array
    pos : int
    key: jax.Array

@partial(jax.jit, donate_argnums = (0,))
def update_env_rec(env_rec: EnvRecord, ep_length : int, pf_value: float):
    return env_rec.replace(pos = env_rec.pos + 1, buffer = jax.lax.dynamic_update_slice(env_rec.buffer, jnp.asarray([ep_length, pf_value , env_rec.cumulated_reward]).reshape(1,-1), (env_rec.pos, 0)), 
                           cumulated_reward = 0.0)
    
def moving_average_extractor(reward_arr, timestep_arr, avg_size = 100):
    concat_rew = np.concatenate(reward_arr)
    concat_ep_length = np.concatenate(timestep_arr)
    cumulated_rew = np.cumsum(concat_rew)
    cumulated_lg = np.cumsum(concat_ep_length)

    return (cumulated_rew[avg_size:] - cumulated_rew[:-avg_size])/avg_size, (cumulated_lg[avg_size:] - cumulated_lg[:-avg_size])/avg_size


def done_indices(dones):
    return np.argwhere(dones).reshape(-1)

def reward_extractor(observations, rewards, actions, dones, avg_size = 100):
    delimitation = np.concatenate([[0], done_indices(dones)+1])
    reward_arr = []
    ep_length = []
    pf_arr = []

    for k in range(len(delimitation) - 1):
        ind_begin = delimitation[k]
        ind_end = delimitation[k+1]

        r = np.sum(rewards[ind_begin : ind_end])
        reward_arr.append(r)
        ep_length.append(ind_end - ind_begin)
        pf_arr.append(observations[ind_end-1, 23])
    
    return reward_arr, ep_length, pf_arr, delimitation[1:]

def calc_res(curr_obs, init_obs, init_state, state, pf_value):
    tot = 3*state + init_state

    long_reward = curr_obs / init_obs * pf_value
    short_reward = init_obs / curr_obs * pf_value

    transac_cost = 0.75/100 * pf_value

    match tot:
        case 0:
            new_state_init, new_obs_init, pf_value = state, curr_obs, pf_value
        case 1:
            new_state_init, new_obs_init, pf_value = state, curr_obs, long_reward - transac_cost
        case 2:
            new_state_init, new_obs_init, pf_value = state, curr_obs, short_reward - transac_cost
        case 3:
            new_state_init, new_obs_init, pf_value = state, curr_obs, pf_value
        case 4:
            new_state_init, new_obs_init, pf_value = state, init_obs, pf_value
        case 5:
            new_state_init, new_obs_init, pf_value = init_state, init_obs, pf_value
        case 6:
            new_state_init, new_obs_init, pf_value = state, curr_obs, pf_value
        case 7:
            new_state_init, new_obs_init, pf_value = init_state, init_obs, pf_value
        case 8:
            new_state_init, new_obs_init, pf_value = state, init_obs, pf_value
    return new_state_init, new_obs_init, pf_value

def calc_renta_ep(obs_arr, state_arr):
    obs_init = obs_arr[0]
    state_init = state_arr[0]
    pf_value = 1

    pf_arr = np.zeros(len(obs_arr))
    pf_arr[0] = 1

    pf_value = 1
    
    for k, (obs, state) in enumerate(zip(obs_arr[1:], state_arr[1:])):
        state_init, obs_init, pf_value = calc_res(obs, obs_init, state_init, state, pf_value)
        pf_arr[k+1] = pf_value

    raw_ret = obs_arr / obs_arr[0]
    return pf_arr, raw_ret  
    

def get_returns(env_rec: EnvRecord):
    states = np.array(env_rec.buffer[:,2]).astype(int)
    dones = np.array(env_rec.buffer[:,4]).astype(np.bool_)
    obs = np.array(env_rec.buffer[:,0])

    idx_delim = np.concatenate([[0], np.where(dones)[0]])

    arr_states = [states[idx_delim[i] + (i>0): idx_delim[i+1]+1] for i in range(len(idx_delim) - 1)]
    arr_obs = [obs[idx_delim[i] + (i>0): idx_delim[i+1]+1] for i in range(len(idx_delim) - 1)]

    ret_pf = []
    ret_raw = []
    sharpe_arr = []

    for k in range(len(arr_obs)):
        pf_arr, raw_ret_arr = calc_renta_ep(arr_obs[k], arr_states[k])
        ret_pf.append(pf_arr)
        ret_raw.append(raw_ret_arr)
        sharpe_arr.append(np.mean(pf_arr - raw_ret_arr) / np.std(pf_arr - raw_ret_arr))
    return ret_pf, ret_raw, sharpe_arr


def tensorboard_logger(env_rec: EnvRecord, loss_rec: LossRecord, test_rec: TestRecord, tensorboard_log, tb_log_name, obs_dim, loss_warmup, log_interval = 4, 
                       collect_timesteps = 1, avg_size = 100, test_frequency = 1):
    idx = 1
    base_dir = tensorboard_log + "/" + tb_log_name
    while True:
        candidate = f"{base_dir}_{idx}"
        if not os.path.exists(candidate):
            break
        idx += 1
    writer = SummaryWriter(log_dir= base_dir + "_{}".format(idx))
    
    pi_loss = np.array(loss_rec.history[:,1])
    Q_loss = np.array(loss_rec.history[:,0])
    alpha_loss = np.array(loss_rec.history[:,2])
    alpha = np.array(loss_rec.history[:,3])
    Q_mean = np.array(loss_rec.history[:,4])
    entropy = np.array(loss_rec.history[:,5])
    std = np.array(loss_rec.history[:,6:])

    rewards = np.array(env_rec.buffer[:,2]).reshape(-1)
    ep_length = np.array(env_rec.buffer[:,0]).reshape(-1)
    pf_values = np.array(env_rec.buffer[:,1]).reshape(-1)

    avg_pf_test = np.mean(test_rec.pf_mat, axis=1)
    med_pf_test = np.median(test_rec.pf_mat, axis=1)
    avg_length_test = np.mean(test_rec.length_mat, axis=1)
    avg_rew_test = np.mean(test_rec.reward_mat, axis=1)
    avg_sharpe_test = np.mean(test_rec.sharpe_mat, axis=1)
    avg_opt_ratio = np.mean(test_rec.opt_ratio_mat, axis=1)

    for k in range(0, len(pi_loss)):        
        writer.add_scalar("train/pi_loss", float(pi_loss[k]), k*log_interval)
        writer.add_scalar("train/Q_loss", float(Q_loss[k]), k*log_interval)
        writer.add_scalar("train/alpha_loss", float(alpha_loss[k]), k*log_interval)
        writer.add_scalar("train/alpha", float(alpha[k]), k*log_interval)
        writer.add_scalar("train/Q_mean", float(Q_mean[k]), k*log_interval)
        writer.add_scalar("train/entropy", float(entropy[k]), k*log_interval)
        writer.add_scalar("train/std_1", float(std[k,0]), k*log_interval)
        writer.add_scalar("train/std_2", float(std[k,1]), k*log_interval)
        writer.add_scalar("train/std_3", float(std[k,2]), k*log_interval)

    it_count = 0
    
    for k in range(env_rec.pos-1):
        it_count += ep_length[k] 
        writer.add_scalar("rollout/ep_rew_mean", float(rewards[k]), int(it_count))
        writer.add_scalar("rollout/ep_len_mean", float(ep_length[k]), int(it_count))
        writer.add_scalar("rollout/pf_value", float((pf_values[k]-1)*100), int(it_count))

    for k in range(len(avg_pf_test)):
        writer.add_scalar("test/ep_rew_mean", float(avg_rew_test[k]), int(k*test_frequency))
        writer.add_scalar("test/ep_len_mean", float(avg_length_test[k]), int(k*test_frequency))
        writer.add_scalar("test/ep_pf_mean", float((avg_pf_test[k]-1)*100), int(k*test_frequency))
        writer.add_scalar("test/ep_pf_median", float((med_pf_test[k]-1)*100), int(k*test_frequency))
        writer.add_scalar("test/ep_sharpe_mean", float(avg_sharpe_test[k]), int(k*test_frequency))
        writer.add_scalar("test/ep_opt_ratio_mean", float(avg_opt_ratio[k]), int(k*test_frequency))

    loss_warmup = np.array(loss_warmup)

    for k in range(len(loss_warmup)): 
        writer.add_scalar("warmup/Q_loss", loss_warmup[k,0], k)
        writer.add_scalar("warmup/Q_mean", loss_warmup[k,1], k)
        writer.add_scalar("warmup/pi_loss", loss_warmup[k,2], k)
        writer.add_scalar("warmup/entropy", loss_warmup[k,3], k)


    