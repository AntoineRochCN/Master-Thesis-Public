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


    