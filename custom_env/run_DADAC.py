import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from env import *
from buffer import *
import jax.numpy as jnp
import jax
from copy import deepcopy
from env import EnvDataBinance
from DADAC_JAX import DADAC_JAX
from DSAC_JAX import DSAC_JAX
from SAC_delayed_JAX import SAC_delayed_JAX
from SAC_VC_JAX import SAC_VC_JAX
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

if __name__ == '__main__':
    path = "/home/aroch/Documents/Travail_Local/Uchile/Trabajo_Titulacion/Master-Thesis-Project/binance_scrapper_v0/transformed_data_bis.npy"
    buffer_size = int(10**6)
    pos = 0

    obs_length = 20
    action_length = 3
    shape = 2*obs_length + 3

    ep_length = 3000

    target_entropy = jnp.log(action_length)*0.05
    stop_loss = 0.95
    stop_limit = 1.05
    transaction_cost = 0.02/100
    leverage = 3

    distribution_action = "measured_action"
    dist_action_kwargs = {}
    distribution_obs = "measured_observation"
    dist_obs_kwargs = {}

    seed = 0
    action_dim = 1

    latency_manager = JaxLatencyEnv.create(distribution_action=distribution_action, dist_action_kwargs=dist_action_kwargs,
                                            distribution_obs=distribution_obs, dist_obs_kwargs=dist_obs_kwargs,
                                            seed=seed, action_dim=action_dim, obs_dim=obs_length, max_ep_length=ep_length)
    latency_manager_test = deepcopy(latency_manager)
    max_latency = int(latency_manager.merged_dist.max_latency + 2)
    
    env = EnvDataBinance.create(path, ep_length=ep_length, stop_loss=stop_loss, stop_limit=stop_limit, transaction_cost=transaction_cost, leverage=leverage)
    env_test = EnvDataBinance.create(path, ep_length=ep_length, stop_loss=stop_loss, stop_limit=stop_limit, transaction_cost=transaction_cost, seed=1, leverage=leverage)
    
    train_freq = 20
    
    
    full =  False
    
    
    buffer = jnp.empty((buffer_size, max_latency, shape), dtype=jnp.float32)
    buffer = CustomBufferLatency(buffer=buffer, pos = pos, buffer_size=buffer_size, full=full, max_latency =max_latency)
    model = DADAC_JAX("DiscretePolicy", env, env_test, latency_manager, latency_manager_test, buffer, tensorboard_log="./test/", replay_buffer_class=None,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu, optimizer_kwargs = {"eps": 1e-4}), learning_rate=1e-4, policy_delay=1, buffer_size=1000000, tau = 0.005 ,
                train_freq=train_freq, seed=0, gradient_steps=1, alpha_0=0.1, gamma=0.99, target_entropy=target_entropy, learning_rate_alpha=1e-4, batch_size=256, n_episodes_warmup=10**2, n_training_warmup=10**4)
    carry_out = model.learn_jax(10**5, eval_freq = 5000, n_eval = 10, log_interval=100, save_model=False, save_path="model_save_test.msgpack")
    """
    buffer = jnp.empty((buffer_size, shape), dtype=jnp.float32)
    buffer = CustomBufferBis(buffer=buffer, pos = pos, buffer_size=buffer_size, full=full)
    model = DSAC_JAX("DiscretePolicy", env, env_test, latency_manager, latency_manager_test, buffer, tensorboard_log="./test/", replay_buffer_class=None,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu, optimizer_kwargs = {"eps": 1e-4}), learning_rate=1e-4, policy_delay=1, buffer_size=1000000, tau = 0.005 ,
                train_freq=train_freq, seed=0, gradient_steps=1, alpha_0=0.1, gamma=0.99, target_entropy=target_entropy, learning_rate_alpha=1e-4, batch_size=256, n_episodes_warmup=10**2, n_training_warmup=10**4)
    carry_out = model.learn_jax(10**4, eval_freq = 5000, n_eval = 10, log_interval=100, save_model=False, save_path="model_save_test.msgpack")
    buffer = jnp.empty((buffer_size, shape), dtype=jnp.float32)
    buffer = CustomBufferBis(buffer=buffer, pos = pos, buffer_size=buffer_size, full=full)
    model = SAC_delayed_JAX("DiscretePolicy", env, env_test, latency_manager, latency_manager_test, buffer, tensorboard_log="./test/", replay_buffer_class=None,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu, optimizer_kwargs = {"eps": 1e-4}), learning_rate=1e-4, policy_delay=1, buffer_size=1000000, tau = 0.005 ,
                train_freq=train_freq, seed=0, gradient_steps=1, alpha_0=0.1, gamma=0.99, target_entropy=target_entropy, learning_rate_alpha=1e-4, batch_size=256, n_episodes_warmup=10**2, n_training_warmup=10**4)
    carry_out = model.learn_jax(10**4, eval_freq = 5000, n_eval = 10, log_interval=100, save_model=False, save_path="model_save_test.msgpack")
    buffer = jnp.empty((buffer_size, max_latency, shape), dtype=jnp.float32)
    buffer = CustomBufferLatency(buffer=buffer, pos = pos, buffer_size=buffer_size, full=full, max_latency =max_latency)
    model = SAC_VC_JAX("DiscretePolicy", env, env_test, latency_manager, latency_manager_test, buffer, tensorboard_log="./test/", replay_buffer_class=None,
                policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu, optimizer_kwargs = {"eps": 1e-4}), learning_rate=1e-4, policy_delay=1, buffer_size=1000000, tau = 0.005 ,
                train_freq=train_freq, seed=0, gradient_steps=1, alpha_0=0.1, gamma=0.99, target_entropy=target_entropy, learning_rate_alpha=1e-4, batch_size=256, n_episodes_warmup=10**2, n_training_warmup=10**4)
    carry_out = model.learn_jax(10**4, eval_freq = 5000, n_eval = 10, log_interval=100, save_model=False, save_path="model_save_test.msgpack")
    """