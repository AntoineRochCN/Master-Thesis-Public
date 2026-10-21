"""
Benchmark script comparing training speed before and after optimizations.
"""

import os
import sys
import time

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
custom_env_dir = os.path.join(_repo_root, "custom_env")
if custom_env_dir not in sys.path:
    sys.path.insert(0, custom_env_dir)

from custom_env.env import EnvDataBinance, JaxLatencyEnv
from custom_env.buffer import CustomBufferLatency
from custom_env.DADAC_JAX import DADAC_JAX
import jax
import jax.numpy as jnp
from copy import deepcopy


def run_benchmark(n_timesteps=4000, eval_freq=2000, n_eval=4, warmup_episodes=5, warmup_train=20):
    path_train = os.path.join(_repo_root, "data", "transformed_train_data.npy")
    path_test = os.path.join(_repo_root, "data", "transformed_test_data.npy")

    buffer_size = 50000
    pos = 0
    obs_length = 20
    action_length = 3
    shape = 2 * obs_length + 3
    ep_length = 1000

    target_entropy = jnp.log(action_length) * 0.05
    stop_loss = 0.95
    stop_limit = 1.05
    transaction_cost = 0.02 / 100
    leverage = 3.0

    distribution_action = "measured_action"
    distribution_obs = "measured_observation"
    seed = 0
    action_dim = 1

    latency_manager = JaxLatencyEnv.create(
        distribution_action=distribution_action, dist_action_kwargs={},
        distribution_obs=distribution_obs, dist_obs_kwargs={},
        seed=seed, action_dim=action_dim, obs_dim=obs_length, max_ep_length=ep_length
    )
    latency_manager_test = deepcopy(latency_manager)
    max_latency = int(latency_manager.merged_dist.max_latency + 2)

    env = EnvDataBinance.create(path_train, ep_length=ep_length, stop_loss=stop_loss, stop_limit=stop_limit, transaction_cost=transaction_cost, leverage=leverage)
    env_test = EnvDataBinance.create(path_test, ep_length=ep_length, stop_loss=stop_loss, stop_limit=stop_limit, transaction_cost=transaction_cost, seed=1, leverage=leverage)

    train_freq = 20
    full = False

    buffer = jnp.empty((buffer_size, max_latency, shape), dtype=jnp.float32)
    buffer = CustomBufferLatency(buffer=buffer, pos=pos, buffer_size=buffer_size, full=full, max_latency=max_latency)

    model = DADAC_JAX(
        "DiscretePolicy", env, env_test, latency_manager, latency_manager_test, buffer,
        tensorboard_log=None, replay_buffer_class=None,
        policy_kwargs=dict(net_arch=[128, 128], activation_fn=jax.nn.gelu),
        learning_rate=1e-4, policy_delay=1, buffer_size=buffer_size, tau=0.005,
        train_freq=train_freq, seed=0, gradient_steps=1, alpha_0=0.1, gamma=0.99,
        target_entropy=target_entropy, learning_rate_alpha=1e-4, batch_size=64,
        n_episodes_warmup=warmup_episodes, n_training_warmup=warmup_train
    )

    print(f"Starting training run: {n_timesteps} steps, eval_freq={eval_freq}, n_eval={n_eval}...")
    t0 = time.time()
    carry = model.learn_jax(n_timesteps, eval_freq=eval_freq, n_eval=n_eval, log_interval=100, save_model=False)
    # Block until JAX operations complete
    _ = carry.env.step_carry.timestep.block_until_ready()
    total_time = time.time() - t0

    fps = n_timesteps / total_time
    print(f"Completed in {total_time:.2f} seconds ({fps:.1f} timesteps/s, {total_time/n_timesteps * 1e6:.1f} us/step)")
    return total_time, fps


if __name__ == "__main__":
    run_benchmark()
