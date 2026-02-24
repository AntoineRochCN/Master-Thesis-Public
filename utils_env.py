import gymnasium as gym
import numpy as np
from numpy.typing import DTypeLike
import scipy as sc
from typing import Any, Optional, Union, NamedTuple, Callable
import os
import jax
from stable_baselines3.common.buffers import ReplayBuffer
import jax.numpy as jnp
import warnings
from copy import deepcopy
import time
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise

warnings.filterwarnings("ignore")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class CustomReplayBufferSamplesNp(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray

class CustomReplayBufferSamplesJax(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    dones: jax.Array
    rewards: jax.Array
    discounts: jax.Array
    next_actions: Optional[jax.Array] = None

class CustomReplayBufferSamples(NamedTuple): 
    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    dones: jax.Array
    rewards: jax.Array
    next_actions: Optional[jax.Array] = None
    discounts: Optional[jax.Array] = None

class LatencyDistribution(NamedTuple):
    latency_range: np.array
    latency_probabilities: np.array
    max_latency: np.array

def get_latency_prop(distribution, **kwargs):
    """
    Gets a distribution name with its parameters, and returns a tuple with the values taken by the latency, its probabilities and the value of the maximum delay
    """
    match distribution:
        case "uniform":
            delta_max = kwargs["delta_max"]
            latence_range = np.arange(delta_max + 1)
            latence_probabilities = np.ones_like(latence_range, dtype=np.float32) / len(latence_range)
        case "gamma":
            delta_max = kwargs["delta_max"]
            mean = kwargs["mean"]
            latence_range = np.arange(delta_max + 1)
            latence_probabilities = sc.stats.gamma.pdf(latence_range, mean).astype(np.float32)
            latence_probabilities /= np.sum(latence_probabilities)
        case "gaussian":
            delta_max = kwargs["delta_max"]
            latence_range = np.arange(delta_max + 1)
            mean_1, mean_2 = kwargs["vec_mean"]
            std_1, std_2 = kwargs["vec_std"]
            latence_probabilities = sc.stats.norm.pdf(latence_range, mean_1, std_1).astype(np.float32)
            latence_probabilities += sc.stats.norm.pdf(latence_range, mean_2, std_2).astype(np.float32)
            latence_probabilities /= np.sum(latence_probabilities)
        case 'test':
            latence_range = np.zeros(1, dtype=int)
            latence_probabilities = np.ones(1)
        case "constant":
            delta_max = kwargs["delta_max"]
            latence_range = np.arange(delta_max + 1)
            latence_probabilities = np.zeros_like(latence_range)
            latence_probabilities[-1] = 1
        case 'custom_test_mixture':
            delta_max = 50
            mean = 2
            latence_range = np.arange(delta_max + 1)
            latence_probabilities = sc.stats.gamma.pdf(latence_range, mean).astype(np.float32)
            latence_probabilities[7:] = 0
            latence_probabilities /= np.sum(latence_probabilities)
            latence_probabilities *= 0.99
            latence_probabilities[-1] = 0.01
        case "custom":
            raise NotImplementedError
    return LatencyDistribution(latence_range, latence_probabilities, max(int(np.max(latence_range)) - 1, 0))

def latency_merger(dist_a: LatencyDistribution, dist_b: LatencyDistribution):
    merged_probs = np.convolve(dist_a.latency_probabilities, dist_b.latency_probabilities).astype(np.float64)
    merged_probs /= np.sum(merged_probs, dtype=np.float64)
    merged_probs = merged_probs[1:]
    latency_range = list(range(dist_a.latency_range[0] + dist_b.latency_range[0], dist_a.latency_range[-1] + dist_b.latency_range[-1] + 1))
    max_latency = np.max(latency_range) - 1
    return LatencyDistribution(latency_range, merged_probs, max_latency)

def last_pos_np(values):
    vals, idx = np.unique(values[::-1], return_index=True)
    return vals, idx

class LatencyEnv(gym.Wrapper):
    def __init__(self, env, distribution_action, dist_action_kwargs,distribution_obs, dist_obs_kwargs):
        super().__init__(env)

        self.dist_action = get_latency_prop(distribution_action, **dist_action_kwargs)
        self.dist_observation = get_latency_prop(distribution_obs, **dist_obs_kwargs)

        self.latence_range_action = self.dist_action.latency_range 
        self.latence_probabilities_action = self.dist_action.latency_probabilities
        self.max_latency_action = self.dist_action.max_latency

        self.latence_range_observation = self.dist_observation.latency_range
        self.latence_probabilities_observation = self.dist_observation.latency_probabilities
        self.max_latency_observation = self.dist_observation.max_latency

        self.merged_dist = latency_merger(self.dist_action, self.dist_observation)
    
        assert hasattr(self.spec, "max_episode_steps")
        assert self.spec.max_episode_steps != None

    def _init_time_shifts(self):
        n_max = self.spec.max_episode_steps + (self.max_latency_action + self.max_latency_observation)*3
        tau = np.random.choice(self.latence_range_action, p = self.latence_probabilities_action, size=n_max)
        tau_prime = np.random.choice(self.latence_range_observation, p = self.latence_probabilities_observation, size=n_max)

        self.t_start = tau[0] + tau_prime[0]
        
        self.mask_actions = np.zeros(n_max + self.max_latency_action + 2, dtype=np.bool)
        self.mask_next_observations = np.zeros(n_max + self.max_latency_action + self.max_latency_observation + 3, dtype=np.bool)

        self.action_shift = np.arange(n_max) + tau + 1
        self.next_obs_shift = np.arange(n_max) + tau + tau_prime + 1

        vals, idx_act = last_pos_np(self.action_shift)
        vals_bis, idx_obs = last_pos_np(self.next_obs_shift)
        self.idx_act = n_max - 1 - idx_act
        self.idx_obs = n_max - 1 - idx_obs

        self.vals_action = vals
        self.vals_obs = vals_bis

        self.mask_actions[self.idx_act] = True
        self.mask_next_observations[self.idx_obs] = True
        
    def _init_buffers(self):
        self.action_buffer = np.zeros((self.spec.max_episode_steps + (self.max_latency_action + self.max_latency_observation) * 3, *self.action_space.shape))
        self.obs_buffer = np.zeros((self.spec.max_episode_steps + (self.max_latency_action + self.max_latency_observation) * 3+1, *self.observation_space.shape))
        self.done_buffer = np.zeros((self.spec.max_episode_steps + (self.max_latency_action + self.max_latency_observation) * 3))
        self.reward_buffer = np.zeros((self.spec.max_episode_steps + (self.max_latency_action + self.max_latency_observation) * 3))

    def reset(self, *, seed = None, options = None):
        self._init_time_shifts()
        self._init_buffers()
        self.timestep = 1
        self.past_action = np.zeros( *self.action_space.shape)
        next_obs, info = super().reset(seed=seed, options=options)
        self.obs_buffer[0] = next_obs
        self.ret_ind = 0
        self.counter_shift = 0
        return (next_obs, info)
    
    def _store_action(self, action):
        idx = np.arange(self.vals_obs[self.counter_shift + 1] - self.vals_obs[self.counter_shift]) + self.vals_obs[self.counter_shift] - 1
        self.counter_shift += 1
        self.action_buffer[idx] = action

    def _load_action(self):
        return self.action_buffer[self.timestep - 1]

    def _store_observation(self, observation):
        self.obs_buffer[self.timestep] = observation

    def _store_reward(self, reward):
        self.reward_buffer[self.timestep] = reward

    def _store_done(self, done):
        self.done_buffer[self.timestep] = done

    def step(self, action):
        if "nothing" in action:
            action = self.past_action
        else:
            self._store_action(action)
            
        action = self._load_action()
        self.past_action = action
        
        obs, reward, terminated, truncated, info = super().step(action)
        self._store_observation(obs)
        self._store_reward(reward)
        self._store_done(terminated or truncated)

        ret = []
        info_ret = {}
        info_ret["num_ret"] = 0

        while self.next_obs_shift[self.ret_ind] < self.timestep:
            
            obs = self.obs_buffer[self.ret_ind]
            action = self.action_buffer[self.ret_ind]
            next_obs = self.obs_buffer[self.next_obs_shift[self.ret_ind] - 1]
            reward = self.reward_buffer[self.next_obs_shift[self.ret_ind] - 1]
            done = self.done_buffer[self.next_obs_shift[self.ret_ind] - 1]
            
            info = {}
            ret_tmp = [obs, next_obs, action, reward, done, info]
            ret.append(ret_tmp)

            self.ret_ind += 1
            info_ret["num_ret"] += 1

        info_ret["flag_compute_action"] = self.mask_actions[self.timestep]
        info_ret["observation"] = self.obs_buffer[self.timestep]
        info_ret["done"] = self.done_buffer[self.timestep]
        self.timestep += 1
        return ret, info_ret

class CustomMonitor(Monitor):
    def __init__(self, env, allow_early_resets=True):
        super().__init__(env)
        self.allow_early_resets = allow_early_resets
        self.num_envs = 1
        
    def step(self, action):        
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        if "nothing" in action:
            ret = self.env.step(action)
        else:
            ret = self.env.step(action.reshape(-1))
        
        for obs, next_obs, action, reward, done, infos in ret[0]:
            self.rewards.append(float(reward))
            if done:
                self.needs_reset = True
                ep_rew = sum(self.rewards)
                ep_len = len(self.rewards)
                ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
                for key in self.info_keywords:
                    ep_info[key] = infos[key]
                self.episode_returns.append(ep_rew)
                self.episode_lengths.append(ep_len)
                self.episode_times.append(time.time() - self.t_start)
                ep_info.update(self.current_reset_info)
                if self.results_writer:
                    self.results_writer.write_row(ep_info)
                infos["episode"] = ep_info
            self.total_steps += 1
        return ret

class CustomReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, max_latency = None, device = "auto", n_envs = 1, optimize_memory_usage = False, handle_timeout_termination = True, reward_scale = 1,
                 observation_scale = 1):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)
        self.observations = self.observations.astype(np.float32)
        self.key = jax.random.key(np.random.randint(0,2**10))

    def init_obs_buffer(self, max_latency):
        self.max_latency = max_latency
        self._init_buffers()

        self.next_observations = np.empty((self.buffer_size, self.max_latency + 1, *self.obs_shape), dtype=np.float32)
        self.next_actions = np.empty((self.buffer_size, self.max_latency + 1, self.action_dim), dtype=np.float32)
        self.next_rewards = np.empty((self.buffer_size, self.max_latency + 1), dtype=np.float32)
        self.next_dones = np.empty((self.buffer_size, self.max_latency + 1), dtype=np.float32)
        self.next_timeouts = np.empty((self.buffer_size, self.max_latency + 1), dtype=np.float32)

    def sample(self, batch_size, env = None):
        if self.full:
            batch_inds_obs = (np.random.randint(1, self.buffer_size, size=batch_size, dtype=np.int32) + self.pos) % self.buffer_size
        else:
            batch_inds_obs = np.random.randint(0, self.pos, size=batch_size, dtype=np.int32)
        return self._get_samples(batch_inds_obs)
    
    @staticmethod
    @jax.jit
    def _convert_sample_jax(obs, actions, next_obs, dones, rewards, next_actions):
        return CustomReplayBufferSamples(jnp.array(obs, copy=False, dtype= jnp.float32),
                                         jnp.array(actions, copy=False, dtype= jnp.float32),
                                         jnp.array(next_obs, copy=False, dtype= jnp.float32),
                                         jnp.array(dones, copy=False, dtype= jnp.float32),
                                         jnp.array(rewards, copy=False, dtype= jnp.float32),
                                         next_actions=jnp.array(next_actions, copy=False, dtype= jnp.float32))

    def _get_samples(self, batch_inds_obs) -> ReplayBufferSamples:        
        env_indices = 0
        next_obs = self.next_observations[batch_inds_obs]
        
        next_rewards = self.next_rewards[batch_inds_obs].reshape(-1,1)
        next_dones = (self.next_dones[batch_inds_obs] * (1 - self.next_timeouts[batch_inds_obs])).reshape(-1,1)
        
        return self._convert_sample_jax(self.observations[batch_inds_obs, env_indices, :], self.actions[batch_inds_obs, env_indices, :],
                                        next_obs, next_dones, next_rewards, self.next_actions[batch_inds_obs])

    def _feed_buffers(self, obs, next_obs, action, reward, done, infos):
        self.buffer_observations[self.buffer_pos] = np.array(obs)
        self.buffer_next_observations[self.buffer_pos] = np.array(next_obs)
        self.buffer_actions[self.buffer_pos] = np.array(action)
        self.buffer_rewards[self.buffer_pos] = np.array(reward)
        self.buffer_dones[self.buffer_pos] = np.array(done).squeeze()
        if self.handle_timeout_termination:
            self.buffer_timeouts[self.buffer_pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos]).squeeze()

        self.ep_length += 1
        self.buffer_pos += 1
        self.buffer_pos %= (self.max_latency + 2)

    def _init_buffers(self):
        self.buffer_observations = np.empty((self.max_latency + 2, *self.obs_shape))
        self.buffer_next_observations = np.empty((self.max_latency + 2, *self.obs_shape))
        self.buffer_actions = np.empty((self.max_latency + 2, self.action_dim))
        self.buffer_rewards = np.empty((self.max_latency + 2))
        self.buffer_dones = np.empty((self.max_latency + 2))
        self.buffer_timeouts = np.empty((self.max_latency + 2))
        self.buffer_pos = 0
        self.ep_length = 0
        self.flag_done = False

    def _reset_buffers(self):
        idx_past_buffer = (self.buffer_pos + 1)%(self.max_latency + 2)
        self.buffer_observations[0] = np.array(self.buffer_observations[idx_past_buffer], copy=True)
        self.buffer_observations[1:] = np.empty((self.max_latency + 1, *self.obs_shape))

        self.buffer_next_observations[0] = deepcopy(np.array(self.buffer_next_observations[idx_past_buffer], copy=True))
        self.buffer_next_observations[1:] = np.empty((self.max_latency + 1, *self.obs_shape))

        self.buffer_actions[0] = deepcopy(np.array(self.buffer_actions[idx_past_buffer], copy=True))
        self.buffer_actions[1:] = np.empty((self.max_latency + 1, self.action_dim))

        self.buffer_rewards[0] = np.array(self.buffer_rewards[idx_past_buffer], copy=True)
        self.buffer_rewards[1:] = np.empty((self.max_latency + 1))        
        
        self.buffer_dones[0] = np.array(self.buffer_dones[idx_past_buffer], copy=True)
        self.buffer_dones[1:] = np.empty((self.max_latency + 1))

        self.buffer_timeouts[0] = np.array(self.buffer_timeouts[idx_past_buffer], copy=True)
        self.buffer_timeouts[1:] = np.empty((self.max_latency + 1))

        self.buffer_pos = 1
        self.ep_length = 1
        self.flag_done = False

    def _store_buffers(self):
        self.observations[self.pos] = np.array(self.buffer_observations[self.buffer_pos]) 
        self.actions[self.pos] = np.array(self.buffer_actions[self.buffer_pos])
        self.rewards[self.pos] = np.array(self.buffer_rewards[self.buffer_pos])
        self.dones[self.pos] = np.array(self.buffer_dones[self.buffer_pos])
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(self.buffer_timeouts[self.buffer_pos])

        idx_obs = np.arange(self.buffer_pos, self.buffer_pos + self.max_latency + 1) % (self.max_latency + 2)
        idx_reward = idx_obs

        self.next_observations[self.pos] = np.array(self.buffer_next_observations[idx_obs]) 
        
        self.next_actions[self.pos] = np.array(self.buffer_actions[idx_reward])
        self.next_rewards[self.pos] = np.array(self.buffer_rewards[idx_reward]) 
        self.next_dones[self.pos] = np.array(self.buffer_dones[idx_reward])
        if self.handle_timeout_termination:
            self.next_timeouts[self.pos] = np.array(self.buffer_timeouts[idx_reward])

        self.pos += 1

    def add(self, obs, next_obs, action, reward, done, infos):
        self._feed_buffers(obs, next_obs, action, reward, done, infos)
        if self.ep_length >= self.max_latency + 2:
            self._store_buffers()
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, callback_on_new_best = None, callback_after_eval = None, n_eval_episodes = 5, 
                 eval_freq = 10000, log_path = None, best_model_save_path = None, deterministic = True, render = False, 
                 verbose = 1, warn = True, learning_timesteps = None):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, n_eval_episodes, eval_freq, log_path, best_model_save_path, deterministic, render, verbose, warn)
        self.eval_env = eval_env
        self.past_timestep = -1
        self.learning_timesteps = learning_timesteps
        self.last_10_mean_reward = -np.inf

    def _on_step(self) -> bool:
        continue_training = True

        flag = not (self.past_timestep == self.num_timesteps)

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0 and flag:
            self.past_timestep = self.num_timesteps
            self._is_success_buffer = []

            episode_rewards, episode_lengths = self.evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.learning_timesteps != None and self.num_timesteps > 0.9*self.learning_timesteps and mean_reward > self.last_10_mean_reward:
                self.last_10_mean_reward = mean_reward

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def evaluate_policy(
        self,
        model,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
    ) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
        episode_rewards = np.zeros(n_eval_episodes)
        episode_lengths = np.zeros(n_eval_episodes)

        for k in range(n_eval_episodes):
            current_rewards = 0
            current_lengths = 0
            res = env.reset()
            first_it = True
            info_ret = {"flag_compute_action": False, "observation": np.array(res[0]).reshape(-1)}
            done = False
            while not done:
                if info_ret["flag_compute_action"]:
                    actions, buffer_actions = model.predict(obs, deterministic=deterministic)
                    first_it = False
                elif first_it:
                    actions = env.action_space.sample()
                else:
                    actions = {"nothing": 0}

                ret, info_ret = env.step(actions)

                for r in ret:
                    obs, next_obs, action, reward, done, infos = r
                    current_rewards += reward
                    current_lengths += 1
                    
                obs = info_ret["observation"]
            episode_rewards[k] = current_rewards
            episode_lengths[k] = current_lengths

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        if reward_threshold is not None:
            assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
        if return_episode_rewards:
            return episode_rewards, episode_lengths
        return mean_reward, std_reward

class Buffered_NormalActionNoise(ActionNoise):
    def __init__(self, mean: np.ndarray, sigma: np.ndarray, size : int  = 10**4, dtype: DTypeLike = np.float32) -> None:
        self._mu = mean
        self._sigma = sigma
        self._dtype = dtype
        self.size = size
        super().__init__()
        self._reset_buffer()

    def _reset_buffer(self):
        self.buffer_pos = 0
        self.buffer_vals = np.random.normal(self._mu, self._sigma, size = (self.size, len(self._mu))).astype(self._dtype)

    def __call__(self) -> np.ndarray:
        ret = self.buffer_vals[self.buffer_pos]
        self.buffer_pos += 1
        if self.buffer_pos >= self.size:
            self._reset_buffer()
        return ret

    def __repr__(self) -> str:
        return f"NormalActionNoise(mu={self._mu}, sigma={self._sigma})"
    


class ReplayBuffer_DelayedSAC(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device = "auto", n_envs = 1, optimize_memory_usage = False, handle_timeout_termination = True):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:        
        if self.full:
            batch_inds_obs = (np.random.randint(1, self.buffer_size, size=batch_size, dtype=np.int32) + self.pos) % self.buffer_size
        else:
            batch_inds_obs = np.random.randint(0, self.pos , size=batch_size, dtype=np.int32)
        return self._get_samples(batch_inds_obs,env)
    
    
    @staticmethod
    @jax.jit
    def _convert_sample_jax(obs, actions, next_obs, dones, rewards, next_actions):
        return CustomReplayBufferSamples(jnp.array(obs, copy=False, dtype= jnp.float32),
                                         jnp.array(actions, copy=False, dtype= jnp.float32),
                                         jnp.array(next_obs, copy=False, dtype= jnp.float32),
                                         jnp.array(dones, copy=False, dtype= jnp.float32),
                                         jnp.array(rewards, copy=False, dtype= jnp.float32),
                                         next_actions=jnp.array(next_actions, copy=False, dtype= jnp.float32))
    
    def _get_samples(self, batch_inds_obs,env) -> ReplayBufferSamples:        
        env_indices = 0
        next_obs = self.next_observations[batch_inds_obs, env_indices]
        
        rewards = self.rewards[batch_inds_obs, env_indices].reshape(-1,1)
        dones = (self.dones[batch_inds_obs, env_indices] * (1 - self.timeouts[batch_inds_obs, env_indices])).reshape(-1,1)
        return self._convert_sample_jax(self.observations[batch_inds_obs, env_indices, :], self.actions[batch_inds_obs, env_indices, :],
                                        next_obs, dones, rewards, self.actions[batch_inds_obs, env_indices, :])


