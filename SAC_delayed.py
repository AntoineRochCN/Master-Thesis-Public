import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from sbx.sac.policies import SACPolicy
from sbx import SAC
from typing import Optional, TypeVar, ClassVar
from utils_env import *
from utils_SAC import CustomSACPolicy
import warnings
warnings.filterwarnings("ignore")
SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")

class SAC_delayed(SAC):
    """
    DSAC algorithm with updated collect rollouts to manage delayed envs
    """
    policy_aliases: ClassVar[dict[str, type[SACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SACPolicy,
        # Implement the DSAC NN
        "CustomMlpPolicy": CustomSACPolicy,
    }
    def __init__(self, policy, env, learning_rate = 0.0003, qf_learning_rate = None, buffer_size = 1000000, learning_starts = 100, 
                 batch_size = 256, tau = 0.005, gamma = 0.99, train_freq = 1, gradient_steps = 1, policy_delay = 1, action_noise = None, 
                 replay_buffer_class = None, replay_buffer_kwargs = None, n_steps = 1, ent_coef = "auto", target_entropy = "auto", 
                 use_sde = False, sde_sample_freq = -1, use_sde_at_warmup = False, stats_window_size = 100, tensorboard_log = None, 
                 policy_kwargs = None, param_resets = None, verbose = 0, seed = None, device = "auto", _init_setup_model = True,
                 learning_rate_alpha = 3e-4, alpha_0 = 0.2):
        
        super().__init__(policy, env, learning_rate, qf_learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, 
                         gradient_steps, policy_delay, action_noise, replay_buffer_class, replay_buffer_kwargs, n_steps, ent_coef, 
                         target_entropy, use_sde, sde_sample_freq, use_sde_at_warmup, stats_window_size, tensorboard_log, policy_kwargs, 
                         param_resets, verbose, seed, device, _init_setup_model)

        self.env = CustomMonitor(env)
        # Implementation of a different learning rate for the entropy
        self.key, ent_key = jax.random.split(self.key, 2)
        params = {"log_ent_coef": jnp.log(alpha_0)}
        self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=params,
                tx=optax.adam(
                    learning_rate=learning_rate_alpha,
                ),
        )

        self.reset_env()
    
    def get_action(self, learning_starts, action_noise, num_envs):
        if self.info_ret["flag_compute_action"]:
            #print(self._last_obs, self.num_timesteps)
            actions, _ = self._sample_action(learning_starts, action_noise, num_envs)
            self.first_it = False
        elif self.first_it:
            actions = self.env.action_space.sample()
        else:
            actions = {"nothing": 0}
        return actions
    
    def update_current_obs(self, obs):
        self._last_obs = np.array(obs)

    def reset_env(self):
        res = self.env.reset()
        self.first_it = True
        self.info_ret = {"flag_compute_action": False, "observation": np.array(res[0]).reshape(-1)}
        self.update_current_obs(self.info_ret["observation"])

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        self.policy.set_training_mode(False)
        num_collected_steps, num_collected_episodes = 0, 0
        
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."
        if self.use_sde:
            self.actor.reset_noise(env.num_envs)  # type: ignore[operator]
        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

            actions = self.get_action(learning_starts, action_noise, env.num_envs)
            ret, self.info_ret = self.env.step(actions)

            for r in ret:
                obs, next_obs, action, reward, done, infos = r
                self.update_current_obs(obs)
                buffer_actions = action
                num_collected_steps += 1
                
                callback.update_locals(locals())

                if not callback.on_step():
                    return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)
                
                self._update_info_buffer([infos], [done])
                self._store_transition(replay_buffer, buffer_actions, next_obs, reward, [done], [infos])  # type: ignore[arg-type]
                
                if done:
                    self.reset_env()
                    

                for idx, done in enumerate([done]):
                    if done:
                        # Update stats
                        num_collected_episodes += 1
                        self._episode_num += 1
                        if action_noise is not None:
                            kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                            action_noise.reset(**kwargs)
                        if log_interval is not None and self._episode_num % log_interval == 0:
                            self.dump_logs()
                if done:
                    break
            self.update_current_obs(self.info_ret["observation"])
        
        num_collected_steps += 1 
        self.num_timesteps += env.num_envs
        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
        self._on_step()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
    
    @staticmethod
    def _convert_to_jax_tuple(obs, actions, next_obs, dones, rewards, discounts):
        return CustomReplayBufferSamplesJax(
                jnp.array(obs, dtype=jnp.float32, copy=False),
                jnp.array(actions, dtype=jnp.float32, copy=False),
                jnp.array(next_obs, dtype=jnp.float32, copy=False),
                jnp.array(dones, dtype=jnp.float32, copy=False).reshape(-1),
                jnp.array(rewards, dtype=jnp.float32, copy=False).reshape(-1),
                jnp.array(discounts, dtype=jnp.float32, copy=False),
            )

    def train(self, gradient_steps: int, batch_size: int) -> None:
        assert self.replay_buffer is not None
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)

        self._update_learning_rate(
            self.policy.actor_state.opt_state,
            learning_rate=self.lr_schedule(self._current_progress_remaining),
            name="learning_rate_actor",
        )
        self._update_learning_rate(
            self.policy.qf_state.opt_state,
            learning_rate=self.initial_qf_learning_rate or self.lr_schedule(self._current_progress_remaining),
            name="learning_rate_critic",
        )
        self._maybe_reset_params()
        
        if data.discounts is None:
            if not hasattr(self, 'discounts'):
                if data.discounts is None:
                    self.discounts = jnp.full((batch_size * gradient_steps), self.gamma, dtype=jnp.float32).reshape(-1)   
        else:
            self.discounts = data.discounts.reshape(-1)
        
        data = self._convert_to_jax_tuple(data.observations, data.actions, data.next_observations, data.dones, 
                                                  data.rewards, self.discounts)

        (
        self.policy.qf_state,
        self.policy.actor_state,
        self.ent_coef_state,
        self.key,
        (actor_loss_value, qf_loss_value, ent_coef_loss_value, ent_coef_value),
        ) = self._train(
            self.tau,
            self.target_entropy,
            gradient_steps,
            data,
            self.policy_delay,
            (self._n_updates + 1) % self.policy_delay,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/ent_coef_loss", ent_coef_loss_value.item())
        self.logger.record("train/ent_coef", ent_coef_value.item())  