from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from stable_baselines3.common.type_aliases import Schedule
from sbx.common.type_aliases import RLTrainState
from sbx.common.policies import SquashedGaussianActor
from sbx.sac.policies import SACPolicy
from sbx.common.policies import Flatten
from collections.abc import Sequence
from typing import Optional, Callable
import warnings
from sbx import SAC
import optax
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.buffers import ReplayBuffer

from utils_env import *

warnings.filterwarnings("ignore")

class CustomContinuousCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activation_fn_final: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.identity
    output_dim: int = 2
    log_std_min: float = -8
    log_std_max: float = 8
    kernel_init: Callable = jax.nn.initializers.orthogonal()

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        for n_units in self.net_arch:
            x = nn.Dense(n_units, kernel_init=self.kernel_init)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)

        
        mean = nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)
        mean = self.activation_fn_final(mean)
        log_std = nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)
        log_std = self.activation_fn_final(log_std)
        log_std = nn.softplus(log_std)
        return jnp.concatenate([mean, log_std], axis = 1)

class CustomVectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activation_fn_final: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.identity
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            CustomContinuousCritic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            activation_fn_final=self.activation_fn_final,
            output_dim=self.output_dim,
        )(obs, action)
        return q_values    

class CustomSACPolicy(SACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch = None, dropout_rate = 0, layer_norm = False, 
                 activation_fn = nn.relu, use_sde = False, log_std_init = -3, use_expln = False, clip_mean = 2, 
                 features_extractor_class=None, features_extractor_kwargs = None, normalize_images = True, optimizer_class = optax.adam, 
                 optimizer_kwargs = None, n_critics = 2, share_features_extractor = False, actor_class = SquashedGaussianActor, vector_critic_class = CustomVectorCritic,
                 actor_min_log_std = -20, actor_max_log_std = 0.5):
        """
        Modification of the vector critic definition to use a CustomVectorCritic (output of dim 2)
        """
        super().__init__(observation_space, action_space, lr_schedule, net_arch, dropout_rate, layer_norm, activation_fn, use_sde, log_std_init, 
                         use_expln, clip_mean, features_extractor_class, features_extractor_kwargs, normalize_images, optimizer_class, 
                         optimizer_kwargs, n_critics, share_features_extractor, actor_class, vector_critic_class)
        
        self.actor_min_log_std = actor_min_log_std
        self.actor_max_log_std = actor_max_log_std
        
    def build(self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float) -> jax.Array:
        key, actor_key, qf_key, dropout_key = jax.random.split(key, 4)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.actor = self.actor_class(
            action_dim=int(np.prod(self.action_space.shape)),
            net_arch=self.net_arch_pi,
            activation_fn=self.activation_fn,
            log_std_min = self.actor_min_log_std,
            log_std_max = self.actor_max_log_std
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # Inject hyperparameters to be able to modify it later
        # See https://stackoverflow.com/questions/78527164
        optimizer_class = optax.inject_hyperparams(self.optimizer_class)(learning_rate=lr_schedule(1), **self.optimizer_kwargs)

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optimizer_class,
        )

        self.qf = self.vector_critic_class(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            net_arch=self.net_arch_qf,
            n_critics=self.n_critics,
            activation_fn=self.activation_fn,
            output_dim = 1
        )

        optimizer_class_qf = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=qf_learning_rate, **self.optimizer_kwargs
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
            ),
            target_params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
            ),
            tx=optimizer_class_qf,
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm"),
        )

        return key
    

class delayed_SAC_wrapper(SAC):
    def __init__(self, policy, env, learning_rate = 0.0003, qf_learning_rate = None, buffer_size = 1000000, learning_starts = 100, batch_size = 256, 
                 tau = 0.005, gamma = 0.99, train_freq = 1, gradient_steps = 1, policy_delay = 1, action_noise = None, replay_buffer_class = None, 
                 replay_buffer_kwargs = None, n_steps = 1, ent_coef = "auto", target_entropy = "auto", use_sde = False, sde_sample_freq = -1, use_sde_at_warmup = False, 
                 stats_window_size = 100, tensorboard_log = None, policy_kwargs = None, param_resets = None, verbose = 0, seed = None, device = "auto", _init_setup_model = True, 
                 learning_rate_alpha = 3e-4, alpha_0 = 0.2):
        super().__init__(policy, env, learning_rate, qf_learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, policy_delay, 
                         action_noise, replay_buffer_class, replay_buffer_kwargs, n_steps, ent_coef, target_entropy, use_sde, sde_sample_freq, use_sde_at_warmup, stats_window_size, 
                         tensorboard_log, policy_kwargs, param_resets, verbose, seed, device, _init_setup_model)

        self.env = CustomMonitor(env)
        # Implementation of a different learning rate for the entropy
        
        params = {"log_ent_coef": jnp.log(alpha_0)}
        self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=params,
                tx=optax.adam(
                    learning_rate=learning_rate_alpha,
                ),
        )
        self.reset_env()

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
    @jax.jit
    def _convert_to_jax_tuple(obs, actions, next_obs, dones, rewards, discounts):
        return CustomReplayBufferSamplesJax(
                jnp.array(obs, dtype=jnp.float32, copy=False),
                jnp.array(actions, dtype=jnp.float32, copy=False),
                jnp.array(next_obs, dtype=jnp.float32, copy=False),
                jnp.array(dones, dtype=jnp.float32, copy=False).reshape(-1),
                jnp.array(rewards, dtype=jnp.float32, copy=False).reshape(-1),
                jnp.array(discounts, dtype=jnp.float32, copy=False),
            )

    def get_action(self, learning_starts, action_noise, num_envs):
        if self.info_ret["flag_compute_action"]:
            actions, _ = self._sample_action(learning_starts, action_noise, num_envs)
            self.first_it = False
        elif self.first_it:
            actions = self.env.action_space.sample()
        else:
            actions = {"nothing": 0}
        return actions