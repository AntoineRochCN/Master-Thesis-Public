import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
import optax
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from sbx.sac.policies import SACPolicy
from sbx.common.type_aliases import RLTrainState
from typing import Any, TypeVar, ClassVar
from functools import partial
from utils_env import *
from utils_SAC import CustomSACPolicy, delayed_SAC_wrapper
import warnings
warnings.filterwarnings("ignore")
SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")

class DSAC(delayed_SAC_wrapper):
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
                         param_resets, verbose, seed, device, _init_setup_model, learning_rate_alpha=learning_rate_alpha, alpha_0=alpha_0)
        
        self.past_std = jnp.ones((2), dtype=jnp.float32)
        
    @staticmethod
    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        discounts: jax.Array,
        key: jax.Array,
        past_std: jax.Array, 
        tau: float
    ):
        key, noise_key, dropout_key_target, dropout_key_current, normal_key = jax.random.split(key, 5)
        dist = actor_state.apply_fn(actor_state.params, next_observations)
        
        next_actions_sample = dist.sample(seed = noise_key)
        next_log_prob = dist.log_prob(next_actions_sample) 
        next_log_prob = next_log_prob.reshape((-1))

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
        qf_next_values = qf_state.apply_fn(
            qf_state.target_params,
            next_observations,
            next_actions_sample,
            rngs={"dropout": dropout_key_target},
        ).reshape((2,-1,2))
        
        normal_values = jax.random.normal(normal_key, qf_next_values.shape[:-1], dtype=jnp.float32) 
        normal_values = jnp.clip(normal_values,-3,3) * jax.lax.stop_gradient(qf_next_values[..., 1]) + jax.lax.stop_gradient(qf_next_values[..., 0])
        
        C1 = ent_coef_value * next_log_prob
        C2 = (1 - dones) * discounts

        idx_min = jnp.argmin(qf_next_values[..., 0], axis=0)

        target_Q_values =  jax.lax.stop_gradient(qf_next_values[idx_min, jnp.arange(qf_next_values.shape[1])[:, None], 0])
        target_Z_values = normal_values[idx_min, jnp.arange(qf_next_values.shape[1])[:, None]]

        target_Z_values = target_Z_values - C1
        target_Z_values = rewards + C2 * target_Z_values

        target_Q_values = target_Q_values - C1
        target_Q_values = rewards + C2 * target_Q_values
        
        def mse_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
            current_q_values = qf_state.apply_fn(params, observations, actions, rngs={"dropout": dropout_key})

            new_means_std = (1 - tau) * past_std + tau *  jax.lax.stop_gradient(current_q_values[:,:,1]).mean(axis = (1)) #Insert tau here to make it evolve smoother (polyak)

            td_bound_1 = 3 *  jax.lax.stop_gradient(new_means_std[0])
            td_bound_2 = 3 *  jax.lax.stop_gradient(new_means_std[1])
            
            difference_1 = jnp.clip(jax.lax.stop_gradient(target_Z_values) -  jax.lax.stop_gradient(current_q_values[0,:,0]), -td_bound_1, td_bound_1)
            difference_2 = jnp.clip(jax.lax.stop_gradient(target_Z_values) -  jax.lax.stop_gradient(current_q_values[1,:,0]), -td_bound_2, td_bound_2)

            target_q_bound_1 =  jax.lax.stop_gradient(current_q_values[0,:,0]) + difference_1
            target_q_bound_2 =  jax.lax.stop_gradient(current_q_values[1,:,0]) + difference_2

            q1_std = current_q_values[0,:,1]
            q2_std = current_q_values[1,:,1]

            q1_std_detach =  jax.lax.stop_gradient(jnp.clip(q1_std, min=0.0))
            q2_std_detach =  jax.lax.stop_gradient(jnp.clip(q2_std, min=0.0))
            bias = 0.1
            
            ratio1 = jnp.clip(new_means_std[0] ** 2 / (q1_std_detach ** 2 + bias), 0.1,10)
            ratio2 = jnp.clip(new_means_std[1] ** 2 / (q2_std_detach ** 2 + bias), 0.1,10)
            
            q1_loss = (ratio1 * (optax.losses.huber_loss(current_q_values[0,:,0] , target_Q_values, 50) + 
                       q1_std * (q1_std_detach**2 - optax.losses.huber_loss( jax.lax.stop_gradient(current_q_values[0,:,0]), target_q_bound_1, 50)) / (q1_std_detach + bias))).mean()
            
            q2_loss = (ratio2 * (optax.losses.huber_loss(current_q_values[1,:,0] , target_Q_values, 50) + 
                       q2_std * (q2_std_detach**2 - optax.losses.huber_loss( jax.lax.stop_gradient(current_q_values[1,:,0]), target_q_bound_2, 50)) / (q2_std_detach + bias))).mean()
            
            
            return q1_loss + q2_loss , (new_means_std, current_q_values[:,:,0].mean())

        (qf_loss_value, (new_std, current_mean)), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, ent_coef_value),
            key,
            new_std,
            current_mean
        )
    
    @classmethod
    @partial(jax.jit, static_argnames=["cls", "gradient_steps", "policy_delay", "policy_delay_offset"])
    def _train(
        cls,
        tau: float,
        target_entropy: ArrayLike,
        gradient_steps: int,
        data: CustomReplayBufferSamplesNp,
        policy_delay: int,
        policy_delay_offset: int,
        qf_state: RLTrainState,
        actor_state: TrainState,
        ent_coef_state: TrainState,
        key: jax.Array,
        past_std: jax.Array
    ):
        
        assert data.observations.shape[0] % gradient_steps == 0
        batch_size = data.observations.shape[0] // gradient_steps

        carry = {
            "actor_state": actor_state,
            "qf_state": qf_state,
            "ent_coef_state": ent_coef_state,
            "key": key,
            "info": {
                "actor_loss": jnp.array(0.0),
                "qf_loss": jnp.array(0.0),
                "ent_coef_loss": jnp.array(0.0),
                "ent_coef_value": jnp.array(0.0),
            },
            "past_std": jnp.array(past_std),
            "current_mean": jnp.array(0.0)
        }

        def one_update(i: int, carry: dict[str, Any]) -> dict[str, Any]:
            actor_state = carry["actor_state"]
            qf_state = carry["qf_state"]
            ent_coef_state = carry["ent_coef_state"]
            key = carry["key"]
            info = carry["info"]
            past_std = carry["past_std"]
            
            batch_obs = jax.lax.dynamic_slice_in_dim(data.observations, i * batch_size, batch_size)
            batch_actions = jax.lax.dynamic_slice_in_dim(data.actions, i * batch_size, batch_size)
            batch_next_obs = jax.lax.dynamic_slice_in_dim(data.next_observations, i * batch_size , batch_size)
            batch_rewards = jax.lax.dynamic_slice_in_dim(data.rewards, i * batch_size, batch_size)
            batch_dones = jax.lax.dynamic_slice_in_dim(data.dones, i * batch_size, batch_size)
            batch_discounts = jax.lax.dynamic_slice_in_dim(data.discounts, i * batch_size, batch_size)
            (
                qf_state,
                (qf_loss_value, ent_coef_value),
                key,
                past_std,
                current_mean
            ) = cls.update_critic(
                actor_state,
                qf_state,
                ent_coef_state,
                batch_obs,
                batch_actions,
                batch_next_obs,
                batch_rewards,
                batch_dones,
                batch_discounts,
                key,
                past_std, 
                tau
            )
            qf_state = cls.soft_update(tau, qf_state)
            (actor_state, qf_state, ent_coef_state, actor_loss_value, ent_coef_loss_value, key) = jax.lax.cond(
                (policy_delay_offset + i) % policy_delay == 0,
                # If True:
                cls.update_actor_and_temperature,
                # If False:
                lambda *_: (actor_state, qf_state, ent_coef_state, info["actor_loss"], info["ent_coef_loss"], key),
                actor_state,
                qf_state,
                ent_coef_state,
                batch_obs,
                target_entropy,
                key,
            )
            info = {
                "actor_loss": actor_loss_value,
                "qf_loss": qf_loss_value,
                "ent_coef_loss": ent_coef_loss_value,
                "ent_coef_value": ent_coef_value,
            }

            return {
                "actor_state": actor_state,
                "qf_state": qf_state,
                "ent_coef_state": ent_coef_state,
                "key": key,
                "info": info,
                "past_std": past_std,
                "current_mean": current_mean
            }

        update_carry = jax.lax.fori_loop(0, gradient_steps, one_update, carry)

        return (
            update_carry["qf_state"],
            update_carry["actor_state"],
            update_carry["ent_coef_state"],
            update_carry["key"],
            (
                update_carry["info"]["actor_loss"],
                update_carry["info"]["qf_loss"],
                update_carry["info"]["ent_coef_loss"],
                update_carry["info"]["ent_coef_value"],
                update_carry["past_std"],
                update_carry["current_mean"]
            ),
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

        data = self._convert_to_jax_tuple(data.observations, data.actions, data.next_observations, data.dones, data.rewards, self.discounts)

        (
        self.policy.qf_state,
        self.policy.actor_state,
        self.ent_coef_state,
        self.key,
        (actor_loss_value, qf_loss_value, ent_coef_loss_value, ent_coef_value, self.past_std, current_mean),
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
            self.past_std
            )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/ent_coef_loss", ent_coef_loss_value.item())
        self.logger.record("train/ent_coef", ent_coef_value.item())
        self.logger.record("train/std", float(self.past_std[0]))
        self.logger.record("train/mean", float(current_mean))

    @staticmethod
    @jax.jit
    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        key: jax.Array,
    ):
        key, dropout_key, noise_key = jax.random.split(key, 3)

        def actor_loss(params: flax.core.FrozenDict) -> tuple[jax.Array, jax.Array]:
            dist = actor_state.apply_fn(params, observations)
            actor_actions = dist.sample(seed=noise_key)
            log_prob = dist.log_prob(actor_actions).reshape(-1)
            qf_pi = qf_state.apply_fn(
                qf_state.params,
                observations,
                actor_actions,
                rngs={"dropout": dropout_key},
            )
            # Take min among all critics (mean for droq)
            min_qf_pi = jnp.min(qf_pi[..., 0], axis=0)
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
            
            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean()
            return actor_loss, -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, qf_state, actor_loss_value, key, entropy