import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
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

class SAC_VC(delayed_SAC_wrapper):
    policy_aliases: ClassVar[dict[str, type[SACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SACPolicy,
        # Implement the DSAC NN
        "CustomMlpPolicy": CustomSACPolicy,
    }
    def __init__(self, policy, env, distribution: LatencyDistribution, learning_rate = 0.0003, qf_learning_rate = None, buffer_size = 1000000, learning_starts = 100, 
                 batch_size = 256, tau = 0.005, gamma = 0.99, train_freq = 1, gradient_steps = 1, policy_delay = 1, action_noise = None, 
                 replay_buffer_class = CustomReplayBuffer, replay_buffer_kwargs = None, n_steps = 1, ent_coef = "auto", target_entropy = "auto", 
                 use_sde = False, sde_sample_freq = -1, use_sde_at_warmup = False, stats_window_size = 100, tensorboard_log = None, 
                 policy_kwargs = None, param_resets = None, verbose = 0, seed = None, device = "auto", _init_setup_model = True,
                 learning_rate_alpha = 3e-4, alpha_0 = 0.2):
        
        super().__init__(policy, env, learning_rate, qf_learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, 
                         gradient_steps, policy_delay, action_noise, replay_buffer_class, replay_buffer_kwargs, n_steps, ent_coef, 
                         target_entropy, use_sde, sde_sample_freq, use_sde_at_warmup, stats_window_size, tensorboard_log, policy_kwargs, 
                         param_resets, verbose, seed, device, _init_setup_model, learning_rate_alpha=learning_rate_alpha, alpha_0=alpha_0)
        
        self.latency_probabilities = distribution.latency_probabilities
        self.max_latency = distribution.max_latency
        
        self.replay_buffer.init_obs_buffer(self.max_latency)
        

    @staticmethod
    @partial(jax.jit, static_argnames=["max_latency"])
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
        latencies_prob: jax.Array,
        max_latency: int,
        key: jax.Array,
    ):
        dones = jnp.cumsum(dones, axis = 1)
        dones = (dones > 0).astype(jnp.float32)
        
        key, noise_key, dropout_key_target, dropout_key_current = jax.random.split(key, 4)
        dist = actor_state.apply_fn(actor_state.params, next_observations)
        
        next_actions_sample = dist.sample(seed = noise_key)
        next_log_prob = dist.log_prob(next_actions_sample) 
        next_log_prob = next_log_prob.reshape((-1, max_latency+1))

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
        qf_next_values = qf_state.apply_fn(
            qf_state.target_params,
            next_observations,
            next_actions_sample,
            rngs={"dropout": dropout_key_target},
        ).reshape((2, -1,max_latency+1))        
        next_q_values = jnp.min(qf_next_values, axis=0)
        # td error + entropy term
        next_q_values = next_q_values - ent_coef_value * next_log_prob
        target_q_values = rewards + (1 - dones) * discounts * next_q_values
        
        target_q_values = target_q_values * latencies_prob
        target_q_values = jnp.sum(target_q_values, axis=1)[:, None]

        

        def mse_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
            current_q_values = qf_state.apply_fn(params, observations, actions, rngs={"dropout": dropout_key})
            
            return 0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()

        qf_loss_value, grads = jax.value_and_grad(mse_loss, has_aux=False)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, ent_coef_value),
            key,
        )
    
    @classmethod
    @partial(jax.jit, static_argnames=["cls", "gradient_steps", "policy_delay", "policy_delay_offset", "max_latency"])
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
        max_latency: int,
        latency_probabilities: tuple, 
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
        }

        def one_update(i: int, carry: dict[str, Any]) -> dict[str, Any]:
            actor_state = carry["actor_state"]
            qf_state = carry["qf_state"]
            ent_coef_state = carry["ent_coef_state"]
            key = carry["key"]
            info = carry["info"]
            
            batch_obs = jax.lax.dynamic_slice_in_dim(data.observations, i * batch_size, batch_size)
            batch_actions = jax.lax.dynamic_slice_in_dim(data.actions, i * batch_size, batch_size)
            batch_next_obs = jax.lax.dynamic_slice_in_dim(data.next_observations, i * batch_size , batch_size).reshape((batch_size*(max_latency+1), -1))
            batch_rewards = jax.lax.dynamic_slice_in_dim(data.rewards, i * batch_size, batch_size*(max_latency+1)).reshape((batch_size, (max_latency+1)))
            batch_dones = jax.lax.dynamic_slice_in_dim(data.dones, i * batch_size, batch_size*(max_latency+1)).reshape((batch_size, (max_latency+1)))
            batch_discounts = jax.lax.dynamic_slice_in_dim(data.discounts, i * batch_size, batch_size)
            batch_latencies = jnp.array(latency_probabilities, copy = False)
            (
                qf_state,
                (qf_loss_value, ent_coef_value),
                key,
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
                batch_latencies,
                max_latency,
                key,
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
                self.discounts = jnp.full((batch_size * gradient_steps,), self.gamma, dtype=jnp.float32)
                self.discounts = jnp.vander(self.discounts, self.max_latency + 2, increasing=True)[:, 1:]
                
        else:
            self.discounts = data.discounts.reshape(-1)

        if self.latency_probabilities.shape != (batch_size, self.max_latency + 1):
            self.latency_probabilities = jnp.tile(self.latency_probabilities, (batch_size, 1))
        
        data = self._convert_to_jax_tuple(data.observations, data.actions, data.next_observations, data.dones, data.rewards, self.discounts)
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
            self.max_latency, 
            self.latency_probabilities,
            )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/ent_coef_loss", ent_coef_loss_value.item())
        self.logger.record("train/ent_coef", ent_coef_value.item())
