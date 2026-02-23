import jax.numpy as jnp
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from sbx.sac.policies import SACPolicy
from typing import TypeVar, ClassVar
from utils_env import *
from utils_SAC import CustomSACPolicy, delayed_SAC_wrapper
import warnings
warnings.filterwarnings("ignore")
SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")

class SAC_delayed(delayed_SAC_wrapper):
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