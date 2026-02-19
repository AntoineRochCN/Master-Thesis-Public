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