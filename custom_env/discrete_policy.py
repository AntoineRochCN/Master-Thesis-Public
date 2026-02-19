from collections.abc import Sequence
from typing import Callable, Optional
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from sbx.sac.policies import SACPolicy
tfd = tfp.distributions
import optax
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces
from flax.training.train_state import TrainState
from sbx.common.type_aliases import RLTrainState

class Flatten(nn.Module):
    """
    Equivalent to PyTorch nn.Flatten() layer.
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape((x.shape[0], -1))

class DiscreteCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
        x = nn.Dense(self.output_dim)(x)
        return x

class DiscreteVectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        vmap_critic = nn.vmap(
            DiscreteCritic,
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
            output_dim=self.output_dim,
        )(obs)
        return q_values

class DiscreteDistributionalCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1
    kernel_init: Callable = jax.nn.initializers.orthogonal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        for n_units in self.net_arch:
            x = nn.Dense(n_units, kernel_init=self.kernel_init)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
        mean = nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)
        log_std = nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)
        log_std = nn.softplus(log_std)
        
        return mean, log_std

class DiscreteDistributionalVectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        vmap_critic = nn.vmap(
            DiscreteDistributionalCritic,
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
            output_dim=self.output_dim,
        )(obs)
        return q_values


class DiscreteActor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    action_column: int
    mask: jax.Array
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  
        z = Flatten()(x)
        for n_units in self.net_arch:
            z = nn.Dense(n_units)(z)
            z = self.activation_fn(z)
        z = nn.Dense(self.action_dim)(z)
        z = z + jax.lax.stop_gradient(self.mask[x[:,self.action_column].astype(jnp.int32)])
        
        dist = tfd.Categorical(logits=z)
        
        return dist
    
class SAC_DPolicy(SACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch = None, dropout_rate = 0, layer_norm = False, 
                 activation_fn = nn.relu, use_sde = False, log_std_init = -3, use_expln = False, clip_mean = 2, 
                 features_extractor_class=None, features_extractor_kwargs = None, normalize_images = True, optimizer_class = optax.adam, 
                 optimizer_kwargs = None, n_critics = 2, share_features_extractor = False, actor_class = DiscreteActor, vector_critic_class = DiscreteVectorCritic,
                 action_dim = None, action_col = 15):
        """
        Modification of the vector critic definition to use a CustomVectorCritic (output of dim 2)
        """
        super().__init__(observation_space, action_space, lr_schedule, net_arch, dropout_rate, layer_norm, activation_fn, use_sde, log_std_init, 
                         use_expln, clip_mean, features_extractor_class, features_extractor_kwargs, normalize_images, optimizer_class, 
                         optimizer_kwargs, n_critics, share_features_extractor, actor_class, vector_critic_class)
        
        self.action_dim = action_dim
        self.action_col = action_col
        
        
    def build(self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float) -> jax.Array:
        key, actor_key, qf_key, dropout_key = jax.random.split(key, 4)
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])

        self.actor = self.actor_class(
            action_dim=self.action_dim,
            net_arch=self.net_arch_pi,
            activation_fn=self.activation_fn,
            mask = jnp.array([[0,0,0], [0,0,1], [0,1,0]], dtype=jnp.float32) * (-1e9),
            action_column = self.action_col
        )
        self.actor.reset_noise = self.reset_noise

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
            output_dim = self.action_dim
        )

        optimizer_class_qf = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=qf_learning_rate, **self.optimizer_kwargs
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
            ),
            target_params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
            ),
            tx=optimizer_class_qf,
        )

        self.actor.apply = jax.jit(self.actor.apply)  
        self.qf.apply = jax.jit( 
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm"),
        )

        return key
    
