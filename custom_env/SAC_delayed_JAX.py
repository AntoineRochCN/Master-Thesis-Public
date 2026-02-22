import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
import optax
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from sbx.sac.policies import SACPolicy
from sbx.common.type_aliases import RLTrainState
from sbx import SAC
from typing import TypeVar, ClassVar
from functools import partial
from discrete_policy import SAC_DPolicy
import warnings
from env import *
from buffer import *
from jax import lax
from record_utils import *
from SAC_common import *
from sbx.common.type_aliases import RLTrainState
import time
from flax import serialization

warnings.filterwarnings("ignore")
SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")

class SAC_delayed_JAX(SAC):
    
    policy_aliases: ClassVar[dict[str, type[SACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SACPolicy,
        "DiscretePolicy": SAC_DPolicy
    }
    
    def __init__(self, policy, env, env_rec, latency_manager: JaxLatencyEnv, latency_manager_test: JaxLatencyEnv,buffer : CustomBufferBis, learning_rate = 0.0003, 
                 qf_learning_rate = None, buffer_size = 1000000, learning_starts = 100, 
                 batch_size = 256, tau = 0.005, gamma = 0.99, train_freq = 1, gradient_steps = 1, policy_delay = 1, action_noise = None, 
                 replay_buffer_class = None, replay_buffer_kwargs = None, n_steps = 1, ent_coef = "auto", target_entropy = "auto", 
                 use_sde = False, sde_sample_freq = -1, use_sde_at_warmup = False, stats_window_size = 100, tensorboard_log = None, 
                 policy_kwargs = None, param_resets = None, verbose = 0, seed = None, device = "auto", _init_setup_model = True,
                 learning_rate_alpha = 3e-4, alpha_0 = 0.2, n_episodes_warmup = 3 * 10**2, n_training_warmup = 10**5):
        
        if policy_kwargs == None:
            policy_kwargs = {"action_dim": env.action_length}
        else:
            policy_kwargs.update({"action_dim": env.action_length})
        
        replacement_env = foolish_env(env.obs_length, env.action_length)

        super().__init__(policy, replacement_env, learning_rate, qf_learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, 
                         gradient_steps, policy_delay, action_noise, replay_buffer_class, replay_buffer_kwargs, n_steps, ent_coef, 
                         target_entropy, use_sde, sde_sample_freq, use_sde_at_warmup, stats_window_size, tensorboard_log, policy_kwargs, 
                         param_resets, verbose, seed, device, _init_setup_model)

        self.env = env
        self.env_rec = env_rec
        self.buffer = buffer
        
        params = {"log_ent_coef": jnp.log(alpha_0)}
        self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=params,
                tx=optax.adam(
                    learning_rate=learning_rate_alpha,
                ),
        )

        self.action_dim = env.action_length
        self.latency_manager = latency_manager
        self.latency_manager_test = latency_manager_test
        self.start_entropy = 0.5 * target_entropy
        self.n_episodes_warmup, self.n_training_warmup = n_episodes_warmup, n_training_warmup

    @classmethod
    @partial(jax.jit, static_argnames = ["cls","n_episodes", "n_networks_training", "act_dim", "tau", "batch_size", "obs_dim"], 
             donate_argnames = ["env", "actor_state", "qf_state"])
    def rollout_warmup(cls, env: EnvDataBinance, n_episodes, n_networks_training, actor_state, qf_state, ent_coef_state, act_dim, tau, batch_size, past_entropy, obs_dim,
                       discounts):
        step_carry = env.step_carry
        
        def run_one_ep(carry, items):
            step_carry, mask = carry
            obs = step_carry.current_obs
            action = step_carry.opt_pf_states[step_carry.timestep-1]
            
            new_obs, reward, done, truncated, step_carry = step_env(step_carry, jnp.asarray([action]))
            done_or_trunc = jnp.logical_or(done, truncated)

            transition_array = jnp.concatenate([obs.ravel(), new_obs.ravel(), jnp.atleast_1d(action), 
                                                jnp.atleast_1d(reward), jnp.atleast_1d(done_or_trunc)])

            transition_array = transition_array * mask
            new_mask = mask & ~done_or_trunc

            return (step_carry, new_mask), (transition_array, new_mask)
        
        def run_all_eps(step_carry: StepCarry, current_keys):
            reset_key = current_keys
            
            (step_carry, _), (tmp_buffer, mask_array) = lax.scan(run_one_ep, (step_carry, True), length=env.ep_length)
            step_carry = reset_step_carry(env, reset_key)
            return step_carry, (tmp_buffer, mask_array)
        
        keys = jax.random.split(env.key, n_episodes * 1 + 1)
        key, prepared_keys = keys[0], keys[1:]

        step_carry, (transition_buffer, mask_array) = jax.lax.scan(run_all_eps, step_carry, prepared_keys)
        
        transition_buffer = transition_buffer.reshape(-1, transition_buffer.shape[-1])
        mask_array = mask_array.reshape(-1)
        indices = jnp.argsort(mask_array, descending=True)
        transition_buffer = transition_buffer[indices]
        max_pos = jnp.sum(mask_array).astype(jnp.int32) - 1

        def train_networks(carry, key):
            qf_state = carry[0]
            actor_state = carry[1]

            idx_sampling = jax.random.randint(key, shape = (batch_size), minval= 0 , maxval=max_pos)
            
            batch = transition_buffer[idx_sampling]
            batch_obs = batch[:, :obs_dim]
            batch_next_obs = batch[:, obs_dim:obs_dim*2].reshape(-1, obs_dim)
            batch_actions = batch[:, obs_dim*2: obs_dim*2 + 1]
            batch_rewards = batch[:, obs_dim*2 + 1]
            batch_dones = batch[:, obs_dim*2 + 2]
             
            (
                qf_state,
                (qf_loss_value, Q_mean),
                key,
            ) = cls.warmup_train_critic(actor_state, qf_state, batch_obs,
                                    batch_actions, batch_next_obs, batch_rewards,
                                    batch_dones, discounts, key, act_dim)
            qf_state = cls.soft_update(tau, qf_state)

            (actor_state, qf_state, actor_loss_value, entropy) = cls.warmup_update_actor(
                actor_state,
                qf_state,
                batch_obs,
                batch_actions
            )
            
            return (qf_state, actor_state, entropy), jnp.hstack([qf_loss_value, Q_mean, actor_loss_value, entropy])
        keys = jax.random.split(key, n_networks_training + 1)
        key, sample_keys = keys[0], keys[1:]

        (qf_state, actor_state, entropy), rec_losses = jax.lax.scan(train_networks, (qf_state, actor_state, past_entropy), sample_keys)

        env = env.replace(
                step_carry = step_carry,
                key = key
            )

        return env, actor_state, qf_state, entropy, rec_losses
    
    @classmethod
    @partial(jax.jit, static_argnames = ["cls", "total_timesteps", "buffer_shape", "obs_dim", "act_dim", "gradient_steps", 
                                         "tau", "policy_delay", "batch_size", "collect_timesteps", "init_timesteps", 
                                         "eval_freq", "n_eval", "target_entropy", "log_interval", "n_episodes_warmup", "n_networks_training_warmup"], donate_argnames = ["buffer"])
    def learning_wrapper(cls, total_timesteps, collect_timesteps, buffer_shape, env, eval_env, buffer: CustomBufferLatency, obs_dim, act_dim, gradient_steps, discounts,
                                 tau, target_entropy, policy_delay, batch_size, qf_state: RLTrainState, actor_state: TrainState, ent_coef_state: TrainState,
                                 key: jax.Array, init_timesteps, eval_freq: int, n_eval: int, log_interval: int, latency_manager: JaxLatencyEnv, latency_manager_test: JaxLatencyEnv, start_entropy,
                                 n_episodes_warmup, n_networks_training_warmup):
        
        env_rec = EnvRecord(pos = 0, buffer=jnp.zeros(((init_timesteps + collect_timesteps * total_timesteps) // 10, 3), dtype=jnp.float32)) #10: empirical assumption that an env will last in mean at least 10 timesteps
        
        loss_rec = LossRecord(pos = 0, history= jnp.zeros((total_timesteps // log_interval, 6 + act_dim), dtype=jnp.float32))

        env, actor_state, qf_state, start_entropy, rec_losses = cls.rollout_warmup(env, n_episodes_warmup, n_networks_training_warmup, actor_state, 
                                                                                              qf_state, ent_coef_state, act_dim, tau, batch_size, start_entropy, obs_dim, 
                                                                                              discounts, 
                                                                                              )
        
        key, test_key = jax.random.split(key)

        test_rec = TestRecord(pos=0, reward_mat=jnp.zeros((total_timesteps // eval_freq, n_eval)), length_mat=jnp.zeros((total_timesteps // eval_freq, n_eval)), 
                              pf_mat = jnp.zeros((total_timesteps // eval_freq, n_eval)), sharpe_mat = jnp.zeros((total_timesteps // eval_freq, n_eval)), 
                              opt_ratio_mat = jnp.zeros((total_timesteps // eval_freq, n_eval)), key = test_key)

        
        carry = SAC_Main_Carry(env = env, env_test = eval_env, buffer=buffer, qf_state= qf_state, actor_state= actor_state, ent_coef_state=ent_coef_state, n_updates=0, key = key, 
                        env_rec = env_rec, loss_rec = loss_rec, test_rec=test_rec, latency_manager=latency_manager, std = None, entropy= start_entropy) 
        
        num_outer_loops = total_timesteps // eval_freq
        num_loop_rec = eval_freq // log_interval
        num_loop_policy = log_interval // policy_delay
        num_loop_rollout = policy_delay
        num_loop_critic = gradient_steps

        default_entry_rollout = jnp.zeros((batch_size, obs_dim), dtype=jnp.float32)
        default_entry_rollout_bis = (0,0,0)
        default_entry_policy = (0,0,0,0,0)

        
        def critic_loop(items, _):
            carry = items[0]
            actor_state = carry.actor_state
            qf_state = carry.qf_state
            buffer = carry.buffer
            ent_coef_state = carry.ent_coef_state
            key = carry.key
            past_entropy = carry.entropy
            data, key = buffer.sample(buffer.buffer, batch_size, buffer.pos, key, discounts, obs_dim, buffer.full, buffer.buffer_size)
            
            batch_obs = jax.lax.dynamic_slice_in_dim(data.observations, batch_size, batch_size)
            batch_actions = jax.lax.dynamic_slice_in_dim(data.actions, batch_size, batch_size)
            batch_next_obs = jax.lax.dynamic_slice_in_dim(data.next_observations, batch_size , batch_size)
            batch_rewards = jax.lax.dynamic_slice_in_dim(data.rewards, batch_size, batch_size)
            batch_dones = jax.lax.dynamic_slice_in_dim(data.dones, batch_size, batch_size)
            batch_discounts = discounts

            (
                qf_state,
                (qf_loss_value, ent_coef_value, Q_mean),
                key,
            ) = cls.update_critic(actor_state, qf_state, ent_coef_state, batch_obs,
                                    batch_actions, batch_next_obs, batch_rewards,
                                    batch_dones, batch_discounts, key, act_dim,  
                                    past_entropy)
            qf_state = cls.soft_update(tau, qf_state)

            carry = carry.replace(qf_state = qf_state, key=key)

            return (carry, batch_obs, (qf_loss_value, ent_coef_value, Q_mean)), None

        def rollout_loop(items, _):
            carry = items[0]
            env = carry.env
            buffer = carry.buffer
            actor_state = carry.actor_state
            env_rec = carry.env_rec
            latency_manager = carry.latency_manager

            (tmp_buffer, buf_pos, buf_pos_init, step_carry, env_rec, latency_manager, done_or_trunc, full, reset_key, key) = rollout_std(
                        env, collect_timesteps, buffer_shape, actor_state, env_rec, latency_manager, 
                        buffer.buffer, buffer.pos, buffer.past_begin_pos, buffer.full)
            
            tmp_buffer, env_rec, step_carry, latency_manager = jax.lax.cond(
                done_or_trunc,
                lambda tb, er, sc, lm: on_done_processor_std(tb, er, sc, lm, reset_key, env), #new_init_pos
                lambda tb, er, sc, lm: (tb, er, sc, lm),
                tmp_buffer, env_rec, step_carry, latency_manager
            )

            env = env.replace(
                    step_carry = step_carry,
                    key = key
                )
                        
            buffer = buffer.replace(buffer=tmp_buffer, pos=buf_pos, past_begin_pos=buf_pos_init, full = full)

            carry = carry.replace(env=env, buffer=buffer, env_rec=env_rec, latency_manager=latency_manager)

            (carry, batch_obs, (qf_loss_value, ent_coef_value, Q_mean)), _ = jax.lax.scan(critic_loop, (carry, default_entry_rollout, default_entry_rollout_bis), None, length=num_loop_critic)

            return (carry, batch_obs, (qf_loss_value, ent_coef_value, Q_mean)), None

        def policy_loop(items, _):
            carry = items[0]
            actor_state = carry.actor_state
            qf_state = carry.qf_state
            ent_coef_state = carry.ent_coef_state
            key = carry.key
            past_entropy = carry.entropy    

            (carry, batch_obs, (qf_loss_value, ent_coef_value, Q_mean)), _ = jax.lax.scan(rollout_loop, (carry, default_entry_rollout, default_entry_rollout_bis), None, length=num_loop_rollout)

            (actor_state, qf_state, ent_coef_state, 
            actor_loss_value, ent_coef_loss_value, 
            key, entropy) = cls.update_actor_and_temperature(actor_state, qf_state, ent_coef_state, batch_obs,
                                                            target_entropy, key,  past_entropy)
            carry = carry.replace(actor_state = actor_state, qf_state = qf_state, ent_coef_state = ent_coef_state,
                                key = key, entropy = entropy)

            return (carry, (qf_loss_value, ent_coef_value, Q_mean, actor_loss_value, ent_coef_loss_value)), None

        def env_rec_loop(carry: SAC_Main_Carry, _):
            (carry, (qf_loss_value, ent_coef_value, Q_mean, actor_loss_value, ent_coef_loss_value)), _ = jax.lax.scan(policy_loop, (carry, default_entry_policy), None, length=num_loop_policy)

            return carry, (qf_loss_value, ent_coef_value, Q_mean, actor_loss_value, ent_coef_loss_value, carry.entropy, 0.0)
        
        def outer_step(items, _):
            carry, latency_manager = items[0], items[1]
            carry, (qf_loss_value, ent_coef_value, Q_mean, actor_loss_value, ent_coef_loss_value, entropies, stds) = jax.lax.scan(env_rec_loop, carry, None, length=num_loop_rec)

            loss_rec = carry.loss_rec
            loss_rec = loss_rec.replace(history=jax.lax.dynamic_update_slice(loss_rec.history,
                                                                             jnp.column_stack((qf_loss_value, actor_loss_value, ent_coef_loss_value, ent_coef_value, Q_mean, entropies, stds)),
                                                                             (loss_rec.pos, 0)),
                                        pos = loss_rec.pos + num_loop_rec)
            
            carry = carry.replace(loss_rec = loss_rec)
            
            env_test = carry.env_test
            actor_state = carry.actor_state
            test_rec = carry.test_rec
            
            test_rec, env_test, latency_manager = eval_policy_env(env_test, n_eval, actor_state, test_rec, latency_manager)
            carry = carry.replace(env_test=env_test, test_rec=test_rec)
            
            return (carry, latency_manager), None
        
        (carry, _), _ = jax.lax.scan(outer_step, (carry, latency_manager_test), None, length=num_outer_loops)
        
        return carry, rec_losses
    
    def learn_jax(
        self: SelfOffPolicyAlgorithm,
        total_timesteps: int,
        n_eval = 10, 
        eval_freq = 2500,
        log_interval: int = 4,
        tb_log_name: str = "run",
        save_model: bool = False, 
        save_path: str = None
    ):
        
        t1 = time.time()
        carry, loss_warmup = self.learning_wrapper(total_timesteps, self.train_freq.frequency, self.buffer.buffer.shape[1], self.env, self.env_rec, self.buffer, self.env.obs_length, self.env.action_length, self.gradient_steps, self.gamma,
                                 self.tau, self.target_entropy, self.policy_delay, self.batch_size, self.policy.qf_state, self.policy.actor_state, self.ent_coef_state, self.key, self.learning_starts, 
                                 eval_freq, n_eval, log_interval, self.latency_manager, self.latency_manager_test, self.start_entropy, self.n_episodes_warmup, self.n_training_warmup)
        total_time = time.time() - t1
        print("Total training time: {:.2f}s, time per iteration: {:.3f}us".format(total_time, total_time/(total_timesteps * self.train_freq.frequency)*10**6) )
        
        if self.tensorboard_log != None:
            tensorboard_logger(carry.env_rec, carry.loss_rec, carry.test_rec, self.tensorboard_log, tb_log_name, self.env.obs_length, log_interval = log_interval, 
                               collect_timesteps=self.train_freq.frequency, test_frequency=eval_freq, loss_warmup=loss_warmup)
        if save_model:
            model_data = {"actor": carry.actor_state.params}
            serialized_weights = serialization.to_bytes(model_data)

            with open(save_path, "wb") as file:
                file.write(serialized_weights)
        return carry

    @staticmethod
    @partial(jax.jit, static_argnames = ["action_dim"])
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
        action_dim: int, 
        past_entropy: jax.Array
    ):  
    
        key, dropout_key_target, dropout_key_current = jax.random.split(key, 3)
        dist = actor_state.apply_fn(actor_state.params, next_observations)
        prob = jnp.clip(dist.probs_parameter(), 1e-7, 1.0 - 1e-7).reshape((-1, action_dim)) # (B, |A|)

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
        mean_next = qf_state.apply_fn(
            qf_state.target_params,
            next_observations,
            rngs={"dropout": dropout_key_target},
        )
        mean_next = mean_next.reshape(2, -1, action_dim) # (2, B, |A|)

        C1 = ent_coef_value * past_entropy # (1)
        C2 = (1 - dones) * discounts # (B)

        target_Q_values = (mean_next * prob[None, ...]).sum(axis = 2) # (2, B)
        target_Q_values = jnp.min(target_Q_values, axis=0) # (B)

        target_Q_values = target_Q_values - C1 #(B)
        target_Q_values = jax.lax.stop_gradient(rewards + C2 * target_Q_values) #(B)
        
        def mse_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
            
            current_q_values = qf_state.apply_fn(params, observations, rngs={"dropout": dropout_key}) #(2, B, |A|) x2 
            current_q_values = current_q_values.reshape(2, -1, action_dim)

            observed_q_values = current_q_values[:,jnp.arange(actions.shape[0]), actions.astype(jnp.int32).squeeze(-1)] #(2, B)

            q_loss = (0.5*(target_Q_values[jnp.newaxis, :] - observed_q_values)**2).mean(axis = 1).sum() 
            
            return q_loss, observed_q_values.mean()
        
        (qf_loss_value, Q_mean), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)
        
        return (
            qf_state,
            (qf_loss_value, ent_coef_value, Q_mean),
            key,
        )
    
    @staticmethod
    @jax.jit
    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        key: jax.Array,
        past_entropy: jax.Array
    ):
        key, dropout_key = jax.random.split(key, 2)
        beta = 0.5

        def actor_loss(params: flax.core.FrozenDict) -> tuple[jax.Array, jax.Array]:
            dist = actor_state.apply_fn(params, observations)
            prob = jnp.clip(dist.probs_parameter(), 1e-7, 1.0 - 1e-7)
            log_prob = jnp.log(prob)
            
            qf_pi = qf_state.apply_fn(
                qf_state.params,
                observations,
                rngs={"dropout": dropout_key},
            )
            min_qf_pi = jnp.min(qf_pi, axis=0)
            ent_coef_value =  lax.stop_gradient(ent_coef_state.apply_fn({"params": ent_coef_state.params}))
            
            entropy = lax.stop_gradient(-(prob * log_prob).sum(axis = 1).mean(axis = 0))
            loss =  (prob * (ent_coef_value * log_prob - min_qf_pi)).sum(axis = 1).mean() + (entropy - past_entropy) ** 2 * beta / 2
            
            return loss, entropy

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, qf_state, actor_loss_value, key, entropy
    
    @classmethod
    @partial(jax.jit, static_argnames = ["cls"])
    def update_actor_and_temperature(
        cls,
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        target_entropy: ArrayLike,
        key: jax.Array,
        past_entropy: jax.Array
        
    ):
        (actor_state, qf_state, actor_loss_value, key, entropy) = cls.update_actor(
            actor_state,
            qf_state,
            ent_coef_state,
            observations,
            key,
            past_entropy,
        )
        
        ent_coef_state, ent_coef_loss_value = update_temperature(target_entropy, ent_coef_state, entropy)
        
        return actor_state, qf_state, ent_coef_state, actor_loss_value, ent_coef_loss_value, key, entropy
    
    @staticmethod
    @partial(jax.jit, static_argnames = ["act_dim"])
    def warmup_train_critic(actor_state, qf_state, batch_obs,
                            batch_actions, batch_next_obs, batch_rewards,
                            batch_dones, discounts, key, act_dim, 
                            ):
        
        key, dropout_key_target, dropout_key_current = jax.random.split(key, 3)
        mean_next = qf_state.apply_fn(
            qf_state.target_params,
            batch_next_obs,
            rngs={"dropout": dropout_key_target},
        )
        mean_next = mean_next.reshape(2, -1, act_dim)

        dist = actor_state.apply_fn(actor_state.params, batch_next_obs)
        prob = jnp.clip(dist.probs_parameter(), 1e-7, 1.0 - 1e-7).reshape((-1, act_dim))

        target_Q_values = (jnp.min(mean_next, axis=0) * prob).sum(axis = -1)
        target_Q_values = jax.lax.stop_gradient(target_Q_values * (1-batch_dones) * discounts + batch_rewards)
        
        def mse_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
            current_q_values = qf_state.apply_fn(params, batch_obs, rngs={"dropout": dropout_key}) #(2, B, |A|) x2 
            current_q_values = current_q_values.reshape(2, -1, act_dim)
            observed_q_values = current_q_values[:,jnp.arange(batch_actions.shape[0]), batch_actions.astype(jnp.int32).squeeze(-1)] #(2, B)

            loss = (0.5 * (target_Q_values[None, ...] - observed_q_values)**2).mean(axis = 1).sum()
            
            return loss , observed_q_values.mean()
        
        (qf_loss_value, Q_mean), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, Q_mean),
            key,
        )        

    def warmup_update_actor(actor_state,
                qf_state,
                batch_obs,
                batch_actions):
        
        def actor_loss(params: flax.core.FrozenDict) -> tuple[jax.Array, jax.Array]:
            dist = actor_state.apply_fn(params, batch_obs)
            logits = dist.logits
            
            entropy = dist.entropy().mean()
            
            actor_loss = optax.losses.softmax_cross_entropy_with_integer_labels( logits, batch_actions.reshape(-1).astype(jnp.int32)).sum()
                
            return actor_loss, entropy
        
        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        
        actor_state = actor_state.apply_gradients(grads=grads)
        
        return actor_state, qf_state, actor_loss_value, entropy