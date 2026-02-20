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
import warnings
try: 
    from discrete_policy import SAC_DPolicy, DiscreteDistributionalVectorCritic
    from record_utils import *
    from env import *
    from buffer import *
    from SAC_common import *
except:
    from .discrete_policy import SAC_DPolicy, DiscreteDistributionalVectorCritic
    from .record_utils import *
    from .env import *
    from .buffer import *
    from .SAC_common import *

from jax import lax
from sbx.common.type_aliases import RLTrainState
import time
from flax import serialization

warnings.filterwarnings("ignore")
SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")

class DADAC_JAX(SAC):
    
    policy_aliases: ClassVar[dict[str, type[SACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SACPolicy,
        "DiscretePolicy": SAC_DPolicy
    }
    
    def __init__(self, policy, env: EnvDataBinance, env_rec, latency_manager: JaxLatencyEnv, latency_manager_test: JaxLatencyEnv,buffer : CustomBufferLatency, learning_rate = 0.0003, 
                 qf_learning_rate = None, buffer_size = 1000000, learning_starts = 100, 
                 batch_size = 256, tau = 0.005, gamma = 0.99, train_freq = 1, gradient_steps = 1, policy_delay = 1, action_noise = None, 
                 replay_buffer_class = None, replay_buffer_kwargs = None, n_steps = 1, ent_coef = "auto", target_entropy = "auto", 
                 use_sde = False, sde_sample_freq = -1, use_sde_at_warmup = False, stats_window_size = 100, tensorboard_log = None, 
                 policy_kwargs = None, param_resets = None, verbose = 0, seed = None, device = "auto", _init_setup_model = True,
                 learning_rate_alpha = 3e-4, alpha_0 = 0.2, n_episodes_warmup = 3 * 10**2, n_training_warmup = 10**5):
        
        if policy_kwargs == None:
            policy_kwargs = {"action_dim": env.action_length, "vector_critic_class": DiscreteDistributionalVectorCritic}
        else:
            policy_kwargs.update({"action_dim": env.action_length, "vector_critic_class": DiscreteDistributionalVectorCritic})
        
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
        self.start_std, self.start_entropy = jnp.ones((2, self.action_dim)), 0.5 * target_entropy
        self.n_episodes_warmup, self.n_training_warmup = n_episodes_warmup, n_training_warmup


    @classmethod
    @partial(jax.jit, static_argnames = ["cls","n_episodes", "n_networks_training", "act_dim", "tau", "batch_size", "obs_dim"], 
             donate_argnames = ["env", "actor_state", "qf_state"])
    def rollout_warmup(cls, env: EnvDataBinance, n_episodes, n_networks_training, actor_state, qf_state, act_dim, tau, batch_size, past_entropy, obs_dim, past_std,
                       discounts):
        step_carry = env.step_carry
        
        def run_one_ep(carry, items):
            step_carry, mask = carry
            obs = step_carry.current_obs
            action = step_carry.opt_pf_states[step_carry.timestep-1]

            new_obs, reward, done, truncated, step_carry = EnvDataBinance.step_jax(step_carry, jnp.asarray([action]))
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
            past_std = carry[2]

            idx_sampling = jax.random.randint(key, shape = (batch_size), minval= 0 , maxval=max_pos)
            
            batch = transition_buffer[idx_sampling]
            batch_obs = batch[:, :obs_dim]
            batch_next_obs = batch[:, obs_dim:obs_dim*2].reshape(-1, obs_dim)
            batch_actions = batch[:, obs_dim*2: obs_dim*2 + 1]
            batch_rewards = batch[:, obs_dim*2 + 1]
            batch_dones = batch[:, obs_dim*2 + 2]
            
            (
                qf_state,
                (qf_loss_value, Q_mean, new_std),
                key,
            ) = cls.warmup_train_critic(actor_state, qf_state, batch_obs,
                                    batch_actions, batch_next_obs, batch_rewards,
                                    batch_dones, discounts, key, act_dim, 
                                    past_std, tau)
            qf_state = cls.soft_update(tau, qf_state)

            (actor_state, qf_state, actor_loss_value, entropy) = cls.warmup_update_actor(
                actor_state,
                qf_state,
                batch_obs,
                batch_actions
            )
            
            return (qf_state, actor_state, new_std, entropy), jnp.hstack([qf_loss_value, Q_mean, actor_loss_value, entropy])
        keys = jax.random.split(key, n_networks_training + 1)
        key, sample_keys = keys[0], keys[1:]

        (qf_state, actor_state, past_std, entropy), rec_losses = jax.lax.scan(train_networks, (qf_state, actor_state, past_std, past_entropy), sample_keys)

        env = env.replace(
                step_carry = step_carry,
                key = key
            )

        return env, actor_state, qf_state, past_std, entropy, rec_losses, transition_buffer, max_pos
    
    @classmethod
    @partial(jax.jit, static_argnames = ["cls", "total_timesteps", "buffer_shape", "obs_dim", "act_dim", "gradient_steps", 
                                         "tau", "policy_delay", "batch_size", "collect_timesteps", "init_timesteps", 
                                         "eval_freq", "n_eval", "target_entropy", "log_interval", "n_episodes_warmup", "n_networks_training_warmup"], donate_argnames = ["buffer"])
    def learning_wrapper(cls, total_timesteps, collect_timesteps, buffer_shape, env, eval_env, buffer: CustomBufferLatency, obs_dim, act_dim, gradient_steps, discounts,
                                 tau, target_entropy, policy_delay, batch_size, qf_state: RLTrainState, actor_state: TrainState, ent_coef_state: TrainState,
                                 key: jax.Array, init_timesteps, eval_freq: int, n_eval: int, log_interval: int, latency_manager: JaxLatencyEnv, 
                                 latency_manager_test: JaxLatencyEnv, start_std, start_entropy, 
                                 n_episodes_warmup: int, n_networks_training_warmup: int):
        
        env_rec = EnvRecord(pos = 0, buffer=jnp.zeros(((init_timesteps + collect_timesteps * total_timesteps) // 10, 3), dtype=jnp.float32)) #10: empirical assumption that an env will last in mean at least 10 timesteps
    
        loss_rec = LossRecord(pos = 0, history= jnp.zeros((total_timesteps // log_interval, 6 + act_dim), dtype=jnp.float32))
        
        
        env, actor_state, qf_state, start_std, start_entropy, rec_losses, transtion_buffer, max_pos = cls.rollout_warmup(env, n_episodes_warmup, n_networks_training_warmup, actor_state, 
                                                                                              qf_state, act_dim, tau, batch_size, start_entropy, obs_dim, 
                                                                                              start_std, discounts
                                                                                              )
        
        
        buffer = buffer.replace(buffer = update_latency_buffer(buffer.buffer, buffer.pos, buffer.max_latency, buffer_shape,
                                                               jnp.arange(transtion_buffer.shape[0]) < max_pos, transtion_buffer, transtion_buffer.shape[0]), 
                                                               pos = max_pos, past_begin_pos = max_pos)
        
        key, test_key = jax.random.split(key)

        test_rec = TestRecord(pos=0, reward_mat=jnp.zeros((total_timesteps // eval_freq, n_eval)), length_mat=jnp.zeros((total_timesteps // eval_freq, n_eval)), 
                              pf_mat = jnp.zeros((total_timesteps // eval_freq, n_eval)), sharpe_mat = jnp.zeros((total_timesteps // eval_freq, n_eval)), 
                              opt_ratio_mat = jnp.zeros((total_timesteps // eval_freq, n_eval)),key = test_key)

        
        carry = SAC_Main_Carry(env = env, env_test = eval_env, buffer=buffer, qf_state= qf_state, actor_state= actor_state, ent_coef_state=ent_coef_state, n_updates=0, key = key, 
                        env_rec = env_rec, loss_rec = loss_rec, test_rec=test_rec, latency_manager=latency_manager, std= start_std, entropy= start_entropy) 
        
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
            latency_manager = carry.latency_manager
            past_std = carry.std
            
            data, key = buffer.sample(buffer.buffer, batch_size, buffer.pos, key, discounts, obs_dim, buffer.full, buffer.buffer_size, buffer.max_latency, latency_manager.action_dim)
            
            batch_obs = jax.lax.dynamic_slice_in_dim(data.observations, batch_size, batch_size)
            batch_actions = jax.lax.dynamic_slice_in_dim(data.actions, batch_size, batch_size)
            batch_next_obs = jax.lax.dynamic_slice_in_dim(data.next_observations, batch_size , batch_size).reshape((batch_size*(buffer.max_latency), -1))
            batch_rewards = jax.lax.dynamic_slice_in_dim(data.rewards, batch_size, batch_size).reshape((batch_size, (buffer.max_latency)))
            batch_dones = jax.lax.dynamic_slice_in_dim(data.dones, batch_size, batch_size).reshape((batch_size, (buffer.max_latency)))
            batch_discounts = jax.lax.dynamic_slice_in_dim(data.discounts, batch_size, batch_size)
            batch_latencies = jnp.array(latency_manager.merged_dist.latency_probabilities, copy = False)

            (
                qf_state,
                (qf_loss_value, ent_coef_value, Q_mean, new_std),
                key,
            ) = cls.update_critic(actor_state, qf_state, ent_coef_state, batch_obs,
                                    batch_actions, batch_next_obs, batch_rewards,
                                    batch_dones, batch_discounts, key, act_dim, 
                                    buffer.max_latency, past_std, tau, batch_latencies)
            qf_state = cls.soft_update(tau, qf_state)

            carry = carry.replace(qf_state = qf_state, key=key, std = new_std)

            return (carry, batch_obs, (qf_loss_value, ent_coef_value, Q_mean)), None

        def rollout_loop(items, _):
            carry = items[0]
            env = carry.env
            buffer = carry.buffer
            actor_state = carry.actor_state
            env_rec = carry.env_rec
            latency_manager = carry.latency_manager

            (tmp_buffer, buf_pos, buf_pos_init, step_carry, env_rec, latency_manager, done_or_trunc, full, reset_key, key) = rollout_VC(
                        env, collect_timesteps, buffer_shape, actor_state, env_rec, latency_manager, 
                        buffer.buffer, buffer.pos, buffer.past_begin_pos, buffer.full)
            
            tmp_buffer, env_rec, step_carry, latency_manager = jax.lax.cond(
                done_or_trunc,
                lambda tb, er, sc, lm: on_done_processor_VC(tb, er, sc, lm, reset_key, 
                                                       buf_pos + buffer.max_latency, buffer.max_latency, buffer_shape, buf_pos_init, env.ep_length, env), #new_init_pos
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
            (carry, batch_obs, (qf_loss_value, ent_coef_value, Q_mean)), _ = jax.lax.scan(rollout_loop, (carry, default_entry_rollout, default_entry_rollout_bis), None, length=num_loop_rollout)
            actor_state = carry.actor_state
            qf_state = carry.qf_state
            ent_coef_state = carry.ent_coef_state
            key = carry.key
        
            (actor_state, qf_state, ent_coef_state, 
            actor_loss_value, ent_coef_loss_value, 
            key, entropy) = cls.update_actor_and_temperature(actor_state, qf_state, ent_coef_state, batch_obs,
                                                            target_entropy, key)
            carry = carry.replace(actor_state = actor_state, qf_state = qf_state, ent_coef_state = ent_coef_state,
                                key = key, entropy = entropy)

            return (carry, (qf_loss_value, ent_coef_value, Q_mean, actor_loss_value, ent_coef_loss_value)), None

        def env_rec_loop(carry: SAC_Main_Carry, _):
            (carry, (qf_loss_value, ent_coef_value, Q_mean, actor_loss_value, ent_coef_loss_value)), _ = jax.lax.scan(policy_loop, (carry, default_entry_policy), None, length=num_loop_policy)

            return carry, (qf_loss_value, ent_coef_value, Q_mean, actor_loss_value, ent_coef_loss_value, carry.entropy, carry.std.mean(axis=0))
        
        def outer_step(items, _):
            carry, latency_manager_test = items[0], items[1]
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
            
            test_rec, env_test, latency_manager = eval_policy_env(env_test, n_eval, actor_state, test_rec, latency_manager_test)
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
        
        self.gamma_arr = jnp.full((self.batch_size) , self.gamma)[:, None]**jnp.arange(self.buffer.max_latency)
        
        t1 = time.time()
        carry, loss_warmup = self.learning_wrapper(total_timesteps, self.train_freq.frequency, self.buffer.buffer.shape[2], self.env, self.env_rec, self.buffer, self.env.obs_length, self.env.action_length, self.gradient_steps, self.gamma_arr,
                                 self.tau, self.target_entropy, self.policy_delay, self.batch_size, self.policy.qf_state, self.policy.actor_state, self.ent_coef_state, self.key, self.learning_starts, 
                                 eval_freq, n_eval, log_interval, self.latency_manager, self.latency_manager_test, self.start_std, self.start_entropy, self.n_episodes_warmup, self.n_training_warmup)
        total_time = time.time() - t1
        print("Total training time: {:.2f}s, time per iteration: {:.3f}us".format(total_time, total_time/(total_timesteps * self.train_freq.frequency)*10**6) )
        
        if self.tensorboard_log != None:
            tensorboard_logger(carry.env_rec, carry.loss_rec, carry.test_rec, self.tensorboard_log, tb_log_name, self.env.obs_length, loss_warmup, log_interval = log_interval, 
                               collect_timesteps=self.train_freq.frequency, test_frequency=eval_freq)

        if save_model:
            model_data = {"actor": carry.actor_state.params}
            serialized_weights = serialization.to_bytes(model_data)

            with open(save_path, "wb") as file:
                file.write(serialized_weights)

        return carry
    
    @staticmethod
    @partial(jax.jit, static_argnames = ["action_dim", "max_latency"])
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
        max_latency: int, 
        past_std: jax.Array, 
        tau: float,
        latencies_prob: jax.Array
    ):  
        dones = jnp.cumsum(dones, axis = 1)
        dones = (dones > 0).astype(jnp.float32)
        
        key, dropout_key_target, dropout_key_current, normal_key = jax.random.split(key, 4)
        dist = actor_state.apply_fn(actor_state.params, next_observations)
        prob = jnp.clip(dist.probs_parameter(), 1e-7, 1.0 - 1e-7).reshape((-1, max_latency, action_dim)) # (B, T, |A|)
        log_prob = jnp.log(prob)
        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
        mean_next, std_next = qf_state.apply_fn(
            qf_state.target_params,
            next_observations,
            rngs={"dropout": dropout_key_target},
        )
        mean_next = mean_next.reshape(2, -1, max_latency, action_dim)
        std_next = std_next.reshape(2, -1, max_latency, action_dim)
        
        normal_values = jax.random.normal(normal_key, mean_next.shape, dtype=jnp.float32) 
        normal_values = jnp.clip(normal_values,-3,3) *  jax.lax.stop_gradient(std_next) + jax.lax.stop_gradient(mean_next) #(2, |B|, T, |A|) 

        C1 = ent_coef_value * log_prob # (B, T, |A|)
        C2 = (1.0 - dones) * discounts # (B, T)

        target_Q_values = jnp.min(mean_next, axis=0) # (B, T, |A|)

        target_Z_vector = jnp.min(normal_values, axis = 0) # (B, T, |A|)
        
        target_Q_values = (prob * (target_Q_values - C1)).sum(axis = -1) #(B, T)
        target_Q_values = rewards + C2 * target_Q_values #(B, T)
        target_Q_values = target_Q_values * latencies_prob #(B, T)
        target_Q_values = jax.lax.stop_gradient(jnp.sum(target_Q_values, axis=1)) #(B)
        
        target_Z_vector = target_Z_vector - C1 # (B, T, |A|)
        target_Z_vector = rewards[..., None] + C2[..., None] * target_Z_vector # (B, T, |A|)
        target_Z_vector = target_Z_vector * latencies_prob[..., None] # (B, T, |A|)
        target_Z_vector = jax.lax.stop_gradient(jnp.sum(target_Z_vector, axis=1)) # (B, |A|)
                
        def mse_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
            current_q_values, current_std_values = qf_state.apply_fn(params, observations, rngs={"dropout": dropout_key}) #(2, B, |A|) x2 
            current_q_values = current_q_values.reshape(2, -1, action_dim)
            current_std_values = current_std_values.reshape(2, -1, action_dim)
            observed_q_values = current_q_values[:,jnp.arange(actions.shape[0]), actions.astype(jnp.int32).squeeze(-1)] #(2, B)
            
            new_means_std = (1 - tau) * past_std + tau *  jax.lax.stop_gradient(current_std_values).mean(axis = 1) # (2, |A|)

            td_bound_1 = 3.0 *  jax.lax.stop_gradient(new_means_std[0]) # (|A|)
            td_bound_2 = 3.0 *  jax.lax.stop_gradient(new_means_std[1]) # (|A|)
            
            difference_1 = jnp.clip(jax.lax.stop_gradient(target_Z_vector) -  jax.lax.stop_gradient(current_q_values[0]), -td_bound_1, td_bound_1) # (B,|A|)
            difference_2 = jnp.clip(jax.lax.stop_gradient(target_Z_vector) -  jax.lax.stop_gradient(current_q_values[1]), -td_bound_2, td_bound_2) # (B,|A|)

            target_q_bound_1 =  jax.lax.stop_gradient(current_q_values[0]) + difference_1 # (B,|A|)
            target_q_bound_2 =  jax.lax.stop_gradient(current_q_values[1]) + difference_2 # (B,|A|)
            
            q1_std = current_std_values[0] # (B, |A|)
            q2_std = current_std_values[1] # (B, |A|)

            q1_std_detach =  jax.lax.stop_gradient(jnp.clip(q1_std, min=0.0)) # (B, |A|)
            q2_std_detach =  jax.lax.stop_gradient(jnp.clip(q2_std, min=0.0)) # (B, |A|)
            bias = 0.1
            
            
            ratio1 = jnp.clip(new_means_std[None, 0, :] ** 2 / (q1_std_detach ** 2 + bias), 0.1,10) # (B, |A|)
            ratio2 = jnp.clip(new_means_std[None, 1, :] ** 2 / (q2_std_detach ** 2 + bias), 0.1,10) # (B, |A|)

            L_mean_1 = ratio1[jnp.arange(actions.shape[0]), actions.astype(jnp.int32).squeeze(-1)] * optax.losses.huber_loss(observed_q_values[0] , target_Q_values, 1) # (B)
            L_mean_2 = ratio2[jnp.arange(actions.shape[0]), actions.astype(jnp.int32).squeeze(-1)] * optax.losses.huber_loss(observed_q_values[1] , target_Q_values, 1) # (B)
            
            L_var_1 = ratio1 * q1_std / (q1_std_detach + bias) * (q1_std_detach**2 - optax.losses.huber_loss( jax.lax.stop_gradient(current_q_values[0]), target_q_bound_1, 1)) #(B, |A|)
            L_var_2 = ratio2 * q2_std / (q2_std_detach + bias) * (q2_std_detach**2 - optax.losses.huber_loss( jax.lax.stop_gradient(current_q_values[1]), target_q_bound_2, 1)) # (B, |A|)
            
            q1_loss = L_mean_1.mean() + L_var_1.sum(axis = 1).mean()
            q2_loss = L_mean_2.mean() + L_var_2.sum(axis = 1).mean()
            
            return q1_loss + q2_loss , (new_means_std, observed_q_values.mean())
        
        (qf_loss_value, (new_std, Q_mean)), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)
        
        return (
            qf_state,
            (qf_loss_value, ent_coef_value, Q_mean, new_std),
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
    ):
        key, dropout_key = jax.random.split(key, 2)

        def actor_loss(params: flax.core.FrozenDict) -> tuple[jax.Array, jax.Array]:
            dist = actor_state.apply_fn(params, observations)
            prob = jnp.clip(dist.probs_parameter(), 1e-7, 1.0 - 1e-7)
            log_prob = jnp.log(prob)
            
            qf_pi, _ = qf_state.apply_fn(
                qf_state.params,
                observations,
                rngs={"dropout": dropout_key},
            )
            min_qf_pi = jnp.min(qf_pi, axis=0)
            ent_coef_value =  lax.stop_gradient(ent_coef_state.apply_fn({"params": ent_coef_state.params}))
            
            entropy = dist.entropy().mean()
            actor_loss =  (prob * (ent_coef_value * log_prob - min_qf_pi)).sum(axis = 1).mean()
            
            return actor_loss, entropy

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
    ):
        (actor_state, qf_state, actor_loss_value, key, entropy) = cls.update_actor(
            actor_state,
            qf_state,
            ent_coef_state,
            observations,
            key,
        )
        
        ent_coef_state, ent_coef_loss_value = cls.update_temperature(target_entropy, ent_coef_state, entropy)
        
        return actor_state, qf_state, ent_coef_state, actor_loss_value, ent_coef_loss_value, key, entropy
    
    @staticmethod
    @jax.jit
    def update_temperature(target_entropy: ArrayLike, ent_coef_state: TrainState, entropy: float):
        def temperature_loss(temp_params: flax.core.FrozenDict) -> jax.Array:
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params})
            ent_coef_loss = jnp.log(ent_coef_value) * lax.stop_gradient(entropy - target_entropy).mean()
            return ent_coef_loss
        
        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    @staticmethod
    @partial(jax.jit, static_argnames = ["tau", "act_dim"])
    def warmup_train_critic(actor_state, qf_state, batch_obs,
                            batch_actions, batch_next_obs, batch_rewards,
                            batch_dones, discounts, key, act_dim, 
                            past_std, tau):
        
        key, dropout_key_target, dropout_key_current, normal_key = jax.random.split(key, 4)
        mean_next, std_next = qf_state.apply_fn(
            qf_state.target_params,
            batch_next_obs,
            rngs={"dropout": dropout_key_target},
        )
        mean_next = mean_next.reshape(2, -1, act_dim)
        std_next = std_next.reshape(2, -1, act_dim)

        dist = actor_state.apply_fn(actor_state.params, batch_next_obs)
        prob = jnp.clip(dist.probs_parameter(), 1e-7, 1.0 - 1e-7).reshape((-1, act_dim))
        
        target_Q_values = (jnp.min(mean_next, axis=0) * prob).sum(axis=-1)
        
        target_Q_values = target_Q_values * (1-batch_dones) * discounts[:, 1] + batch_rewards

        normal_values = jax.random.normal(normal_key, mean_next.shape, dtype=jnp.float32) 
        normal_values = jnp.clip(normal_values,-3,3) *  jax.lax.stop_gradient(std_next) + jax.lax.stop_gradient(mean_next) #(2, |B|, |A|) 
        target_Z_vector = normal_values * (1-batch_dones[:, None]) * discounts[:, [1]] + batch_rewards[:, None]
        
        def mse_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
            current_q_values, current_std_values = qf_state.apply_fn(params, batch_obs, rngs={"dropout": dropout_key}) #(2, B, |A|) x2 
            current_q_values = current_q_values.reshape(2, -1, act_dim)
            observed_q_values = current_q_values[:,jnp.arange(batch_actions.shape[0]), batch_actions.astype(jnp.int32).squeeze(-1)] #(2, B)
            new_means_std = (1 - tau) * past_std + tau *  jax.lax.stop_gradient(current_std_values).mean(axis = 1) # (2, |A|)

            td_bound = 3.0 *  jax.lax.stop_gradient(new_means_std)
            
            difference = jnp.clip(jax.lax.stop_gradient(target_Z_vector) -  jax.lax.stop_gradient(current_q_values), -td_bound[:, None, :], td_bound[:, None, :]) # (B,|A|)

            target_q_bound =  jax.lax.stop_gradient(current_q_values) + difference # (B,|A|)

            loss = (0.5 * (target_Q_values[None, ...] - observed_q_values)**2).sum(axis = 1).mean() + (jax.lax.stop_gradient(new_means_std[:, None, :]**2) - (current_q_values - target_q_bound)**2 ).mean(axis = 1).sum()
            
            return loss , (new_means_std, observed_q_values.mean())
        
        (qf_loss_value, (new_std, Q_mean)), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, Q_mean, new_std),
            key,
        )        

    @staticmethod
    @jax.jit
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