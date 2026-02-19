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
from flax import struct
from discrete_policy import SAC_DPolicy
import warnings
from env import *
from buffer import *
from jax import lax
from record_utils import *
from sbx.common.type_aliases import RLTrainState
import time

warnings.filterwarnings("ignore")
SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")

@struct.dataclass
class Carry():
    step_carry: StepCarry
    env_rec: EnvRecord
    it : int = 0
    optionnal_arr: jax.Array = None
    latency_manager: JaxLatencyEnv = None
       
@struct.dataclass
class CarryInit():
    tmp_buffer: jax.Array 
    timestep : jax.Array
    timestep_entry: jax.Array
    state: jax.Array
    current_obs: jax.Array
    key: jax.Array
    curve: jax.Array

@struct.dataclass
class SACarry():
    env : EnvDataBinance
    env_test: EnvDataBinance
    buffer : CustomBufferBis
    qf_state : RLTrainState
    actor_state : TrainState
    ent_coef_state : TrainState
    n_updates : int
    key: jax.Array
    env_rec: EnvRecord
    loss_rec: LossRecord
    test_rec: TestRecord
    latency_manager: JaxLatencyEnv
    std: jax.Array
    entropy: float
    
@struct.dataclass
class EvalCarry():
    current_obs : jax.Array
    is_done : bool
    timestep : int
    timestep_entry : int
    curve : jax.Array
    state : float
    reward_arr : jax.Array
    length_arr : jax.Array
    pf_arr : jax.Array
    cum_reward : float
    length : int 
    pf_val: float
    actor_state : TrainState
    env : EnvDataBinance
    key: jax.Array

@partial(jax.jit, static_argnames = ["max_latency", "buffer_shape", "ep_length"], donate_argnames=["tmp_buffer", "env_rec", "step_carry", "latency_manager"])
def on_done_processor(tmp_buffer, env_rec, step_carry, latency_manager, reset_key, 
                     tmp_buffer_pos, max_latency, buffer_shape, init_pos, ep_length, env):
    
    new_env_rec = update_env_rec(env_rec, step_carry.timestep, step_carry.pf_value)
    new_step_carry = reset_step_carry(env, reset_key)
    new_latency_manager, _, _ = reset_latency_env(latency_manager, step_carry.current_obs)
    
    return tmp_buffer, new_env_rec, new_step_carry, new_latency_manager

class SAC_delayed_JAX(SAC):
    
    policy_aliases: ClassVar[dict[str, type[SACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SACPolicy,
        "DiscretePolicy": SAC_DPolicy
    }
    
    def __init__(self, policy, foolish_env, env, env_rec, latency_manager: JaxLatencyEnv, latency_manager_test: JaxLatencyEnv,buffer : CustomBufferBis, learning_rate = 0.0003, 
                 qf_learning_rate = None, buffer_size = 1000000, learning_starts = 100, 
                 batch_size = 256, tau = 0.005, gamma = 0.99, train_freq = 1, gradient_steps = 1, policy_delay = 1, action_noise = None, 
                 replay_buffer_class = None, replay_buffer_kwargs = None, n_steps = 1, ent_coef = "auto", target_entropy = "auto", 
                 use_sde = False, sde_sample_freq = -1, use_sde_at_warmup = False, stats_window_size = 100, tensorboard_log = None, 
                 policy_kwargs = None, param_resets = None, verbose = 0, seed = None, device = "auto", _init_setup_model = True,
                 learning_rate_alpha = 3e-4, alpha_0 = 0.2):
        
        if policy_kwargs == None:
            policy_kwargs = {"action_dim": env.action_length}
        else:
            policy_kwargs.update({"action_dim": env.action_length})
        
        super().__init__(policy, foolish_env, learning_rate, qf_learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, 
                         gradient_steps, policy_delay, action_noise, replay_buffer_class, replay_buffer_kwargs, n_steps, ent_coef, 
                         target_entropy, use_sde, sde_sample_freq, use_sde_at_warmup, stats_window_size, tensorboard_log, policy_kwargs, 
                         param_resets, verbose, seed, device, _init_setup_model)

        self.env = env
        self.env_rec = env_rec
        self.buffer = buffer
        
        self.key, ent_key = jax.random.split(self.key, 2)
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

    @classmethod
    @partial(jax.jit, static_argnames = ["cls","n_episodes", "n_networks_training", "act_dim", "tau", "batch_size", "obs_dim"], 
             donate_argnames = ["env", "actor_state", "qf_state"])
    def rollout_warmup(cls, env: EnvDataBinance, n_episodes, n_networks_training, actor_state, qf_state, ent_coef_state, act_dim, tau, batch_size, past_entropy, obs_dim,
                       discounts, latency_probs):
        step_carry = env.step_carry
        
        def run_one_ep(carry, items):
            step_carry, mask = carry
            obs = step_carry.current_obs
            action = step_carry.opt_pf_arr[step_carry.timestep-1]
            
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
            past_entropy = carry[2]

            idx_sampling = jax.random.randint(key, shape = (batch_size), minval= 0 , maxval=max_pos)
            
            batch = transition_buffer[idx_sampling]
            batch_obs = batch[:, :obs_dim]
            batch_next_obs = batch[:, obs_dim:obs_dim*2].reshape(-1, obs_dim)
            batch_actions = batch[:, obs_dim*2: obs_dim*2 + 1]
            batch_rewards = batch[:, obs_dim*2 + 1]
            batch_dones = batch[:, obs_dim*2 + 2]
            
            (
                qf_state,
                (qf_loss_value, ent_coef_value, Q_mean),
                key,
            ) = cls.warmup_train_critic(actor_state, qf_state, ent_coef_state, batch_obs,
                                    batch_actions, batch_next_obs, batch_rewards,
                                    batch_dones, discounts, key, act_dim, 
                                    tau, latency_probs, 
                                    past_entropy)
            qf_state = cls.soft_update(tau, qf_state)

            (actor_state, qf_state, actor_loss_value, key, entropy) = cls.warmup_update_actor(
                actor_state,
                qf_state,
                ent_coef_state,
                batch_obs,
                key,
                act_dim,
                past_entropy, 
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


    @staticmethod
    @partial(jax.jit, static_argnames = ["n_steps", "buffer_shape"], donate_argnames = ["env", "tmp_buffer", "env_rec"])
    def rollout(env: EnvDataBinance, n_steps, buffer_shape, actor_state, env_rec: EnvRecord, latency_manager: JaxLatencyEnv, tmp_buffer, buf_pos, buf_pos_init):

        max_latency = latency_manager.buffer_length // 2
        step_carry = env.step_carry

        keys = jax.random.split(env.key, 1 + n_steps + 1)
        key, all_keys, reset_key = keys[0], keys[1:], keys[-1]

        def scan_body(items, keys_for_step):
            tmp_buffer = items[0]
            tmp_buffer_pos = items[1]
            init_pos = items[2]
            step_carry = items[3]
            current_obs = step_carry.current_obs
            env_rec = items[4]
            latency_manager = items[5]
            past_done = items[6]
            
            sample_key = keys_for_step

            dist = actor_state.apply_fn(actor_state.params, current_obs)
            action = jax.lax.select(latency_manager.carry.do_action, dist.sample(1, seed = sample_key)[0], latency_manager.carry.past_action)
            
            new_obs, reward, done, truncated, step_carry = EnvDataBinance.step_jax(step_carry, action)
            done_or_trunc = jnp.logical_or(done, truncated)

            latency_manager, return_arr, mask_return_arr, n_iterations = step_latency_env(latency_manager, current_obs, new_obs, action, reward, done_or_trunc, past_done)
            
            tmp_buffer = jax.lax.dynamic_update_slice(tmp_buffer, jnp.where(mask_return_arr[:, None], return_arr,jax.lax.dynamic_slice(
                                                        tmp_buffer, (tmp_buffer_pos, 0),  (2*max_latency, buffer_shape))), 
                                                        (tmp_buffer_pos, 0))
            
            env_rec = env_rec.replace(cumulated_reward = env_rec.cumulated_reward + reward[0] * (1-past_done))
            
            past_begin_pos = jnp.where(done_or_trunc | past_done, (tmp_buffer_pos + latency_manager.carry.n_iter_mask - max_latency*(1-past_done) ) % tmp_buffer.shape[0], init_pos) 
            pos = (tmp_buffer_pos + latency_manager.carry.n_iter_mask - max_latency*done_or_trunc*(1-past_done) ) % tmp_buffer.shape[0]
            
            return (tmp_buffer, pos, past_begin_pos, step_carry, env_rec, latency_manager, done_or_trunc | past_done), None
        
        (tmp_buffer, pos, init_pos, step_carry, env_rec, latency_manager, done_or_trunc), _ = lax.scan(scan_body, (tmp_buffer, buf_pos, buf_pos_init, step_carry, env_rec, latency_manager, False), all_keys)
        
        tmp_buffer, env_rec, step_carry, latency_manager = jax.lax.cond(
                done_or_trunc,
                lambda tb, er, sc, lm: on_done_processor(tb, er, sc, lm, reset_key, 
                                                       pos + max_latency, max_latency, buffer_shape, buf_pos_init, env.ep_length, env), #new_init_pos
                lambda tb, er, sc, lm: (tb, er, sc, lm),
                tmp_buffer, env_rec, step_carry, latency_manager
            )
        
        env = env.replace(
                step_carry = step_carry,
                key = key
            )
        return env, env_rec, latency_manager, tmp_buffer, pos, init_pos
    
    @classmethod
    @partial(jax.jit, static_argnames = ["cls", "total_timesteps", "buffer_shape", "obs_dim", "act_dim", "gradient_steps", 
                                         "tau", "policy_delay", "batch_size", "collect_timesteps", "init_timesteps", 
                                         "eval_freq", "n_eval", "target_entropy", "log_interval"], donate_argnames = ["buffer"])
    def learning_wrapper(cls, total_timesteps, collect_timesteps, buffer_shape, env, eval_env, buffer: CustomBufferLatency, obs_dim, act_dim, gradient_steps, discounts,
                                 tau, target_entropy, policy_delay, batch_size, qf_state: RLTrainState, actor_state: TrainState, ent_coef_state: TrainState,
                                 key: jax.Array, init_timesteps, eval_freq: int, n_eval: int, log_interval: int, latency_manager: JaxLatencyEnv, latency_manager_test: JaxLatencyEnv, start_entropy,
                                 n_episodes_warmup = 10**2, n_networks_training_warmup = 10**4*5):
        
        env_rec = EnvRecord(pos = 0, buffer=jnp.zeros(((init_timesteps + collect_timesteps * total_timesteps) // 10, 3), dtype=jnp.float32)) #10: empirical assumption that an env will last in mean at least 10 timesteps
        
        loss_rec = LossRecord(pos = 0, history= jnp.zeros((total_timesteps // log_interval, 6 + act_dim), dtype=jnp.float32))

        env, actor_state, qf_state, start_entropy, rec_losses = cls.rollout_warmup(env, n_episodes_warmup, n_networks_training_warmup, actor_state, 
                                                                                              qf_state, ent_coef_state, act_dim, tau, batch_size, start_entropy, obs_dim, 
                                                                                              discounts, jnp.array(latency_manager.merged_dist.latency_probabilities), 
                                                                                              )
        
        env, env_rec, latency_manager, tmp_buf, buf_pos, buf_pos_init = cls.rollout(env, init_timesteps, buffer_shape, actor_state, env_rec, latency_manager, buffer.buffer, buffer.pos, buffer.past_begin_pos)
        buffer = buffer.replace(buffer = tmp_buf, pos = buf_pos, past_begin_pos= buf_pos_init)
        
        key, test_key = jax.random.split(key)

        test_rec = TestRecord(pos=0, reward_mat=jnp.zeros((total_timesteps // eval_freq, n_eval)), length_mat=jnp.zeros((total_timesteps // eval_freq, n_eval)), 
                              pf_mat = jnp.zeros((total_timesteps // eval_freq, n_eval)), key = test_key)

        
        carry = SACarry(env = env, env_test = eval_env, buffer=buffer, qf_state= qf_state, actor_state= actor_state, ent_coef_state=ent_coef_state, n_updates=0, key = key, 
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
            latency_manager = carry.latency_manager
            past_entropy = carry.entropy
            data, key = buffer.sample(buffer.buffer, batch_size, buffer.pos, key, discounts, obs_dim, buffer.full, buffer.buffer_size)
            
            batch_obs = jax.lax.dynamic_slice_in_dim(data.observations, batch_size, batch_size)
            batch_actions = jax.lax.dynamic_slice_in_dim(data.actions, batch_size, batch_size)
            batch_next_obs = jax.lax.dynamic_slice_in_dim(data.next_observations, batch_size , batch_size)
            batch_rewards = jax.lax.dynamic_slice_in_dim(data.rewards, batch_size, batch_size)
            batch_dones = jax.lax.dynamic_slice_in_dim(data.dones, batch_size, batch_size)
            #batch_discounts = jax.lax.dynamic_slice_in_dim(data.discounts, batch_size, batch_size)
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

            env, env_rec, latency_manager, tmp_buf, buf_pos, buf_pos_init = cls.rollout(
                        env, collect_timesteps, buffer_shape, actor_state, env_rec, latency_manager, 
                        buffer.buffer, buffer.pos, buffer.past_begin_pos)
                        
            buffer = buffer.replace(buffer=tmp_buf, pos=buf_pos, past_begin_pos=buf_pos_init)

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
                                                            target_entropy, key, act_dim,  past_entropy)
            carry = carry.replace(actor_state = actor_state, qf_state = qf_state, ent_coef_state = ent_coef_state,
                                key = key, entropy = entropy)

            return (carry, (qf_loss_value, ent_coef_value, Q_mean, actor_loss_value, ent_coef_loss_value)), None

        def env_rec_loop(carry: SACarry, _):
            loss_rec = carry.loss_rec

            (carry, (qf_loss_value, ent_coef_value, Q_mean, actor_loss_value, ent_coef_loss_value)), _ = jax.lax.scan(policy_loop, (carry, default_entry_policy), None, length=num_loop_policy)

            loss_rec = cls.update_loss_rec(loss_rec, qf_loss_value, actor_loss_value, ent_coef_loss_value, ent_coef_value, Q_mean, carry.entropy, jnp.zeros(3))
            carry = carry.replace(loss_rec = loss_rec)

            return carry, None
        
        def outer_step(items, _):
            carry, latency_manager = items[0], items[1]
            carry, _ = jax.lax.scan(env_rec_loop, carry, None, length=num_loop_rec)
            
            env_test = carry.env_test
            actor_state = carry.actor_state
            test_rec = carry.test_rec
            
            test_rec, env_test, latency_manager = cls.eval_env(env_test, n_eval, actor_state, test_rec, latency_manager)
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
    ):
        
        t1 = time.time()
        carry, rec_losses = self.learning_wrapper(total_timesteps, self.train_freq.frequency, self.buffer.buffer.shape[1], self.env, self.env_rec, self.buffer, self.env.obs_length, self.env.action_length, self.gradient_steps, self.gamma,
                                 self.tau, self.target_entropy, self.policy_delay, self.batch_size, self.policy.qf_state, self.policy.actor_state, self.ent_coef_state, self.key, self.learning_starts, 
                                 eval_freq, n_eval, log_interval, self.latency_manager, self.latency_manager_test, self.start_entropy)
        total_time = time.time() - t1
        print("Total training time: {:.2f}s, time per iteration: {:.3f}us".format(total_time, total_time/(total_timesteps * self.train_freq.frequency)*10**6) )
        
        if self.tensorboard_log != None:
            tensorboard_logger(carry.env_rec, carry.loss_rec, carry.test_rec, self.tensorboard_log, tb_log_name, self.env.obs_length, log_interval = log_interval, 
                               collect_timesteps=self.train_freq.frequency, test_frequency=eval_freq, loss_warmup=rec_losses)

        return carry

    @staticmethod
    @partial(jax.jit, donate_argnums=(0,))
    def update_loss_rec(loss_rec: LossRecord, qf_loss_value, actor_loss_value, 
                        ent_coef_loss_value, ent_coef_value, Q_mean, entropy, std):
        
        new_metrics = jnp.concatenate([
        jnp.array([
            qf_loss_value, 
            actor_loss_value, 
            ent_coef_loss_value, 
            ent_coef_value, 
            Q_mean, 
            entropy
        ]),
        jnp.ravel(std)
        ])

        return loss_rec.replace(
            history=loss_rec.history.at[loss_rec.pos].set(new_metrics),
            pos=loss_rec.pos + 1
        )

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
    
        key, noise_key, dropout_key_target, dropout_key_current, normal_key = jax.random.split(key, 5)
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
    @partial(jax.jit, static_argnames = ["action_dim"])
    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        key: jax.Array,
        action_dim: int,
        past_entropy: jax.Array
    ):
        key, dropout_key, noise_key, target_key = jax.random.split(key, 4)
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
            actor_loss =  (prob * (ent_coef_value * log_prob - min_qf_pi)).sum(axis = 1).mean() + (entropy - past_entropy) ** 2 * beta / 2
            
            return actor_loss, entropy

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, qf_state, actor_loss_value, key, entropy
    
    @classmethod
    @partial(jax.jit, static_argnames = ["cls", "action_dim"])
    def update_actor_and_temperature(
        cls,
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        target_entropy: ArrayLike,
        key: jax.Array,
        action_dim: int,
        past_entropy: jax.Array
        
    ):
        (actor_state, qf_state, actor_loss_value, key, entropy) = cls.update_actor(
            actor_state,
            qf_state,
            ent_coef_state,
            observations,
            key,
            action_dim,
            past_entropy
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
    
    @classmethod
    @partial(jax.jit, static_argnames=["cls", "eval_num"])
    def eval_env(cls, env_test: EnvDataBinance, eval_num, actor_state, test_rec: TestRecord, latency_manager_test: JaxLatencyEnv):
        
        def run_single_episode(items, nothin):
            env = items[0]
            latency_manager = items[1]
            env, latency_manager, final_reward, ep_len, pf_value = cls.rollout_for_test(env, actor_state, latency_manager)

            return (env, latency_manager), (final_reward, ep_len, pf_value)

        (env_test, latency_manager_test), (ret_reward, ret_len, ret_pf) = lax.scan(run_single_episode, (env_test, latency_manager_test), xs = jnp.arange(eval_num))
        pos = test_rec.pos
        
        new_rew_mat = jax.lax.dynamic_update_slice(
            test_rec.reward_mat, 
            ret_reward.reshape(1,-1), 
            (pos,0)
        )
        
        new_l_mat = jax.lax.dynamic_update_slice(
            test_rec.length_mat, 
            ret_len.reshape(1,-1), 
            (pos,0)
        )
        
        new_pf_mat = jax.lax.dynamic_update_slice(
            test_rec.pf_mat, 
            ret_pf.reshape(1,-1), 
            (pos,0)
        )

        new_test_rec = test_rec.replace(
            length_mat=new_l_mat, 
            reward_mat=new_rew_mat, 
            pf_mat=new_pf_mat,
            pos=pos + 1
        )

        return new_test_rec, env_test, latency_manager_test
    
    @staticmethod
    @partial(jax.jit, donate_argnames = ["env_test"])
    def rollout_for_test(env_test: EnvDataBinance, actor_state, latency_manager_test: JaxLatencyEnv):

        step_carry = env_test.step_carry
        ep_length = env_test.ep_length

        keys = jax.random.split(env_test.key, 1 + ep_length + 1)
        key, all_keys, reset_key = keys[0], keys[1:], keys[-1]

        def scan_body(items, keys_for_step):
            step_carry = items[0]
            current_obs = step_carry.current_obs
            latency_manager_test = items[1]
            past_done = items[2]
            cumulated_reward = items[3]
            current_ep_len = items[4]
            pf_value = items[5]
            
            sample_key = keys_for_step

            def true_fun(action, current_obs):
                dist = actor_state.apply_fn(actor_state.params, current_obs)
                return dist.sample(1, seed = sample_key)[0]
            def false_fun(action, current_obs):
                return action
            
            action = jax.lax.select(latency_manager_test.carry.do_action, true_fun(latency_manager_test.carry.past_action, current_obs), false_fun(latency_manager_test.carry.past_action, current_obs))
            
            new_obs, reward, done, truncated, step_carry = EnvDataBinance.step_jax(step_carry, action)
            done_or_trunc = jnp.logical_or(done, truncated)

            latency_manager_test, _, _, _ = step_latency_env(latency_manager_test, current_obs, new_obs, action, reward, done_or_trunc, past_done)

            cumulated_reward = cumulated_reward + reward[0] * (1-past_done)
            pf_value = new_obs[0, 16] * (1-past_done) + pf_value * past_done
            
            return (step_carry, latency_manager_test, done_or_trunc | past_done, cumulated_reward, current_ep_len + (1-past_done), pf_value), None
        
        (step_carry, latency_manager_test, done_or_trunc, final_reward, ep_len, pf_value), _ = lax.scan(scan_body, (step_carry, latency_manager_test, False, 0.0, 0.0, 1.0), all_keys)
        

        step_carry = reset_step_carry(env_test, reset_key)
        latency_manager_test, _, _ = reset_latency_env(latency_manager_test, step_carry.current_obs)
        
        env_test = env_test.replace(
                step_carry = step_carry,
                key = key
            )
        return env_test, latency_manager_test, final_reward, ep_len, pf_value

    @staticmethod
    @partial(jax.jit, static_argnames = ["tau", "act_dim"])
    def warmup_train_critic(actor_state, qf_state, ent_coef_state, batch_obs,
                                    batch_actions, batch_next_obs, batch_rewards,
                                    batch_dones, discounts, key, act_dim, 
                                    tau, latency_probs, 
                                    past_entropy):
        
        key, noise_key, dropout_key_target, dropout_key_current, normal_key = jax.random.split(key, 5)
        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
        mean_next = qf_state.apply_fn(
            qf_state.target_params,
            batch_next_obs,
            rngs={"dropout": dropout_key_target},
        )
        mean_next = mean_next.reshape(2, -1, act_dim)

        target_Q_values = jnp.min(mean_next, axis=0)
        
        target_Q_values = target_Q_values * (1-batch_dones[:, None]) * discounts + batch_rewards[:, None]
        
        def mse_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
            current_q_values = qf_state.apply_fn(params, batch_obs, rngs={"dropout": dropout_key}) #(2, B, |A|) x2 
            current_q_values = current_q_values.reshape(2, -1, act_dim)
            observed_q_values = current_q_values[:,jnp.arange(batch_actions.shape[0]), batch_actions.astype(jnp.int32).squeeze(-1)] #(2, B)

            loss = (0.5 * (target_Q_values[None, ...] - current_q_values)**2).sum(axis = 1).mean()
            
            return loss , observed_q_values.mean()
        
        (qf_loss_value, Q_mean), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, ent_coef_value, Q_mean),
            key,
        )        

    def warmup_update_actor(actor_state,
                qf_state,
                ent_coef_state,
                batch_obs,
                key,
                act_dim,
                past_entropy, 
                batch_actions):
        
        key, dropout_key, noise_key, target_key = jax.random.split(key, 4)

        def actor_loss(params: flax.core.FrozenDict) -> tuple[jax.Array, jax.Array]:
            dist = actor_state.apply_fn(params, batch_obs)
            logits = dist.logits
            
            entropy = dist.entropy().mean()
            
            actor_loss = optax.losses.softmax_cross_entropy_with_integer_labels( logits, batch_actions.reshape(-1).astype(jnp.int32)).sum()
                
            return actor_loss, entropy
        
        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        
        actor_state = actor_state.apply_gradients(grads=grads)
        
        return actor_state, qf_state, actor_loss_value, key, entropy