from env import EnvDataBinance, JaxLatencyEnv, reset_latency_env, step_latency_env, reset_step_carry, update_latency_buffer, erase_coming_obs, erase_first_obs, step_env
from record_utils import TestRecord, LossRecord, EnvRecord, update_env_rec
from buffer import CustomBufferBis
from discrete_policy import RLTrainState, TrainState
from functools import partial
import jax
from jax import lax
import jax.numpy as jnp
from flax import struct
import flax

@struct.dataclass
class SAC_Main_Carry():
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

        dist = actor_state.apply_fn(actor_state.params, current_obs)
        action = jax.lax.select(latency_manager_test.carry.do_action, dist.sample(1, seed = sample_key)[0], latency_manager_test.carry.past_action)
        
        new_obs, reward, done, truncated, step_carry = step_env(step_carry, action)
        done_or_trunc = jnp.logical_or(done, truncated)

        latency_manager_test, _, _, _ = step_latency_env(latency_manager_test, current_obs, new_obs, action, reward, done_or_trunc, past_done)

        cumulated_reward = cumulated_reward + reward[0] * (1-past_done)
        pf_value = new_obs[0, 16] * (1-past_done) + pf_value * past_done
        market_val = (new_obs[0, 0] / 100 + 1) * (1-past_done) + pf_value * past_done
        
        return (step_carry, latency_manager_test, done_or_trunc | past_done, cumulated_reward, current_ep_len + (1-past_done), pf_value), (pf_value, market_val)
    
    (step_carry, latency_manager_test, done, final_reward, ep_len, pf_value), (pf_vals, raw_rets) = lax.scan(scan_body, (step_carry, latency_manager_test, False, 0.0, 0.0, 1.0), all_keys)
    
    mean = jnp.sum(pf_vals - raw_rets) / ep_len
    mean_2 = jnp.sum((pf_vals - raw_rets) ** 2) / ep_len
    
    sharpe =  mean / (mean_2 - mean**2) ** 0.5 
    ratio_to_opt = pf_value / (step_carry.opt_pf_states[-1] / 100 + 1)

    step_carry = reset_step_carry(env_test, reset_key)
    latency_manager_test, _, _ = reset_latency_env(latency_manager_test, step_carry.current_obs)
    
    env_test = env_test.replace(
            step_carry = step_carry,
            key = key
        )
    return env_test, latency_manager_test, final_reward, ep_len, pf_value, sharpe, ratio_to_opt

@partial(jax.jit, static_argnames=["eval_num"])
def eval_policy_env(env_test: EnvDataBinance, eval_num, actor_state, test_rec: TestRecord, latency_manager_test: JaxLatencyEnv):
    
    def run_single_episode(items, nothing):
        env = items[0]
        latency_manager = items[1]
        env, latency_manager, final_reward, ep_len, pf_value, sharpe, opt_ratio = rollout_for_test(env, actor_state, latency_manager)

        return (env, latency_manager), (final_reward, ep_len, pf_value, sharpe, opt_ratio)

    (env_test, latency_manager_test), (ret_reward, ret_len, ret_pf, ret_sharpe, ret_opt_ratio) = lax.scan(run_single_episode, (env_test, latency_manager_test), xs = jnp.arange(eval_num))
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

    new_sharpe_mat = jax.lax.dynamic_update_slice(
        test_rec.sharpe_mat, 
        ret_sharpe.reshape(1,-1), 
        (pos,0)
    )

    new_opt_ratio_mat = jax.lax.dynamic_update_slice(
        test_rec.opt_ratio_mat, 
        ret_opt_ratio.reshape(1,-1), 
        (pos,0)
    )

    new_test_rec = test_rec.replace(
        length_mat=new_l_mat, 
        reward_mat=new_rew_mat, 
        pf_mat=new_pf_mat,
        sharpe_mat = new_sharpe_mat,
        opt_ratio_mat = new_opt_ratio_mat, 
        pos=pos + 1
    )

    return new_test_rec, env_test, latency_manager_test


@partial(jax.jit, static_argnames = ["n_steps", "buffer_shape"], donate_argnames = ["env", "tmp_buffer", "env_rec"])
def rollout_VC(env: EnvDataBinance, n_steps, buffer_shape, actor_state, env_rec: EnvRecord, latency_manager: JaxLatencyEnv, tmp_buffer, buf_pos, buf_pos_init, full):

    max_latency = latency_manager.buffer_length // 2
    step_carry = env.step_carry

    keys = jax.random.split(env.key, 1 + n_steps + 1)
    key, all_keys, reset_key = keys[0], keys[1:-1], keys[-1]

    def scan_body(items, keys_for_step):
        tmp_buffer = items[0]
        tmp_buffer_pos = items[1]
        init_pos = items[2]
        step_carry = items[3]
        current_obs = step_carry.current_obs
        env_rec = items[4]
        latency_manager = items[5]
        past_done = items[6]
        full = items[7]
        
        sample_key = keys_for_step

        dist = actor_state.apply_fn(actor_state.params, current_obs)
        action = jax.lax.select(latency_manager.carry.do_action, dist.sample(1, seed = sample_key)[0], latency_manager.carry.past_action)
        
        new_obs, reward, done, truncated, step_carry = step_env(step_carry, action)
        done_or_trunc = jnp.logical_or(done, truncated)

        latency_manager, return_arr, mask_return_arr, _ = step_latency_env(latency_manager, current_obs, new_obs, action, reward, done_or_trunc, past_done)

        tmp_buffer = update_latency_buffer(tmp_buffer, tmp_buffer_pos, max_latency, buffer_shape, mask_return_arr, return_arr, 2*max_latency)
        env_rec = env_rec.replace(cumulated_reward = env_rec.cumulated_reward + reward[0] * (1-past_done))
                    
        past_begin_pos = jnp.where(done_or_trunc | past_done, (tmp_buffer_pos + latency_manager.carry.n_iter_mask - max_latency*(1-past_done) ) % tmp_buffer.shape[0], init_pos) 
        full = jnp.logical_or(tmp_buffer_pos + latency_manager.carry.n_iter_mask - max_latency*done_or_trunc*(1-past_done) >= tmp_buffer.shape[0] , full )
        pos = (tmp_buffer_pos + latency_manager.carry.n_iter_mask - max_latency*done_or_trunc*(1-past_done) ) % tmp_buffer.shape[0]
    
        return (tmp_buffer, pos, past_begin_pos, step_carry, env_rec, latency_manager, done_or_trunc | past_done, full), None
    
    (tmp_buffer, pos, init_pos, step_carry, env_rec, latency_manager, done_or_trunc, full), _ = lax.scan(scan_body, 
                                                                                                            (tmp_buffer, buf_pos, buf_pos_init, step_carry, env_rec, latency_manager, False, full), 
                                                                                                            all_keys, 
                                                                                                            unroll=2)
    
    return (tmp_buffer, pos, init_pos, step_carry, env_rec, latency_manager, done_or_trunc, full, reset_key, key) 

@partial(jax.jit, static_argnames = ["n_steps", "buffer_shape"], donate_argnames = ["env", "tmp_buffer", "env_rec"])
def rollout_std(env: EnvDataBinance, n_steps, buffer_shape, actor_state, env_rec: EnvRecord, latency_manager: JaxLatencyEnv, tmp_buffer, buf_pos, buf_pos_init, full):

    max_latency = latency_manager.buffer_length // 2
    step_carry = env.step_carry

    keys = jax.random.split(env.key, 1 + n_steps + 1)
    key, all_keys, reset_key = keys[0], keys[1:-1], keys[-1]

    def scan_body(items, keys_for_step):
        tmp_buffer = items[0]
        tmp_buffer_pos = items[1]
        init_pos = items[2]
        step_carry = items[3]
        current_obs = step_carry.current_obs
        env_rec = items[4]
        latency_manager = items[5]
        past_done = items[6]
        full = items[7]
        
        sample_key = keys_for_step

        dist = actor_state.apply_fn(actor_state.params, current_obs)
        action = jax.lax.select(latency_manager.carry.do_action, dist.sample(1, seed = sample_key)[0], latency_manager.carry.past_action)
        
        new_obs, reward, done, truncated, step_carry = step_env(step_carry, action)
        done_or_trunc = jnp.logical_or(done, truncated)

        latency_manager, return_arr, mask_return_arr, _ = step_latency_env(latency_manager, current_obs, new_obs, action, reward, done_or_trunc, past_done)

        tmp_buffer = jax.lax.dynamic_update_slice(tmp_buffer, jnp.where(mask_return_arr[:, None], return_arr,jax.lax.dynamic_slice(
                                                        tmp_buffer, (tmp_buffer_pos, 0),  (2*max_latency, buffer_shape))), 
                                                        (tmp_buffer_pos, 0))
        env_rec = env_rec.replace(cumulated_reward = env_rec.cumulated_reward + reward[0] * (1-past_done))
                    
        past_begin_pos = jnp.where(done_or_trunc | past_done, (tmp_buffer_pos + latency_manager.carry.n_iter_mask - max_latency*(1-past_done) ) % tmp_buffer.shape[0], init_pos) 
        full = jnp.logical_or(tmp_buffer_pos + latency_manager.carry.n_iter_mask - max_latency*done_or_trunc*(1-past_done) >= tmp_buffer.shape[0] , full )
        pos = (tmp_buffer_pos + latency_manager.carry.n_iter_mask - max_latency*done_or_trunc*(1-past_done) ) % tmp_buffer.shape[0]
    
        return (tmp_buffer, pos, past_begin_pos, step_carry, env_rec, latency_manager, done_or_trunc | past_done, full), None
    
    (tmp_buffer, pos, init_pos, step_carry, env_rec, latency_manager, done_or_trunc, full), _ = lax.scan(scan_body, 
                                                                                                            (tmp_buffer, buf_pos, buf_pos_init, step_carry, env_rec, latency_manager, False, full), 
                                                                                                            all_keys, 
                                                                                                            unroll=2)
    
    return (tmp_buffer, pos, init_pos, step_carry, env_rec, latency_manager, done_or_trunc, full, reset_key, key) 

@partial(jax.jit, donate_argnames=["tmp_buffer", "env_rec", "step_carry", "latency_manager"])
def on_done_processor_std(tmp_buffer, env_rec, step_carry, latency_manager, reset_key, env):
    new_env_rec = update_env_rec(env_rec, step_carry.timestep, step_carry.pf_value)
    new_step_carry = reset_step_carry(env, reset_key)
    new_latency_manager, _, _ = reset_latency_env(latency_manager, step_carry.current_obs)
    
    return tmp_buffer, new_env_rec, new_step_carry, new_latency_manager

@staticmethod    
@partial(jax.jit, static_argnames = ["max_latency", "buffer_shape", "ep_length"], donate_argnames=["tmp_buffer", "env_rec", "step_carry", "latency_manager"])
def on_done_processor_VC(tmp_buffer, env_rec, step_carry, latency_manager, reset_key, 
                    tmp_buffer_pos, max_latency, buffer_shape, init_pos, ep_length, env):
    
    tmp_buffer = erase_coming_obs(tmp_buffer, tmp_buffer_pos + 1, max_latency, buffer_shape)
    tmp_buffer = erase_first_obs(tmp_buffer, tmp_buffer_pos, max_latency, buffer_shape, init_pos, ep_length)
    
    env_rec = update_env_rec(env_rec, step_carry.timestep, step_carry.pf_value)
    step_carry = reset_step_carry(env, reset_key)
    latency_manager, _, _ = reset_latency_env(latency_manager, step_carry.current_obs)
    
    return tmp_buffer, env_rec, step_carry, latency_manager

@jax.jit
def update_temperature(target_entropy, ent_coef_state: TrainState, entropy: float):
    def temperature_loss(temp_params: flax.core.FrozenDict) -> jax.Array:
        ent_coef_value = ent_coef_state.apply_fn({"params": temp_params})
        ent_coef_loss = jnp.log(ent_coef_value) * lax.stop_gradient(entropy - target_entropy).mean()
        return ent_coef_loss
    
    ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
    ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

    return ent_coef_state, ent_coef_loss