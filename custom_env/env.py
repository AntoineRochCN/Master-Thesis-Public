from functools import partial
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import jax.numpy as jnp
try: 
    from reward_function import *
    from buffer import *
except:
    from .reward_function import *
    from .buffer import *

from flax import struct
import jax
import scipy as sc

class foolish_env(gym.Env):
    def __init__(self, obs_shape, action_shape):
        super().__init__()

        self.observation_space = Box(-np.inf, np.inf, shape = (obs_shape,))
        self.action_space = Box(-1, 1, shape = (action_shape,))

@struct.dataclass
class StepCarry():
    timestep : int
    timestep_entry : int
    pf_value : float
    state : float
    curve : jax.Array
    current_obs: jax.Array
    opt_pf_arr: jax.Array
    opt_pf_states: jax.Array
    
    ep_length : int = struct.field(pytree_node=False)
    stop_loss: float = struct.field(pytree_node=False)
    stop_limit: float = struct.field(pytree_node=False)
    obs_length: float = struct.field(pytree_node=False)
    leverage: float = struct.field(pytree_node=False, default = 1.0)
    transaction_cost : float = struct.field(pytree_node=False, default = 0.075/100)

class LatencyDistribution(NamedTuple):
    latency_range: np.array
    latency_probabilities: np.array
    max_latency: np.array

@struct.dataclass
class CarryLatency():
    timestep : int
    ret_ind: int 
    mask_actions : jax.Array
    action_shift : jax.Array
    mask_next_observations : jax.Array
    next_obs_shift : jax.Array
    buffer_latency: jax.Array #shape: (ep_length + max_tau_action + max_tau_obs + 3, action_dim + obs_dim + 2)
    mask_return : jax.Array
    past_action : jax.Array
    past_obs : jax.Array
    new_obs_positions : jax.Array
    do_action: bool = False
    pos_init_mask: int = 0 
    n_iter_mask: int = 0
    counter_shift: int = 0

@struct.dataclass(frozen = True)
class EnvDataBinance:
    '''
    Obs : 
    0: bids (normalized, centered and %), 
    1: cumulated_quantity_bids 
    2: running_mean10_bids (normalized, centered and %)
    3: running_mean50_bids (normalized, centered and %)
    4: running_mean100_bids (normalized, centered and %)

    5: asks
    6: cumulated_quantity_asks
    7: running_mean10_asks (normalized, centered and %)
    8: running_mean50_asks (normalized, centered and %)
    9: running_mean100_asks (normalized, centered and %)

    10: high_1s (normalized, centered and %)
    11: low_1s (normalized, centered and %)
    12: log10(spread + 1)

    13: entry_value 
    14: past_value
    15: state
    16: pf_value
    17: time_since_entry
    18: dist_to_stop_loss
    19: dist_to_stop_limit
    '''

    key: jax.Array = struct.field(pytree_node=True)
    step_carry: StepCarry = struct.field(pytree_node=True)
    obs_length: int = struct.field(pytree_node=False)
    data: jax.Array = struct.field(pytree_node=True)
    action_length : int = struct.field(pytree_node=False, default = 3)
    ep_length: int = struct.field(pytree_node=False, default = 10000)
    stop_loss: float = struct.field(pytree_node=False, default = 0.9)
    stop_limit: float = struct.field(pytree_node=False, default = 1.1)
    leverage: float = struct.field(pytree_node=False, default = 1.0)
    transaction_cost: float = struct.field(pytree_node=False, default = 0.075/100)

    @classmethod
    def create(cls, path : str,
               ep_length: int = 10000, seed: int = 0, 
               stop_loss: float = 0.9, stop_limit: float = 1.1, 
               leverage: float = 1.0, transaction_cost: float = 0.075/100):

        key = jax.random.key(seed)
        data = jnp.load(path)
        data = data[~jnp.any(jnp.isnan(data), axis = 1)]
        
        curve, key = curve_generator(data, ep_length, key)
        curve = normalize_curve(curve)
        opt_pf_states, opt_pf_arr = solve_constrained_oracle_pf(curve[:,0], transaction_cost, leverage)
        obs_length = curve.shape[1] + 7
        current_obs = jnp.hstack([curve[0], (curve[0, 0] + curve[0, 5])/2, (curve[0, 0] + curve[0, 5])/2, 0.0, 1.0, 0.0, 1-stop_loss, stop_limit-1]).reshape(1,-1)
        step_carry = StepCarry(timestep=1, timestep_entry=0, curve=curve, current_obs=current_obs, ep_length=ep_length, stop_loss=stop_loss, stop_limit=stop_limit, 
                               obs_length=obs_length, state = 0.0, pf_value = 1.0, leverage=leverage, transaction_cost=transaction_cost, opt_pf_arr=opt_pf_arr, opt_pf_states = opt_pf_states)
        env = cls(
            obs_length=obs_length,
            ep_length=ep_length,
            key=key,
            stop_loss=stop_loss,
            stop_limit=stop_limit, 
            data = data, 
            step_carry = step_carry,
            leverage = leverage,
            transaction_cost = transaction_cost
        )

        return env

@partial(jax.jit, static_argnames = ["ep_length"])
def curve_generator(data: jax.Array, ep_length: int, key: jax.Array):
    shape = data.shape[0]
    key, gen_key = jax.random.split(key)

    start_time = jax.random.randint(gen_key, (1,), minval=0, maxval= shape-ep_length)
    idx = np.arange(ep_length) + start_time

    cols = jnp.array([0,1,2,3,4,8,9,10,11,12,16,17,18,19])
    return data[jnp.ix_(idx, cols)], key

@partial(jax.jit, donate_argnames = ["curve"])
def normalize_curve(curve: jax.Array):
    cols = jnp.arange(13)
    
    curve = curve.at[:, [10, 11]].set( (curve[:, [10, 11]] / (curve[0, 0] +  curve[0, 5]) * 2 - 1) * 100)
    curve = curve.at[:, 12].set( jnp.log10(curve[:, 12] - curve[:, 13] + 1) )
    curve = curve.at[:, [2,3,4]].set( (curve[:, [2,3,4]]/ curve[0,0] -1)*100)
    curve = curve.at[:, 0].set( (curve[:, 0]/ curve[0,0] - 1) * 100)
    
    curve = curve.at[:, [7,8,9]].set( (curve[:, [7,8,9]] / curve[0,5] - 1)*100)
    curve = curve.at[:, 5].set( (curve[:, 5]/ curve[0,5] - 1) * 100 )
    return curve[:, cols]

@partial(jax.jit, static_argnames = ["cost", "leverage", "initial_capital"])
def solve_constrained_oracle_pf(prices, cost, leverage, initial_capital=1.0):
    
    p_diff_arr = jnp.diff(prices) 
    r_long = p_diff_arr / (prices[:-1] + 100.0)
    r_short = -(prices[:-1] + 100.0) * (p_diff_arr) / ((prices[1:] + 100.0) * (prices[:-1] + 100.0))
    
    def backward_step(val_next, W):
        v_long, v_short = W
        x = val_next[None, :] * jnp.asarray([[1.0, 1.0 + leverage *(v_long - cost), 1.0 + leverage * (v_short - cost)], 
                                           [1.0 - leverage * cost, 1.0 + leverage * v_long, -jnp.inf], 
                                           [1.0 - leverage * cost, -jnp.inf, 1.0 + leverage * v_short]])
        return jnp.max(x, axis=1), jnp.argmax(x, axis=1)
    
    init_val = jnp.ones(3)
    _, best_choices = jax.lax.scan(backward_step, init_val, (r_long, r_short), reverse=True, unroll=5)
    
    def forward_step(items, items_bis):
        curr_state_idx, pf_val = items
        best_choice, v_long, v_short = items_bis
        next_state_idx = best_choice[curr_state_idx]
        new_pf_val = pf_val * jnp.asarray([[1.0, 1.0 + leverage *(v_long - cost), 1.0 + leverage * (v_short - cost)], 
                                           [1.0 - leverage * cost, 1.0 + leverage * v_long, -jnp.inf], 
                                           [1.0 - leverage * cost, -jnp.inf, 1.0 + leverage * v_short]])[curr_state_idx, next_state_idx]
        return (next_state_idx, new_pf_val), (next_state_idx, new_pf_val)
    
    _, (states, pf_vals) = jax.lax.scan(forward_step, (0, 1.0), (best_choices, r_long, r_short), unroll=10)
    
    return jnp.concatenate([jnp.array([0.0]), states]), jnp.concatenate([jnp.array([1.0]), pf_vals])

@jax.jit
def reset_step_carry(env: EnvDataBinance, key: jax.Array):
    new_curve, key = curve_generator(env.data, env.ep_length, key)
    new_curve = normalize_curve(new_curve)
    opt_pf_states, opt_pf_arr = solve_constrained_oracle_pf(new_curve[:,0], env.transaction_cost, env.leverage)
    current_obs = jnp.hstack([new_curve[0], (new_curve[0, 0] + new_curve[0, 5])/2, (new_curve[0, 0] + new_curve[0, 5])/2, 0.0, 1.0, 0.0, 1.0-env.stop_loss, env.stop_limit-1.0]).reshape(1,-1)
    step_carry = env.step_carry.replace(timestep=1, timestep_entry=0, curve=new_curve, current_obs=current_obs, state = 0.0, pf_value = 1.0, opt_pf_arr = opt_pf_arr, opt_pf_states = opt_pf_states)

    return step_carry

@partial(jax.jit, donate_argnames = ["step_carry"])
def step_env(step_carry : StepCarry, action: float):

    timestep = step_carry.timestep
    timestep_entry = step_carry.timestep_entry
    transaction_cost = step_carry.transaction_cost
    past_market_val = step_carry.current_obs[0, 14]
    market_val_init = step_carry.current_obs[0, 13]
    state = step_carry.current_obs[0, 15]
    pf_value = step_carry.current_obs[0, 16]
    time_since_entry = step_carry.current_obs[0, 17]
    idx = 3.0*action + state

    pf_opt = step_carry.opt_pf_arr[timestep-1]

    new_market_obs = step_carry.curve[timestep] ## vec of size 13
    ep_length = step_carry.ep_length
    
    new_market_val = (idx == 0.0) * (new_market_obs[0] + new_market_obs[5]) / 2 + jnp.isin(idx, jnp.asarray([2.0,3.0,8.0])) * new_market_obs[5]+ jnp.isin(idx, jnp.asarray([1.0,4.0,6.0])) * new_market_obs[0]
    truncated = jnp.asarray(timestep >= ep_length, dtype=jnp.bool_, copy=False)
    
    reward, done, new_pf_value = reward_function(new_market_val, past_market_val, market_val_init, action, state, pf_value, truncated, time_since_entry, step_carry.stop_loss, 
                                                 step_carry.stop_limit, step_carry.leverage, transaction_cost, pf_opt)
    
    flag = action == state
    new_timestep_entry = flag * timestep_entry + (1-flag) * timestep
    
    new_entry_obs = flag * market_val_init + (1-flag) * new_market_val
    
    new_obs = jnp.hstack([new_market_obs, new_entry_obs, new_market_val, action, new_pf_value, (timestep +1 - new_timestep_entry)/ep_length, 
                       new_pf_value - step_carry.stop_loss, step_carry.stop_limit - new_pf_value]).reshape(1,step_carry.obs_length)
    
    return new_obs, reward, done, truncated, step_carry.replace(timestep = timestep + 1, timestep_entry = new_timestep_entry[0].astype(jnp.int32), 
                                                                state = action[0].astype(jnp.float32), pf_value = new_pf_value[0],
                                                                current_obs = new_obs)

def get_latency_prop(distribution, **kwargs):
    """
    Gets a distribution name with its parameters, and returns a tuple with the values taken by the latency, its probabilities and the value of the maximum delay
    """
    match distribution:
        case "uniform":
            delta_max = kwargs["delta_max"]
            latence_range = np.arange(delta_max + 1)
            latence_probabilities = np.ones_like(latence_range, dtype=np.float32) / len(latence_range)
        case "gamma":
            delta_max = kwargs["delta_max"]
            mean = kwargs["mean"]
            latence_range = np.arange(delta_max + 1)
            latence_probabilities = sc.stats.gamma.pdf(latence_range, mean).astype(np.float32)
            latence_probabilities /= np.sum(latence_probabilities)
        case "gaussian":
            delta_max = kwargs["delta_max"]
            latence_range = np.arange(delta_max + 1)
            mean_1, mean_2 = kwargs["vec_mean"]
            std_1, std_2 = kwargs["vec_std"]
            latence_probabilities = sc.stats.norm.pdf(latence_range, mean_1, std_1).astype(np.float32)
            latence_probabilities += sc.stats.norm.pdf(latence_range, mean_2, std_2).astype(np.float32)
            latence_probabilities /= np.sum(latence_probabilities)
        case 'test':
            latence_range = np.zeros(1, dtype=int)
            latence_probabilities = np.ones(1)
        case "constant":
            delta_max = kwargs["delta_max"]
            latence_range = np.arange(delta_max + 1)
            latence_probabilities = np.zeros_like(latence_range)
            latence_probabilities[-1] = 1
        case 'custom_test_mixture':
            delta_max = 50
            mean = 2
            latence_range = np.arange(delta_max + 1)
            latence_probabilities = sc.stats.gamma.pdf(latence_range, mean).astype(np.float32)
            latence_probabilities[7:] = 0
            latence_probabilities /= np.sum(latence_probabilities)
            latence_probabilities *= 0.99
            latence_probabilities[-1] = 0.01
        case "measured_observation":
            latence_probabilities = np.array([0.0, 0.97067666, 0.01768216, 0.01164118])
            latence_range = np.arange(4)

        case "measured_action":
            latence_probabilities = np.array([0.0, 0.25225046, 0.74449818, 0.00325136])
            latence_range = np.arange(4)
            
    return LatencyDistribution(latence_range, latence_probabilities, max(int(np.max(latence_range)) - 1, 0))

def latency_merger(dist_a: LatencyDistribution, dist_b: LatencyDistribution):
    merged_probs = np.convolve(dist_a.latency_probabilities, dist_b.latency_probabilities).astype(np.float64)
    merged_probs /= np.sum(merged_probs, dtype=np.float64)
    latency_range = list(range(dist_a.latency_range[0] + dist_b.latency_range[0], dist_a.latency_range[-1] + dist_b.latency_range[-1] + 1))
    max_latency = np.max(latency_range) - 1
    return LatencyDistribution(latency_range, merged_probs, max_latency)

@struct.dataclass
class JaxLatencyEnv:

    key : jax.Array
    latency_range_action: jax.Array = struct.field(pytree_node=True)
    latency_probabilities_action: jax.Array = struct.field(pytree_node=True)
    max_latency_action: int = struct.field(pytree_node=False)

    latency_range_observation : jax.Array = struct.field(pytree_node=True)
    latency_probabilities_observation : jax.Array = struct.field(pytree_node=True)
    max_latency_observation : jax.Array = struct.field(pytree_node=False)

    merged_dist: jax.Array = struct.field(pytree_node=True)
    max_ep_length: int = struct.field(pytree_node=False)

    action_dim: int = struct.field(pytree_node=False)
    obs_dim: int = struct.field(pytree_node=False)
    buffer_length: int = struct.field(pytree_node=False)

    carry : CarryLatency

    @classmethod
    def create(cls, distribution_action, dist_action_kwargs,distribution_obs, dist_obs_kwargs, max_ep_length, seed, action_dim, obs_dim):

        dist_action = get_latency_prop(distribution_action, **dist_action_kwargs)
        dist_observation = get_latency_prop(distribution_obs, **dist_obs_kwargs)
        
        latency_range_action = dist_action.latency_range 
        latency_probabilities_action = dist_action.latency_probabilities
        max_latency_action = dist_action.max_latency

        latency_range_observation = dist_observation.latency_range
        latency_probabilities_observation = dist_observation.latency_probabilities
        max_latency_observation = dist_observation.max_latency

        merged_dist = latency_merger(dist_action, dist_observation)

        key = jax.random.key(seed)
        buffer_length = 2*(max_latency_action + max_latency_observation + 1 + 1)
        
        buffer_latency = jnp.zeros((buffer_length, action_dim + obs_dim*2 + 2), dtype=jnp.float32)

        carry = CarryLatency(timestep=0, ret_ind=0, buffer_latency=buffer_latency, mask_return=jnp.zeros(buffer_length, dtype=jnp.bool), past_action=jnp.zeros(action_dim, dtype=jnp.int32), 
                             mask_actions=None, action_shift=None, mask_next_observations=None, next_obs_shift=None, past_obs=jnp.zeros(obs_dim), new_obs_positions=None)

        latency_env = cls(latency_range_action = latency_range_action, latency_probabilities_action = latency_probabilities_action, max_latency_action = max_latency_action,
                   latency_range_observation = latency_range_observation, latency_probabilities_observation = latency_probabilities_observation, max_latency_observation = max_latency_observation,
                   merged_dist = merged_dist, max_ep_length = max_ep_length, key = key, carry=carry, action_dim = action_dim, obs_dim = obs_dim, buffer_length = buffer_length)
        
        latency_env = mask_generator(latency_env)
        return latency_env
    
@partial(jax.jit, donate_argnums = (0,))
def mask_generator(LatencyEnv: JaxLatencyEnv):

    key, tau_key, taup_key = jax.random.split(LatencyEnv.key, 3)
    n_max = LatencyEnv.max_ep_length + LatencyEnv.max_latency_action + LatencyEnv.max_latency_observation + 3

    tau = jax.random.choice(tau_key, LatencyEnv.latency_range_action, p = LatencyEnv.latency_probabilities_action, shape=(n_max,))
    tau_prime = jax.random.choice(taup_key, LatencyEnv.latency_range_observation, p = LatencyEnv.latency_probabilities_observation, shape=(n_max,))
        
    mask_actions = jnp.zeros(n_max + LatencyEnv.max_latency_action + 2, dtype=jnp.bool)
    mask_next_observations = jnp.zeros(n_max + LatencyEnv.max_latency_action + LatencyEnv.max_latency_observation + 3, dtype=jnp.bool)

    action_shift = np.arange(n_max) + tau + 1
    next_obs_shift = np.arange(n_max) + tau + tau_prime + 1
    
    vals, idx_act = jnp.unique(action_shift[::-1],size=n_max, fill_value=0, return_index=True)
    vals_bis, idx_obs = jnp.unique(next_obs_shift[::-1], size=n_max, fill_value=0, return_index=True)

    idx_act = n_max - 1 - idx_act
    idx_obs = n_max - 1 - idx_obs

    mask_actions = mask_actions.at[idx_act].set(True)
    mask_next_observations = mask_next_observations.at[idx_obs].set(True)

    mask_actions = mask_actions.at[0].set(False)
    mask_next_observations = mask_next_observations.at[0].set(False)

    carry = LatencyEnv.carry
    carry = carry.replace( mask_actions = mask_actions, buffer_latency = jnp.zeros_like(carry.buffer_latency), ret_ind = 0,
                            action_shift = action_shift, mask_next_observations = mask_next_observations, next_obs_shift = next_obs_shift, 
                            timestep = 0, mask_return = jnp.zeros_like(carry.mask_return), past_action=jnp.zeros(LatencyEnv.action_dim, dtype=jnp.int32), 
                            new_obs_positions = vals_bis, counter_shift = 0)
    
    LatencyEnv = LatencyEnv.replace(key = key, carry=carry)

    return LatencyEnv

@partial(jax.jit, static_argnames=["buffer_length", "obs_dim", "act_dim", "max_latency"], donate_argnums=(0,))
def store_action(buffer_latency, flag_action, buffer_length, pos_a, pos_b, action, obs_dim, act_dim, max_latency):
    
    def update_buffer(pos_a, pos_b, buffer_latency, action):

        update_len = pos_b - pos_a
        grid_dist = (jnp.arange(buffer_length) - (pos_a + 1)) % buffer_length
        mask = grid_dist < update_len

        slice = jnp.where(mask[:, jnp.newaxis], action, buffer_latency[:, 2*obs_dim : 2*obs_dim+act_dim])

        return jax.lax.dynamic_update_slice(buffer_latency, slice, (0, 2*obs_dim))
    
    return jax.lax.select(
        flag_action, 
        update_buffer(pos_a, pos_b, buffer_latency, action),
        buffer_latency)
    
@partial(jax.jit, static_argnames=["obs_dim", "act_dim"], donate_argnums=(1,))
def store_obs(current_timestep, latency_buffer, obs, next_obs, reward, done, obs_dim, act_dim):
    obs_block = jnp.concatenate([obs.reshape(-1), next_obs.reshape(-1)])

    meta_block = jnp.concatenate([reward.reshape(-1), done.astype(jnp.float32).reshape(-1)])
    latency_buffer = jax.lax.dynamic_update_slice(
        latency_buffer,
        obs_block.reshape(1, -1), 
        (current_timestep, 0)
    )
    latency_buffer = jax.lax.dynamic_update_slice(
        latency_buffer,
        meta_block.reshape(1, -1),
        (current_timestep, 2 * obs_dim + act_dim)
    )

    return latency_buffer

@partial(jax.jit, donate_argnums = (0,))
def step_latency_env(LatencyEnv: JaxLatencyEnv, obs, next_obs, action, reward, done, past_done):
    obs_dim = LatencyEnv.obs_dim
    act_dim = LatencyEnv.action_dim
    carry = LatencyEnv.carry
    buf_len = LatencyEnv.buffer_length

    current_timestep = carry.timestep % buf_len
    
    act_shift = carry.new_obs_positions[carry.counter_shift] - 1
    next_act_shift = carry.new_obs_positions[carry.counter_shift + 1] - 1
    
    latency_buffer = carry.buffer_latency
    latency_buffer = store_action(latency_buffer, carry.do_action, buf_len, act_shift, next_act_shift, action, obs_dim, act_dim, LatencyEnv.buffer_length//2)
    latency_buffer = store_obs(current_timestep, latency_buffer, obs, next_obs, reward, done, obs_dim, act_dim)

    candidates_offset = jnp.arange(buf_len)
    global_indices = carry.ret_ind + candidates_offset

    shifts = carry.next_obs_shift[global_indices]
    is_ready = shifts < carry.timestep

    mask_return_arr = jnp.cumprod(is_ready).astype(bool)
    mask_return_arr = jnp.logical_and(mask_return_arr, jnp.logical_not(past_done))
    n_iterations = jnp.sum(mask_return_arr)

    src_idx_curr = global_indices % buf_len
    src_idx_next = (shifts - 1) % buf_len

    batch_obs = latency_buffer[src_idx_curr, :obs_dim]
    batch_next_obs = latency_buffer[src_idx_curr, obs_dim:2*obs_dim]
    batch_act = latency_buffer[src_idx_curr, 2*obs_dim:2*obs_dim+act_dim]
    batch_rew = latency_buffer[src_idx_curr, 2*obs_dim+act_dim]
    batch_done = latency_buffer[src_idx_curr, 2*obs_dim+act_dim + 1]

    rec_data = jnp.concatenate([
        batch_obs, 
        batch_next_obs, 
        batch_act, 
        batch_rew[:, None], 
        batch_done[:, None]
    ], axis=1)

    return_arr = jnp.where(mask_return_arr[:, None], rec_data, jnp.zeros_like(rec_data))

    pos_init = carry.ret_ind % buf_len

    process_next_action = carry.mask_actions[current_timestep]
    new_ret_ind = carry.ret_ind + n_iterations
    
    new_carry = carry.replace(
        timestep = carry.timestep + 1 - (past_done | done),
        ret_ind = new_ret_ind,
        past_action = action,
        do_action = process_next_action & jnp.logical_not(past_done),
        past_obs = next_obs.reshape(-1),
        pos_init_mask = pos_init,
        n_iter_mask = n_iterations,
        buffer_latency = latency_buffer,
        counter_shift = carry.counter_shift + carry.do_action.astype(jnp.int32)
    )

    LatencyEnv = LatencyEnv.replace(carry = new_carry)

    return LatencyEnv, return_arr, mask_return_arr, n_iterations


@partial(jax.jit, donate_argnums=(0,))
def reset_latency_env(LatencyEnv: JaxLatencyEnv, obs):
    """
    Unloads the pending steps to be processed before a reset - and resets the env
    """
    
    carry = LatencyEnv.carry
    idx = jnp.arange(LatencyEnv.buffer_length)
    idx = jnp.logical_and(idx >= carry.ret_ind % LatencyEnv.buffer_length, idx <= carry.timestep % LatencyEnv.buffer_length)
    tbp_obs = carry.buffer_latency * idx[:, None]
    switch_indices = (jnp.arange(LatencyEnv.buffer_length) + carry.ret_ind) % LatencyEnv.buffer_length
    tbp_obs = tbp_obs[switch_indices]
    idx = idx[switch_indices]
    LatencyEnv = mask_generator(LatencyEnv)
    carry = LatencyEnv.carry
    carry = carry.replace(past_obs = obs.reshape(-1))
    LatencyEnv = LatencyEnv.replace(carry = carry)
    
    return LatencyEnv, tbp_obs, idx