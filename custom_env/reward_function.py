import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames = ["stop_loss", "stop_limit", "R", "C"])
def born_loss(x, stop_loss, stop_limit, R = 1.0, C = 0.3):
    segment = x <= 1.0
    lin_val = R * (x-1) * (segment/(1-stop_loss) + (1-segment)/(stop_limit - 1))
    return jnp.arctan(lin_val) * C

@jax.jit
def positional_loss(new_pf, pf, state, fees, diff_norm, inv_diff, ratio_fwd, ratio_inv):
    return (new_pf - pf) + (state == 0.0) * jnp.clip(jnp.maximum(ratio_fwd, ratio_inv) - fees, min = 0) + (state == 1.0) * jnp.maximum(diff_norm - fees, ratio_fwd) + (state == 2.0) * jnp.maximum(inv_diff - fees, ratio_inv)

@partial(jax.jit, static_argnames = ["stop_loss", "stop_limit", "leverage", "transaction_cost"])
def reward_function(current, past, init, action, state, pf_value, truncated, current_time, stop_loss, stop_limit, leverage, transaction_cost, pf_opt):
    current = current[0]
    past = past
    init = init
    
    ratio_fwd = (current - past) / (init + 100) * pf_value * leverage
    ratio_inv = (init + 100) * (past - current) / ((current + 100) * (past + 100))* pf_value * leverage
    M = 1.0 + leverage * jnp.array([[0.0, - transaction_cost + ratio_fwd, - transaction_cost + ratio_inv], 
                                    [- transaction_cost + ratio_fwd, ratio_fwd, -jnp.inf], 
                                    [- transaction_cost + ratio_inv, -jnp.inf, ratio_inv]])
    new_pf_value = pf_value * M[jnp.astype(state, jnp.int32), jnp.astype(action, jnp.int32)]
    
    done = truncated | (new_pf_value >= stop_limit) | (new_pf_value <= stop_loss)
    
    global_reward = (new_pf_value - 1.0) * 100.0
    return global_reward, done[0], new_pf_value
    
