import jax
import jax.numpy as jnp
from functools import partial

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
    
