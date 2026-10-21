import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames = ["stop_loss", "stop_limit", "leverage", "transaction_cost"])
def reward_function(current, past, init, action, state, pf_value, truncated, current_time, stop_loss, stop_limit, leverage, transaction_cost, pf_opt):
    current = current[0]
    past = past
    init = init
    
    ratio_fwd = (current - past) / (past + 100.0)
    ratio_inv = (past - current) / (current + 100.0)
    M = 1.0 + leverage * jnp.array([[0.0, ratio_fwd - transaction_cost, ratio_inv - transaction_cost], 
                                    [ratio_fwd - transaction_cost, ratio_fwd, -jnp.inf], 
                                    [ratio_inv - transaction_cost, -jnp.inf, ratio_inv]])
    new_pf_value = pf_value * M[jnp.astype(state, jnp.int32), jnp.astype(action, jnp.int32)]
    
    done = truncated | (new_pf_value >= stop_limit) | (new_pf_value <= stop_loss)
    
    global_reward = (new_pf_value - 1.0) * 100.0
    return global_reward, done[0], new_pf_value
    
