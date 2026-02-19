import flashbax as fbx
import jax.numpy as jnp
import os
from flax import struct
from typing import NamedTuple
import jax
from functools import partial

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class CustomBuffer:
    def __init__(self, obs_dim, action_dim, buffer_shape, batch_size, ep_length):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.ep_length = ep_length
        self.memory = fbx.make_item_buffer(
            max_length=buffer_shape, min_length=batch_size, sample_batch_size=batch_size, add_batches=True
        )

        self.init_memory()
        
    def init_memory(self):
        self.example_transition = {
        "obs": jnp.arange(self.ep_length, dtype= jnp.float32).reshape(-1,self.obs_dim),
        "action": jnp.arange(self.ep_length, dtype= jnp.float32).reshape(-1,self.action_dim),
        "reward": jnp.ones(self.ep_length, dtype= jnp.float32).reshape(-1,1),
        "next_obs": jnp.arange(self.ep_length, dtype= jnp.float32).reshape(-1,self.obs_dim),
        "done": jnp.zeros(self.ep_length, dtype= jnp.float32), 
        }
        self.buffer = self.memory.init(self.example_transition)

    def add(self, batch):
        self.buffer = self.memory.add(self.buffer, batch)


class ReplayBufferSamplesJAX(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    dones: jax.Array
    rewards: jax.Array
    discounts: jax.Array

@struct.dataclass
class CustomBufferBis:
    buffer: jax.Array
    buffer_size: int
    full : jax.Array
    pos: int = 0
    past_begin_pos: int = 0

    @staticmethod
    @partial(jax.jit, static_argnames = ["batch_size", "obs_dim"])
    def sample(buf, batch_size, pos, key, discounts, obs_dim, full, buffer_size):
        key, gen_key = jax.random.split(key, 2)
        idx = jax.random.randint(gen_key, batch_size, 0, pos * (1-full) + full * buffer_size)
        
        batch = ReplayBufferSamplesJAX(
            observations= buf[idx, :obs_dim],
            actions= buf[idx, 2*obs_dim:2*obs_dim + 1].reshape(batch_size, 1),
            next_observations= buf[idx, obs_dim:2*obs_dim],
            rewards=buf[idx, 2*obs_dim+ 1],
            dones=buf[idx, 2*obs_dim+2],
            discounts=discounts
        )
        return batch, key
    
@struct.dataclass
class CustomBufferLatency:
    buffer: jax.Array
    
    buffer_size: int
    full : jax.Array
    max_latency: int = struct.field(pytree_node=False)
    pos: int = 0
    past_begin_pos: int = 0

    @staticmethod
    @partial(jax.jit, static_argnames = ["batch_size", "obs_dim", "max_latency", "act_dim"])
    def sample(buf, batch_size, pos, key, discounts, obs_dim, full, buffer_size, max_latency, act_dim):
        key, gen_key_1, gen_key_2 = jax.random.split(key, 3)
        v_min_1 = 0
        v_max_1 = jnp.maximum(pos - max_latency, 0)

        q_2 = jnp.floor(((1 - (pos + max_latency)/ buffer_size) * batch_size)  * full)
        q_1 = batch_size - q_2

        idx_1 = jax.random.randint(gen_key_1, batch_size, v_min_1, v_max_1)

        v_min_2 = jnp.minimum(pos + max_latency, buffer_size)
        v_max_2 = buffer_size
        
        idx_2 = jax.random.randint(gen_key_2, batch_size, v_min_2, v_max_2)

        arr_idx = jnp.arange(batch_size)
        mask = arr_idx < q_1
        idx = jnp.where(mask, idx_1, idx_2)
        
        batch = ReplayBufferSamplesJAX(
            observations= buf[idx, 0, :obs_dim],
            next_observations= buf[idx, :, obs_dim:2*obs_dim],
            actions= buf[idx, 0, 2*obs_dim: 2*obs_dim+ act_dim],
            rewards=buf[idx, :, 2*obs_dim + 1],
            dones=buf[idx, :, 2*obs_dim + 2],
            discounts=discounts
        )
        return batch, key

@partial(jax.jit, static_argnames = ["window_len"])
def create_sliding_windows(arr, window_len):
    N, D = arr.shape
    num_windows = N - window_len + 1
    start_indices = jnp.arange(num_windows)
    
    def get_window(i):
        return jax.lax.dynamic_slice(arr, (i, 0), (window_len, D))
    
    return jax.vmap(get_window)(start_indices)

@jax.jit
def update_unstored_obs(tbp_obs, idx, tmp_buffer, pos_buffer):
    tmp_buffer = jax.lax.dynamic_update_slice(tmp_buffer, tbp_obs, (pos_buffer, 0))    
    return tmp_buffer

@partial(jax.jit, static_argnames=["max_latency", "buffer_length", "patch_height"], donate_argnums = (0,))
def update_latency_buffer(buffer, buffer_pos, max_latency, buffer_length, mask_return_arr, return_arr, patch_height):
    adaptated_tmp_buffer = jnp.where(mask_return_arr[:, None], return_arr, jax.lax.dynamic_slice(buffer[:, max_latency, :], (buffer_pos, 0), (patch_height, buffer_length)))
    
    local_n = jnp.arange(patch_height)
    local_t = jnp.arange(max_latency)
    
    input_row_map = local_n[:, None] - (max_latency - 1 - local_t[None, :])
    
    mask = (input_row_map >= 0) & (input_row_map < patch_height)

    safe_indices = jnp.clip(input_row_map, 0, adaptated_tmp_buffer.shape[0] - 1)
    patch_updates = adaptated_tmp_buffer[safe_indices]

    return jax.lax.dynamic_update_slice(buffer, jnp.where(mask[:, :, None], patch_updates,jax.lax.dynamic_slice(
        buffer, 
        (buffer_pos, 0, 0), 
        (patch_height, max_latency, buffer_length)
    )), (buffer_pos, 0, 0))

@partial(jax.jit, static_argnames = ["max_latency", "buffer_length"], donate_argnums = (0,))
def erase_coming_obs(buffer, buffer_pos, max_latency, buffer_length):
    return jax.lax.dynamic_update_slice(buffer, jnp.zeros((max_latency, max_latency, buffer_length)), (buffer_pos, 0, 0))

@partial(jax.jit, static_argnames = ["max_latency", "buffer_length", "ep_length"], donate_argnums = (0,))
def erase_first_obs(buffer, buffer_pos, max_latency, buffer_length, pos_init, ep_length):
    slice_size = ep_length + max_latency
    relative_indices = jnp.arange(slice_size)
    threshold = buffer_pos - pos_init
    mask = relative_indices < threshold
    return jax.lax.dynamic_update_slice(buffer, jnp.where(mask[:, None, None], 
                                                          jax.lax.dynamic_slice(buffer, (pos_init + max_latency, 0,0), (slice_size, max_latency, buffer_length)), 
                                                          jax.lax.dynamic_slice(buffer, (pos_init, 0,0), (slice_size, max_latency, buffer_length))), 
                                                          (pos_init, 0,0))