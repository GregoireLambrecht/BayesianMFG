import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp

def vector_torus_block_shift(key, shape, nb_states, eta):
    key_trigger, key_start, key_shift = jax.random.split(key, 3)


    max_shift = nb_states // 3
    shared_shift = jax.random.randint(
        key_shift, shape=shape,
        minval=-max_shift,
        maxval=max_shift + 1
    )

    # IMPORTANT: start_idx has shape (batch, H)
    start_idx = jax.random.randint(
        key_start, shape=shape,
        minval=0, maxval=nb_states
    )

    def apply_block_noise(s_shift_scalar,  start_idx_scalar):
        vec = jnp.zeros(nb_states)

        block_indices = (jnp.arange(2) + start_idx_scalar) % nb_states
        val = s_shift_scalar.astype(jnp.float32)

        return vec.at[block_indices].set(val)

    vmapped_fn = jax.vmap(jax.vmap(apply_block_noise))
    final_noise = vmapped_fn(shared_shift, start_idx)*((eta)**(1/3))

    return jnp.int32(final_noise)


def vector_box_asymetric(key, shape, nb_states, eta):
    """
    JAX implementation of asymmetric common noise.s
    
    Logic:
    - States < midpoint: Random push [0, nb_states//2]
    - States >= midpoint: Hard push of -nb_states//2
    - The noise only activates with probability 'eta'
    """
    key_val, key_mask = jax.random.split(key, 2)
    size_ = shape + (nb_states,)
    midpoint = nb_states // 2
    
    # 1. Generate random base noise [0, 10]
    # jax.random.randint maxval is exclusive, so 11 gives us 0-10
    noise_values = jax.random.randint(key_val, shape=size_, minval=0, maxval=nb_states//2).astype(jnp.float32)
    
    # 2. Apply Asymmetric Logic
    # If state index >= midpoint, force the value to -10.0
    indices = jnp.arange(nb_states)
    # We use jnp.where to handle the vectorized batch dimensions correctly
    noise_values = jnp.where(indices >= midpoint, -nb_states//2, noise_values)
    
    # 3. Generate the activation mask (Bernoulli(eta))
    mask = jax.random.uniform(key_mask, shape=size_) < eta
    
    # 4. Apply mask and round
    return jnp.int32(noise_values * mask.astype(jnp.float32))


def vector_box_symetric(key, shape, nb_states, eta):
    """
    JAX implementation of asymmetric common noise.s
    
    Logic:
    - States < midpoint: Random push [0, nb_states//2]
    - States >= midpoint: Hard push of -nb_states//2
    - The noise only activates with probability 'eta'
    """
    key_val, key_mask = jax.random.split(key, 2)
    size_ = shape + (nb_states,)
    midpoint = nb_states // 2
    
    # 1. Generate random base noise [0, 10]
    # jax.random.randint maxval is exclusive, so 11 gives us 0-10
    noise_values = jax.random.randint(key_val, shape=size_, minval=0, maxval=nb_states//2).astype(jnp.float32)
    
    # 2. Apply Asymmetric Logic
    # If state index >= midpoint, force the value to -10.0
    indices = jnp.arange(nb_states)
    # We use jnp.where to handle the vectorized batch dimensions correctly
    noise_values = jnp.where(indices >= midpoint, -noise_values, noise_values)
    
    # 3. Generate the activation mask (Bernoulli(eta))
    mask = jax.random.uniform(key_mask, shape=size_) < eta
    
    # 4. Apply mask and round
    return jnp.int32(noise_values * mask.astype(jnp.float32))



def vector_torus_uniform_displaced(key, shape, nb_states, eta):
    """
    For each state x: eps0(x) = eps(x) * u(x)
    eps(x) ~ Bernoulli(eta)
    u(x) ~ Uniform({-S//4, ..., S//4})
    
    shape: tuple, e.g., (B, H) or (H,)
    """
    k1, k2 = jax.random.split(key)
    
    # 1. Generate the Bernoulli mask (0 or 1)
    # Shape: (*shape, nb_states)
    eps = jax.random.bernoulli(k1, p=eta, shape=(*shape, nb_states))
    
    # 2. Generate the Uniform displacement u(x)
    # Range: [-nb_states//4, nb_states//4]
    # Note: jax.random.randint is [low, high), so we add 1 to the high bound
    low = -(nb_states // 4)
    high = (nb_states // 4) + 1
    u = jax.random.randint(k2, shape=(*shape, nb_states), minval=low, maxval=high)
    
    # 3. Combine and return
    return (eps * u).astype(jnp.int32)



