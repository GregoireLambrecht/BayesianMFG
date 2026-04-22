import jax
import jax.numpy as jnp
import equinox as eqx
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt 
import numpy as np
import optax
import distrax
import time

from jax.scipy.special import logsumexp

import os
import pickle

from utils import *

from envs.beachbar import *
from envs.common_noise_script import *


# class BayesianPolicyNN(eqx.Module):
#     layers: list
#     nb_states: int = eqx.static_field()
#     vanilla: bool = eqx.static_field()
#     activation: callable = eqx.static_field()

#     def __init__(self, env, theta_dim = 1, vanilla=False, key=None):
#         if key is None:
#             raise ValueError("A jax.random.PRNGKey must be provided as 'key'.")
            
#         self.nb_states = env.nb_states
#         self.vanilla = vanilla
#         self.activation = jax.nn.tanh
#         nb_actions = env.nb_actions
        
#         # input: time(1) + x_onehot(nb_states) [+ rho(nb_states)]
#         input_dim = theta_dim + 1 + self.nb_states if vanilla else theta_dim + 1 + self.nb_states + self.nb_states
        
#         # Split keys for 6 Linear layers
#         keys = jax.random.split(key, 6)
        
#         self.layers = [
#             eqx.nn.Linear(input_dim, 64, key=keys[0]),
#             eqx.nn.Linear(64, 64, key=keys[1]),
#             eqx.nn.Linear(64, 64, key=keys[2]),
#             eqx.nn.Linear(64, 64, key=keys[3]),
#             eqx.nn.Linear(64, 64, key=keys[4]),
#             eqx.nn.Linear(64, nb_actions, key=keys[5])
#         ]

#     def __call__(self, t, x, rho,theta):
#         """
#         t: scalar time step
#         x: scalar state index
#         rho: array (NB_STATES,) distribution
#         """
#         # 1. Prepare Inputs
#         x_onehot = jax.nn.one_hot(x, self.nb_states)
#         t_input = jnp.atleast_1d(t).astype(jnp.float32)
        
#         if self.vanilla:
#             h = jnp.concatenate([t_input, x_onehot,theta])
#         else:
#             h = jnp.concatenate([t_input, x_onehot, rho,theta])
            
#         # 2. Forward Pass through layers with activation
#         h = self.activation(self.layers[0](h))
#         h = self.activation(self.layers[1](h))
#         h = self.activation(self.layers[2](h))
#         h = self.activation(self.layers[3](h))
#         h = self.activation(self.layers[4](h))
#         logits = self.layers[5](h)
        
#         # 3. Output Probabilities
#         return jax.nn.softmax(logits)

# class BayesianPolicyNN(eqx.Module):
#     layers: list
#     film_layers: list
#     # --- ADD THESE FIELDS ---
#     depth: int = eqx.static_field() 
#     nb_states: int = eqx.static_field()
#     vanilla: bool = eqx.static_field()
#     activation: callable = eqx.static_field()
#     output_layer: eqx.nn.Linear # Dedicated final layer

#     def __init__(self, env, theta_dim=1, depth=3, vanilla=False, key=None):
#         if key is None:
#             raise ValueError("key must be provided.")
        
#         # Now these assignments will work
#         self.depth = depth 
#         self.nb_states = env.nb_states
#         self.vanilla = vanilla
#         self.activation = jax.nn.tanh
        
#         nb_actions = env.nb_actions
#         hidden_dim = 64
        
#         input_dim = 1 + self.nb_states if vanilla else 1 + self.nb_states + self.nb_states
        
#         # We need depth keys for backbone, depth for FiLM, and 1 for output
#         keys = jax.random.split(key, 2 * depth + 1) 
        
#         # 1. Main Policy Backbone (Hidden Layers)
#         # Note: i=0 takes input_dim, others take hidden_dim
#         self.layers = []
#         for i in range(depth):
#             in_d = input_dim if i == 0 else hidden_dim
#             self.layers.append(eqx.nn.Linear(in_d, hidden_dim, key=keys[i]))
        
#         # 2. Output Layer (Separate from the FiLM loop)
#         self.output_layer = eqx.nn.Linear(hidden_dim, nb_actions, key=keys[2 * depth])
        
#         # 3. FiLM Generators
#         self.film_layers = [
#             eqx.nn.Linear(theta_dim, hidden_dim * 2, key=keys[depth + i]) 
#             for i in range(depth)
#         ]

#     def __call__(self, t, x, rho, theta):
#         x_onehot = jax.nn.one_hot(x, self.nb_states)
#         t_input = jnp.atleast_1d(t).astype(jnp.float32)
#         theta_input = jnp.atleast_1d(theta)
        
#         h = jnp.concatenate([t_input, x_onehot]) if self.vanilla else jnp.concatenate([t_input, x_onehot, rho])
            
#         for i in range(self.depth):
#             h = self.layers[i](h)
            
#             film_params = self.film_layers[i](theta_input)
#             gamma, beta = jnp.split(film_params, 2)
            
#             # Apply FiLM
#             h = h * (1.0 + gamma) + beta
#             h = self.activation(h)
            
#         # Use the dedicated output layer instead of self.layers[5]
#         logits = self.output_layer(h)
#         return jax.nn.softmax(logits)

class BayesianPolicyNN(eqx.Module):
    layers: list
    film_layers_1: list  # first linear of each FiLM MLP
    film_layers_2: list  # second linear of each FiLM MLP
    output_layer: eqx.nn.Linear
    depth: int = eqx.static_field()
    nb_states: int = eqx.static_field()
    vanilla: bool = eqx.static_field()
    activation: callable = eqx.static_field()

    def __init__(self, env, theta_dim=1, depth=3, film_hidden=64,
                 vanilla=False, key=None):
        if key is None:
            raise ValueError("key must be provided.")

        self.depth = depth
        self.nb_states = env.nb_states
        self.vanilla = vanilla
        self.activation = jax.nn.tanh

        nb_actions = env.nb_actions
        hidden_dim = 64
        input_dim = 1 + self.nb_states if vanilla else 1 + 2 * self.nb_states

        # Keys: depth backbone + 2*depth film + 1 output
        keys = jax.random.split(key, depth + 2 * depth + 1)

        # Backbone
        self.layers = []
        for i in range(depth):
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(eqx.nn.Linear(in_d, hidden_dim, key=keys[i]))

        self.output_layer = eqx.nn.Linear(hidden_dim, nb_actions, key=keys[depth])

        # FiLM MLPs: theta_dim -> film_hidden -> hidden_dim*2
        self.film_layers_1 = []
        self.film_layers_2 = []
        for i in range(depth):
            k1, k2 = jax.random.split(keys[depth + 1 + 2 * i])
            self.film_layers_1.append(eqx.nn.Linear(theta_dim,   film_hidden,    key=k1))
            self.film_layers_2.append(eqx.nn.Linear(film_hidden,  hidden_dim * 2, key=k2))

    def __call__(self, t, x, rho, theta):
        x_onehot = jax.nn.one_hot(x, self.nb_states)
        t_input  = jnp.atleast_1d(t).astype(jnp.float32)
        theta    = jnp.atleast_1d(theta)

        h = jnp.concatenate([t_input, x_onehot]) if self.vanilla \
            else jnp.concatenate([t_input, x_onehot, rho])

        for i in range(self.depth):
            h = self.layers[i](h)

            # FiLM MLP: Linear -> tanh -> Linear, all on raw flat theta
            film_h      = jax.nn.tanh(self.film_layers_1[i](theta))
            film_params = self.film_layers_2[i](film_h)
            gamma, beta = jnp.split(film_params, 2)

            h = h * (1.0 + gamma) + beta
            h = self.activation(h)

        return jax.nn.softmax(self.output_layer(h))

def train_best_response_fictitious_bayesian(
    env, 
    list_policy, 
    rho0, 
    generate_theta, 
    n_iterations=100, 
    lr=1e-3, 
    batch_size=32, 
    key=None
):
    if key is None: key = jax.random.PRNGKey(0)
    key, model_key = jax.random.split(key)

    # 1. Learner uses the Augmented State (Physical + Theta)
    # env.theta_dim is the dimension of the parameter vector theta
    model = BayesianPolicyNN(env, theta_dim=env.theta_dim, key=model_key)
    # optimizer = optax.adam(lr)
    lr_scheduler = optax.cosine_decay_schedule(
        init_value=lr, 
        decay_steps=n_iterations, 
        alpha=1e-2  # Final LR will be 0.2% of the initial LR (e.g., 1e-4 -> 1e-6)
    )
    optimizer = optax.chain(
        optax.adam(learning_rate=lr_scheduler)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    

    # 2. History Pre-processing
    # These are PolicyNN instances (only physical state)
    params_list = [eqx.filter(p, eqx.is_array) for p in list_policy]
    batched_params = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *params_list)
    static_struct = eqx.filter(list_policy[0], lambda x: not eqx.is_array(x))
    batched_history = eqx.combine(batched_params, static_struct)

    def loss_fn(model, rho_init_batch, batched_hist, eps0_batch, theta_batch):        
        def single_universe_loss(r0, e0, theta):
            # 1. Specialize the environment for this universe's theta
            local_env = env.set_theta(theta)
            
            # 2. Wrap the LEARNER to match the 3-arg signature (t, x, rho)
            # generate_mean_field_scan expects a 3-arg policy.
            pi_theta = lambda t, x, rho: model(t, x, rho, theta)
            
            # 3. Compute the ENSEMBLE MEAN FIELD from history
            # We define a helper that takes ONE historical policy 'p' 
            # and runs a simulation using the current universe's theta and noise.
            def compute_hist_rho(p):
                # Wrap the historical policy to use the current theta
                p_fixed = lambda t, x, r: p(t, x, r, theta)
                return generate_mean_field_scan(local_env, r0, p_fixed, e0)

            # Map 'compute_hist_rho' over all policies in the historical batch
            # batched_hist is a PyTree where the first axis of arrays is the history index
            all_rhos = jax.vmap(compute_hist_rho)(batched_hist)
            
            # Average the distributions across the history to get the Mean Field
            rho_mf = jnp.mean(all_rhos, axis=0)
            
            # 4. Compute the LEARNER'S trajectory (Best Response)
            # How our current model performs against the historical average rho_mf
            rho_ag = generate_MF_agent_scan(local_env, r0, pi_theta, e0, rho_mf)
            
            # 5. Calculate Total Reward
            # Returns a scalar representing the performance in this specific universe
    
            # 1. Compute History Performance (The Baseline)
            def compute_hist_performance(p_params):
                p_combined = eqx.combine(p_params, static_struct)
                p_fixed = lambda t, x, r: p_combined(t, x, r, theta)
                # Each historical agent acts in their own simulation
                rho_p = generate_mean_field_scan(local_env, r0, p_fixed, e0)
                # But they all face the aggregate rho_mf
                return compute_total_reward(local_env, rho_p, p_fixed, rho_mf)
            
            # Average reward of the ensemble at this specific theta
            v_history_mean = jax.vmap(compute_hist_performance)(batched_params).mean()
            v_learner = compute_total_reward(local_env, rho_ag, pi_theta, rho_mf)

            # 3. YOUR PROPOSED NORMALIZATION
            # We use subtraction (Advantage) or division (Ratio). 
            # Subtraction is usually more stable for gradients.
            return (v_learner - v_history_mean) # Maximize the 'Gap'
            return compute_total_reward(local_env, rho_ag, pi_theta, rho_mf)

        # Vectorize the entire logic across the batch of universes
        # in_axes=(0, 0, 0) corresponds to (rho_init, eps0, theta)
        rewards = jax.vmap(single_universe_loss, in_axes=(0, 0, 0))(
            rho_init_batch, eps0_batch, theta_batch
        )
        
        # We minimize the negative expected reward
        return -jnp.mean(rewards)

    # 3. Training Loop Step
    def scan_step(carry, _):
        current_model, current_opt_state, current_key = carry
        
        sk, nk, tk = jax.random.split(current_key, 3)
        # eps0 = env.common_noise(nk, batch_size) 
        eps0 = env.common_noise(nk, (batch_size, env.H)) 
        thetas = generate_theta(tk, batch_size)
        # theta = generate_theta(tk, 1).reshape(-1) 
        # thetas = jnp.tile(theta, (batch_size, 1))
        
        # Initial distribution r0 might need to be augmented depending on your env setup
        r0_batch = jnp.tile(rho0, (batch_size, 1))
        
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
            current_model, r0_batch, batched_history, eps0, thetas
        )
        
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_model)
        next_model = eqx.apply_updates(current_model, updates)
        
        return (next_model, next_opt_state, sk), -loss_val

    @eqx.filter_jit
    def run_training(m, s, k):
        final_carry, rewards_history = jax.lax.scan(
            scan_step, (m, s, k), xs=None, length=n_iterations
        )
        return final_carry, rewards_history

    (final_model, _, _), rewards = run_training(model, opt_state, key)
    return final_model, rewards



def compute_exploitability_ficitious_bayesian(
    env,
    rho0,
    list_policy,      # Changed from policy_history to match call
    best_response,
    generate_theta,
    key,
    size_mc,         # Changed from mc_size to match call
    nb_batch_mc,
):
    """
    Computes Bayesian Exploitability averaged over common noise AND theta samples.
    """
    batch_size = size_mc // nb_batch_mc # Use size_mc here
    
    # Pre-process history into a batched PyTree
    # Use the new variable name 'list_policy'
    params_list = [eqx.filter(p, eqx.is_array) for p in list_policy]
    batched_params = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *params_list)
    static_struct = eqx.filter(list_policy[0], lambda x: not eqx.is_array(x))
    batched_hist_pytree = eqx.combine(batched_params, static_struct)

    @jax.jit
    def calculate_gap_single(r0, eps, theta):
        local_env = env.set_theta(theta)
        pi_BR_theta = lambda t, x, r: best_response(t, x, r, theta)

        def compute_hist_rho(p):
            p_fixed = lambda t, x, r: p(t, x, r, theta)
            return generate_mean_field_scan(local_env, r0, p_fixed, eps)

        all_rhos = jax.vmap(compute_hist_rho)(batched_hist_pytree)
        rho_MF = jnp.mean(all_rhos, axis=0)

        def welfare_fn(p):
            p_fixed = lambda t, x, r: p(t, x, r, theta)
            rho_p = generate_mean_field_scan(local_env, r0, p_fixed, eps)
            return compute_total_reward(local_env, rho_p, p_fixed, rho_MF)

        v_welfare = jnp.mean(jax.vmap(welfare_fn)(batched_hist_pytree))

        rho_BR = generate_MF_agent_scan(local_env, r0, pi_BR_theta, eps,rho_MF)
        v_BR = compute_total_reward(local_env, rho_BR, pi_BR_theta, rho_MF)

        return v_BR - v_welfare

    vmap_gap = jax.jit(jax.vmap(calculate_gap_single, in_axes=(0, 0, 0)))

    def mc_batch_step(carry, step_key):
        k_noise, k_theta = jax.random.split(step_key)
        eps_batch = env.common_noise(k_noise, (batch_size, env.H))
        theta_batch = generate_theta(k_theta, batch_size) 
        r0_batch = jnp.tile(rho0, (batch_size, 1))

        gaps = vmap_gap(r0_batch, eps_batch, theta_batch)
        return carry, jnp.mean(gaps)

    keys = jax.random.split(key, nb_batch_mc)
    _, batch_means = jax.lax.scan(mc_batch_step, key, keys)

    return jnp.mean(batch_means)


def run_fictitious_play_recursive_bayesian(
    env, 
    K_steps, 
    initial_policy, 
    rho0, 
    generate_theta, # Added this argument
    n_train_iters=100, 
    batch_size_train=32, 
    size_mc=1000, 
    nb_batch_mc=10,
    lr=1e-3,
    key=jax.random.PRNGKey(0),
    plot_report=False
):
    policy_history = [initial_policy]
    nash_gaps = []

    for k in range(1, K_steps + 1):
        print(f"--- FP Round {k}/{K_steps} ---")
        
        # Split keys for the three distinct operations
        key, train_key, mc_key = jax.random.split(key, 3)
        
        # 1. Train the Bayesian Best Response
        # Note: 'generate_theta' is passed here to sample alpha_cong during SGD
        new_best_response, rewards = train_best_response_fictitious_bayesian(
            env=env,
            list_policy=policy_history,
            rho0=rho0,
            generate_theta=generate_theta,
            n_iterations=n_train_iters,
            lr=lr,
            batch_size=batch_size_train,
            key=train_key
        )
        
        if plot_report:
            plt.plot(rewards)
            plt.title(f"Training Reward - Round {k}")
            plt.show()

        # 2. Compute Bayesian Exploitability
        # This function should sample thetas using mc_key and 
        # compute the average gap across the parameter space.
        gap = compute_exploitability_ficitious_bayesian(
            env=env, 
            rho0=rho0, 
            list_policy=policy_history, 
            best_response=new_best_response, 
            generate_theta=generate_theta,
            key=mc_key, 
            size_mc=size_mc,
            nb_batch_mc=nb_batch_mc
        )
        
        nash_gaps.append(gap)
        print(f"   Nash Gap: {gap:.6f}")

        # 3. Add to history
        policy_history.append(new_best_response)

    return policy_history, jnp.array(nash_gaps)



def learn_fictitious_policy_bayesian(
    env, 
    rho0, 
    list_fictitious, 
    generate_theta, 
    n_iterations, 
    batch_size, 
    lr, 
    key=None, 
):
    """
    Learns a single BayesianPolicyNN that mimics the average behavior 
    of the FP history across the entire theta distribution.
    """
    if key is None: key = jax.random.PRNGKey(0)
    key, model_key = jax.random.split(key)

    # 1. Initialize the Bayesian learner (augmented state: x + theta)
    learner = BayesianPolicyNN(env, theta_dim=env.theta_dim, key=model_key)
    
    # optimizer = optax.adam(lr)
    
    # opt_state = optimizer.init(eqx.filter(learner, eqx.is_array))
    lr_scheduler = optax.cosine_decay_schedule(
        init_value=lr, 
        decay_steps=n_iterations, 
        alpha=1e-2  # Final LR will be 1% of the initial LR (e.g., 1e-4 -> 1e-6)
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_scheduler)
    )
    opt_state = optimizer.init(eqx.filter(learner, eqx.is_array))

    # 2. Batch the history (Assumes history contains BayesianPolicyNNs)
    params_list = [eqx.filter(p, eqx.is_array) for p in list_fictitious]
    batched_params = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *params_list)
    static_struct = eqx.filter(list_fictitious[0], lambda x: not eqx.is_array(x))
    batched_history = eqx.combine(batched_params, static_struct)

    def loss_fn(model, r0_batch, batched_hist, eps0_batch, theta_batch):
        def single_universe_loss(r0, e0, theta):
            # A. Specialize the environment for this sampled theta
            local_env = env.set_theta(theta)
            
            # B. Compute the target: Average mu from history at THIS theta
            # Every policy in history is evaluated at the SAME theta
            def get_hist_mu(p):
                p_fixed = lambda t, x, r: p(t, x, r, theta)
                return generate_mu_scan(local_env, r0, p_fixed, e0)

            all_mu_hist = jax.vmap(get_hist_mu)(batched_hist)
            mu_target = jnp.mean(all_mu_hist, axis=0) # (H, S, A)

            # C. Compute the learner's specialized distribution
            pi_learner_theta = lambda t, x, r: model(t, x, r, theta)
            mu_learner = generate_mu_scan(local_env, r0, pi_learner_theta, e0)

            return jnp.sum(jnp.abs(mu_learner - mu_target))

        # Vmap over the batch of (r0, noise, theta)
        losses = jax.vmap(single_universe_loss, in_axes=(0, 0, 0))(
            r0_batch, eps0_batch, theta_batch
        )
        return jnp.mean(losses)

    def scan_step(carry, _):
        current_model, current_opt_state, current_key = carry
        
        # Sample theta, noise, and setup batch
        k1, k2, k3 = jax.random.split(current_key, 3)
        theta_batch = generate_theta(k1, batch_size)

        # theta = generate_theta(k1, 1).reshape(-1) 
        # theta_batch = jnp.tile(theta, (batch_size, 1))

        eps0_batch = env.common_noise(k2, (batch_size, env.H)) 
        r0_batch = jnp.tile(rho0, (batch_size, 1))
        
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
            current_model, r0_batch, batched_history, eps0_batch, theta_batch
        )
        
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_model)
        next_model = eqx.apply_updates(current_model, updates)
        
        return (next_model, next_opt_state, k3), loss_val

    # 3. Training Loop
    (final_learner, _, _), loss_history = jax.lax.scan(
        scan_step, 
        (learner, opt_state, key), 
        None, 
        length=n_iterations
    )

    return final_learner, loss_history




def compute_single_policy_exploitability_bayesian(env, rho0, target_policy,generate_theta, n_iterations=200, mc_size=10000, nb_batch_mc = 10, lr=1e-3, batch_size = 100, key = None):
    """
    1. Trains a Best Response against the 'target_policy'.
    2. Computes the Gap: V(BR vs target_policy) - V(target_policy vs target_policy).
    """
    if key is None: 
        # Fallback only if no key is provided, but we should always pass one
        key = jax.random.PRNGKey(0)

    key, model_key,eval_key = jax.random.split(key,3)
    # Step 1: Wrap the single policy into a list so we can reuse your FP training function
    # Your training function expects a list (the history) to build the Mean Field
    policy_list = [target_policy]

    # Step 2: Train the Best Response against this specific policy
    # print("Training Best Response against the target policy...")
    br_model, _ = train_best_response_fictitious_bayesian(
        env, policy_list, rho0, generate_theta, n_iterations=n_iterations, lr=lr,batch_size=batch_size,key=model_key
    )

    # Step 3: Compute the exploitability gap using Monte Carlo
    # We create a new key for the evaluation
    
    # Re-using your compute_exploitability function
    gap = compute_exploitability_ficitious_bayesian(
        env, rho0, policy_list, br_model,generate_theta, eval_key, mc_size, nb_batch_mc
    )
    
    return gap, br_model




def compute_agg_MF_bayesian_theta_fixed(env, rho0, pi, eps0, theta):
    """
    Computes the aggregate Mean Field trajectory for a Bayesian Ensemble.
    """
    local_env = env.set_theta(theta)
    # Ensure theta is at least 1D (e.g., shape (1,))
    theta_vec = jnp.atleast_1d(theta)
    def p_fixed(t, x, r):
        return pi(t, x, r, theta_vec)

    return generate_mean_field_scan(local_env, rho0, p_fixed, eps0)


def sample_rho(env,rho0, pi, key, N):
    eps0s = env.common_noise(key, (N, env.H)) # (N, H) 
    def get_single_trajectory(eps0):
        rho_mf = generate_mean_field_scan(env, rho0, pi, eps0)
        rho_flat = rho_mf.reshape(-1)
        return rho_flat
    dataset = jax.vmap(get_single_trajectory)(eps0s)
    return dataset

def sample_mu(env,rho0, pi, key, N):
    eps0s = env.common_noise(key, (N, env.H)) # (N, H) 
    def get_single_trajectory(eps0):
        mu_mf = generate_mu_scan(env, rho0, pi, eps0)
        mu_flat= mu_mf.reshape(-1)
        return mu_flat
    dataset = jax.vmap(get_single_trajectory)(eps0s)
    return dataset

def filter_samples(env, samples, indices_I,use_mu = False):
    if use_mu: 
        samples_reshaped = samples.reshape(samples.shape[0], -1, env.nb_states, env.nb_actions)
        samples_filtered = samples_reshaped[:, indices_I, :]
        samples = samples_filtered.reshape(samples.shape[0], -1)
    else :
        samples_reshaped = samples.reshape(samples.shape[0], -1, env.nb_states)
        samples_filtered = samples_reshaped[:, indices_I, :]
        samples = samples_filtered.reshape(samples.shape[0], -1)
    return samples



def sample_theta_rho_bayesian(env,rho0, generate_theta, pi, key, N):
    """
    Samples N pairs of (theta, rho) and concatenates them into a single dataset.
    rho is the full mean field trajectory flattened into a vector.
    
    Returns:
        Data: Array of shape (N, theta_dim + H * nb_states)
    """
    # 1. Split keys for theta and common noise
    key_theta, key_noise = jax.random.split(key)
    
    # 2. Sample N thetas and N noise realizations
    thetas = generate_theta(key_theta, N)  # (N, theta_dim)
    eps0s = env.common_noise(key_noise, (N, env.H)) # (N, H)
    
    # 3. Define a function to get rho for a single (theta, eps) pair
    def get_single_trajectory(theta_sample, eps0):
        # Specialize the environment
        local_env = env.set_theta(theta_sample)
        
        p_fixed = lambda t, x, r: pi(t, x, r, theta_sample)

        rho_mf = generate_mean_field_scan(local_env, rho0, p_fixed, eps0)
        
        # Flatten rho into a 1D vector: (H * nb_states,)
        rho_flat = rho_mf.reshape(-1)

        return jnp.concatenate([theta_sample, rho_flat])

    dataset = jax.vmap(get_single_trajectory)(thetas, eps0s)
    
    return dataset


def sample_theta_mu_bayesian(env,rho0, generate_theta, pi, key, N):
    """
    Samples N pairs of (theta, rho) and concatenates them into a single dataset.
    rho is the full mean field trajectory flattened into a vector.
    
    Returns:
        Data: Array of shape (N, theta_dim + H * nb_states)
    """
    # 1. Split keys for theta and common noise
    key_theta, key_noise = jax.random.split(key)
    
    # 2. Sample N thetas and N noise realizations
    thetas = generate_theta(key_theta, N)  # (N, theta_dim)
    eps0s = env.common_noise(key_noise, (N, env.H)) # (N, H)
    
    # 3. Define a function to get rho for a single (theta, eps) pair
    def get_single_trajectory(theta_sample, eps0):
        # Specialize the environment
        local_env = env.set_theta(theta_sample)
        
        p_fixed = lambda t, x, r: pi(t, x, r, theta_sample)

        mu_mf = generate_mu_scan(local_env, rho0, p_fixed, eps0)
        
        # Flatten rho into a 1D vector: (H * nb_states,)
        rho_flat = mu_mf.reshape(-1)

        return jnp.concatenate([theta_sample, rho_flat])

    dataset = jax.vmap(get_single_trajectory)(thetas, eps0s)
    
    return dataset



# class ConditionalMAF(eqx.Module):
#     conditioners: list 
#     base_dist: distrax.Distribution = eqx.field(static=True)
#     event_dim: int = eqx.field(static=True)
#     num_layers: int = eqx.field(static=True)
#     context_dim: int = eqx.field(static=True)

#     def __init__(self, event_dim, context_dim, hidden_dim, num_layers, key):
#         self.event_dim = event_dim
#         self.context_dim = context_dim
#         self.num_layers = num_layers
        
#         # Base distribution is a simple Standard Normal
#         self.base_dist = distrax.MultivariateNormalDiag(
#             loc=jnp.zeros(event_dim), 
#             scale_diag=jnp.ones(event_dim)
#         )
        
#         split = event_dim // 2
#         keys = jax.random.split(key, num_layers)
#         self.conditioners = []
        
#         for i in range(num_layers):
#             # Define MLP layers
#             l1 = eqx.nn.Linear(split + context_dim, hidden_dim, key=keys[i])
#             l2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=jax.random.split(keys[i])[0])
#             l3 = eqx.nn.Linear(hidden_dim, (event_dim - split) * 2, key=jax.random.split(keys[i])[1])
            
#             # STABILITY: Zero-init the final layer so the flow starts as Identity
#             l3 = eqx.tree_at(lambda l: l.weight, l3, jnp.zeros_like(l3.weight))
#             l3 = eqx.tree_at(lambda l: l.bias, l3, jnp.zeros_like(l3.bias))
            
#             # Use eqx.nn.Lambda to wrap jax.nn.tanh for Sequential compatibility
#             conditioner = eqx.nn.Sequential([
#                 l1, eqx.nn.Lambda(jax.nn.tanh),
#                 l2, eqx.nn.Lambda(jax.nn.tanh),
#                 l3
#             ])
#             self.conditioners.append(conditioner)

#     def log_prob(self, rho_flat, theta):
#         """Calculates log P(rho | theta)"""
#         # 1. PRE-PROCESS: Logit Transform for [0, 1] bounded data
#         eps = 1e-6
#         rho_clamped = jnp.clip(rho_flat, eps, 1.0 - eps)
#         x = jnp.log(rho_clamped / (1.0 - rho_clamped))
        
#         # Log-det for the logit transform
#         # Change of variables: log(dy/dx) = log(1/x + 1/(1-x))
#         logit_log_det = jnp.sum(-jnp.log(rho_clamped) - jnp.log(1.0 - rho_clamped))
        
#         total_log_det = logit_log_det
#         split = self.event_dim // 2

#         # 2. FORWARD PASS: Flow through Coupling Layers
#         for i in range(self.num_layers):
#             x1, x2 = x[:split], x[split:]
            
#             # Conditioner sees half the trajectory and the environment parameter theta
#             params = self.conditioners[i](jnp.concatenate([x1, theta], axis=-1))
#             shift, log_scale = jnp.split(params, 2, axis=-1)
            
#             # STABILITY: Clamp log_scale to prevent numerical explosion
#             log_scale = 1 * jnp.tanh(log_scale / 3.0) 
            
#             # Apply transformation
#             y2 = x2 * jnp.exp(log_scale) + shift
#             total_log_det += jnp.sum(log_scale)
            
#             # Combine and Permute (Reverse) for the next layer
#             x = jnp.concatenate([x1, y2], axis=-1)[::-1]
            
#         return self.base_dist.log_prob(x) + total_log_det

#     def sample(self, theta, key, num_samples=1):
#         """Generates rho ~ P(rho | theta)"""
#         # 1. Sample from Gaussian base distribution
#         z_samples = self.base_dist.sample(seed=key, sample_shape=(num_samples,))
#         split = self.event_dim // 2

#         def single_inverse(z):
#             x = z
#             # 2. INVERSE PASS: Go backwards through the flow
#             for i in reversed(range(self.num_layers)):
#                 # Undo Permutation (Reverse is its own inverse)
#                 x = x[::-1]
                
#                 x1, x2 = x[:split], x[split:]
#                 params = self.conditioners[i](jnp.concatenate([x1, theta], axis=-1))
#                 shift, log_scale = jnp.split(params, 2, axis=-1)
                
#                 # Apply the same tanh clamp used in forward
#                 log_scale = 3.0 * jnp.tanh(log_scale / 3.0)
                
#                 # Inverse: x = (y - shift) * exp(-log_scale)
#                 original_x2 = (x2 - shift) * jnp.exp(-log_scale)
#                 x = jnp.concatenate([x1, original_x2], axis=-1)
            
#             # 3. POST-PROCESS: Inverse Logit (Sigmoid) to return to [0, 1]
#             return jax.nn.sigmoid(x)
            
#         return jax.vmap(single_inverse)(z_samples)
    
class ConditionalMAF(eqx.Module):
    conditioners_l1: list
    conditioners_l2: list
    conditioners_l3: list
    theta_encoders: list      
    base_dist: distrax.Distribution = eqx.field(static=True)
    event_dim:      int  = eqx.field(static=True)
    num_layers:     int  = eqx.field(static=True)
    context_dim:    int  = eqx.field(static=True)
    nb_states:      int  = eqx.field(static=True)
    theta_embed_dim: int = eqx.field(static=True)
    use_simplex:    bool = eqx.field(static=True)  # ← new flag

    def __init__(self, event_dim, context_dim, hidden_dim, num_layers,
                 nb_states, key, theta_embed_dim=64, use_simplex=True):
        self.event_dim        = event_dim
        self.context_dim      = context_dim
        self.num_layers       = num_layers
        self.nb_states        = nb_states
        self.theta_embed_dim  = theta_embed_dim
        self.use_simplex      = use_simplex   # True for mu, False for rho

        self.base_dist = distrax.MultivariateNormalDiag(
            loc        = jnp.zeros(event_dim),
            scale_diag = jnp.ones(event_dim)
        )

        keys = jax.random.split(key, num_layers * 4)
        self.conditioners_l1 = []
        self.conditioners_l2 = []
        self.conditioners_l3 = []
        self.theta_encoders  = []

        for i in range(num_layers):
            split_i  = event_dim // 2 if i % 2 == 0 else event_dim - event_dim // 2
            output_i = event_dim - split_i

            theta_enc = eqx.nn.Linear(context_dim, theta_embed_dim, key=keys[4*i])
            l1 = eqx.nn.Linear(split_i + theta_embed_dim, hidden_dim, key=keys[4*i+1])
            l2 = eqx.nn.Linear(hidden_dim, hidden_dim,               key=keys[4*i+2])
            l3 = eqx.nn.Linear(hidden_dim, output_i * 2,             key=keys[4*i+3])

            l3 = eqx.tree_at(lambda l: l.weight, l3, jnp.zeros_like(l3.weight))
            l3 = eqx.tree_at(lambda l: l.bias,   l3, jnp.zeros_like(l3.bias))

            self.theta_encoders.append(theta_enc)
            self.conditioners_l1.append(l1)
            self.conditioners_l2.append(l2)
            self.conditioners_l3.append(l3)

    def _conditioner(self, i, x1, theta):
        theta_emb = jax.nn.tanh(self.theta_encoders[i](theta))
        h = jax.nn.tanh(self.conditioners_l1[i](jnp.concatenate([x1, theta_emb])))
        h = jax.nn.tanh(self.conditioners_l2[i](h))
        return self.conditioners_l3[i](h)

    def _preprocess(self, rho_flat):
        """Simplex -> R^n  (forward, with log-det)"""
        eps = 1e-6
        if self.use_simplex:
            # Centered log transform — preserves simplex structure
            # log-det: -sum(log(rho)) - nb_states * log(mean) per block (approx const, ignored)
            rho_blocks = rho_flat.reshape(-1, self.nb_states)
            log_rho    = jnp.log(jnp.clip(rho_blocks, eps, 1.0))
            x          = (log_rho - jnp.mean(log_rho, axis=-1, keepdims=True)).reshape(-1)
            # Log-det of centered log transform
            log_det    = jnp.sum(-jnp.log(jnp.clip(rho_flat, eps, 1.0)))
        else:
            # Logit transform — for independent [0,1] variables
            rho_c   = jnp.clip(rho_flat, eps, 1.0 - eps)
            x       = jnp.log(rho_c / (1.0 - rho_c))
            log_det = jnp.sum(-jnp.log(rho_c) - jnp.log(1.0 - rho_c))
        return x, log_det

    def _postprocess(self, x):
        """R^n -> Simplex (inverse)"""
        if self.use_simplex:
            x_blocks   = x.reshape(-1, self.nb_states)
            rho_blocks = jax.nn.softmax(x_blocks, axis=-1)
            return rho_blocks.reshape(-1)
        else:
            return jax.nn.sigmoid(x)

    def log_prob(self, rho_flat, theta):
        theta = jnp.atleast_1d(theta)
        x, total_log_det = self._preprocess(rho_flat)

        for i in range(self.num_layers):
            split_i          = self.event_dim // 2 if i % 2 == 0 else self.event_dim - self.event_dim // 2
            x1, x2           = x[:split_i], x[split_i:]
            params            = self._conditioner(i, x1, theta)
            shift, log_scale  = jnp.split(params, 2, axis=-1)
            log_scale         = 1 * jnp.tanh(log_scale / 3.0)
            y2                = x2 * jnp.exp(log_scale) + shift
            total_log_det    += jnp.sum(log_scale)
            x                 = jnp.concatenate([x1, y2], axis=-1)[::-1]

        return self.base_dist.log_prob(x) + total_log_det

    def sample(self, theta, rho0, key, num_samples=1):
        theta     = jnp.atleast_1d(theta)
        z_samples = self.base_dist.sample(seed=key, sample_shape=(num_samples,))

        def single_inverse(z):
            x = z
            for i in reversed(range(self.num_layers)):
                x                = x[::-1]
                split_i          = self.event_dim // 2 if i % 2 == 0 else self.event_dim - self.event_dim // 2
                x1, x2           = x[:split_i], x[split_i:]
                params            = self._conditioner(i, x1, theta)
                shift, log_scale  = jnp.split(params, 2, axis=-1)
                log_scale         = 1 * jnp.tanh(log_scale / 3.0)
                original_x2       = (x2 - shift) * jnp.exp(-log_scale)
                x                 = jnp.concatenate([x1, original_x2], axis=-1)
            return self._postprocess(x)

        samples    = jax.vmap(single_inverse)(z_samples)
        rho0_tiled = jnp.tile(rho0, (num_samples, 1))
        return jnp.concatenate([rho0_tiled, samples], axis=-1)
    

def apply_simplex_noise(key, rho_flat, nb_states, strength=1):
    """
    Perturbs probability distributions on the simplex using Dirichlet noise.
    
    rho_flat: (batch, num_indices * nb_states)
    nb_states: dimensionality of the simplex (S or S*A)
    epsilon_strength: how much to weight the noise (higher = more perturbation)
    """
    batch_size = rho_flat.shape[0]
    # Reshape to (batch, num_time_steps, nb_states)
    rho_blocks = rho_flat.reshape(batch_size, -1, nb_states)
    
    # 1. Sample Dirichlet(1, ..., 1) for each block
    # alpha=1 makes it a uniform distribution over the simplex
    noise_key, _ = jax.random.split(key)
    epsilon = jax.random.dirichlet(noise_key, strength*jnp.ones(nb_states), shape=(batch_size, rho_blocks.shape[1]))
    
    rho_perturbed = rho_blocks * epsilon
    
    # 3. Renormalize over the state dimension
    rho_perturbed = rho_perturbed / jnp.sum(rho_perturbed, axis=-1, keepdims=True)
    
    return rho_perturbed.reshape(batch_size, -1)



def train_nle_online(
    env, 
    model, 
    rho0,
    generate_theta,
    pi, 
    indices_I,        # [NEW] Time steps to include in training
    use_mu=False,     # [NEW] Toggle: True uses sample_theta_mu, False uses rho
    n_steps=10000,   
    lr=1e-4,         
    batch_size=128, 
    key=None
):
    if key is None: key = jax.random.PRNGKey(0)
    
    # 1. Setup Optimizer & Partition
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr)
    )
    model_params, model_static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(model_params)

    theta_dim = env.theta_dim

    # --- The Core Step: Generate + Train ---
    def train_step(carry, step_key):
        params, opt_s = carry
        
        # 2. DATA GENERATION LOGIC
        if use_mu:
            # Uses the expected density simulation
            dataset = sample_theta_mu_bayesian(
                env, rho0, generate_theta, pi, step_key, batch_size
            )
            # 3. SLICE AND FILTER BY TIME STEPS
            thetas = dataset[:, :theta_dim]
            full_traj = dataset[:, theta_dim:] # (batch, H * nb_states)
            
            rho_flat = filter_samples(env, full_traj, indices_I, use_mu)
            dim_xi = env.nb_states*env.nb_actions
            rho_flat = apply_simplex_noise(key, rho_flat,dim_xi, strength=np.sqrt(dim_xi))
            # print(rho_flat.shape)
        else:
            # Uses the stochastic realization simulation
            dataset = sample_theta_rho_bayesian(
                env, rho0, generate_theta, pi, step_key, batch_size
            )
            # 3. SLICE AND FILTER BY TIME STEPS
            thetas = dataset[:, :theta_dim]
            full_traj = dataset[:, theta_dim:] # (batch, H * nb_states)
            
            rho_flat = filter_samples(env, full_traj, indices_I, use_mu)

        # 4. COMPUTE LOSS AND GRADIENT
        def loss_fn(p):
            m = eqx.combine(p, model_static)
            log_p = jax.vmap(m.log_prob)(rho_flat, thetas)

            # 1. Standard Negative Log Likelihood
            nll = -jnp.mean(log_p)
            # 2. LASSO / L1 Penalty (Encourages Sparsity)
            # We apply it to the conditioners to find the most important features
            # l1_sum = sum(jnp.sum(jnp.abs(l.weight)) for l in model.conditioners_l1)
            # 3. Combined Loss
            return nll 
            # return -jnp.mean(log_p)

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params)
        
        # 5. UPDATE
        updates, next_opt_s = optimizer.update(grads, opt_s, params)
        next_params = eqx.apply_updates(params, updates)
        
        return (next_params, next_opt_s), loss_val

    # --- 6. Execution via lax.scan ---
    keys = jax.random.split(key, n_steps)
    
    (final_params, _), loss_history = jax.lax.scan(
        train_step, (model_params, opt_state), keys
    )
    
    return eqx.combine(final_params, model_static), loss_history    



def generate_uniform(key, batch_size, theta_dim=1, low=0.0, high=2):
    """
    Returns an array of shape (batch_size, theta_dim).
    For theta_dim=1, this looks like [[val1], [val2], ..., [valN]].
    """
    return jax.random.uniform(
        key, 
        shape=(batch_size, theta_dim), 
        minval=low, 
        maxval=high
    )




def compute_likelihood_uniform_prior(thetas_grid, samples, model_flow):
    batch_log_prob = jax.vmap(
    lambda t: jax.vmap(lambda r: model_flow.log_prob(r, t))(samples)
    )(thetas_grid)
    sum_scan_ll = jnp.sum(batch_log_prob, axis=1)
    log_evidence = logsumexp(sum_scan_ll)
    log_like = sum_scan_ll - log_evidence
    likelihood = jnp.exp(log_like) 
    theta_map = thetas_grid[log_like.argmax()]
    
    return log_like, likelihood, theta_map


def train_best_response_vs_bayesian_theta_fixed(
    env, 
    rho0, 
    pi_bays, 
    theta_fixed, 
    n_iterations=1000, 
    lr=5e-4, 
    batch_size=128, 
    key=None
):
    """
    Trains a Best Response against the expected Mean Field of a 
    Bayesian Ensemble at a specific theta.
    """
    if key is None: key = jax.random.PRNGKey(42)
    key, model_key = jax.random.split(key)

    # 1. Initialize the Best Response Model
    # Note: Use the same architecture as your ensemble policies
    model = PolicyNN(env, key=model_key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Standardize theta to 1D array
    theta_vec = jnp.atleast_1d(theta_fixed)

    def loss_fn(model_params, r0_batch, eps0_batch):
        current_br = model_params

        def single_universe_loss(r0, e0):

            p_fixed = lambda t, x, r: pi_bays(t, x, r, theta_vec)
            rho_mf = generate_mean_field_scan(env, r0, p_fixed, e0)

            br_fixed = lambda t, x, r: current_br(t, x, r)
            rho_ag = generate_MF_agent_scan(env, r0, br_fixed, e0, rho_mf)

            return compute_total_reward(env, rho_ag, br_fixed, rho_mf)

        rewards = jax.vmap(single_universe_loss)(r0_batch, eps0_batch)
        return -jnp.mean(rewards)

    @eqx.filter_jit
    def scan_step(carry, _):
        m, s, k = carry
        k, noise_key = jax.random.split(k)
        
        eps0 = env.common_noise(noise_key, (batch_size, env.H))
        r0_batch = jnp.tile(rho0, (batch_size, 1))
        
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(m, r0_batch, eps0)
        updates, next_s = optimizer.update(grads, s, m)
        next_m = eqx.apply_updates(m, updates)
        
        return (next_m, next_s, k), -loss_val

    # Run the loop on device
    (final_model, _, _), rewards_history = jax.lax.scan(
        scan_step, (model, opt_state, key), length=n_iterations
    )
    
    return final_model, rewards_history



def compute_exploitability_bayesian_fixed_theta(
    env,
    rho0,
    pi_bays,  # List of BayesianPolicyNN (ensemble)
    best_response,   # The current BayesianPolicyNN being trained
    theta_fixed,     # The specific theta to evaluate (scalar or 1D array)
    key,
    mc_size,
    nb_batch_mc,
):

    batch_size = mc_size // nb_batch_mc
    # Ensure theta is correctly shaped for the NN calls
    theta_input = jnp.atleast_1d(theta_fixed)

    @jax.jit
    def calculate_gap_single(r0, eps):

        p_fixed = lambda t, x, rho: pi_bays(t, x, rho, theta_input)
        rho_mf = generate_mean_field_scan(env, r0, p_fixed, eps)
        v_mf = compute_total_reward(env, rho_mf, p_fixed, rho_mf)

        rho_br = generate_MF_agent_scan(env, r0, best_response, eps, rho_mf)
        v_br = compute_total_reward(env, rho_br, best_response, rho_mf)

        return v_br, v_mf

    vmap_gap = jax.vmap(calculate_gap_single)

    def mc_batch_step(carry, k):
        k, subkey = jax.random.split(k)
        eps_batch = env.common_noise(subkey, (batch_size, env.H))
        r0_batch = jnp.tile(rho0, (batch_size, 1))

        v_BR, v_welfare = vmap_gap(r0_batch, eps_batch)
        return k, (jnp.mean(v_BR - v_welfare), jnp.mean(v_welfare))

    keys = jax.random.split(key, nb_batch_mc)
    _, (batch_gaps, batch_welfare) = jax.lax.scan(mc_batch_step, key, keys)

    return jnp.mean(batch_gaps), jnp.mean(batch_welfare)




def compute_reward_bays_theta_fixed_vs_determinist(
    env, 
    rho0, 
    pi_bays, 
    pi_det, 
    theta_fixed,
    mc_size=1000, 
    nb_batch_mc=10, 
    key=None
):
    """
    1. V(Bays vs Det): Bayesian ensemble members playing against the Det-Ensemble Mean Field.
    2. V(Det vs Det): Deterministic ensemble members playing against their own Mean Field.
    """
    if key is None: key = jax.random.PRNGKey(0)
    batch_size = mc_size // nb_batch_mc
    theta_input = jnp.atleast_1d(theta_fixed)

    @jax.jit
    def eval_step(r0, eps):
        p_fixed = lambda t, x, rho: pi_bays(t, x, rho, theta_input)
        rho_mf = generate_mean_field_scan(env, r0, pi_det, eps)
        rho_ag = generate_MF_agent_scan(env, r0, p_fixed, eps, rho_mf)

        v_mf = compute_total_reward(env, rho_mf, pi_det, rho_mf)
        v_ag = compute_total_reward(env, rho_ag, p_fixed, rho_mf)

        return v_ag, v_mf

    vmap_eval = jax.vmap(eval_step)

    def mc_batch_step(carry, subkey):
        eps_batch = env.common_noise(subkey, (batch_size, env.H))
        r0_batch = jnp.tile(rho0, (batch_size, 1))
        res_bays, res_det = vmap_eval(r0_batch, eps_batch)
        return carry, (jnp.mean(res_bays), jnp.mean(res_det))

    keys = jax.random.split(key, nb_batch_mc)
    _, (final_bays, final_det) = jax.lax.scan(mc_batch_step, None, keys)

    return jnp.mean(final_bays), jnp.mean(final_det)





def train_best_response_vs_bma(
    env, 
    rho0, 
    pi_bays, 
    theta_grid,
    theta_probs,
    n_iterations=1000, 
    lr=5e-4, 
    batch_size=32, 
    key=None
):
    """
    Trains a Best Response against a Mean Field generated by the 
    weighted average of historical Bayesian policies.
    """
    if key is None: key = jax.random.PRNGKey(42)
    key, model_key = jax.random.split(key)

    model = PolicyNN(env, key=model_key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(model_params, r0_batch, eps0_batch):
        current_br = model_params

        def single_universe_loss(r0, e0):
            def crowd_policy(t, x, r):
                # Inner function to get average action across history for a fixed theta
                def get_hist_action(single_policy):
                    # vmap this over the theta grid
                    def get_theta_action(single_theta):
                        return single_policy(t, x, r, single_theta)
                    
                    actions_theta = jax.vmap(get_theta_action)(theta_grid)
                    return jnp.tensordot(theta_probs, actions_theta, axes=1)
                
                actions_history = get_hist_action(pi_bays)
                return actions_history


            rho_mf = generate_mean_field_scan(env, r0, crowd_policy, e0)   
            rho_ag = generate_MF_agent_scan(env, r0, current_br, e0, rho_mf)
            return compute_total_reward(env, rho_ag, current_br, rho_mf)

        rewards = jax.vmap(single_universe_loss)(r0_batch, eps0_batch)
        return -jnp.mean(rewards)

    @eqx.filter_jit
    def scan_step(carry, _):
        m, s, k = carry
        k, noise_key = jax.random.split(k)
        eps0 = env.common_noise(noise_key, (batch_size, env.H))
        r0_batch = jnp.tile(rho0, (batch_size, 1))
        
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(m, r0_batch, eps0)
        updates, next_s = optimizer.update(grads, s, m)
        next_m = eqx.apply_updates(m, updates)
        
        return (next_m, next_s, k), -loss_val

    (final_model, _, _), rewards_history = jax.lax.scan(
        scan_step, (model, opt_state, key), length=n_iterations
    )
    
    return final_model, rewards_history



def compute_exploitability_bma(
    env,
    rho0,
    pi_bays,  # List/Ensemble of BayesianPolicyNN
    best_response,   # Current BayesianPolicyNN
    theta_grid,      # (N_grid, theta_dim)
    theta_probs,     # (N_grid,)
    key,
    mc_size,
    nb_batch_mc,
):

    batch_size = mc_size // nb_batch_mc

    @jax.jit
    def calculate_gap_single(r0, eps):
        def crowd_policy(t, x, r):
            # Inner function to get average action across history for a fixed theta
            def get_hist_action(single_policy):
                # vmap this over the theta grid
                def get_theta_action(single_theta):
                    return single_policy(t, x, r, single_theta)
                
                actions_theta = jax.vmap(get_theta_action)(theta_grid)
                return jnp.tensordot(theta_probs, actions_theta, axes=1)
            
            actions_history = get_hist_action(pi_bays)
            return actions_history

        rho_mf = generate_mean_field_scan(env, r0, crowd_policy, eps)   
        rho_ag = generate_MF_agent_scan(env, r0, best_response, eps, rho_mf)

        v_br = compute_total_reward(env, rho_ag, best_response, rho_mf)
        v_mf = compute_total_reward(env, rho_mf, crowd_policy, rho_mf)

        return v_br, v_mf

    vmap_gap = jax.vmap(calculate_gap_single)

    def mc_batch_step(carry, k):
        k, subkey = jax.random.split(k)
        eps_batch = env.common_noise(subkey, (batch_size, env.H))
        r0_batch = jnp.tile(rho0, (batch_size, 1))

        v_BR, v_welf = vmap_gap(r0_batch, eps_batch)
        return k, (jnp.mean(v_BR - v_welf), jnp.mean(v_welf))

    keys = jax.random.split(key, nb_batch_mc)
    _, (gaps, welfares) = jax.lax.scan(mc_batch_step, key, keys)

    return jnp.mean(gaps), jnp.mean(welfares)



def compute_reward_bma_vs_deterministic(
    env,
    rho0,
    pi_bays,  # List/Ensemble of BayesianPolicyNN
    pi_det,   # Current BayesianPolicyNN
    theta_grid,      # (N_grid, theta_dim)
    theta_probs,     # (N_grid,)
    key,
    mc_size,
    nb_batch_mc,
):

    batch_size = mc_size // nb_batch_mc

    @jax.jit
    def calculate_gap_single(r0, eps):
        def crowd_policy(t, x, r):
            # Inner function to get average action across history for a fixed theta
            def get_hist_action(single_policy):
                # vmap this over the theta grid
                def get_theta_action(single_theta):
                    return single_policy(t, x, r, single_theta)
                
                actions_theta = jax.vmap(get_theta_action)(theta_grid)
                return jnp.tensordot(theta_probs, actions_theta, axes=1)
            
            actions_history = get_hist_action(pi_bays)
            return actions_history

        rho_mf = generate_mean_field_scan(env, r0, pi_det, eps)   
        rho_ag = generate_MF_agent_scan(env, r0, crowd_policy, eps, rho_mf)

        v_ag = compute_total_reward(env, rho_ag, crowd_policy, rho_mf)
        v_mf = compute_total_reward(env, rho_mf, pi_det, rho_mf)

        return v_ag, v_mf

    vmap_eval = jax.vmap(calculate_gap_single)

    def mc_batch_step(carry, subkey):
        eps_batch = env.common_noise(subkey, (batch_size, env.H))
        r0_batch = jnp.tile(rho0, (batch_size, 1))
        res_bays, res_det = vmap_eval(r0_batch, eps_batch)
        return carry, (jnp.mean(res_bays), jnp.mean(res_det))

    keys = jax.random.split(key, nb_batch_mc)
    _, (final_bays, final_det) = jax.lax.scan(mc_batch_step, None, keys)

    return jnp.mean(final_bays), jnp.mean(final_det)


def ensemble_log_prob(models, rho_flat, thetas_grid):
    all_log_probs = []
    
    for m in models:
        lp, _, _ = compute_likelihood_uniform_prior(thetas_grid, rho_flat, m)
        all_log_probs.append(lp)

    log_probs_stack = jnp.stack(all_log_probs)
    mean_log = jnp.mean(log_probs_stack, axis = 0)
    log_like = mean_log - mean_log.max()
    map_idx = jnp.argmax(log_like)
    theta_MAP = thetas_grid[map_idx]
    return log_like, theta_MAP



def first_experiment(config, seed):
    config['seed'] = seed
    folder = config['folder_name']
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, f"eta={config['eta']}_seed={config['seed']}.pkl")
    
    results = {"config": config}

    KEY = jax.random.PRNGKey(config['seed'])
    k_pi0_bays, k_fic_bays, k_nash_bays, k_br_nash_bays, k_flow, k_train_flow, k_pi0, k_fic, k_nash, k_br_nash_theta, k_samples, k_br_map, k_gap_map, k_rew_map, k_br_bma, k_gap_bma, k_rew_bma = jax.random.split(KEY, 17)

    rho0 = jnp.ones(config['NB_STATES']) / config['NB_STATES']
    env_Theta = BeachBarEnv(
        generate_common_noise=vector_torus_uniform_displaced,
        rho0=rho0,
        nb_states=config['NB_STATES'],
        H=config['H'],
        eta=config['eta'],
        alpha_cong=1,
        alpha_dist=1 / config['NB_STATES'],
        bar_threshold=1
    )

    generate_theta = lambda k, b: generate_uniform(k, b, theta_dim=1, low=config['theta_low'], high=config['theta_high'])

    # ── Bayesian Fictitious Play ──────────────────────────────────────────────
    print(f"[seed={seed}] Starting Bayesian Fictitious Play (K={config['K_bays']} rounds)...", flush=True)
    t0 = time.time()
    pi0_bays = BayesianPolicyNN(env_Theta, key=k_pi0_bays)
    fictitious_ensemble_bays, nash_gaps_fic_bays = run_fictitious_play_recursive_bayesian(
        env_Theta,
        K_steps=config['K_bays'], initial_policy=pi0_bays,
        rho0=rho0, generate_theta=generate_theta,
        n_train_iters=config['epochs_fic_bays'],
        batch_size_train=config['batch_size_fic_bays'],
        size_mc=config['size_mc'], nb_batch_mc=config['nb_batch_mc'],
        lr=config['lr_fic_bays'], plot_report=False, key=k_fic_bays
    )
    results['nash_gaps_fic_bays'] = nash_gaps_fic_bays
    print(f"[seed={seed}] Bayesian FP done in {time.time()-t0:.1f}s | Nash gaps: {nash_gaps_fic_bays}", flush=True)

    print(f"[seed={seed}] Compressing Bayesian Nash policy ({config['epochs_nash_bays']} steps)...", flush=True)
    t0 = time.time()
    pi_nash_bays, loss_nash_bays = learn_fictitious_policy_bayesian(
        env_Theta, rho0, fictitious_ensemble_bays, generate_theta,
        config['epochs_nash_bays'], config['batch_size_nash_bays'], config['lr_nash_bays'],
        k_nash_bays
    )
    results['loss_nash_bays'] = loss_nash_bays
    print(f"[seed={seed}] Nash compression done in {time.time()-t0:.1f}s | Final loss: {loss_nash_bays[-1]:.4f}", flush=True)

    print(f"[seed={seed}] Computing Bayesian exploitability...", flush=True)
    t0 = time.time()
    gap_nash_bays, _ = compute_single_policy_exploitability_bayesian(
        env_Theta, rho0, pi_nash_bays, generate_theta,
        n_iterations=config['epochs_fic_bays'],
        mc_size=config['size_mc'], nb_batch_mc=config['nb_batch_mc'],
        lr=config['lr_fic_bays'], batch_size=config['batch_size_fic_bays'],
        key=k_br_nash_bays
    )
    results['gap_nash_bays'] = gap_nash_bays
    print(f"[seed={seed}] Bayesian exploitability done in {time.time()-t0:.1f}s | Gap: {gap_nash_bays:.6f}", flush=True)

    # ── Deterministic Nash per theta ──────────────────────────────────────────
    pi_nash_theta_dic = {}
    det_results = {}
    print(f"\n[seed={seed}] Starting deterministic Nash for {5} thetas...", flush=True)
    for idx, theta_true in enumerate(jnp.linspace(0.5, 2, 5)):
        theta_key = float(theta_true)
        theta_data = {}
        print(f"[seed={seed}]   theta {idx+1}/5 = {theta_key:.3f}", flush=True)
        env_true = env_Theta.set_theta(jnp.array([theta_true]))
        pi0 = PolicyNN(env_true, key=k_pi0)

        t0 = time.time()
        fictitious_ensemble_theta, nash_gaps_fic_theta = run_fictitious_play_recursive(
            env_true, config['K'], pi0, rho0,
            n_train_iters=config['epochs_fic'],
            batch_size_train=config['batch_size_fic'],
            size_mc=config['size_mc'], nb_batch_mc=config['nb_batch_mc'],
            lr=config['lr_fic'], plot_report=False, key=k_fic
        )
        theta_data['nash_gaps_fic_theta'] = nash_gaps_fic_theta
        print(f"[seed={seed}]     FP done in {time.time()-t0:.1f}s | Nash gaps: {nash_gaps_fic_theta}", flush=True)

        t0 = time.time()
        pi_nash_theta, loss_nash_theta = learn_fictitious_policy(
            env_true, rho0, fictitious_ensemble_theta,
            config['epochs_nash'], config['batch_size_nash'], config['lr_nash'],
            key=k_nash
        )
        pi_nash_theta_dic[theta_key] = pi_nash_theta
        theta_data['loss_nash_theta'] = loss_nash_theta
        print(f"[seed={seed}]     Nash compression done in {time.time()-t0:.1f}s | Final loss: {loss_nash_theta[-1]:.4f}", flush=True)

        t0 = time.time()
        gap_nash_theta, _ = compute_single_policy_exploitability(
            env_true, rho0, pi_nash_theta,
            n_iterations=config['epochs_fic'],
            mc_size=config['size_mc'], nb_batch_mc=config['nb_batch_mc'],
            lr=config['lr_fic'], batch_size=config['batch_size_fic'],
            key=k_br_nash_theta
        )
        theta_data['gap_nash_theta'] = gap_nash_theta
        det_results[theta_key] = theta_data
        print(f"[seed={seed}]     Exploitability done in {time.time()-t0:.1f}s | Gap: {gap_nash_theta:.6f}", flush=True)

    results['det_results'] = det_results
    print(f"[seed={seed}] All deterministic Nash done.", flush=True)

    # ── Flow ensemble × obs type × indices ───────────────────────────────────
    list_indices = [
        list(range(1, config['H'])),
        list(range(1, config['H'], 2)) + [config['H']-1],
        list(range(3, config['H'], 4)) + [config['H']-1],
        [config['H'] - 1],
    ]
    use_mu_map = {'rho': False, 'mu': True}
    flow_results = {}

    total_combos = len(use_mu_map) * len(list_indices)
    combo_idx = 0

    for obs, do_mu in use_mu_map.items():
        for indices_I in list_indices:
            combo_idx += 1
            indices_key = str(indices_I)
            if do_mu: 
                dim_xi = env_Theta.nb_actions * config['NB_STATES']
                hidden_dim = 256
            else: 
                dim_xi = config['NB_STATES']
                hidden_dim = 128
            event_dim = len(indices_I) * dim_xi
            print(f"\n[seed={seed}] === Combo {combo_idx}/{total_combos}: obs={obs}, |I|={len(indices_I)}, event_dim={event_dim} ===", flush=True)

            num_models = 5
            ensemble_keys = jax.random.split(k_flow, num_models)
            ensemble_flows = []
            ensemble_flows_losses = []

            for i in range(num_models):
                print(f"[seed={seed}]   Training flow {i+1}/{num_models}...", flush=True)
                t0 = time.time()
                m_key, train_key = jax.random.split(ensemble_keys[i])

                model = ConditionalMAF(
                    event_dim=event_dim,
                    context_dim=1,
                    nb_states=dim_xi,
                    hidden_dim=hidden_dim,
                    num_layers=5,
                    key=m_key,
                    use_simplex=do_mu
                )

                trained_model, losses = train_nle_online(
                    env_Theta,
                    model=model,
                    rho0=rho0,
                    generate_theta=generate_theta,
                    pi=pi_nash_bays,
                    indices_I=indices_I,
                    use_mu=do_mu,
                    n_steps=config['epochs_flow'],
                    lr=config['lr_flow'],
                    batch_size=config['batch_size_flow'],
                    key=train_key
                )
                ensemble_flows.append(trained_model)
                ensemble_flows_losses.append(losses)
                print(f"[seed={seed}]   Flow {i+1} done in {time.time()-t0:.1f}s | Final loss: {losses[-1]:.4f}", flush=True)

            obs_indices_results = {}
            for idx, theta_true in enumerate(jnp.linspace(0.5, 2, 5)):
                theta_key = float(theta_true)
                env_true = env_Theta.set_theta(jnp.array([theta_true]))
                pi_nash_theta = pi_nash_theta_dic[theta_key]
                print(f"[seed={seed}]   Evaluating theta {idx+1}/5 = {theta_key:.3f}", flush=True)

                n_samples_data = {}
                for N in [1, 10, 100]:
                    print(f"[seed={seed}]     N={N}...", flush=True)
                    t0 = time.time()
                    n_data = {}

                    if do_mu:
                        samples = sample_mu(env_true, rho0, pi_nash_theta, k_samples, N)
                    else:
                        samples = sample_rho(env_true, rho0, pi_nash_theta, k_samples, N)

                    samples = samples.reshape(N, -1, env_Theta.nb_states)
                    samples = samples[:, indices_I, :]
                    samples = samples.reshape(N, -1)

                    thetas_grid = jnp.linspace(config['theta_low'], config['theta_high'], 500).reshape(-1, 1)
                    log_like, theta_MAP = ensemble_log_prob(ensemble_flows, samples, thetas_grid)
                    likelihood = jnp.exp(log_like)
                    n_data['log_like']  = log_like
                    n_data['theta_map'] = theta_MAP
                    # print(f"[seed={seed}]       theta_MAP={float(theta_MAP):.4f} (true={theta_key:.3f})", flush=True)

                    br_to_map, _ = train_best_response_vs_bayesian_theta_fixed(
                        env_true, rho0, pi_nash_bays, theta_MAP,
                        n_iterations=config['epochs_fic'], lr=config['lr_fic'],
                        batch_size=config['batch_size_fic'], key=k_br_map
                    )
                    gap_map, rew_map = compute_exploitability_bayesian_fixed_theta(
                        env_true, rho0, pi_nash_bays, br_to_map, theta_MAP,
                        k_gap_map, config['size_mc'], config['nb_batch_mc']
                    )
                    n_data['gap_map'] = gap_map
                    n_data['rew_map'] = rew_map

                    rew_map_vs_nash_true, rew_det_true_map = compute_reward_bays_theta_fixed_vs_determinist(
                        env_true, rho0, pi_nash_bays, pi_nash_theta, theta_MAP,
                        config['size_mc'], config['nb_batch_mc'], key=k_rew_map
                    )
                    n_data['rew_map_vs_nash_true'] = rew_map_vs_nash_true
                    n_data['rew_det_true_map']     = rew_det_true_map

                    br_to_bma, _ = train_best_response_vs_bma(
                        env_true, rho0, pi_nash_bays, thetas_grid, likelihood,
                        n_iterations=config['epochs_fic'], lr=config['lr_fic'],
                        batch_size=config['batch_size_fic'], key=k_br_bma
                    )
                    gap_bma, rew_bma = compute_exploitability_bma(
                        env_true, rho0, pi_nash_bays, br_to_bma, thetas_grid, likelihood,
                        k_gap_bma, config['size_mc'], config['nb_batch_mc']
                    )
                    n_data['gap_bma'] = gap_bma
                    n_data['rew_bma'] = rew_bma

                    rew_bma_vs_nash_true, rew_det_true_bma = compute_reward_bma_vs_deterministic(
                        env_true, rho0, pi_nash_bays, pi_nash_theta, thetas_grid, likelihood,
                        key=k_rew_bma, mc_size=config['size_mc'], nb_batch_mc=config['nb_batch_mc']
                    )
                    n_data['rew_bma_vs_nash_true'] = rew_bma_vs_nash_true
                    n_data['rew_det_true_bma']     = rew_det_true_bma

                    n_samples_data[N] = n_data
                    print(f"[seed={seed}]     N={N} done in {time.time()-t0:.1f}s | gap_map={float(gap_map):.4f} | gap_bma={float(gap_bma):.4f}", flush=True)

                obs_indices_results[theta_key] = {'n_samples_evals': n_samples_data}

            flow_results[(obs, indices_key)] = {
                'losses': ensemble_flows_losses,
                'evals':  obs_indices_results,
            }
            print(f"[seed={seed}] Combo {combo_idx}/{total_combos} done.", flush=True)

    results['flow_results'] = flow_results
    print(f"\n[seed={seed}] All done. Saving to {file_path}...", flush=True)

    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"[seed={seed}] Saved.", flush=True)




