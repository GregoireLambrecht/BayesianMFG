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


class BayesianPolicyNN(eqx.Module):
    layers: list
    film_layers: list  # New: Generators for gamma and beta
    nb_states: int = eqx.static_field()
    vanilla: bool = eqx.static_field()
    activation: callable = eqx.static_field()

    def __init__(self, env, theta_dim=1, vanilla=False, key=None):
        if key is None:
            raise ValueError("key must be provided.")
            
        self.nb_states = env.nb_states
        self.vanilla = vanilla
        self.activation = jax.nn.tanh
        nb_actions = env.nb_actions
        hidden_dim = 64
        
        # Base input dimension (no theta here, theta will modulate the hidden layers)
        input_dim = 1 + self.nb_states if vanilla else 1 + self.nb_states + self.nb_states
        
        keys = jax.random.split(key, 12) # More keys for FiLM generators
        
        # 1. Main Policy Backbone
        self.layers = [
            eqx.nn.Linear(input_dim, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[2]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[3]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[4]),
            eqx.nn.Linear(hidden_dim, nb_actions, key=keys[5])
        ]
        
        # 2. FiLM Generators: Small networks that map theta to (gamma, beta)
        # We create one generator for each hidden layer (total of 5)
        self.film_layers = []
        for i in range(5):
            # Output is hidden_dim * 2 (one gamma and one beta per neuron)
            self.film_layers.append(eqx.nn.Linear(theta_dim, hidden_dim * 2, key=keys[6+i]))

    def __call__(self, t, x, rho, theta):
        # 1. Prepare Inputs (Standard)
        x_onehot = jax.nn.one_hot(x, self.nb_states)
        t_input = jnp.atleast_1d(t).astype(jnp.float32)
        theta_input = jnp.atleast_1d(theta)
        
        if self.vanilla:
            h = jnp.concatenate([t_input, x_onehot])
        else:
            h = jnp.concatenate([t_input, x_onehot, rho])
            
        # 2. Forward Pass with FiLM Modulation
        for i in range(5):
            # A. Standard Linear + Activation
            h = self.layers[i](h)
            
            # B. FiLM Modulation: Generate Gamma and Beta from Theta
            film_params = self.film_layers[i](theta_input)
            gamma, beta = jnp.split(film_params, 2)
            
            # C. Apply Modulation: h = h * gamma + beta
            # (Adding 1.0 to gamma makes it start near identity)
            h = h * (1.0 + gamma) + beta
            
            # D. Activation
            h = self.activation(h)
            
        # 3. Output
        logits = self.layers[5](h)
        return jax.nn.softmax(logits)
    


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
    optimizer = optax.adam(lr)
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

    generate_mean_field_scan(local_env, rho0, p_fixed, eps0)


def sample_rho(env,rho0, pi, key, N):
    eps0s = env.common_noise(key, (N, env.H)) # (N, H) 
    def get_single_trajectory(eps0):
        rho_mf = generate_mean_field_scan(env, rho0, pi, eps0)
        rho_flat = rho_mf.reshape(-1)
        return rho_flat
    dataset = jax.vmap(get_single_trajectory)(eps0s)
    return dataset

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



class ConditionalMAF(eqx.Module):
    conditioners: list 
    base_dist: distrax.Distribution = eqx.field(static=True)
    event_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)

    def __init__(self, event_dim, context_dim, hidden_dim, num_layers, key):
        self.event_dim = event_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        
        # Base distribution is a simple Standard Normal
        self.base_dist = distrax.MultivariateNormalDiag(
            loc=jnp.zeros(event_dim), 
            scale_diag=jnp.ones(event_dim)
        )
        
        split = event_dim // 2
        keys = jax.random.split(key, num_layers)
        self.conditioners = []
        
        for i in range(num_layers):
            # Define MLP layers
            l1 = eqx.nn.Linear(split + context_dim, hidden_dim, key=keys[i])
            l2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=jax.random.split(keys[i])[0])
            l3 = eqx.nn.Linear(hidden_dim, (event_dim - split) * 2, key=jax.random.split(keys[i])[1])
            
            # STABILITY: Zero-init the final layer so the flow starts as Identity
            l3 = eqx.tree_at(lambda l: l.weight, l3, jnp.zeros_like(l3.weight))
            l3 = eqx.tree_at(lambda l: l.bias, l3, jnp.zeros_like(l3.bias))
            
            # Use eqx.nn.Lambda to wrap jax.nn.tanh for Sequential compatibility
            conditioner = eqx.nn.Sequential([
                l1, eqx.nn.Lambda(jax.nn.tanh),
                l2, eqx.nn.Lambda(jax.nn.tanh),
                l3
            ])
            self.conditioners.append(conditioner)

    def log_prob(self, rho_flat, theta):
        """Calculates log P(rho | theta)"""
        # 1. PRE-PROCESS: Logit Transform for [0, 1] bounded data
        eps = 1e-6
        rho_clamped = jnp.clip(rho_flat, eps, 1.0 - eps)
        x = jnp.log(rho_clamped / (1.0 - rho_clamped))
        
        # Log-det for the logit transform
        # Change of variables: log(dy/dx) = log(1/x + 1/(1-x))
        logit_log_det = jnp.sum(-jnp.log(rho_clamped) - jnp.log(1.0 - rho_clamped))
        
        total_log_det = logit_log_det
        split = self.event_dim // 2

        # 2. FORWARD PASS: Flow through Coupling Layers
        for i in range(self.num_layers):
            x1, x2 = x[:split], x[split:]
            
            # Conditioner sees half the trajectory and the environment parameter theta
            params = self.conditioners[i](jnp.concatenate([x1, theta], axis=-1))
            shift, log_scale = jnp.split(params, 2, axis=-1)
            
            # STABILITY: Clamp log_scale to prevent numerical explosion
            log_scale = 3.0 * jnp.tanh(log_scale / 3.0) 
            
            # Apply transformation
            y2 = x2 * jnp.exp(log_scale) + shift
            total_log_det += jnp.sum(log_scale)
            
            # Combine and Permute (Reverse) for the next layer
            x = jnp.concatenate([x1, y2], axis=-1)[::-1]
            
        return self.base_dist.log_prob(x) + total_log_det

    def sample(self, theta, key, num_samples=1):
        """Generates rho ~ P(rho | theta)"""
        # 1. Sample from Gaussian base distribution
        z_samples = self.base_dist.sample(seed=key, sample_shape=(num_samples,))
        split = self.event_dim // 2

        def single_inverse(z):
            x = z
            # 2. INVERSE PASS: Go backwards through the flow
            for i in reversed(range(self.num_layers)):
                # Undo Permutation (Reverse is its own inverse)
                x = x[::-1]
                
                x1, x2 = x[:split], x[split:]
                params = self.conditioners[i](jnp.concatenate([x1, theta], axis=-1))
                shift, log_scale = jnp.split(params, 2, axis=-1)
                
                # Apply the same tanh clamp used in forward
                log_scale = 3.0 * jnp.tanh(log_scale / 3.0)
                
                # Inverse: x = (y - shift) * exp(-log_scale)
                original_x2 = (x2 - shift) * jnp.exp(-log_scale)
                x = jnp.concatenate([x1, original_x2], axis=-1)
            
            # 3. POST-PROCESS: Inverse Logit (Sigmoid) to return to [0, 1]
            return jax.nn.sigmoid(x)
            
        return jax.vmap(single_inverse)(z_samples)
    

def train_nle_online(
    env, 
    model, 
    rho0,
    generate_theta,
    pi,  # The Bayesian BMA policy ensemble
    n_steps=10000,   # Total number of gradient steps
    lr=1e-4,         
    batch_size=128, 
    key=None
):
    if key is None: key = jax.random.PRNGKey(0)
    
    # 1. Setup Optimizer & Partition
    # Using a chain with gradient clipping for stability in 'Infinite Data' mode
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr)
    )
    model_params, model_static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(model_params)

    # Pre-calculate dimensions for slicing the concatenated dataset
    theta_dim = env.theta_dim

    # --- The Core Step: Generate + Train ---
    def train_step(carry, step_key):
        params, opt_s = carry
        
        # 2. GENERATE FRESH DATA FOR THIS STEP
        # This function runs N simulations in parallel via vmap
        # Returns dataset of shape (batch_size, theta_dim + H * nb_states)
        dataset = sample_theta_rho_bayesian(
            env, rho0, generate_theta, pi, step_key, batch_size
        )

        # 3. SLICE THE DATA
        # Slicing out thetas and rhos from the concatenated batch
        thetas = dataset[:, :theta_dim]
        rho_flat = dataset[:, theta_dim:]

        # 4. COMPUTE LOSS AND GRADIENT
        def loss_fn(p):
            # Reassemble model to use log_prob
            m = eqx.combine(p, model_static)
            
            # vmap log_prob(rho, theta) across the batch
            log_p = jax.vmap(m.log_prob)(rho_flat, thetas)
            
            # Return negative log-likelihood (NLL)
            return -jnp.mean(log_p)

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params)
        
        # 5. UPDATE
        updates, next_opt_s = optimizer.update(grads, opt_s, params)
        next_params = eqx.apply_updates(params, updates)
        
        return (next_params, next_opt_s), loss_val

    # --- 6. Execution via lax.scan ---
    keys = jax.random.split(key, n_steps)
    
    # This will compile the simulator AND the trainer into one optimized XLA graph.
    # Expect a few minutes for the first compilation.
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


def first_experiment(config, seed):
    config['seed'] = seed
    # Setup Saving Path
    folder = config['folder_name']
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, f"eta={config['eta']}_seed={config['seed']}.pkl")
    
    # Storage dictionary
    results = {"config": config}

    KEY = jax.random.PRNGKey(config['seed'])
    k_pi0_bays, k_fic_bays, k_nash_bays, k_br_nash_bays, k_flow, k_train_flow, k_pi0, k_fic, k_nash,k_br_nash_theta, k_samples, k_br_map,k_gap_map, k_rew_map, k_br_bma, k_gap_bma, k_rew_bma = jax.random.split(KEY, 17)

    rho0 = jnp.ones(config['NB_STATES'])/config['NB_STATES']
    env_Theta = BeachBarEnv(
    generate_common_noise=vector_torus_uniform_displaced,
    rho0=rho0,
    nb_states=config['NB_STATES'],
    H=config['H'],
    eta=config['eta'],
    alpha_cong=1,
    alpha_dist= 1,
    bar_threshold=1
    )

    generate_theta = lambda k,b : generate_uniform(k, b, theta_dim = 1, low = config['theta_low'], high = config['theta_high'])

    pi0_bays = BayesianPolicyNN(env_Theta, key = k_pi0_bays)
    ficititious_ensemble_bays, nash_gaps_fic_bays = run_fictitious_play_recursive_bayesian(env_Theta,
                                                                                        K_steps= config['K_bays'],initial_policy= pi0_bays,
                                                                                        rho0 = rho0,
                                                                                        generate_theta = generate_theta,
                                                                                        n_train_iters=config['epochs_fic_bays'],
                                                                                        batch_size_train=config['batch_size_fic_bays'],size_mc=config['size_mc'], #64
                                                                                        nb_batch_mc=config['nb_batch_mc'],
                                                                                        lr =config['lr_fic_bays'],
                                                                                        plot_report=False, 
                                                                                        key = k_fic_bays
                                                                                        )
    #SAVE nash_gaps_fic_bays
    results['nash_gaps_fic_bays'] = nash_gaps_fic_bays

    pi_nash_bays, loss_nash_bays = learn_fictitious_policy_bayesian(env_Theta, rho0, ficititious_ensemble_bays,generate_theta,
                                                                    config['epochs_nash_bays'], config['batch_size_nash_bays'], config['lr_nash_bays'], 
                                                                    k_nash_bays)
    
    #SAVE loss_nash_bays
    results['loss_nash_bays'] = loss_nash_bays
    
    gap_nash_bays, _ = compute_single_policy_exploitability_bayesian(env_Theta, rho0, pi_nash_bays, generate_theta, 
                                                                  n_iterations=config['epochs_fic_bays'], 
                                                                  mc_size= config['size_mc'], nb_batch_mc=config['nb_batch_mc'], 
                                                                  lr = config['lr_fic_bays'], batch_size=config['batch_size_fic_bays'], 
                                                                  key = k_br_nash_bays)
    results['gap_nash_bays'] = gap_nash_bays

    model_flow = ConditionalMAF(
    event_dim=config['H']*config['NB_STATES'], 
    context_dim=1, 
    hidden_dim=256, 
    num_layers=5, 
    key=k_flow
    )

    model_flow, loss_flow = train_nle_online(env_Theta,
        model=model_flow,
        rho0=rho0, 
        generate_theta=generate_theta,
        pi=pi_nash_bays,
        n_steps=config['epochs_flow'],
        lr=config['lr_flow'],
        batch_size=config['batch_size_flow'],
        key=k_train_flow
    )
    #SAVE loss_flow
    results['loss_flow'] = loss_flow

    # Modification: Dictionary for eval_results
    eval_results = {}
    for theta_true in jnp.linspace(0, 2, 5):
        theta_key = float(theta_true)
        theta_data = {}
        env_true = env_Theta.set_theta(jnp.array([theta_true]))
        pi0  = PolicyNN(env_true, key=k_pi0)

        fictitious_ensemble_theta, nash_gaps_fic_theta = run_fictitious_play_recursive(env_true, config['K'], pi0,rho0,
                                                                       n_train_iters=config['epochs_fic'],
                                                                       batch_size_train=config['batch_size_fic'],
                                                                       size_mc=config['size_mc'], nb_batch_mc=config['nb_batch_mc'],
                                                                       lr = config['lr_fic'],
                                                                       plot_report=False, 
                                                                       key = k_fic
                                                                       )
        #SAVE nash_gaps_fic_theta
        theta_data['nash_gaps_fic_theta'] = nash_gaps_fic_theta

        pi_nash_theta, loss_nash_theta = learn_fictitious_policy(env_true, rho0, fictitious_ensemble_theta, 
                                                                 config['epochs_nash'], config['batch_size_nash'], config['lr_nash'], 
                                                                 key = k_nash)
        #SAVE loss_nash_theta
        theta_data['loss_nash_theta'] = loss_nash_theta

        gap_nash_theta, _ = compute_single_policy_exploitability(env_true, rho0, pi_nash_theta,
                                                                  n_iterations=config['epochs_fic'], 
                                                                  mc_size= config['size_mc'], nb_batch_mc=config['nb_batch_mc'], 
                                                                  lr = config['lr_fic'], batch_size=config['batch_size_fic'], 
                                                                  key = k_br_nash_theta)
        results['gap_nash_theta'] = gap_nash_theta

        n_samples_data = {}
        for N in [1, 10, 100]:
            n_data = {}
            samples = sample_rho(env_true,rho0,pi_nash_theta, k_samples, N)
            thetas_grid = jnp.linspace(config['theta_low'], config['theta_high'], 500).reshape(-1, 1)
            log_like, likelihood, theta_map = compute_likelihood_uniform_prior(thetas_grid, samples, model_flow)
            #SAVE log_like, theta_map
            n_data['log_like'] = log_like
            n_data['theta_map'] = theta_map

            br_to_map, _ = train_best_response_vs_bayesian_theta_fixed(
            env_true, rho0, pi_nash_bays, theta_map, 
            n_iterations= config['epochs_fic'], lr=config['lr_fic'],batch_size=config['batch_size_fic'],
            key=k_br_map
            )

            gap_map, rew_map = compute_exploitability_bayesian_fixed_theta(env_true, rho0, pi_nash_bays, br_to_map, theta_map, 
                                                                                    k_gap_map,
                                                                                    config['size_mc'], config['nb_batch_mc'])
            #SAVE gap_map, rew_map
            n_data['gap_map'] = gap_map
            n_data['rew_map'] = rew_map

            rew_map_vs_nash_true, rew_det_true_map = compute_reward_bays_theta_fixed_vs_determinist(env_true, rho0, pi_nash_bays, pi_nash_theta, theta_map, 
                                                                                    config['size_mc'], config['nb_batch_mc'], 
                                                                                    key = k_rew_map)
            #SAVE rew_map_vs_nash_true, rew_det_true_map
            n_data['rew_map_vs_nash_true'] = rew_map_vs_nash_true
            n_data['rew_det_true_map'] = rew_det_true_map

            br_to_bma, _ = train_best_response_vs_bma(
            env_true, rho0, pi_nash_bays, thetas_grid, likelihood,
            n_iterations= config['epochs_fic'], lr=config['lr_fic'],batch_size=config['batch_size_fic'],
            key=k_br_bma
            )
            
            gap_bma, rew_bma = compute_exploitability_bma(env_true, rho0, pi_nash_bays, br_to_bma, thetas_grid, likelihood, 
                                                                                    k_gap_bma,
                                                                                    config['size_mc'], config['nb_batch_mc'])
            #SAVE gap_bma, rew_bma
            n_data['gap_bma'] = gap_bma
            n_data['rew_bma'] = rew_bma

            rew_bma_vs_nash_true, rew_det_true_bma = compute_reward_bma_vs_deterministic(env_true, rho0, pi_nash_bays, pi_nash_theta, thetas_grid, likelihood, 
                                                                                        key = k_rew_bma,
                                                                                        mc_size=config['size_mc'], nb_batch_mc= config['nb_batch_mc'], 
                                                                                        )
            #SAVE rew_bma_vs_nash_true, rew_det_true_bma
            n_data['rew_bma_vs_nash_true'] = rew_bma_vs_nash_true
            n_data['rew_det_true_bma'] = rew_det_true_bma
            
            n_samples_data[N] = n_data # Dict by N
        
        theta_data['n_samples_evals'] = n_samples_data
        eval_results[theta_key] = theta_data # Dict by theta

    results['evaluations'] = eval_results
    
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)