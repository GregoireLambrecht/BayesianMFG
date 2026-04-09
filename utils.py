import jax
import jax.numpy as jnp
import equinox as eqx
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt 
import numpy as np
import optax

import os
import pickle


class PolicyNN(eqx.Module):
    layers: list
    nb_states: int = eqx.static_field()
    vanilla: bool = eqx.static_field()
    activation: callable = eqx.static_field()

    def __init__(self, env, vanilla=False, key=None):
        if key is None:
            raise ValueError("A jax.random.PRNGKey must be provided as 'key'.")
            
        self.nb_states = env.nb_states
        self.vanilla = vanilla
        self.activation = jax.nn.tanh
        nb_actions = env.nb_actions
        
        # input: time(1) + x_onehot(nb_states) [+ rho(nb_states)]
        input_dim = 1 + self.nb_states if vanilla else 1 + self.nb_states + self.nb_states
        
        # Split keys for 6 Linear layers
        keys = jax.random.split(key, 6)
        
        self.layers = [
            eqx.nn.Linear(input_dim, 64, key=keys[0]),
            eqx.nn.Linear(64, 64, key=keys[1]),
            eqx.nn.Linear(64, 64, key=keys[2]),
            eqx.nn.Linear(64, 64, key=keys[3]),
            eqx.nn.Linear(64, 64, key=keys[4]),
            eqx.nn.Linear(64, nb_actions, key=keys[5])
        ]

    def __call__(self, t, x, rho):
        """
        t: scalar time step
        x: scalar state index
        rho: array (NB_STATES,) distribution
        """
        # 1. Prepare Inputs
        x_onehot = jax.nn.one_hot(x, self.nb_states)
        t_input = jnp.atleast_1d(t).astype(jnp.float32)
        
        if self.vanilla:
            h = jnp.concatenate([t_input, x_onehot])
        else:
            h = jnp.concatenate([t_input, x_onehot, rho])
            
        # 2. Forward Pass through layers with activation
        h = self.activation(self.layers[0](h))
        h = self.activation(self.layers[1](h))
        h = self.activation(self.layers[2](h))
        h = self.activation(self.layers[3](h))
        h = self.activation(self.layers[4](h))
        logits = self.layers[5](h)
        
        # 3. Output Probabilities
        return jax.nn.softmax(logits)
    


def generate_rho_one_step(env, rho0, pi, t, eps0):
    """
    rho0: (NB_STATES,)  - 1D vector
    eps0: (NB_STATES,)  - 1D vector (noise per state)
    """
    # 1. Get Policy for all states: (S, A)
    # Returns the probability of taking each action 'a' in state 's'
    u_pi = jax.vmap(pi, in_axes=(None, 0, None))(t, env.states, rho0)
    P_matrix = env.get_P_matrix(eps0) 
    
    # 3. Compute joint mass at every (state, action) pair: (S, A)
    # rho_sa[s, a] = Prob(State=s) * Prob(Action=a | State=s)
    rho_sa = rho0[:, jnp.newaxis] * u_pi
    
    # 4. Push-forward via Matrix Multiplication:
    # We sum over the current state 's' and current action 'a'.
    rho_next = jnp.einsum('sa,saj->j', rho_sa, P_matrix)
    
    return rho_next / (jnp.sum(rho_next) + 1e-10)

def generate_mu_from_rho_one_step(env, rho, pi, t):
    """
    Computes the joint state-action distribution mu_t(x, a).
    
    rho: (NB_STATES,) - The state distribution at time t
    pi: Policy function (t, x, rho) -> (NB_ACTIONS,)
    t: Current timestep
    
    Returns:
        mu: (NB_STATES, NB_ACTIONS) matrix where mu[x, a] = rho(x) * pi(a|x, rho)
    """
    # 1. Get the policy (action probabilities) for every state
    # We vmap over the environment's state space
    # in_axes: (None, 0, None) means we only vary the state 'x'
    u_pi = jax.vmap(pi, in_axes=(None, 0, None))(t, env.states, rho)
    
    # 2. Compute the joint distribution mu(x, a)
    # We multiply rho(x) [shape (S,)] by u_pi(a|x) [shape (S, A)]
    # Using newaxis to align (S, 1) with (S, A) for broadcasting
    mu = rho[:, jnp.newaxis] * u_pi
    
    # Optional: Safety normalization to ensure sum(mu) == 1.0
    return mu / (jnp.sum(mu) + 1e-10)


def generate_rho_one_step_agent(env, rho_agent, pi, t, eps0, rho_MF):
    """
    Simulates one step of an individual agent's mass distribution.
    
    rho_agent: (NB_STATES,) - The agent's current distribution
    pi: Policy function (t, x, rho) -> (NB_ACTIONS,)
    t: Current timestep
    eps0: (NB_STATES,) - Noise for this step
    rho_MF: (NB_STATES,) - The FIXED crowd distribution at time t
    """
    
    # 1. Get Policy based on CROWD density (rho_MF)
    # The individual decides what to do by looking at the crowd.
    u_pi = jax.vmap(pi, in_axes=(None, 0, None))(t, env.states, rho_MF)
    
    # 2. Get the Transition Probability Matrix: (S_curr, A, S_next)
    # This matrix captures the physics (transitions + noise)
    P_matrix = env.get_P_matrix(eps0) 
    
    # 3. Compute joint mass at every (state, action) pair for the AGENT: (S, A)
    # The agent's mass at (s, a) is their mass at 's' times the policy's decision
    rho_sa = rho_agent[:, jnp.newaxis] * u_pi
    
    # 4. Push-forward via Matrix Multiplication:
    # rho_next[s_next] = sum_{s, a} rho_sa[s, a] * P_matrix[s, a, s_next]
    rho_next = jnp.einsum('sa,saj->j', rho_sa, P_matrix)
    
    # 5. Safety Normalization
    return rho_next / (jnp.sum(rho_next) + 1e-10)


def generate_mean_field_scan(env, rho0, pi, eps0):
    """
    rho0: (NB_STATES,)
    eps0: (H, NB_STATES)
    """
    
    # 1. Define the step function for scan
    # scan expects: (carry, input) -> (next_carry, output)
    def step_fn(current_rho, t_and_eps):
        t, current_eps = t_and_eps
        
        # Calculate next state
        next_rho = generate_rho_one_step(env, current_rho, pi, t, current_eps)
        
        # We carry next_rho to the next t, and also output it to be stacked
        return next_rho, next_rho

    # 2. Prepare the inputs for the scan
    # We want to iterate from t=0 to H-2
    # We use eps0[:-1] because the last noise vector moves us to the last rho
    timesteps = jnp.arange(env.H - 1)
    scan_inputs = (timesteps, eps0[:-1])

    # 3. Run the scan
    # final_rho: the rho at t = H-1
    # trajectory: a stacked array of all rhos from t=1 to H-1
    final_rho, trajectory = jax.lax.scan(step_fn, rho0, scan_inputs)

    # 4. Combine the initial rho0 with the rest of the trajectory
    # This results in an array of shape (H, NB_STATES)
    return jnp.concatenate([rho0[None, :], trajectory], axis=0)


def generate_mu_from_rho_scan(env, rho_traj, pi):
    """
    Maps the full state trajectory to the state-action trajectory.
    """
    timesteps = jnp.arange(env.H)
    mu_traj = jax.vmap(generate_mu_from_rho_one_step, in_axes=(None, 0, None, 0))(
        env, rho_traj, pi, timesteps
    )
    
    return mu_traj

def generate_mu_scan(env, rho0, pi, eps0):
    """
    Generates rho flow then converts to mu flow.
    """
    rho_traj = generate_mean_field_scan(env, rho0, pi, eps0)
    mu_traj = generate_mu_from_rho_scan(env, rho_traj, pi)
    return mu_traj

def generate_MF_agent_scan(env, rho0, pi, eps0, rho_MF):
    """
    rho0: (NB_STATES,)
    eps0: (H, NB_STATES)
    """
    
    # 1. Define the step function for scan
    # scan expects: (carry, input) -> (next_carry, output)
    def step_fn(current_rho, t_and_eps):
        t, current_eps = t_and_eps
        
        # Calculate next state
        next_rho = generate_rho_one_step_agent(env, current_rho, pi, t, current_eps, rho_MF[t])
        
        # We carry next_rho to the next t, and also output it to be stacked
        return next_rho, next_rho

    # 2. Prepare the inputs for the scan
    # We want to iterate from t=0 to H-2
    # We use eps0[:-1] because the last noise vector moves us to the last rho
    timesteps = jnp.arange(env.H - 1)
    scan_inputs = (timesteps, eps0[:-1])

    # 3. Run the scan
    # final_rho: the rho at t = H-1
    # trajectory: a stacked array of all rhos from t=1 to H-1
    final_rho, trajectory = jax.lax.scan(step_fn, rho0, scan_inputs)

    # 4. Combine the initial rho0 with the rest of the trajectory
    # This results in an array of shape (H, NB_STATES)
    return jnp.concatenate([rho0[None, :], trajectory], axis=0)


def compute_reward_one_step(env, rho_agent, pi, t, rho_mf):
    """
    rho_agent: (NB_STATES,)  - 1D vector
    rho_mf: (NB_STATES,)  - 1D vector
    """
    u_pi = jax.vmap(pi, in_axes=(None, 0, None))(t, env.states, rho_mf)
    R = env.get_R_matrix(rho_mf)
    return jnp.sum(rho_agent[:, jnp.newaxis] * u_pi * R)



def compute_total_reward(env, rho_agent_traj, pi, rho_mf_traj):
    """
    rho_agent_traj: (H, NB_STATES)
    rho_mf_traj: (H, NB_STATES)
    """
    
    def scan_body(carry_reward, inputs):
        t, rho_a, rho_m = inputs
        
        # Calculate reward for this specific step
        step_reward = compute_reward_one_step(env, rho_a, pi, t, rho_m)
        
        # New carry is the accumulated reward
        new_carry = carry_reward + step_reward
        
        # We return (new_carry, None) because we only care about the final sum,
        # not the intermediate cumulative sums.
        return new_carry, None

    # Prepare inputs: time indices, agent trajectory, and population trajectory
    timesteps = jnp.arange(env.H)
    inputs = (timesteps, rho_agent_traj, rho_mf_traj)

    # Initial reward is 0.0
    # scan returns (final_carry, stacked_outputs)
    total_reward, _ = jax.lax.scan(scan_body, 0.0, inputs)
    
    return total_reward


def compute_expected_reward(env, rho0, pi_A, pi_E, mc_size, nb_batch_mc, key):
    batch_size = mc_size // nb_batch_mc

    @jax.jit
    def eval_batch(r0_batch, eps_batch):
        @jax.vmap
        def single_eval(e0):
            rho_mf = generate_mean_field_scan(env, r0_batch[0], pi_E, e0)
            rho_a  = generate_mean_field_scan(env, r0_batch[0], pi_A, e0)
            return compute_total_reward(env, rho_a, pi_A, rho_mf)
        return jnp.mean(single_eval(eps_batch))

    def mc_step(carry, subkey):
        key, _ = carry
        key, subkey = jax.random.split(key)

        eps_batch = env.common_noise(subkey, (batch_size, env.H))
        r0_batch  = jnp.tile(rho0, (batch_size, 1))

        batch_mean = eval_batch(r0_batch, eps_batch)
        return (key, None), batch_mean

    keys = jax.random.split(key, nb_batch_mc)
    (_, _), batch_means = jax.lax.scan(mc_step, (key, None), keys)

    return jnp.mean(batch_means)




def batch_policy_history(list_policy):
    """
    Converts a list of PolicyNN modules into a single PolicyNN 
    where each parameter is batched along axis 0.
    """
    if not list_policy:
        return None
        
    # 1. Separate arrays (params) from static structure (metadata)
    # We use eqx.filter to ensure we only try to stack actual JAX arrays
    params_list = [eqx.filter(p, eqx.is_array) for p in list_policy]
    
    # 2. Stack the arrays along a new leading dimension
    batched_params = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *params_list)
    
    # 3. Get the static structure from the first policy
    # (nb_states, vanilla flag, activation function, etc.)
    static_struct = eqx.filter(list_policy[0], lambda x: not eqx.is_array(x))
    
    # 4. Combine them back into a single PolicyNN object
    return eqx.combine(batched_params, static_struct)




def compute_agg_MF(env, rho0, batched_history, eps0):
    """
    Computes the Mean Field (average distribution) using a batched PyTree.
    
    rho0: (NB_STATES,)
    batched_history: A single PolicyNN where weights are (K, ...)
    eps0: (H, NB_STATES)
    """
    # Define how to get a trajectory for ONE policy out of the batch
    def get_single_trajectory(single_policy):
        # generate_mean_field_scan expects (env, rho0, policy, eps0)
        return generate_mean_field_scan(env, rho0, single_policy, eps0)

    # Vectorize the trajectory generation over the policy batch (axis 0)
    # Resulting shape: (K, H, NB_STATES)
    all_trajectories = jax.vmap(get_single_trajectory)(batched_history)

    # Average across the K policies
    # Resulting shape: (H, NB_STATES)
    return jnp.mean(all_trajectories, axis=0)


def train_best_response_fictitious(env, list_policy, rho0, n_iterations=100, lr=1e-3, batch_size=32, key = None):
    if key is None: key = jax.random.PRNGKey(0)
    key, model_key = jax.random.split(key)

    # 1. Initialize Learner
    model = PolicyNN(env, key=model_key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # 2. Robust Pre-processing for History
    params_list = [eqx.filter(p, eqx.is_array) for p in list_policy]
    batched_params = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *params_list)
    static_struct = eqx.filter(list_policy[0], lambda x: not eqx.is_array(x))
    batched_history = eqx.combine(batched_params, static_struct)

    def loss_fn(model, rho_init_batch, batched_hist, eps0_batch):
        def single_universe_loss(r0, e0):
            all_rhos = jax.vmap(generate_mean_field_scan, in_axes=(None, None, 0, None))(
                env, r0, batched_hist, e0
            )
            rho_mf = jnp.mean(all_rhos, axis=0)
            rho_ag = generate_MF_agent_scan(env, r0, model, e0, rho_mf)
            return compute_total_reward(env, rho_ag, model, rho_mf)

        rewards = jax.vmap(single_universe_loss, in_axes=(0, 0))(rho_init_batch, eps0_batch)
        return -jnp.mean(rewards)

    # 3. Define the step function for lax.scan
    def scan_step(carry, _):
        current_model, current_opt_state, current_key = carry
        
        # Internal step logic
        step_key, noise_key = jax.random.split(current_key)
        eps0 = env.common_noise(noise_key, (batch_size, env.H)) 
        r0_batch = jnp.tile(rho0, (batch_size, 1))
        
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
            current_model, r0_batch, batched_history, eps0
        )
        
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_model)
        next_model = eqx.apply_updates(current_model, updates)
        
        # Carry the state, and output the loss (converted to reward)
        return (next_model, next_opt_state, step_key), -loss_val

    # 4. Run the full training inside JIT
    @eqx.filter_jit
    def run_training(m, s, k):
        # scan runs n_iterations on device and returns the stacked rewards
        final_carry, rewards_history = jax.lax.scan(
            scan_step, (m, s, k), xs=None, length=n_iterations
        )
        return final_carry, rewards_history

    (final_model, _, _), rewards = run_training(model, opt_state, key)
            
    return final_model, rewards



# def compute_exploitability_fictitious(env, rho0, policy_history, best_response, key, mc_size):
#     """
#     Computes Exploitability = V(BestResponse vs History) - V(History vs History)
#     using a large Monte Carlo sample of common noise.
#     """
#     # 1. Generate huge batch of common noise
#     # eps_batch shape: (mc_size, H, S)
#     eps_batch = env.common_noise(key, (mc_size, env.H))
#     r_init_batch = jnp.tile(rho0, (mc_size, 1))

#     @jax.jit
#     def calculate_gap_batch(r0_b, e0_b):
#         # A. Environment Trajectory (Average of history)
#         # Using list comprehension inside JIT is fine for small/medium history
#         all_rhos = jnp.stack([generate_mean_field_scan(env, r0_b, p, e0_b) for p in policy_history])
#         rho_MF = jnp.mean(all_rhos, axis=0) # (H, S)

#         # B. Population Welfare (Current History Performance)
#         # V(pi_avg, rho_MF)
#         def welfare_fn(p):
#             return compute_total_reward(env, generate_mean_field_scan(env, r0_b, p, e0_b), p, rho_MF)
        
#         v_welfare = jnp.mean(jnp.stack([welfare_fn(p) for p in policy_history]))

#         # C. Best Response Performance
#         # V(BR, rho_MF)
#         rho_BR = generate_mean_field_scan(env, r0_b, best_response, e0_b)
#         v_BR = compute_total_reward(env, rho_BR, best_response, rho_MF)

#         return v_BR - v_welfare

#     # Vectorize the entire calculation over the MC samples
#     gaps = jax.vmap(calculate_gap_batch)(r_init_batch, eps_batch)
    
#     return jnp.mean(gaps)

def compute_exploitability_fictitious(
    env,
    rho0,
    policy_history,
    best_response,
    key,
    mc_size,
    nb_batch_mc,
):
    """
    Computes Exploitability = V(BestResponse vs History) - V(History vs History)
    using batched Monte Carlo over common noise.
    """

    batch_size = mc_size // nb_batch_mc

    @jax.jit
    def calculate_gap_single(r0, eps):
        # A. Mean-field trajectory from history
        all_rhos = jnp.stack([
            generate_mean_field_scan(env, r0, p, eps)
            for p in policy_history
        ])  # (n_hist, H, S)

        rho_MF = jnp.mean(all_rhos, axis=0)  # (H, S)

        # B. Welfare of history
        def welfare_fn(p):
            return compute_total_reward(
                env,
                generate_mean_field_scan(env, r0, p, eps),
                p,
                rho_MF,
            )

        v_welfare = jnp.mean(jnp.stack([welfare_fn(p) for p in policy_history]))

        # C. Best response
        rho_BR = generate_MF_agent_scan(env, rho0, best_response,eps, rho_MF )
        v_BR = compute_total_reward(env, rho_BR, best_response, rho_MF)

        return v_BR - v_welfare

    vmap_gap = jax.jit(jax.vmap(calculate_gap_single))

    def mc_batch_step(carry, key):
        key, subkey = jax.random.split(key)

        eps_batch = env.common_noise(subkey, (batch_size, env.H))
        r0_batch = jnp.tile(rho0, (batch_size, 1))

        gaps = vmap_gap(r0_batch, eps_batch)
        batch_mean = jnp.mean(gaps)

        return key, batch_mean

    # Generate keys for each MC batch
    keys = jax.random.split(key, nb_batch_mc)

    _, batch_means = jax.lax.scan(
        mc_batch_step,
        key,
        keys,
    )

    # Final MC estimate
    return jnp.mean(batch_means)



def run_fictitious_play_recursive(
    env, 
    K_steps, 
    initial_policy, 
    rho0, 
    n_train_iters=100, 
    batch_size_train=32, 
    size_mc=1000, 
    nb_batch_mc = 10,
    lr=1e-3,
    key=None,
    plot_report = False
):
    """
    Handles keys explicitly to ensure reproducibility across FP rounds.
    """
    if key is None: 
        # Fallback only if no key is provided, but we should always pass one
        key = jax.random.PRNGKey(0)
    
    policy_history = [initial_policy]
    nash_gaps = []

    for k in range(1, K_steps + 1):
        # print(f"FP Round {k}/{K_steps}...")
        
        # 1. Split the key for this specific round
        # We need one key for training and one for the gap calculation
        # and one to carry over to the next 'k' iteration.
        key, train_key, mc_key = jax.random.split(key, 3)
        
        # 2. Train the Best Response
        # We pass train_key so the common noise and SGD batches are deterministic
        new_best_response, reward = train_best_response_fictitious(
            env=env,
            list_policy=policy_history,
            rho0=rho0,
            n_iterations=n_train_iters,
            lr=lr,
            batch_size=batch_size_train,
            key=train_key # Ensure your train function accepts this!
        )
        if plot_report:
            plt.plot(reward)
            plt.show()
        # 3. Compute Exploitability
        # We use mc_key so the Monte Carlo noise is independent of the training noise
        gap = compute_exploitability_fictitious(
            env, 
            rho0, 
            policy_history, 
            new_best_response, 
            mc_key, 
            size_mc,
            nb_batch_mc=nb_batch_mc
        )

        
        nash_gaps.append(gap)
        print(f"   Nash Gap: {gap:.6f}")

        # 4. Add to history
        policy_history.append(new_best_response)

    return policy_history, jnp.array(nash_gaps)


def plot_mean_field_trajectory(trajectory, save = False, folder = None, file = None):
    """
    trajectory: Tensor of shape (H, B, NB_STATES)
    """
    data = trajectory
    nb_s = data.shape[1]
    
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(data.T, aspect='auto', origin='lower', cmap='viridis')
    clb = plt.colorbar()
    clb.set_label(label=r'$\rho^{\pi^E}_n(x)$', fontsize = 15)
    plt.xlabel('Time Step (n)', fontsize = 20)
    plt.ylabel('State (X)', fontsize = 20)
    # plt.title(f'Mean Field Evolution for Batch {batch_idx}')
    # plt.xticks(range(data.shape[0]))
    plt.yticks(range(nb_s))
    if save: 
        os.makedirs(folder, exist_ok=True)
        # --- Save full figure ---
        full_path = os.path.join(folder, f"{file}_meanfield.pdf")
        fig.savefig(full_path, dpi=300)
    plt.show()

def get_scheduler(max_lr, total_steps):
    # This single function handles the 0 -> max_lr ramp 
    # AND the max_lr -> (0.01 * max_lr) decay.
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=max_lr,
        warmup_steps=int(0.1 * total_steps), # 10% Warmup
        decay_steps=total_steps,             # Total duration
        end_value=0.1*max_lr                           # End at 10% of peak
    )
    return lr_schedule

def learn_fictitious_policy(env, rho0, list_fictitious, n_iterations, batch_size, lr,key = None, scheduler = False):
    """
    Learns a single PolicyNN that mimics the average behavior of the FP history.
    """
    if key is None: 
        # Fallback only if no key is provided, but we should always pass one
        key = jax.random.PRNGKey(0)
    key, model_key = jax.random.split(key)

    # 1. Initialize the single learner policy
    learner = PolicyNN(env, key=model_key)
    if scheduler:
        lr_schedule = get_scheduler(max_lr=lr, total_steps=n_iterations)
        optimizer = optax.adam(learning_rate=lr_schedule) 
    else:
        optimizer = optax.adam(lr)
    
    opt_state = optimizer.init(eqx.filter(learner, eqx.is_array))

    # 2. Batch the history (once) to avoid re-compilation
    batched_history = batch_policy_history(list_fictitious)

    def loss_fn(model, r0_batch, batched_hist, eps0_batch):
        def single_universe_loss(r0, e0):
            # A. Compute the target: Average distribution from history (the "mu_agg")
            # We calculate rhos for each policy in history and average them
            all_mu_hist = jax.vmap(generate_mu_scan, in_axes=(None, None, 0, None))(
                env, r0, batched_hist, e0
            )
            mu_mf_agg = jnp.mean(all_mu_hist, axis=0) # (H, S)

            # B. Compute the learner's distribution
            mu_learner = generate_mu_scan(env, r0, model, e0) # (H, S)

            # C. L1 Loss between the distributions (H, S)
            # In MFG, matching the state distributions rho usually suffices if 
            # the policy is also regularized, but we use L1 on rho here.
            return jnp.sum(jnp.abs(mu_learner - mu_mf_agg))

        losses = jax.vmap(single_universe_loss)(r0_batch, eps0_batch)
        return jnp.mean(losses)

    def scan_step(carry, _):
        current_model, current_opt_state, current_key = carry
        
        step_key, noise_key = jax.random.split(current_key)
        eps0 = env.common_noise(noise_key, (batch_size, env.H)) 
        r0_batch = jnp.tile(rho0, (batch_size, 1))
        
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
            current_model, r0_batch, batched_history, eps0
        )
        
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_model)
        next_model = eqx.apply_updates(current_model, updates)
        
        return (next_model, next_opt_state, step_key), loss_val

    # 3. Run the training loop inside JIT
    @eqx.filter_jit
    def run_training(m, s, k):
        final_carry, loss_history = jax.lax.scan(
            scan_step, (m, s, k), xs=None, length=n_iterations
        )
        return final_carry, loss_history

    (final_learner, _, _), losses = run_training(learner, opt_state, key)
            
    return final_learner, losses



def compute_single_policy_exploitability(env, rho0, target_policy, n_iterations=200, mc_size=10000, nb_batch_mc = 10, lr=1e-3, batch_size = 100, key = None):
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
    br_model, _ = train_best_response_fictitious(
        env, policy_list, rho0, n_iterations=n_iterations, lr=lr,batch_size=batch_size,key=model_key
    )

    # Step 3: Compute the exploitability gap using Monte Carlo
    # We create a new key for the evaluation
    
    # Re-using your compute_exploitability function
    gap = compute_exploitability_fictitious(
        env, rho0, policy_list, br_model, eval_key, mc_size, nb_batch_mc
    )
    
    return gap, br_model



def generate_expert_trajectories(env, rho0, pi_expert, eps0_batch, m_agent, key):
    """
    Simulates M agents across N noise universes.
    
    eps0_batch: (N_traj, H, NB_STATES)
    returns: 
        X: (H, N_traj, M_agent) - state indices
        A: (H, N_traj, M_agent) - action indices
        rho_emp: (H, N_traj, NB_STATES) - empirical distributions
    """
    n_traj = eps0_batch.shape[0]
    h_horizon = env.H

    def simulate_one_universe(eps0, universe_key):
        # eps0 shape: (H, NB_STATES)
        
        # 1. Initialize agents: sample M positions from rho0
        k_init, k_run = jax.random.split(universe_key)
        start_X = jax.random.categorical(k_init, jnp.log(rho0 + 1e-10), shape=(m_agent,))

        def step_fn(current_X, t_and_eps0):
            t, current_eps0 = t_and_eps0
            k_t = jax.random.fold_in(k_run, t)
            k_act, k_idio = jax.random.split(k_t)

            # A. Compute Empirical Density rho_t (needed for the policy input)
            # We count how many agents are in each state
            rho_t = jnp.bincount(current_X, length=env.nb_states) / m_agent

            # B. Get Action Probabilities from Expert: (M, NB_ACTIONS)
            # Expert takes (time, state, current_rho)
            probs = jax.vmap(lambda x: pi_expert(t, x, rho_t))(current_X)
            
            # C. Sample discrete actions for all M agents
            actions = jax.random.categorical(k_act, jnp.log(probs + 1e-10))

            # D. Sample Idiosyncratic Noise for all M agents
            # Using your idio_noise [-1, 0, 1] and law [1/3, 1/3, 1/3]
            idio_idx = jax.random.categorical(k_idio, jnp.log(env.law_idio_noise), shape=(m_agent,))
            eps_idio = env.idio_noise[idio_idx]

            # E. Compute Next States
            # dynamics(x, a, idio, common_vec) -> uses common_vec[x]
            next_X = jax.vmap(lambda x, a, e: env.dynamics(x, a, e, current_eps0))(
                current_X, actions, eps_idio
            )

            return next_X, (current_X, actions, rho_t)

        # Carry out the simulation over H steps
        timesteps = jnp.arange(h_horizon)
        scan_input = (timesteps, eps0)
        _, (X_traj, A_traj, rho_traj) = jax.lax.scan(step_fn, start_X, scan_input)
        
        return X_traj, A_traj, rho_traj

    # Vectorize over the N common noise universes
    # We need a unique key for each universe to ensure different idiosyncratic paths
    universe_keys = jax.random.split(key, n_traj)
    X, A, rho_emp = jax.vmap(simulate_one_universe)(eps0_batch, universe_keys)

    # Reshape to (H, N, M) for the learner
    # vmap returns (N, H, ...), we transpose to (H, N, ...) to match your code
    X = jnp.transpose(X, (1, 0, 2))
    A = jnp.transpose(A, (1, 0, 2))
    rho_emp = jnp.transpose(rho_emp, (1, 0, 2))

    return X, A, rho_emp




def learn_policy_streaming(env, rho0, pi_expert,vanilla, n_iterations, lr, n_traj, m_agent, key = None, scheduler = False):
    """
    Imitation Learning by sampling expert trajectories and matching action distributions.
    """
    if key is None: 
        # Fallback only if no key is provided, but we should always pass one
        key = jax.random.PRNGKey(0)
    key, model_key = jax.random.split(key)

    # 1. Initialize Student Policy
    learner = PolicyNN(env, key=model_key,vanilla=vanilla)

    if scheduler:
        lr_schedule = get_scheduler(max_lr=lr, total_steps=n_iterations)
        optimizer = optax.adam(learning_rate=lr_schedule) 
    else:
        optimizer = optax.adam(lr)

    opt_state = optimizer.init(eqx.filter(learner, eqx.is_array))

    def loss_fn(model, x_batch, a_batch, rho_emp_batch):
        H, N, M = x_batch.shape
        S = env.nb_states
        A_dim = env.nb_actions

        # 1. Target calculation (Same as before, this part is correct)
        def get_empirical_pi(x_slice, a_slice):
            counts = jnp.zeros((S, A_dim)).at[x_slice].add(jax.nn.one_hot(a_slice, A_dim))
            state_counts = counts.sum(axis=-1, keepdims=True)
            return counts / (state_counts + 1e-10)

        # targets: (H, N, S, A_dim)
        target_pi = jax.vmap(jax.vmap(get_empirical_pi))(x_batch, a_batch)

        # 2. Learner Predictions
        # We need to call model(t, x, rho) for every t, every universe N, and every state S.
        
        # We define a helper that computes all S states for a FIXED time and FIXED rho
        def predict_all_states(t_val, rho_val):
            # rho_val: (S,) | t_val: scalar
            # vmap over the state index (0, 1, ..., S-1)
            return jax.vmap(lambda s_idx: model(t_val, s_idx, rho_val))(jnp.arange(S))

        # Now we vmap this over Universes (N) and Time (H)
        # rho_emp_batch is (H, N, S)
        # t_vec is (H,)
        
        # vmap over N (Universe) - t_val is shared, rho_val varies
        v_universe = jax.vmap(predict_all_states, in_axes=(None, 0))
        
        # vmap over H (Time) - both vary
        v_time = jax.vmap(v_universe, in_axes=(0, 0))
        
        # pred_probs shape: (H, N, S, A_dim)
        t_vec = jnp.arange(H)
        pred_probs = v_time(t_vec, rho_emp_batch)

        # 3. Loss Calculation
        # We weight the L1 distance by rho_emp_batch (the probability of being in that state)
        diff = jnp.abs(pred_probs - target_pi).sum(axis=-1) # (H, N, S)
        weighted_loss = diff * rho_emp_batch
    
        return jnp.mean(weighted_loss) # Mean over all dimensions

    def scan_step(carry, _):
        current_model, current_opt_state, current_key = carry
        
        # 1. Generate Expert Data (Stochastic)
        k1, k2, next_k = jax.random.split(current_key, 3)
        eps0 = env.common_noise(k1, (n_traj, env.H))
        
        # Note: You need a JAX version of generate_trajectory_mean_field_fast
        # X: (H, N, M), A: (H, N, M), rho_emp: (H, N, S)
        X, A, rho_emp = generate_expert_trajectories(env, rho0, pi_expert, eps0, m_agent, k2)
        
        # 2. Update Learner
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
            current_model, X, A, rho_emp
        )
        
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_model)
        next_model = eqx.apply_updates(current_model, updates)
        
        return (next_model, next_opt_state, next_k), loss_val

    # 3. JIT Compiled Training
    @eqx.filter_jit
    def run_training(m, s, k):
        final_carry, loss_history = jax.lax.scan(
            scan_step, (m, s, k), xs=None, length=n_iterations
        )
        return final_carry, loss_history

    (final_learner, _, _), losses = run_training(learner, opt_state, key)
    return final_learner, losses



def compute_BC_ADV_proxies(env, rho0, pi_A, pi_E, n_mc=1000, key = None):
    """
    Computes Behavioral Cloning (BC) and Advection (ADV) proxy metrics.
    """
    if key is None: 
        # Fallback only if no key is provided, but we should always pass one
        key = jax.random.PRNGKey(0)
    eps0_batch = env.common_noise(key, (n_mc, env.H))
    
    # We define the states once for reuse
    states = jnp.arange(env.nb_states)

    def step_fn(carry, t_and_eps0):
        rho_E, rho_A = carry
        t, eps0_t = t_and_eps0 # eps0_t shape: (n_mc, S)

        def compute_metrics_single_universe(r_E, r_A, e0_t):
            # 1. BC Proxy: E[ |pi_A(t, x, rho_E) - pi_E(t, x, rho_E)| ]
            # vmap over states to get policy for every state in the universe
            probs_A_on_E = jax.vmap(lambda x: pi_A(t, x, r_E))(states)
            probs_E_on_E = jax.vmap(lambda x: pi_E(t, x, r_E))(states)
            
            # L1 diff weighted by the expert density rho_E
            diff_bc = jnp.sum(jnp.abs(probs_A_on_E - probs_E_on_E), axis=-1)
            bc_val = jnp.sum(r_E * diff_bc)

            # 2. ADV Proxy: L1 dist between mu_A and mu_E
            # mu(s, a) = rho(s) * pi(a|s)
            probs_A_on_A = jax.vmap(lambda x: pi_A(t, x, r_A))(states)
            mu_A = r_A[:, None] * probs_A_on_A
            mu_E = r_E[:, None] * probs_E_on_E
            
            adv_val = jnp.sum(jnp.abs(mu_A - mu_E))

            # 3. Evolve distributions to next time step
            # We reuse your existing one-step transition logic
            next_r_E = generate_rho_one_step(env, r_E, pi_E, t, e0_t)
            next_r_A = generate_rho_one_step(env, r_A, pi_A, t, e0_t)
            
            return (next_r_E, next_r_A), (bc_val, adv_val)

        # Vectorize across all MC universes
        (next_rho_E, next_rho_A), (bc_batch, adv_batch) = jax.vmap(compute_metrics_single_universe)(
            rho_E, rho_A, eps0_t
        )

        # Average results across universes for this time step
        return (next_rho_E, next_rho_A), (jnp.mean(bc_batch), jnp.mean(adv_batch))

    # Initial state: expand rho0 for all MC universes
    init_rhos = (jnp.tile(rho0, (n_mc, 1)), jnp.tile(rho0, (n_mc, 1)))
    
    # Run scan across time
    # Transpose eps0_batch to (H, n_mc, S) for scanning over time
    scan_eps = (jnp.arange(env.H), jnp.transpose(eps0_batch, (1, 0, 2)))
    
    _, (delta_BC, delta_ADV) = jax.lax.scan(step_fn, init_rhos, scan_eps)

    # Return the maximum error over the horizon as per your original logic
    return jnp.max(delta_BC), jnp.max(delta_ADV)


def run_full_experiment(env, all_params):
    main_key = jax.random.PRNGKey(all_params['seed'])
    # 1. Setup paths and parameters
    folder_name = all_params['folder_name']
    file_name = all_params['file_name']
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    H = env.H
    S = env.nb_states
    rho0 = env.rho0
    
    # Generate unique filename
    full_filename = f"{file_name}_key={all_params['seed']}_eta={env.eta}_H={H}_S={S}_idx={all_params['idx']}.pkl"
    save_path = os.path.join(folder_name, full_filename)

    main_key, fp_key, exp_key, vanilla_key, adapt_key, eval_key, proxy_key = jax.random.split(main_key, 7)

    # 2. Training Phase
    print("--- Starting Fictitious Play ---")
    # Start with a random policy (kaiming init)
    initial_policy = PolicyNN(env, key=jax.random.PRNGKey(42))
    
    # fictitious_history is the list of policies [pi_0, pi_1, ..., pi_n]
    fictitious_history, fp_gaps = run_fictitious_play_recursive(
        env, K_steps= all_params['K_steps'], rho0 =rho0, initial_policy= initial_policy, 
        n_train_iters=all_params['fp_iters'], 
        batch_size_train= all_params['fp_batch_size'], 
        lr=all_params['fp_lr'],
        size_mc= all_params['exploit_mc'],
        nb_batch_mc= all_params['nb_batch_mc'],
        key = fp_key
    )

    print("--- Learning Expert (Aggregated FP) ---")
    # expert: Single NN that mimics the average of the FP history
    expert, expert_loss = learn_fictitious_policy(
        env, rho0, fictitious_history, 
        n_iterations=all_params['exp_iters'], 
        batch_size=all_params['exp_batch_size'], 
        lr=all_params['exp_lr'],
        key = exp_key,
        scheduler=all_params['il_scheduler']
    )

    print("--- Learning Student: Vanilla (rho-independent) ---")
    pi_vanilla, vanilla_loss = learn_policy_streaming(
        env, rho0, expert, vanilla=True, 
        n_iterations=all_params['il_iters'], 
        lr=all_params['il_lr'], 
        n_traj=all_params['n_traj'], 
        m_agent=all_params['m_agent'], 
        key = vanilla_key,
        scheduler=all_params['il_scheduler']
    )

    print("--- Learning Student: Adaptive (rho-dependent) ---")
    pi_adaptive, adaptive_loss = learn_policy_streaming(
        env, rho0, expert, vanilla=False, 
        n_iterations=all_params['il_iters'], 
        lr=all_params['il_lr'], 
        n_traj=all_params['n_traj'], 
        m_agent=all_params['m_agent'], 
        key = adapt_key,
        scheduler=all_params['il_scheduler']
    )

    # 3. Evaluation Phase: Proxies (BC/ADV)
    print("--- Computing BC/ADV Proxies ---")
    bc_vanilla, adv_vanilla = compute_BC_ADV_proxies(env, rho0, pi_vanilla, expert, n_mc=all_params['proxy_mc'], key=proxy_key)
    bc_adaptive, adv_adaptive = compute_BC_ADV_proxies(env, rho0, pi_adaptive, expert, n_mc=all_params['proxy_mc'], key=proxy_key)

    # 4. Evaluation Phase: Exploitability & Rewards
    # This involves training a BR for each model
    print("--- Computing Exploitability & BR Rewards ---")
    
    def evaluate_model(model):
        # Trains BR against the Mean Field generated by 'model'
        exploit, br_model = compute_single_policy_exploitability(
            env, rho0, model, 
            n_iterations=all_params['br_iters'], 
            batch_size= all_params['br_batch_size'], 
            lr = all_params['br_lr'],
            mc_size=all_params['mc_size'],
            nb_batch_mc=all_params['nb_batch_mc'],
            key=eval_key
        )

        # Compute expected reward of the model vs itself
        rew_model = compute_expected_reward(
            env,
            rho0, 
            model,
            expert,
            mc_size=all_params['mc_size'],
            nb_batch_mc=all_params['nb_batch_mc'],
            key=eval_key
        )

        return float(exploit), float(rew_model)

    exp_expert, rew_expert = evaluate_model(expert)
    exp_vanilla, rew_vanilla = evaluate_model(pi_vanilla)
    exp_adaptive, rew_adaptive = evaluate_model(pi_adaptive)

    # 5. Pack Results
    results = {
        "params": all_params,
        "fp_gaps": [float(r) for r in fp_gaps],
        "expert_loss": [float(l) for l in expert_loss],
        "vanilla_loss": [float(l) for l in vanilla_loss],
        "adaptive_loss": [float(l) for l in adaptive_loss],
        "proxies": {
            "vanilla": {"BC": float(bc_vanilla), "ADV": float(adv_vanilla)},
            "adaptive": {"BC": float(bc_adaptive), "ADV": float(adv_adaptive)}
        },
        "exploitability": {
            "expert": exp_expert,
            "vanilla": exp_vanilla,
            "adaptive": exp_adaptive
        },
        "reward_vs_expert": {
            "expert": rew_expert,
            "vanilla": rew_vanilla,
            "adaptive":rew_adaptive,
        }
    }

    # 6. Save
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"--- Experiment Complete. Results saved to {save_path} ---")
    return results