from envs.mfg import *
import jax
import jax.numpy as jnp
import equinox as eqx

class BeachBarEnv(eqx.Module):
    # --- Dynamic Fields (Leaves of the PyTree) ---
    states: jnp.ndarray
    actions: jnp.ndarray
    rho0: jnp.ndarray
    idio_noise: jnp.ndarray
    law_idio_noise: jnp.ndarray
    alpha_cong: float 
    alpha_dist: float
    bar_threshold: float

    # --- Static Fields (Metadata/Functions) ---
    nb_states: int = eqx.static_field()
    nb_actions: int = eqx.static_field()
    H: int = eqx.static_field()
    eta: float = eqx.static_field()
    bar_x: int = eqx.static_field()
    theta_dim: int = eqx.static_field()
    generate_common_noise: callable = eqx.static_field()

    def __init__(self, generate_common_noise, rho0, nb_states=21, H=20, eta=0.05, 
                 bar_threshold=0.1, alpha_dist=1.0, alpha_cong=0.5):
        
        # 1. Direct assignments (DO NOT call super().__init__)
        self.nb_states = nb_states
        self.nb_actions = 3
        self.H = H
        self.rho0 = rho0
        self.eta = eta
        
        self.states = jnp.arange(nb_states)
        self.actions = jnp.arange(3) - 1
        
        self.bar_x = nb_states // 2
        self.generate_common_noise = generate_common_noise
        self.bar_threshold = bar_threshold
        self.idio_noise = jnp.arange(3) - 1
        self.law_idio_noise = jnp.array([1/3, 1/3, 1/3])
        self.alpha_dist = alpha_dist
        self.alpha_cong = alpha_cong
        self.theta_dim = 1

    def set_theta(self, theta):
        # Squeeze or reshape to ensure the scalar type matches alpha_cong
        return eqx.tree_at(lambda e: e.alpha_cong, self, jnp.reshape(theta, ()))

    def common_noise(self, key, batch_size):
        # Generates the noise vector for the whole state space
        return self.generate_common_noise(key, batch_size, self.nb_states, self.eta)

    def dynamics(self, x, a, eps, eps0):
        """Standard periodic dynamics for Beach Bar."""
        # eps_c_vec is (NB_STATES,)
        return (x + a + eps + eps0[x]) % self.nb_states

    # def reward(self, x, a, rho):
    #     """Beach Bar Reward logic."""
    #     # Density at the bar makes it attractive
    #     dist_to_bar = jnp.abs(x - self.bar_x)
    #     is_attractive = rho[self.bar_x] <= self.bar_threshold
        
    #     # If superior to threshold, the dist_term becomes 0
    #     dist_term = jnp.where(is_attractive, dist_to_bar, 0.0)
        
    #     # Action cost
    #     effort_term = jnp.abs(a) / self.nb_states
        
    #     # Crowding penalty
    #     rho_at_x = rho[x]
    #     congestion_term = jnp.log(rho_at_x + 1e-9)
        
    #     return -self.alpha_dist*dist_term - effort_term - self.alpha_cong*congestion_term
    

    def reward(self, x, a, rho):
        """Beach Bar Reward logic."""
        # Density at the bar makes it attractive
        dist_to_bar = jnp.abs(x - self.bar_x)
        is_attractive = rho[self.bar_x] <= self.bar_threshold
        
        # If superior to threshold, the dist_term becomes 0
        dist_term = jnp.where(is_attractive, dist_to_bar, 0.0)
        
        # Action cost
        effort_term = jnp.abs(a) / self.nb_states
        
        # Crowding penalty
        rho_at_x = rho[x]
        congestion_term = jnp.log(rho_at_x + 1e-9)
        
        return -self.alpha_dist*dist_term - effort_term - self.alpha_cong*congestion_term
    
    def get_P_matrix(self, eps0):
        """
        Returns the transition probability matrix P[s, a, s_next].
        Shape: (NB_STATES, NB_ACTIONS, NB_STATES)
        """
        def single_step_prob(s, a):
            # 1. Compute all possible next states for this (s, a) 
            # based on the 3 possible idiosyncratic noise values.
            # idio_noise is [-1, 0, 1]
            next_states = jax.vmap(lambda e: self.dynamics(s, a, e, eps0))(self.idio_noise)
            
            # 2. Create a probability vector of size (NB_STATES,)
            # We scatter the probabilities (1/3 each) into the calculated next_states
            p_vector = jnp.zeros(self.nb_states)
            p_vector = p_vector.at[next_states].add(self.law_idio_noise)
            return p_vector

        # vmap over states and actions to build the full (S, A, S) matrix
        return jax.vmap(jax.vmap(single_step_prob, in_axes=(None, 0)), in_axes=(0, None))(
            self.states, self.actions
        )

    def get_R_matrix(self, rho):
        """
        Returns a reward table (States, Actions).
        Shape: (NB_STATES, NB_ACTIONS)
        """
        return jax.vmap(lambda s: 
                   jax.vmap(lambda a: 
                       self.reward(s, a, rho)
                   )(self.actions)
               )(self.states)
