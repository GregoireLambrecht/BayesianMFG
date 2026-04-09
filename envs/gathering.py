from envs.mfg import *
import jax
import jax.numpy as jnp

class Gathering(BaseMFGEnv):
    def __init__(self, generate_common_noise,rho0, nb_states=21, H=50, eta=0.05):
        states = jnp.arange(nb_states)
        actions = jnp.arange(3) - 1
        super().__init__(states, actions, H, rho0, eta)
        self.generate_common_noise = generate_common_noise
        self.idio_noise = jnp.arange(3) - 1
        self.law_idio_noise = jnp.array([1/3, 1/3, 1/3])


    def common_noise(self, key, batch_size):
        # Generates the noise vector for the whole state space
        return self.generate_common_noise(key, batch_size, self.nb_states, self.eta)

    def dynamics(self, x, a,eps, eps0):
        """Standard periodic dynamics for Beach Bar."""
        # eps0 is (NB_STATES,)
        return jnp.clip(x + a + eps + eps0[x], 0, self.nb_states -1)

    def reward(self, x, a, rho):
            """
            Gathering Reward:
            r(x, a, rho) = -dist(x, target_x) - abs(a)
            where target_x is the state with max density closest to x.
            """
            # 1. Find the maximum density value
            max_val = jnp.max(rho)
            
            # 2. Identify all states that have this maximum density
            # We use a small epsilon for float comparison safety
            is_max = (rho >= max_val - 1e-7)
            
            # 3. Find the state among the 'max' states closest to the current x
            # We penalize non-max states with a huge distance so they aren't picked by argmin
            distances = jnp.abs(self.states - x)
            masked_distances = jnp.where(is_max, distances, 2*self.nb_states)
            
            # The target bar_x for this specific agent at x
            target_x = jnp.argmin(masked_distances)
            
            dist_term = jnp.abs(x - target_x)
            effort_term = jnp.abs(a)

            return -1.0 * dist_term - 1.0 * effort_term
    
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
    




