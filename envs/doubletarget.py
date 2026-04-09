from envs.mfg import *
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
from envs.mfg import BaseMFGEnv

class DoubleTarget(BaseMFGEnv):
    def __init__(self, generate_common_noise, rho0, nb_states=20, H=50, eta=0.05, alpha=1.0):
        states = jnp.arange(nb_states)
        actions = jnp.arange(3) - 1 # [-1, 0, 1]
        super().__init__(states, actions, H, rho0, eta)
        
        self.generate_common_noise = generate_common_noise
        self.idio_noise = jnp.arange(3) - 1
        self.law_idio_noise = jnp.array([1/3, 1/3, 1/3])
        
        # Fixed Targets
        self.T1 = nb_states // 4 
        self.T2 = 3*(nb_states // 4)
        self.targets = jnp.array([self.T1, self.T2])
        self.alpha = alpha

    def dist_torus(self, x, y):
        """Calculates the shortest distance between x and y on a periodic grid."""
        abs_diff = jnp.abs(x - y)
        return jnp.minimum(abs_diff, self.nb_states - abs_diff) 

    def common_noise(self, key, batch_size):
        return self.generate_common_noise(key, batch_size, self.nb_states, self.eta)

    def dynamics(self, x, a, eps, eps0):
        """Periodic dynamics: agents wrap around the boundaries."""
        return (x + a + eps + eps0[x]) % self.nb_states

    def get_bar_x_torus(self, rho):
        """
        Circular mean on a torus of size nb_states.
        The reference center is nb_states / 2.
        """
        shifted_states = (self.states - self.nb_states // 2) % self.nb_states
        angles = (shifted_states / self.nb_states) * 2 * jnp.pi

        mean_cos = jnp.sum(rho * jnp.cos(angles))
        mean_sin = jnp.sum(rho * jnp.sin(angles))

        R = jnp.sqrt(mean_cos**2 + mean_sin**2)

        def circular_mean():
            mean_angle = jnp.arctan2(mean_sin, mean_cos)
            return (
                mean_angle / (2 * jnp.pi) * self.nb_states
                + self.nb_states / 2
            ) % self.nb_states

        def uniform_case():
            return self.nb_states / 2

        return jnp.where(R < 1e-6, uniform_case(), circular_mean())


    # def reward(self, x, a, rho):
    #     """
    #     r(x, a, rho) = -min(d_torus(x, T))
    #                 - abs(a)
    #                 - alpha * d_torus(x, bar_x(rho))
    #                 + log(rho(x) + 0.005)
    #     """
    #     # 1. Distance to closest fixed target
    #     dist_to_T1 = self.dist_torus(x, self.T1)
    #     dist_to_T2 = self.dist_torus(x, self.T2)
    #     min_dist_fixed = jnp.minimum(dist_to_T1, dist_to_T2)

    #     # 2. Distance to circular mean
    #     bar_x = self.get_bar_x_torus(rho)
    #     dist_to_mean = self.dist_torus(x, bar_x)

    #     # 3. Effort
    #     effort_term = jnp.abs(a)

    #     # 4. Density reward (penalize isolation)
    #     rho_x = rho[x]          # or jnp.take(rho, x) if you prefer
    #     density_term = jnp.log(rho_x + 0.005)

    #     return (-min_dist_fixed - effort_term - self.alpha * dist_to_mean + density_term) / 10
    
    def reward(self, x, a, rho):
            # 1. Check crowd levels
            rho_T1 = rho[self.T1]
            rho_T2 = rho[self.T2]
            
            # 2. Distances
            dist_to_T1 = self.dist_torus(x, self.T1)
            dist_to_T2 = self.dist_torus(x, self.T2)

            mask_T1 = rho_T1 <= 0.1
            mask_T2 = rho_T2 <= 0.1

            cost_T1 = dist_to_T1*mask_T1 + dist_to_T2*(1-mask_T1)
            cost_T2 = dist_to_T2*mask_T2 + dist_to_T1*(1-mask_T2)

            cost_target = jnp.where(dist_to_T1 < dist_to_T2, cost_T1, cost_T2)

            # 4. Social & Effort terms
            bar_x = self.get_bar_x_torus(rho)
            dist_to_mean = self.dist_torus(x, bar_x)
            effort_term = jnp.abs(a)
            
            # 5. Local density (Entropy/Congestion term)
            rho_x = rho[x]
            density_term = jnp.log(rho_x + 0.005)
            # print(cost_target, dist_to_mean)

            return -cost_target - self.alpha * dist_to_mean  #- effort_term

    
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
    




