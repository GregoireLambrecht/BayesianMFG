import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

class BaseMFGEnv(ABC):
    def __init__(self, states, actions, H, rho0, eta):
        # Configuration Constants
        self.states = states
        self.actions = actions
        self.nb_states = len(states)
        self.nb_actions = len(actions)
        self.H = H
        self.rho0 = rho0
        self.eta = eta

    @abstractmethod
    def dynamics(self, x, a, eps_c_vec):
        pass

    @abstractmethod
    def reward(self, x, a, rho):
        pass

    @abstractmethod
    def common_noise(self, key):
        pass