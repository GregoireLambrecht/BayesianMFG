import jax
import jax.numpy as jnp
import equinox as eqx

class CityRelocation(eqx.Module):
    # --- Dynamic Fields ---
    states: jnp.ndarray
    actions: jnp.ndarray
    rho0: jnp.ndarray
    theta_map: jnp.ndarray  
    theta_cong: float       
    theta_move: float     
    eta : float  

    # --- Static Fields ---
    nb_states: int = eqx.static_field()
    nb_actions: int = eqx.static_field()
    grid_size: int = eqx.static_field()
    H: int = eqx.static_field()
    theta_dim: int = eqx.static_field()
    generate_common_noise: callable = eqx.static_field()

    def __init__(self, generate_common_noise, rho0, nb_states=100, H=20, eta=0, theta_cong = 2, theta_move = 0.1):
        self.nb_states = nb_states
        self.grid_size = int(jnp.sqrt(nb_states))
        
        # Actions: 0: Stay, 1: North, 2: South, 3: East, 4: West
        self.nb_actions = 5 
        self.H = H
        self.rho0 = rho0
        self.theta_dim = nb_states  
        
        self.states = jnp.arange(nb_states)
        self.actions = jnp.arange(5)
        
        self.theta_map = jnp.zeros(nb_states)
        self.theta_cong = theta_cong
        self.theta_move = theta_move # Usually lower move cost since actions are incremental
        self.eta = eta
        
        self.generate_common_noise = generate_common_noise

    def set_theta(self, theta):
        new_map = theta
        # new_map = theta[:self.nb_states]
        # new_cong = jnp.reshape(theta[self.nb_states], ())
        # new_move = jnp.reshape(theta[self.nb_states+1], ())
        
        model = eqx.tree_at(lambda e: e.theta_map, self, new_map)
        # model = eqx.tree_at(lambda e: e.theta_cong, model, new_cong)
        # model = eqx.tree_at(lambda e: e.theta_move, model, new_move)
        return model

    def common_noise(self, key, batch_size):
        return self.generate_common_noise(key, batch_size, self.nb_states, self.eta)

    # def dynamics(self, x, a, eps0):
    #     """
    #     1. Convert 1D x to 2D (row, col)
    #     2. Apply action a (N, S, E, W, Stay)
    #     3. Apply eps0[x] (the vector block shift)
    #     4. Wrap around with modulo grid_size
    #     """
    #     r = x // self.grid_size
    #     c = x % self.grid_size

    #     # Action offsets: [Stay, N, S, E, W]
    #     # In grid: N is row-1, S is row+1, E is col+1, W is col-1
    #     dr = jnp.array([0, -1, 1, 0, 0])
    #     dc = jnp.array([0, 0, 0, 1, -1])

    #     new_r = r + dr[a]
    #     new_c = c + dc[a]

    #     # Apply noise shift (eps0[x] is the shift index for state x)
    #     # Note: We assume the block shift is also applied in 1D for simplicity, 
    #     # but the physics wrap correctly.
    #     target_1d = (new_r * self.grid_size + new_c) % self.nb_states
        
    #     # Add the common noise shift
    #     shift = eps0[target_1d]
    #     return (target_1d + shift) % self.nb_states

    def dynamics(self, x, a, eps0, eps):
        # 1. Current position
        r = x // self.grid_size
        c = x % self.grid_size

        # 2. Check if the CURRENT state is a "Storm Trigger"
        # (We check if eps0[x] is 1. If it is, the 2x2 area starting 
        # at this x is a storm zone).
        is_storm_active = eps0[x] == 1

        # 3. Nominal Move (Standard Action)
        dr = jnp.array([0, -1, 1, 0, 0]) # Stay, N, S, E, W
        dc = jnp.array([0, 0, 0, 1, -1])
        target_r = (r + dr[a]) % self.grid_size
        target_c = (c + dc[a]) % self.grid_size
        nominal_next_state = target_r * self.grid_size + target_c

        # 4. Storm Move (Ignore Action)
        # eps is 0, 1, 2, or 3. We map this to a 2x2 offset from CURRENT x.
        # 0: (r,c), 1: (r, c+1), 2: (r+1, c), 3: (r+1, c+1)
        storm_dr = jnp.array([0, 0, 1, 1])
        storm_dc = jnp.array([0, 1, 0, 1])
        
        storm_r = (r + storm_dr[eps]) % self.grid_size
        storm_c = (c + storm_dc[eps]) % self.grid_size
        storm_next_state = storm_r * self.grid_size + storm_c

        # 5. Result
        # If eps0[x] is 1, you go to storm_next_state. 
        # Otherwise, you go to nominal_next_state.
        return jnp.where(is_storm_active, storm_next_state, nominal_next_state)
    

    def reward(self, x, a, rho):
        """
        r = theta_x' - theta_rho * log(rho_x') - theta_move * 1_{a != 0}
        Note: The agent is rewarded based on the utility of where they LAND (x').
        """

        attraction = self.theta_map[x]
        # congestion = self.theta_cong * jnp.log(rho[x_next] + 1e-9)
        #congestion = self.theta_cong * rho[x]
        congestion = self.theta_cong*jnp.log(rho[x] + 1e-5)
        
        # Penalize if action is not 'Stay' (a=0)
        moving_cost = jnp.where(a != 0, self.theta_move, 0.0)
        
        return (100*attraction - congestion - moving_cost)#*(10*self.nb_states/self.H)

    # def get_P_matrix(self, eps0):
    #     def single_step_prob(s, a):
    #         next_s = self.dynamics(s, a, eps0)
    #         return jax.nn.one_hot(next_s, self.nb_states)

    #     return jax.vmap(jax.vmap(single_step_prob, in_axes=(None, 0)), in_axes=(0, None))(
    #         self.states, self.actions
    #     )

    def get_P_matrix(self, eps0):
        def single_step_prob(s, a):
            # All possible idiosyncratic outcomes
            all_eps = jnp.arange(4)
            
            # Vectorize dynamics over the 4 eps values
            next_states = jax.vmap(lambda e: self.dynamics(s, a, eps0, e))(all_eps)
            
            # Create one-hot vectors and average them
            # If not impacted, all 4 next_states are identical (avg remains one-hot)
            # If impacted, next_states are 4 different cells (avg is 0.25 at each)
            probs = jax.nn.one_hot(next_states, self.nb_states)
            return jnp.mean(probs, axis=0)

        # Vmap over states and actions
        return jax.vmap(
            jax.vmap(single_step_prob, in_axes=(None, 0)), 
            in_axes=(0, None)
        )(self.states, self.actions)

    def get_R_matrix(self, rho):
        return jax.vmap(lambda s: 
                   jax.vmap(lambda a: 
                       self.reward(s, a, rho)
                   )(self.actions)
               )(self.states)
    


class BayesianPolicyCityCNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    cnn_linear: eqx.nn.Linear
    layers: list
    film_heads: list
    output_layer: eqx.nn.Linear
    depth: int = eqx.static_field()
    nb_states: int = eqx.static_field()
    grid_size: int = eqx.static_field()
    vanilla: bool = eqx.static_field()

    def __init__(self, env, depth=4, film_hidden=128, vanilla=False, key=None):
        k_back, k_cnn, k_out = jax.random.split(key, 3)
        k_c1, k_c2, k_lin = jax.random.split(k_cnn, 3)

        self.depth      = depth
        self.nb_states  = env.nb_states
        self.grid_size  = int(jnp.sqrt(env.nb_states))
        self.vanilla    = vanilla
        hidden_dim      = 128

        # CNN encoder — stored as separate layers, no Sequential
        self.conv1      = eqx.nn.Conv2d(1,  16, kernel_size=3, padding=1, key=k_c1)
        self.conv2      = eqx.nn.Conv2d(16,  8, kernel_size=3, padding=1, key=k_c2)
        self.cnn_linear = eqx.nn.Linear(8 * self.nb_states, film_hidden,  key=k_lin)

        # Backbone
        input_dim = 1 + self.nb_states if vanilla else 1 + 2 * self.nb_states
        backbone_keys = jax.random.split(k_back, depth)
        self.layers = [
            eqx.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim, key=backbone_keys[i])
            for i in range(depth)
        ]

        # FiLM heads
        head_keys = jax.random.split(key, depth)
        self.film_heads = [
            eqx.nn.Linear(film_hidden, hidden_dim * 2, key=head_keys[i])
            for i in range(depth)
        ]

        self.output_layer = eqx.nn.Linear(hidden_dim, env.nb_actions, key=k_out)

    def _encode_theta(self, theta):
        """CNN encoder — no Sequential, activations applied manually."""
        theta_2d = (theta*self.nb_states).reshape(1, self.grid_size, self.grid_size)
        h = jax.nn.relu(self.conv1(theta_2d))
        h = jax.nn.relu(self.conv2(h))
        h = h.reshape(-1)                        # flatten
        return jax.nn.relu(self.cnn_linear(h))   # (film_hidden,)

    def __call__(self, t, x, rho, theta):
        theta_repr = self._encode_theta(theta)   # (film_hidden,)

        x_onehot = jax.nn.one_hot(x, self.nb_states)
        t_input  = jnp.atleast_1d(t).astype(jnp.float32)
        h = jnp.concatenate([t_input, x_onehot]) if self.vanilla \
            else jnp.concatenate([t_input, x_onehot, rho*self.nb_states])

        for i in range(self.depth):
            h = self.layers[i](h)
            film_params  = self.film_heads[i](theta_repr)
            gamma, beta  = jnp.split(film_params, 2)
            h = h * (1.0 + gamma) + beta
            h = jax.nn.tanh(h)

        return jax.nn.softmax(self.output_layer(h))