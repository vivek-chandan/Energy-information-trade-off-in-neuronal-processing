import numpy as np
from .bursting_neuron import BurstingNeuron

class StochasticBurstingNeuron(BurstingNeuron):
    
    def __init__(self, n_channels_Na=1000, n_channels_K=300, 
                 n_channels_NaP=100, n_channels_K_slow=50, 
                 noise_strength=0.02, **kwargs):
        
        super().__init__(**kwargs)
        
        self.n_Na = n_channels_Na
        self.n_K = n_channels_K
        self.n_NaP = n_channels_NaP
        self.n_K_slow = n_channels_K_slow
        self.noise_strength = noise_strength
        
        self.noise_scale_Na = 1.0 / np.sqrt(max(1, self.n_Na))
        self.noise_scale_K = 1.0 / np.sqrt(max(1, self.n_K))
        self.noise_scale_NaP = 1.0 / np.sqrt(max(1, self.n_NaP))
        self.noise_scale_K_slow = 1.0 / np.sqrt(max(1, self.n_K_slow))
    
    def derivatives_deterministic(self, state, t, I_inj):
       
        return super().derivatives(state, t, I_inj)
    
    def _euler_maruyama(self, state0, t, I_inj, dt):
        
        n_steps = len(t)
        n_vars = len(state0)
        solution = np.zeros((n_steps, n_vars))
        solution[0] = state0
        
        sqrt_dt = np.sqrt(dt)
        
        for i in range(1, n_steps):
            current_state = solution[i-1]
            
            derivs = np.array(self.derivatives_deterministic(current_state, t[i-1], I_inj))
            
            noise_vec = np.zeros(n_vars)
            
            rand_m = np.random.randn()
            rand_h = np.random.randn()
            rand_n = np.random.randn()
            rand_s = np.random.randn()
            
            noise_vec[1] = rand_m * self.noise_strength * self.noise_scale_Na
            noise_vec[2] = rand_h * self.noise_strength * self.noise_scale_Na
            noise_vec[3] = rand_n * self.noise_strength * self.noise_scale_K
            noise_vec[4] = rand_s * self.noise_strength * self.noise_scale_K_slow
            
            solution[i] = current_state + (derivs * dt) + (noise_vec * sqrt_dt)
            
            solution[i, 1:] = np.clip(solution[i, 1:], 0.0, 1.0)
        
        return solution

    def simulate(self, I_inj, t_max=1000, dt=0.01, seed=None):
        
        if seed is not None:
            np.random.seed(seed)
        
        t = np.arange(0, t_max, dt)
        
        V0 = -65.0
        m0 = self.alpha_m(V0) / (self.alpha_m(V0) + self.beta_m(V0))
        h0 = self.alpha_h(V0) / (self.alpha_h(V0) + self.beta_h(V0))
        n0 = self.alpha_n(V0) / (self.alpha_n(V0) + self.beta_n(V0))
        s0 = self.s_inf(V0)
        
        state0 = np.array([V0, m0, h0, n0, s0])
        
        # Run the SDE solver
        solution = self._euler_maruyama(state0, t, I_inj, dt)
        
        # Record currents for analysis
        self._record_all_currents(solution, t)
        
        return t, solution
    
    def get_channel_counts(self):
        return {
            'n_Na': self.n_Na,
            'n_K': self.n_K,
            'n_NaP': self.n_NaP,
            'n_K_slow': self.n_K_slow,
            'noise_strength': self.noise_strength
        }