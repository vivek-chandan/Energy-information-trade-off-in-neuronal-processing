import numpy as np
from scipy.integrate import odeint

class HodgkinHuxley:
    
    def __init__(self):

        self.C_m = 1.0
        self.g_Na = 120.0 
        self.g_K = 36.0 
        self.g_L = 0.3    
        
        self.E_Na = 50.0
        self.E_K = -77.0
        self.E_L = -54.387
        
        self.recorded_currents = None
        
    def alpha_m(self, V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def alpha_n(self, V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def I_Na(self, V, m, h):
        return self.g_Na * m**3 * h * (V - self.E_Na)
    
    def I_K(self, V, n):
        return self.g_K * n**4 * (V - self.E_K)
    
    def I_L(self, V):
        return self.g_L * (V - self.E_L)
    
    def derivatives(self, state, t, I_inj):
        V, m, h, n = state
        dm_dt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dh_dt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dn_dt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        
        dV_dt = (I_inj - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        
        return [dV_dt, dm_dt, dh_dt, dn_dt]
    
    def simulate(self, I_inj, t_max=100, dt=0.01, record_currents=True):

        # Time array
        t = np.arange(0, t_max, dt)
        
        # Initial conditions (resting state)
        V0 = -65.0
        m0 = self.alpha_m(V0) / (self.alpha_m(V0) + self.beta_m(V0))
        h0 = self.alpha_h(V0) / (self.alpha_h(V0) + self.beta_h(V0))
        n0 = self.alpha_n(V0) / (self.alpha_n(V0) + self.beta_n(V0))
        
        state0 = [V0, m0, h0, n0]
        
        # Solve ODEs
        solution = odeint(self.derivatives, state0, t, args=(I_inj,))
        
        # Record currents if requested
        if record_currents:
            self._record_currents(solution, t)
        
        return t, solution
    
    def _record_currents(self, solution, t):
        V = solution[:, 0]
        m = solution[:, 1]
        h = solution[:, 2]
        n = solution[:, 3]
        
        I_Na_arr = np.array([self.I_Na(V[i], m[i], h[i]) for i in range(len(t))])
        I_K_arr = np.array([self.I_K(V[i], n[i]) for i in range(len(t))])
        I_L_arr = np.array([self.I_L(V[i]) for i in range(len(t))])
        
        self.recorded_currents = {
            'I_Na': I_Na_arr,
            'I_K': I_K_arr,
            'I_L': I_L_arr,
            't': t
        }
    
    def get_steady_state_values(self, V):
        
        m_inf = self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))
        h_inf = self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))
        n_inf = self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))
        
        return {'m_inf': m_inf, 'h_inf': h_inf, 'n_inf': n_inf}