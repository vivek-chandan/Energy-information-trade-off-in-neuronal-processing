import numpy as np
from scipy.integrate import odeint
from .hodgkin_huxley import HodgkinHuxley

class BurstingNeuron(HodgkinHuxley):
    
    def __init__(self, g_NaP=0.8, g_K_slow=7.0, tau_s=1000.0):
        
        super().__init__()
        self.g_NaP = g_NaP
        self.g_K_slow = g_K_slow
        self.tau_s = tau_s
        
        self.E_NaP = self.E_Na
        self.E_K_slow = self.E_K
        
    def m_inf_NaP(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 68.0) / 7.0))
    
    def s_inf(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 45.0) / 5.0))
    
    def tau_s_V(self, V):
        return self.tau_s
    
    def I_NaP(self, V):
        return self.g_NaP * self.m_inf_NaP(V) * (V - self.E_NaP)
    
    def I_K_slow(self, V, s):
        return self.g_K_slow * s * (V - self.E_K_slow)
    
    def derivatives(self, state, t, I_inj):
    
        V, m, h, n, s = state
        
        # Original HH gating variables
        dm_dt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dh_dt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dn_dt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        
        # Slow potassium gating (creates bursting)
        ds_dt = (self.s_inf(V) - s) / self.tau_s_V(V)
        
        # Membrane potential with additional currents
        dV_dt = (I_inj 
                 - self.I_Na(V, m, h) 
                 - self.I_K(V, n) 
                 - self.I_L(V)
                 - self.I_NaP(V)
                 - self.I_K_slow(V, s)) / self.C_m
        
        return [dV_dt, dm_dt, dh_dt, dn_dt, ds_dt]
    
    def simulate(self, I_inj, t_max=1000, dt=0.01, record_currents=True):

        # Time array
        t = np.arange(0, t_max, dt)
        
        # Initial conditions (resting state)
        V0 = -65.0
        m0 = self.alpha_m(V0) / (self.alpha_m(V0) + self.beta_m(V0))
        h0 = self.alpha_h(V0) / (self.alpha_h(V0) + self.beta_h(V0))
        n0 = self.alpha_n(V0) / (self.alpha_n(V0) + self.beta_n(V0))
        s0 = self.s_inf(V0)
        
        state0 = [V0, m0, h0, n0, s0]
        
        # Solve ODEs
        solution = odeint(self.derivatives, state0, t, args=(I_inj,))
        
        # Record currents if requested
        if record_currents:
            self._record_all_currents(solution, t)
        
        return t, solution
    
    def _record_all_currents(self, solution, t):
        V = solution[:, 0]
        m = solution[:, 1]
        h = solution[:, 2]
        n = solution[:, 3]
        s = solution[:, 4]
        
        I_Na_arr = np.array([self.I_Na(V[i], m[i], h[i]) for i in range(len(t))])
        I_K_arr = np.array([self.I_K(V[i], n[i]) for i in range(len(t))])
        I_L_arr = np.array([self.I_L(V[i]) for i in range(len(t))])
        I_NaP_arr = np.array([self.I_NaP(V[i]) for i in range(len(t))])
        I_K_slow_arr = np.array([self.I_K_slow(V[i], s[i]) for i in range(len(t))])
        
        self.recorded_currents = {
            'I_Na': I_Na_arr,
            'I_K': I_K_arr,
            'I_L': I_L_arr,
            'I_NaP': I_NaP_arr,
            'I_K_slow': I_K_slow_arr,
            't': t
        }
    
    def get_bursting_parameters(self):
        return {
            'g_NaP': self.g_NaP,
            'g_K_slow': self.g_K_slow,
            'tau_s': self.tau_s
        }