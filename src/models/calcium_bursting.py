import numpy as np
from scipy.integrate import odeint
from .hodgkin_huxley import HodgkinHuxley

class CalciumBurstingNeuron(HodgkinHuxley):
    
    def __init__(self, g_Ca=1.0, g_K_Ca=2.0, **kwargs):
        
        super().__init__()
        
        self.g_Ca = g_Ca
        self.E_Ca = 120.0
        
        self.g_K_Ca = g_K_Ca
        self.E_K_Ca = self.E_K
        
        self.tau_Ca = 200.0
        self.Ca_rest = 0.0001
        self.k_Ca = 0.001
    
    def m_inf_Ca(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 20.0) / 9.0))
    
    def w_inf(self, Ca):
        return Ca / (Ca + 0.001)
    
    def I_Ca(self, V):
        return self.g_Ca * self.m_inf_Ca(V) * (V - self.E_Ca)
    
    def I_K_Ca(self, V, Ca):
        return self.g_K_Ca * self.w_inf(Ca) * (V - self.E_K_Ca)
    
    def derivatives(self, state, t, I_inj):
        V, m, h, n, Ca = state
        
        dm_dt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dh_dt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dn_dt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        
        Ca_influx = -self.k_Ca * self.I_Ca(V) 
        dCa_dt = Ca_influx - (Ca - self.Ca_rest) / self.tau_Ca
        
        dV_dt = (I_inj 
                 - self.I_Na(V, m, h) 
                 - self.I_K(V, n) 
                 - self.I_L(V)
                 - self.I_Ca(V)
                 - self.I_K_Ca(V, Ca)) / self.C_m
        
        return [dV_dt, dm_dt, dh_dt, dn_dt, dCa_dt]
    
    def simulate(self, I_inj, t_max=1000, dt=0.01, record_currents=True):
        t = np.arange(0, t_max, dt)
        
        V0 = -65.0
        m0 = self.alpha_m(V0) / (self.alpha_m(V0) + self.beta_m(V0))
        h0 = self.alpha_h(V0) / (self.alpha_h(V0) + self.beta_h(V0))
        n0 = self.alpha_n(V0) / (self.alpha_n(V0) + self.beta_n(V0))
        Ca0 = self.Ca_rest
        
        state0 = [V0, m0, h0, n0, Ca0]
        
        solution = odeint(self.derivatives, state0, t, args=(I_inj,))
        
        if record_currents:
            self._record_all_currents_ca(solution, t)
        
        return t, solution
    
    def _record_all_currents_ca(self, solution, t):
        V = solution[:, 0]
        m = solution[:, 1]
        h = solution[:, 2]
        n = solution[:, 3]
        Ca = solution[:, 4]
        
        I_Na_arr = np.array([self.I_Na(V[i], m[i], h[i]) for i in range(len(t))])
        I_K_arr = np.array([self.I_K(V[i], n[i]) for i in range(len(t))])
        I_L_arr = np.array([self.I_L(V[i]) for i in range(len(t))])
        I_Ca_arr = np.array([self.I_Ca(V[i]) for i in range(len(t))])
        
        I_K_Ca_arr = np.array([self.I_K_Ca(V[i], Ca[i]) for i in range(len(t))])
        
        self.recorded_currents = {
            'I_Na': I_Na_arr,
            'I_K': I_K_arr,
            'I_L': I_L_arr,
            'I_Ca': I_Ca_arr,
            'I_K_Ca': I_K_Ca_arr,
            't': t
        }