

import numpy as np

class MetabolicAnalyzer:
    
    def __init__(self, neuron_model):
        
        self.model = neuron_model
        
    def calculate_metabolic_load(self, t, solution):
        
        V = solution[:, 0]
        dt = t[1] - t[0]
        total_load = 0.0

        m = solution[:, 1]
        h = solution[:, 2]
        I_Na = np.array([self.model.I_Na(V[i], m[i], h[i]) for i in range(len(t))])
        
        if hasattr(self.model, 'I_NaP'):
            I_NaP = np.array([self.model.I_NaP(V[i]) for i in range(len(t))])
            total_Na_current = np.abs(I_Na + I_NaP)
        else:
            total_Na_current = np.abs(I_Na)
            
        Na_load = np.trapz(total_Na_current, dx=dt)
        total_load += Na_load

        if hasattr(self.model, 'I_Ca'):
            I_Ca = np.array([self.model.I_Ca(V[i]) for i in range(len(t))])
            
            Ca_load = np.trapz(np.abs(I_Ca), dx=dt)
            total_load += (Ca_load * 3.0)
            
        return total_load
    
    def calculate_energy_per_spike(self, t, solution, spike_times):
        
        total_energy = self.calculate_metabolic_load(t, solution)
        n_spikes = len(spike_times)
        
        if n_spikes == 0:
            return np.nan, total_energy, 0
        
        energy_per_spike = total_energy / n_spikes
        
        return energy_per_spike, total_energy, n_spikes
    
    def calculate_energy_in_window(self, t, solution, t_start, t_end):
    
        idx_start = np.searchsorted(t, t_start)
        idx_end = np.searchsorted(t, t_end)
        
        t_window = t[idx_start:idx_end]
        solution_window = solution[idx_start:idx_end]
        
        if len(t_window) < 2:
            return 0.0
        
        window_energy = self.calculate_metabolic_load(t_window, solution_window)
        
        return window_energy
    
    def get_energy_statistics(self, t, solution, spike_times):

        energy_per_spike, total_energy, n_spikes = self.calculate_energy_per_spike(
            t, solution, spike_times
        )
        
        duration_sec = (t[-1] - t[0]) / 1000.0
        
        stats = {
            'total_energy': total_energy,
            'n_spikes': n_spikes,
            'energy_per_spike': energy_per_spike,
            'energy_rate': total_energy / duration_sec if duration_sec > 0 else np.nan,
            'duration_ms': t[-1] - t[0]
        }
        
        return stats

    def calculate_sodium_influx(self, t, solution):
        return self.calculate_metabolic_load(t, solution)