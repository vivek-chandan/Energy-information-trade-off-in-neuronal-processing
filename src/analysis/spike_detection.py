

import numpy as np

class SpikeDetector:
    
    def __init__(self, threshold=0.0, min_isi=2.0):
        
        self.threshold = threshold
        self.min_isi = min_isi
    
    def detect_spikes(self, t, V):

        crossings = np.where((V[:-1] < self.threshold) & (V[1:] >= self.threshold))[0]
        
        if len(crossings) == 0:
            return np.array([]), np.array([])
        
        dt = t[1] - t[0]
        window = int(2.0 / dt)
        
        spike_indices = []
        for cross in crossings:
            end_idx = min(cross + window, len(V))
            peak_idx = cross + np.argmax(V[cross:end_idx])
            spike_indices.append(peak_idx)
        
        spike_indices = np.array(spike_indices)
        if len(spike_indices) > 1:
            isi = np.diff(t[spike_indices])
            valid = np.concatenate([[True], isi >= self.min_isi])
            spike_indices = spike_indices[valid]
        
        spike_times = t[spike_indices]
        
        return spike_times, spike_indices
    
    def calculate_isi(self, spike_times):

        if len(spike_times) < 2:
            return np.array([])
        
        return np.diff(spike_times)
    
    def calculate_firing_rate(self, spike_times, t_start=None, t_end=None):
        
        if len(spike_times) == 0:
            return 0.0
        
        if t_start is None:
            t_start = spike_times[0]
        if t_end is None:
            t_end = spike_times[-1]
        
        duration = (t_end - t_start) / 1000.0 
        
        if duration <= 0:
            return 0.0
        
        n_spikes = np.sum((spike_times >= t_start) & (spike_times <= t_end))
        
        return n_spikes / duration
    
    def calculate_cv_isi(self, spike_times):
        
        isi = self.calculate_isi(spike_times)
        
        if len(isi) < 2:
            return np.nan
        
        return np.std(isi) / np.mean(isi)
    
    def get_spike_statistics(self, t, V):
        
        spike_times, spike_indices = self.detect_spikes(t, V)
        
        stats = {
            'n_spikes': len(spike_times),
            'spike_times': spike_times,
            'spike_indices': spike_indices,
            'firing_rate': self.calculate_firing_rate(spike_times, t[0], t[-1]),
            'mean_isi': np.mean(self.calculate_isi(spike_times)) if len(spike_times) > 1 else np.nan,
            'cv_isi': self.calculate_cv_isi(spike_times)
        }
        
        return stats