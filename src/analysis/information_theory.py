import numpy as np

class InformationAnalyzer:
    
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
    
    def calculate_isi_entropy(self, spike_times):
        if len(spike_times) < 2:
            return 0.0
        
        isi = np.diff(spike_times)
        
        if len(isi) < 5:
            return 0.0
        
        try:
            hist, bin_edges = np.histogram(isi, bins=self.n_bins, density=True)
        except ValueError:
            return 0.0
        
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        bin_width = bin_edges[1] - bin_edges[0]
        p = hist * bin_width
        
        p_sum = np.sum(p)
        if p_sum > 0:
            p = p / p_sum
        else:
            return 0.0
        
        entropy = -np.sum(p * np.log2(p))
        
        return entropy
    
    def calculate_spike_time_entropy(self, spike_times, t_start, t_end, n_bins=None):
        if n_bins is None:
            n_bins = self.n_bins
        
        if len(spike_times) == 0:
            return 0.0
            
        try:
            hist, _ = np.histogram(spike_times, bins=n_bins, range=(t_start, t_end))
        except ValueError:
            return 0.0
        total_spikes = np.sum(hist)
        if total_spikes == 0:
            return 0.0
            
        p = hist / total_spikes
        p = p[p > 0]
        
        if len(p) == 0:
            return 0.0
        entropy = -np.sum(p * np.log2(p))
        
        return entropy
    
    def calculate_burst_pattern_entropy(self, bursts):
        if len(bursts) < 2:
            return 0.0
        
        # Extract the symbol: number of spikes in each burst
        spikes_per_burst = [b['n_spikes'] for b in bursts]
        
        # Create histogram of these discrete symbols
        unique, counts = np.unique(spikes_per_burst, return_counts=True)
        
        # Convert to probabilities
        total_bursts = np.sum(counts)
        if total_bursts == 0:
            return 0.0
            
        p = counts / total_bursts
        entropy = -np.sum(p * np.log2(p))
        
        return entropy
    
    def calculate_information_rate(self, spike_times, t_start, t_end):
        
        entropy = self.calculate_isi_entropy(spike_times)
        
        duration_sec = (t_end - t_start) / 1000.0
        n_spikes = len(spike_times)
        
        if duration_sec > 0 and n_spikes > 0:
            spike_rate = n_spikes / duration_sec
            info_rate = entropy * spike_rate
        else:
            info_rate = 0.0
        
        return info_rate
    
    def get_information_statistics(self, spike_times, bursts=None, t_start=None, t_end=None):
        if t_start is None:
            t_start = spike_times[0] if len(spike_times) > 0 else 0
        if t_end is None:
            t_end = spike_times[-1] if len(spike_times) > 0 else 1000
        
        stats = {
            'isi_entropy': self.calculate_isi_entropy(spike_times),
            'temporal_entropy': self.calculate_spike_time_entropy(spike_times, t_start, t_end),
            'information_rate': self.calculate_information_rate(spike_times, t_start, t_end)
        }
        
        if bursts is not None and len(bursts) > 0:
            stats['burst_pattern_entropy'] = self.calculate_burst_pattern_entropy(bursts)
        else:
            stats['burst_pattern_entropy'] = 0.0
        
        return stats