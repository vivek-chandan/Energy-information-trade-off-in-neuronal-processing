import numpy as np
from .spike_detection import SpikeDetector

class BurstAnalyzer:
    
    def __init__(self, max_isi_in_burst=20.0, min_spikes_per_burst=2):
        self.max_isi = max_isi_in_burst
        self.min_spikes = min_spikes_per_burst
        self.spike_detector = SpikeDetector()
    
    def detect_bursts(self, spike_times):
        if len(spike_times) < self.min_spikes:
            return []
        
        isi = np.diff(spike_times)
        
        bursts = []
        current_burst = [spike_times[0]]
        
        for i in range(len(isi)):
            if isi[i] <= self.max_isi:
                current_burst.append(spike_times[i + 1])
            else:
                if len(current_burst) >= self.min_spikes:
                    bursts.append(self._characterize_burst(current_burst))
                current_burst = [spike_times[i + 1]]
        
        if len(current_burst) >= self.min_spikes:
            bursts.append(self._characterize_burst(current_burst))
        
        return bursts
    
    def _characterize_burst(self, spike_times_in_burst):
        spike_times = np.array(spike_times_in_burst)
        
        n_spikes = len(spike_times)
        burst_start = spike_times[0]
        burst_end = spike_times[-1]
        burst_duration = burst_end - burst_start
        
        if burst_duration > 0:
            intraburst_freq = (n_spikes - 1) / (burst_duration / 1000.0)  # Hz
        else:
            intraburst_freq = 0.0
        
        isi = np.diff(spike_times)
        
        burst_info = {
            'spike_times': spike_times,
            'n_spikes': n_spikes,
            'start_time': burst_start,
            'end_time': burst_end,
            'duration': burst_duration,
            'intraburst_freq': intraburst_freq,
            'mean_isi': np.mean(isi),
            'std_isi': np.std(isi),
            'cv_isi': np.std(isi) / np.mean(isi) if np.mean(isi) > 0 else 0
        }
        
        return burst_info
    
    def calculate_interburst_intervals(self, bursts):
        if len(bursts) < 2:
            return np.array([])
        
        ibi = []
        for i in range(len(bursts) - 1):
            interval = bursts[i + 1]['start_time'] - bursts[i]['end_time']
            ibi.append(interval)
        
        return np.array(ibi)
    
    def get_burst_statistics(self, spike_times):

        bursts = self.detect_bursts(spike_times)
        
        if len(bursts) == 0:
            return {
                'n_bursts': 0,
                'bursts': [],
                'mean_spikes_per_burst': np.nan,
                'mean_burst_duration': np.nan,
                'mean_intraburst_freq': np.nan,
                'mean_interburst_interval': np.nan
            }
        
        ibi = self.calculate_interburst_intervals(bursts)
        
        stats = {
            'n_bursts': len(bursts),
            'bursts': bursts,
            'mean_spikes_per_burst': np.mean([b['n_spikes'] for b in bursts]),
            'std_spikes_per_burst': np.std([b['n_spikes'] for b in bursts]),
            'mean_burst_duration': np.mean([b['duration'] for b in bursts]),
            'mean_intraburst_freq': np.mean([b['intraburst_freq'] for b in bursts]),
            'mean_interburst_interval': np.mean(ibi) if len(ibi) > 0 else np.nan,
            'burst_duty_cycle': self._calculate_duty_cycle(bursts, spike_times)
        }
        
        return stats
    
    def _calculate_duty_cycle(self, bursts, spike_times):

        if len(bursts) == 0 or len(spike_times) < 2:
            return 0.0
        
        total_burst_time = sum([b['duration'] for b in bursts])
        total_time = spike_times[-1] - spike_times[0]
        
        if total_time > 0:
            return total_burst_time / total_time
        else:
            return 0.0
    
    def classify_burst_pattern(self, burst_info):
        
        n_spikes = burst_info['n_spikes']
        freq = burst_info['intraburst_freq']
        
        spike_threshold = 4
        freq_threshold = 100  # Hz
        
        if n_spikes < spike_threshold:
            spike_category = 'short'
        else:
            spike_category = 'long'
        
        if freq < freq_threshold:
            freq_category = 'low'
        else:
            freq_category = 'high'
        
        pattern_type = f"{spike_category}_{freq_category}"
        
        return pattern_type
        