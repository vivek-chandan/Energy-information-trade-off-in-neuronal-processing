
import numpy as np
from .spike_detection import SpikeDetector
from .metabolic_cost import MetabolicAnalyzer
from .burst_analyzer import BurstAnalyzer
from .information_theory import InformationAnalyzer

class MultiScaleAnalyzer:
    
    def __init__(self, neuron_model):
        
        self.model = neuron_model
        self.spike_detector = SpikeDetector()
        self.metabolic = MetabolicAnalyzer(neuron_model)
        self.burst_analyzer = BurstAnalyzer()
        self.info_analyzer = InformationAnalyzer()
    
    def analyze_at_scale(self, t, solution, window_size):
        spike_times, _ = self.spike_detector.detect_spikes(t, solution[:, 0])
        
        t_start = t[0]
        t_end = t[-1]
        
        window_results = []
        current_time = t_start
        
        while current_time + window_size <= t_end:
            window_end = current_time + window_size
            
            window_spikes = spike_times[
                (spike_times >= current_time) & (spike_times < window_end)
            ]
            
            window_energy = self.metabolic.calculate_energy_in_window(
                t, solution, current_time, window_end
            )
            
            n_spikes = len(window_spikes)
            if n_spikes > 0:
                efficiency = window_energy / n_spikes
                firing_rate = n_spikes / (window_size / 1000.0)  # Hz
            else:
                efficiency = np.nan
                firing_rate = 0.0
            
            window_results.append({
                'start': current_time,
                'end': window_end,
                'n_spikes': n_spikes,
                'energy': window_energy,
                'efficiency': efficiency,
                'firing_rate': firing_rate
            })
            
            current_time += window_size
        
        # Aggregate results
        valid_efficiencies = [w['efficiency'] for w in window_results 
                             if not np.isnan(w['efficiency'])]
        
        scale_results = {
            'window_size': window_size,
            'n_windows': len(window_results),
            'windows': window_results,
            'mean_efficiency': np.mean(valid_efficiencies) if valid_efficiencies else np.nan,
            'std_efficiency': np.std(valid_efficiencies) if valid_efficiencies else np.nan,
            'cv_efficiency': (np.std(valid_efficiencies) / np.mean(valid_efficiencies) 
                            if valid_efficiencies and np.mean(valid_efficiencies) > 0 else np.nan)
        }
        
        return scale_results
    
    def analyze_multiple_scales(self, t, solution, scales=None):
        if scales is None:
            scales = [10, 25, 50, 100, 250, 500, 1000]  # ms
        
        results = {}
        
        for scale in scales:
            print(f"  Analyzing scale: {scale} ms")
            results[scale] = self.analyze_at_scale(t, solution, scale)
        
        mean_efficiencies = [results[s]['mean_efficiency'] for s in scales]
        valid_means = [e for e in mean_efficiencies if not np.isnan(e)]
        
        if len(valid_means) > 1:
            scale_dependence = np.std(valid_means) / np.mean(valid_means)
        else:
            scale_dependence = np.nan
        
        multi_scale_results = {
            'scales': scales,
            'scale_results': results,
            'scale_dependence_factor': scale_dependence
        }
        
        return multi_scale_results
    
    def analyze_burst_level_efficiency(self, t, solution):

        spike_times, _ = self.spike_detector.detect_spikes(t, solution[:, 0])
        bursts = self.burst_analyzer.detect_bursts(spike_times)
        
        if len(bursts) == 0:
            return {
                'n_bursts': 0,
                'burst_efficiencies': [],
                'mean_energy_per_burst': np.nan,
                'mean_energy_per_spike_in_burst': np.nan
            }
        
        burst_efficiencies = []
        
        for burst in bursts:
            # Calculate energy during burst
            burst_energy = self.metabolic.calculate_energy_in_window(
                t, solution, burst['start_time'], burst['end_time']
            )
            
            n_spikes = burst['n_spikes']
            
            burst_efficiencies.append({
                'burst': burst,
                'energy': burst_energy,
                'energy_per_spike': burst_energy / n_spikes,
                'n_spikes': n_spikes,
                'duration': burst['duration']
            })
        
        # Aggregate
        mean_energy_per_burst = np.mean([b['energy'] for b in burst_efficiencies])
        mean_energy_per_spike = np.mean([b['energy_per_spike'] for b in burst_efficiencies])
        
        result = {
            'n_bursts': len(bursts),
            'burst_efficiencies': burst_efficiencies,
            'mean_energy_per_burst': mean_energy_per_burst,
            'mean_energy_per_spike_in_burst': mean_energy_per_spike,
            'burst_efficiency_ratio': mean_energy_per_burst / mean_energy_per_spike if mean_energy_per_spike > 0 else np.nan
        }
        
        return result
    
    def analyze_energy_information_tradeoff(self, t, solution):

        spike_times, _ = self.spike_detector.detect_spikes(t, solution[:, 0])
        bursts = self.burst_analyzer.detect_bursts(spike_times)
        
        # Calculate energy
        total_energy = self.metabolic.calculate_sodium_influx(t, solution)
        
        # Calculate information
        info_stats = self.info_analyzer.get_information_statistics(
            spike_times, bursts, t[0], t[-1]
        )
        
        # Calculate energy per bit
        if info_stats['isi_entropy'] > 0:
            energy_per_bit = total_energy / (info_stats['isi_entropy'] * len(spike_times))
        else:
            energy_per_bit = np.nan
        
        tradeoff = {
            'total_energy': total_energy,
            'isi_entropy': info_stats['isi_entropy'],
            'information_rate': info_stats['information_rate'],
            'energy_per_bit': energy_per_bit,
            'n_spikes': len(spike_times)
        }
        
        return tradeoff
    
    def full_analysis(self, t, solution):

        spike_times, spike_indices = self.spike_detector.detect_spikes(t, solution[:, 0])
        spike_stats = self.spike_detector.get_spike_statistics(t, solution[:, 0])
        
        burst_stats = self.burst_analyzer.get_burst_statistics(spike_times)
        
        energy_stats = self.metabolic.get_energy_statistics(t, solution, spike_times)
        
        multi_scale = self.analyze_multiple_scales(t, solution)
        
        burst_efficiency = self.analyze_burst_level_efficiency(t, solution)
        
        tradeoff = self.analyze_energy_information_tradeoff(t, solution)
        
        complete_results = {
            'spike_statistics': spike_stats,
            'burst_statistics': burst_stats,
            'energy_statistics': energy_stats,
            'multi_scale_analysis': multi_scale, 
            'burst_efficiency': burst_efficiency, 
            'energy_information_tradeoff': tradeoff  
        }
        
        return complete_results