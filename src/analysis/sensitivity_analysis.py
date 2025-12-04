
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class SensitivityAnalyzer:
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.results = {}
    
    def parameter_sweep_1d(self, param_name, param_range, I_inj=8.0, 
                          metric='energy_per_spike', n_trials=1):
        
        from src.models.bursting_neuron import BurstingNeuron
        from src.analysis.spike_detection import SpikeDetector
        from src.analysis.metabolic_cost import MetabolicAnalyzer
        
        results = {
            'param_values': [],
            'metric_values': [],
            'firing_rates': [],
            'n_bursts': []
        }
        
        print(f"Sweeping {param_name} from {param_range[0]} to {param_range[-1]}...")
        
        for param_value in tqdm(param_range):

            kwargs = {param_name: param_value}
            model = BurstingNeuron(**kwargs)
            
            t, solution = model.simulate(I_inj=I_inj, t_max=2000, dt=0.01)
            
            detector = SpikeDetector()
            spike_times, _ = detector.detect_spikes(t, solution[:, 0])
            
            if metric == 'energy_per_spike':
                analyzer = MetabolicAnalyzer(model)
                energy_stats = analyzer.get_energy_statistics(t, solution, spike_times)
                metric_value = energy_stats['energy_per_spike']
            elif metric == 'firing_rate':
                metric_value = len(spike_times) / ((t[-1] - t[0]) / 1000.0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            results['param_values'].append(param_value)
            results['metric_values'].append(metric_value)
            results['firing_rates'].append(len(spike_times) / ((t[-1] - t[0]) / 1000.0))
        
        self.results[f'{param_name}_sweep'] = results
        return results
    
    def parameter_sweep_2d(self, param1_name, param1_range, 
                          param2_name, param2_range, 
                          I_inj=8.0, metric='energy_per_spike'):
        
        from src.models.bursting_neuron import BurstingNeuron
        from src.analysis.spike_detection import SpikeDetector
        from src.analysis.metabolic_cost import MetabolicAnalyzer
        
        n1, n2 = len(param1_range), len(param2_range)
        metric_grid = np.zeros((n1, n2))
        
        print(f"2D sweep: {param1_name} x {param2_name}...")
        
        for i, p1 in enumerate(tqdm(param1_range)):
            for j, p2 in enumerate(param2_range):
                kwargs = {param1_name: p1, param2_name: p2}
                model = BurstingNeuron(**kwargs)
                
                try:
                    # Simulate
                    t, solution = model.simulate(I_inj=I_inj, t_max=2000, dt=0.01)
                    
                    # Analyze
                    detector = SpikeDetector()
                    spike_times, _ = detector.detect_spikes(t, solution[:, 0])
                    
                    if metric == 'energy_per_spike' and len(spike_times) > 0:
                        analyzer = MetabolicAnalyzer(model)
                        energy_stats = analyzer.get_energy_statistics(t, solution, spike_times)
                        metric_grid[i, j] = energy_stats['energy_per_spike']
                    else:
                        metric_grid[i, j] = np.nan
                
                except Exception as e:
                    print(f"Error at ({p1}, {p2}): {e}")
                    metric_grid[i, j] = np.nan
        
        self.results[f'{param1_name}_{param2_name}_sweep'] = {
            'param1_range': param1_range,
            'param2_range': param2_range,
            'metric_grid': metric_grid
        }
        
        return metric_grid
    
    def monte_carlo_robustness(self, n_samples=100, noise_level=0.1):
        
        from src.models.bursting_neuron import BurstingNeuron
        from src.analysis.multi_scale_analyzer import MultiScaleAnalyzer
        
        base_params = {
            'g_NaP': 0.5,
            'g_K_slow': 2.0,
            'tau_s': 200.0 
        }
        
        results = []
        
        print(f"Monte Carlo robustness test ({n_samples} samples)...")
        
        for i in tqdm(range(n_samples)):
            noisy_params = {}
            for param, value in base_params.items():
                noise = np.random.uniform(-noise_level, noise_level)
                noisy_params[param] = value * (1 + noise)
            
            try:
                model = BurstingNeuron(**noisy_params)
                t, solution = model.simulate(I_inj=8.0, t_max=2000, dt=0.01)
                
                analyzer = MultiScaleAnalyzer(model)
                multi_scale = analyzer.analyze_multiple_scales(t, solution, scales=[10, 100, 1000])
                
                eff_10 = multi_scale['scale_results'][10]['mean_efficiency']
                eff_100 = multi_scale['scale_results'][100]['mean_efficiency']
                eff_1000 = multi_scale['scale_results'][1000]['mean_efficiency']
                
                scale_change = ((eff_1000 - eff_10) / eff_10) * 100
                
                results.append({
                    'sample': i,
                    **noisy_params, 
                    'eff_10ms': eff_10,
                    'eff_1000ms': eff_1000,
                    'scale_change': scale_change
                })
            
            except Exception as e:
                continue
        
        df = pd.DataFrame(results)
        
        impact_scores = {}
        
        for param in base_params.keys():
            correlation = df[param].corr(df['scale_change'])
            impact_scores[param] = correlation
        sorted_impact = sorted(impact_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for param, score in sorted_impact:
            strength = "Positive" if score > 0 else "Negative"
            print(f"  {param:<12} : {score:+.4f} ({strength} correlation)")
            
        critical_param = sorted_impact[0][0]
        print(f"\nCRITICAL CHANNEL IDENTIFIED: {critical_param}")
        
        self.results['monte_carlo'] = {
            'n_samples': len(results),
            'data': df,
            'impact_scores': impact_scores,
            'critical_parameter': critical_param,
            'summary': {
                'mean_scale_change': df['scale_change'].mean(),
                'std_scale_change': df['scale_change'].std(),
                'robust': abs(df['scale_change'].mean()) < 10.0 
            }
        }
        
        return df
    
    def plot_sensitivity(self, param_name):
        if f'{param_name}_sweep' not in self.results:
            print(f"No results for {param_name}")
            return
        
        results = self.results[f'{param_name}_sweep']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Metric vs parameter
        ax1.plot(results['param_values'], results['metric_values'], 
                'o-', linewidth=2, markersize=8)
        ax1.set_xlabel(f'{param_name}', fontsize=12)
        ax1.set_ylabel('Energy per Spike (a.u.)', fontsize=12)
        ax1.set_title(f'Sensitivity to {param_name}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Firing rate vs parameter
        ax2.plot(results['param_values'], results['firing_rates'], 
                'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel(f'{param_name}', fontsize=12)
        ax2.set_ylabel('Firing Rate (Hz)', fontsize=12)
        ax2.set_title(f'Firing Rate vs {param_name}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_2d_sensitivity(self, param1_name, param2_name):

        key = f'{param1_name}_{param2_name}_sweep'
        if key not in self.results:
            print(f"No results for {param1_name} x {param2_name}")
            return
        
        results = self.results[key]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(results['metric_grid'], aspect='auto', 
                      cmap='RdYlGn_r', origin='lower')
        
        ax.set_xlabel(param2_name, fontsize=12)
        ax.set_ylabel(param1_name, fontsize=12)
        ax.set_title(f'Energy Efficiency: {param1_name} vs {param2_name}', 
                    fontsize=14, fontweight='bold')
        
        # Set tick labels
        n1, n2 = results['metric_grid'].shape
        ax.set_xticks(np.linspace(0, n2-1, 5))
        ax.set_xticklabels([f'{x:.2f}' for x in np.linspace(
            results['param2_range'][0], results['param2_range'][-1], 5)])
        ax.set_yticks(np.linspace(0, n1-1, 5))
        ax.set_yticklabels([f'{y:.2f}' for y in np.linspace(
            results['param1_range'][0], results['param1_range'][-1], 5)])
        
        plt.colorbar(im, ax=ax, label='Energy per Spike (a.u.)')
        plt.tight_layout()
        return fig