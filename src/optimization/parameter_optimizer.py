import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

# Move imports to top for performance
from src.analysis.spike_detection import SpikeDetector
from src.analysis.metabolic_cost import MetabolicAnalyzer

class ParameterOptimizer:

    def __init__(self, model_class=None, analyzer_class=None):
       
        if model_class is None:
            from src.models.bursting_neuron import BurstingNeuron
            self.model_class = BurstingNeuron
        else:
            self.model_class = model_class
            
        if analyzer_class is None:
            from src.analysis.multi_scale_analyzer import MultiScaleAnalyzer
            self.analyzer_class = MultiScaleAnalyzer
        else:
            self.analyzer_class = analyzer_class
            
        self.history = []
    
    def objective_energy_per_spike(self, params):
        g_NaP, g_K_slow, tau_s = params
        
        try:
            model = self.model_class(g_NaP=g_NaP, g_K_slow=g_K_slow, tau_s=tau_s)
            
            t, solution = model.simulate(I_inj=8.0, t_max=2000, dt=0.01)
            
            detector = SpikeDetector()
            spike_times, _ = detector.detect_spikes(t, solution[:, 0])
            
            if len(spike_times) < 10:  
                return 1e6
            
            analyzer = MetabolicAnalyzer(model)
            energy_stats = analyzer.get_energy_statistics(t, solution, spike_times)
            
            energy_per_spike = energy_stats['energy_per_spike']
            
            firing_rate = len(spike_times) / ((t[-1] - t[0]) / 1000.0)
            
            if firing_rate < 10 or firing_rate > 150:  
                return 1e6 + abs(firing_rate - 80)*100
            
            self.history.append({
                'params': params.copy(),
                'energy_per_spike': energy_per_spike,
                'firing_rate': firing_rate,
                'n_spikes': len(spike_times)
            })
            
            return energy_per_spike
        
        except Exception as e:
            return 1e6

    def objective_energy_per_bit(self, params):
        
        g_NaP, g_K_slow, tau_s = params
        
        try:
            model = self.model_class(g_NaP=g_NaP, g_K_slow=g_K_slow, tau_s=tau_s)
            t, solution = model.simulate(I_inj=8.0, t_max=3000, dt=0.01)
            
            analyzer = self.analyzer_class(model)
            results = analyzer.full_analysis(t, solution)
            
            energy_per_bit = results['energy_information_tradeoff']['energy_per_bit']
            
            if np.isnan(energy_per_bit) or energy_per_bit <= 0:
                return 1e6
            
            self.history.append({
                'params': params.copy(),
                'energy_per_bit': energy_per_bit,
                'isi_entropy': results['energy_information_tradeoff']['isi_entropy']
            })
            
            return energy_per_bit
        
        except Exception as e:
            return 1e6

    def optimize_bursting(self, bounds=None, maxiter=50, workers=1):
        
        if bounds is None:
            bounds = [
                (0.2, 1.0),  
                (1.0, 4.0),  
                (50, 500) 
            ]
        
        print(f"Optimizing Energy per Spike...")
        print(f"Bounds: g_NaP={bounds[0]}, g_K_slow={bounds[1]}, tau_s={bounds[2]}")
        
        self.history = []
        
        # Run optimization
        result = differential_evolution(
            self.objective_energy_per_spike,
            bounds=bounds,
            maxiter=maxiter,
            popsize=15,
            workers=workers,
            disp=True,
            seed=42
        )
        
        print(f"Optimal parameters:")
        print(f"  g_NaP: {result.x[0]:.4f}")
        print(f"  g_K_slow: {result.x[1]:.4f}")
        print(f"  tau_s: {result.x[2]:.1f}")
        print(f"Optimal Energy: {result.fun:.4f}")
        
        return result

    def optimize_local(self, initial_params, objective='energy_per_spike'):
       
        if objective == 'energy_per_spike':
            obj_func = self.objective_energy_per_spike
        else:
            obj_func = self.objective_energy_per_bit
        
        print(f"Local optimization from {initial_params}...")
        
        self.history = []
        
        result = minimize(
            obj_func,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 100, 'disp': True}
        )
        
        print(f"Optimal parameters: {result.x}")
        print(f"Optimal {objective}: {result.fun:.4f}")
        
        return result

    def plot_optimization_history(self):
        if not self.history:
            print("No history to plot")
            return
        
        iterations = range(len(self.history))
        
        if 'energy_per_spike' in self.history[0]:
            metric = [h['energy_per_spike'] for h in self.history]
            metric_name = 'Energy per Spike (a.u.)'
        else:
            metric = [h['energy_per_bit'] for h in self.history]
            metric_name = 'Energy per Bit (a.u.)'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(iterations, metric, 'o-', alpha=0.6)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel(metric_name, fontsize=12)
        ax1.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
        best_so_far = [min(metric[:i+1]) for i in range(len(metric))]
        ax1.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best so far')
        ax1.legend()
        params_g_NaP = [h['params'][0] for h in self.history]
        params_tau_s = [h['params'][2] for h in self.history]
        
        ax2.plot(iterations, params_g_NaP, 'o-', label='g_NaP', alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(iterations, params_tau_s, 's-', color='orange', label='tau_s', alpha=0.7)
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('g_NaP', fontsize=12)
        ax2_twin.set_ylabel('tau_s (ms)', fontsize=12)
        ax2.set_title('Parameter Evolution', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def compare_baseline_vs_optimal(self, optimal_params, baseline_params=[0.5, 2.0, 200.0]):
        
        results = {}
        
        for name, params in [('Baseline', baseline_params), ('Optimal', optimal_params)]:
            print(f"\n{name} parameters: {params}")
            
            model = self.model_class(g_NaP=params[0], g_K_slow=params[1], tau_s=params[2])
            t, solution = model.simulate(I_inj=8.0, t_max=3000, dt=0.01)
            
            analyzer = self.analyzer_class(model)
            full_results = analyzer.full_analysis(t, solution)
            
            results[name] = {
                'params': params,
                'energy_per_spike': full_results['energy_statistics']['energy_per_spike'],
                'firing_rate': full_results['spike_statistics']['firing_rate'],
                'n_bursts': full_results['burst_statistics']['n_bursts'],
                'energy_per_bit': full_results['energy_information_tradeoff']['energy_per_bit']
            }
            
            print(f"  Energy/spike: {results[name]['energy_per_spike']:.2f}")
            print(f"  Firing rate:  {results[name]['firing_rate']:.2f} Hz")
            print(f"  N bursts:     {results[name]['n_bursts']}")
            print(f"  Energy/bit:   {results[name]['energy_per_bit']:.2f}")
        
        base_es = results['Baseline']['energy_per_spike']
        opt_es = results['Optimal']['energy_per_spike']
        
        if not np.isnan(base_es) and base_es > 0:
            improvement_eps = ((base_es - opt_es) / base_es) * 100
            print(f"Energy per spike: {improvement_eps:+.2f}%")
        else:
            print("Energy per spike: N/A")
            
        return results