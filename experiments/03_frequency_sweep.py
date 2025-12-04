
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.models.bursting_neuron import BurstingNeuron
from src.analysis.spike_detection import SpikeDetector
from src.analysis.metabolic_cost import MetabolicAnalyzer
from src.visualization.basic_plots import BasicPlotter

def main():
    
    neuron = BurstingNeuron()
    spike_detector = SpikeDetector()
    metabolic = MetabolicAnalyzer(neuron)
    plotter = BasicPlotter()
    
    I_inj_values = np.linspace(0, 20, 25)
    
    results = {
        'I_inj': [],
        'n_spikes': [],
        'firing_rate': [],
        'total_energy': [],
        'energy_per_spike': []
    }
    
    for I_inj in I_inj_values:
        print(f"   Simulating I_inj = {I_inj:.2f} μA/cm²...", end='')
        
        t, solution = neuron.simulate(I_inj=I_inj, t_max=2000, dt=0.01)
        
        spike_times, _ = spike_detector.detect_spikes(t, solution[:, 0])
        stats = spike_detector.get_spike_statistics(t, solution[:, 0])
        
        energy_stats = metabolic.get_energy_statistics(t, solution, spike_times)
        
        results['I_inj'].append(I_inj)
        results['n_spikes'].append(stats['n_spikes'])
        results['firing_rate'].append(stats['firing_rate'])
        results['total_energy'].append(energy_stats['total_energy'])
        results['energy_per_spike'].append(energy_stats['energy_per_spike'])
        
        print(f" {stats['n_spikes']} spikes, {stats['firing_rate']:.2f} Hz")
    
    df = pd.DataFrame(results)
    df.to_csv('data/processed/frequency_sweep_results.csv', index=False)
    print("   Saved: data/processed/frequency_sweep_results.csv")
    
    
    fig1, ax1 = plotter.plot_fi_curve(results['I_inj'], results['firing_rate'])
    plt.savefig('data/figures/06_fi_curve_bursting.png', dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/06_fi_curve_bursting.png")
    
    fig2, ax2 = plotter.plot_energy_vs_frequency(results['firing_rate'], 
                                                  results['energy_per_spike'])
    plt.savefig('data/figures/07_energy_efficiency.png', dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/07_energy_efficiency.png")
    
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(results['I_inj'], results['firing_rate'], 'o-', 
                    linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Input Current (μA/cm²)', fontsize=12)
    axes[0, 0].set_ylabel('Firing Rate (Hz)', fontsize=12)
    axes[0, 0].set_title('F-I Curve', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    valid_idx = [i for i, e in enumerate(results['energy_per_spike']) if not np.isnan(e)]
    axes[0, 1].plot([results['I_inj'][i] for i in valid_idx],
                    [results['energy_per_spike'][i] for i in valid_idx], 
                    'o-', linewidth=2, markersize=8, color='red')
    axes[0, 1].set_xlabel('Input Current (μA/cm²)', fontsize=12)
    axes[0, 1].set_ylabel('Energy per Spike (a.u.)', fontsize=12)
    axes[0, 1].set_title('Energy Efficiency vs Current', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(results['I_inj'], results['total_energy'], 
                    'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Input Current (μA/cm²)', fontsize=12)
    axes[1, 0].set_ylabel('Total Energy (a.u.)', fontsize=12)
    axes[1, 0].set_title('Total Metabolic Cost', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot([results['firing_rate'][i] for i in valid_idx],
                    [results['energy_per_spike'][i] for i in valid_idx], 
                    'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_xlabel('Firing Rate (Hz)', fontsize=12)
    axes[1, 1].set_ylabel('Energy per Spike (a.u.)', fontsize=12)
    axes[1, 1].set_title('Efficiency vs Activity', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Traditional Energy Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/figures/08_traditional_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/08_traditional_analysis.png")

    if valid_idx:
        optimal_idx = valid_idx[np.argmin([results['energy_per_spike'][i] for i in valid_idx])]
        print(f"   Optimal I_inj: {results['I_inj'][optimal_idx]:.2f} μA/cm²")
        print(f"   Optimal firing rate: {results['firing_rate'][optimal_idx]:.2f} Hz")
        print(f"   Minimum energy/spike: {results['energy_per_spike'][optimal_idx]:.2f} a.u.")
    
    
    plt.show()

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/figures', exist_ok=True)
    main()