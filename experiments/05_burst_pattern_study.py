
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.models.bursting_neuron import BurstingNeuron
from src.analysis.multi_scale_analyzer import MultiScaleAnalyzer
from visualization.advanced_plots import NovelPlotter

def main():
    
    neuron = BurstingNeuron()
    analyzer = MultiScaleAnalyzer(neuron)
    plotter = NovelPlotter()
    
    I_values = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    
    all_bursts = []
    all_burst_efficiencies = []
    
    for I_inj in I_values:
        print(f"   Testing I_inj = {I_inj} μA/cm²...", end='')
        
        t, solution = neuron.simulate(I_inj=I_inj, t_max=3000, dt=0.01)
        
        burst_efficiency = analyzer.analyze_burst_level_efficiency(t, solution)
        
        if burst_efficiency['n_bursts'] > 0:
            all_bursts.extend(burst_efficiency['burst_efficiencies'])
            print(f" {burst_efficiency['n_bursts']} bursts detected")
        else:
            print(" No bursts")
    
    # Extract burst information
    print(f"\n3. Total bursts analyzed: {len(all_bursts)}")
    
    if len(all_bursts) == 0:
        print("   ERROR: No bursts detected. Adjust parameters.")
        return
    
    
    burst_data = []
    for be in all_bursts:
        burst = be['burst']
        burst_data.append({
            'n_spikes': burst['n_spikes'],
            'duration': burst['duration'],
            'intraburst_freq': burst['intraburst_freq'],
            'energy': be['energy'],
            'energy_per_spike': be['energy_per_spike']
        })
    
    df_bursts = pd.DataFrame(burst_data)
    
    print("\n   Burst pattern statistics:")
    print(f"   Mean spikes per burst: {df_bursts['n_spikes'].mean():.2f} ± {df_bursts['n_spikes'].std():.2f}")
    print(f"   Mean intraburst freq: {df_bursts['intraburst_freq'].mean():.2f} ± {df_bursts['intraburst_freq'].std():.2f} Hz")
    print(f"   Mean energy per burst: {df_bursts['energy'].mean():.4f} ± {df_bursts['energy'].std():.4f}")
    print(f"   Mean energy per spike: {df_bursts['energy_per_spike'].mean():.4f} ± {df_bursts['energy_per_spike'].std():.4f}")
    
    optimal_idx = df_bursts['energy_per_spike'].idxmin()
    optimal_burst = df_bursts.iloc[optimal_idx]
    
    print("\n5. Optimal burst pattern:")
    print(f"   ★ Spikes: {optimal_burst['n_spikes']:.0f}")
    print(f"   ★ Frequency: {optimal_burst['intraburst_freq']:.2f} Hz")
    print(f"   ★ Duration: {optimal_burst['duration']:.2f} ms")
    print(f"   ★ Energy per spike: {optimal_burst['energy_per_spike']:.4f} a.u.")
    
    df_bursts.to_csv('data/processed/burst_patterns.csv', index=False)
    print("   Saved: data/processed/burst_patterns.csv")
    
    bursts_for_plot = [be['burst'] for be in all_bursts]
    
    fig1 = plotter.plot_burst_pattern_space(bursts_for_plot, all_bursts)
    plt.savefig('data/figures/novel_analysis/11_burst_pattern_space.png', 
                dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/novel_analysis/11_burst_pattern_space.png")
    
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].scatter(df_bursts['n_spikes'], df_bursts['energy_per_spike'], 
                      s=80, alpha=0.6, c=df_bursts['intraburst_freq'], 
                      cmap='viridis', edgecolors='black')
    axes[0, 0].set_xlabel('Spikes per Burst', fontsize=12)
    axes[0, 0].set_ylabel('Energy per Spike (a.u.)', fontsize=12)
    axes[0, 0].set_title('Energy vs Burst Size', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Frequency vs energy
    axes[0, 1].scatter(df_bursts['intraburst_freq'], df_bursts['energy_per_spike'], 
                      s=80, alpha=0.6, c=df_bursts['n_spikes'], 
                      cmap='plasma', edgecolors='black')
    axes[0, 1].set_xlabel('Intraburst Frequency (Hz)', fontsize=12)
    axes[0, 1].set_ylabel('Energy per Spike (a.u.)', fontsize=12)
    axes[0, 1].set_title('Energy vs Frequency', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Duration distribution
    axes[1, 0].hist(df_bursts['duration'], bins=20, color='steelblue', 
                   edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Burst Duration (ms)', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Duration Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Energy distribution
    axes[1, 1].hist(df_bursts['energy_per_spike'], bins=20, color='coral', 
                   edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Energy per Spike (a.u.)', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Energy Efficiency Distribution', fontsize=13, fontweight='bold')
    axes[1, 1].axvline(optimal_burst['energy_per_spike'], color='red', 
                      linestyle='--', linewidth=2, label='Optimal')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Burst Pattern Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/figures/novel_analysis/12_burst_analysis.png', 
                dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/novel_analysis/12_burst_analysis.png")
    
    plt.show()

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/figures/novel_analysis', exist_ok=True)
    main()