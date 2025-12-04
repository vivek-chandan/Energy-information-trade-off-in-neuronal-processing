import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bursting_neuron import BurstingNeuron
from src.analysis.multi_scale_analyzer import MultiScaleAnalyzer
from visualization.advanced_plots import NovelPlotter

def main():
    
    neuron = BurstingNeuron() 
    analyzer = MultiScaleAnalyzer(neuron)
    plotter = NovelPlotter()
    
    I_values = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    
    results_list = []
    labels = []
    
    for I_inj in I_values:
        print(f"   Analyzing I_inj = {I_inj} μA/cm²...", end='')
        
        t, solution = neuron.simulate(I_inj=I_inj, t_max=5000, dt=0.01)
        
        results = analyzer.full_analysis(t, solution)
        
        n_spikes = results['spike_statistics']['n_spikes']
        if n_spikes > 10:
            results_list.append(results)
            labels.append(f"I={I_inj:.1f}")
            
            ei = results['energy_information_tradeoff']
            print(f" OK ({n_spikes} spikes)")
            print(f"      Entropy: {ei['isi_entropy']:.3f} bits | E/bit: {ei['energy_per_bit']:.2f}")
        else:
            print(" SKIPPED (Too few spikes/Depolarization Block)")
    
    if not results_list:
        print("\n❌ CRITICAL: No valid bursting data found in this range.")
        print("   Try adjusting I_values lower or checking model parameters.")
        return

    
    trade_off_data = []
    for i, result in enumerate(results_list):
        ei = result['energy_information_tradeoff']
        trade_off_data.append({
            'I_inj': I_values[i], 
            'firing_rate': result['spike_statistics']['firing_rate'],
            'total_energy': ei['total_energy'],
            'isi_entropy': ei['isi_entropy'],
            'information_rate': ei['information_rate'],
            'energy_per_bit': ei['energy_per_bit'],
            'n_spikes': ei['n_spikes']
        })
    
    df_tradeoff = pd.DataFrame(trade_off_data)
    
    df_tradeoff.to_csv('data/processed/energy_information_tradeoff.csv', index=False)
    print("   Saved: data/processed/energy_information_tradeoff.csv")
    
    print("\n5. Finding optimal energy-information trade-off...")
    valid_idx = df_tradeoff['energy_per_bit'].notna()
    if valid_idx.any():
        optimal_idx = df_tradeoff.loc[valid_idx, 'energy_per_bit'].idxmin()
        optimal = df_tradeoff.iloc[optimal_idx]
        
        print(f"   ★ Optimal I_inj: {optimal['I_inj']:.2f} μA/cm²")
        print(f"   ★ Energy per bit: {optimal['energy_per_bit']:.4f} a.u.")
    
    fig1 = plotter.plot_energy_information_tradeoff(results_list, labels)
    plt.savefig('data/figures/novel_analysis/13_energy_information_tradeoff.png', 
                dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/novel_analysis/13_energy_information_tradeoff.png")
    
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].scatter(df_tradeoff['isi_entropy'], df_tradeoff['total_energy'], 
                      s=100, c=df_tradeoff['firing_rate'], cmap='viridis',
                      edgecolors='black')
    axes[0, 0].set_xlabel('ISI Entropy (bits)')
    axes[0, 0].set_ylabel('Total Energy (a.u.)')
    axes[0, 0].set_title('Energy vs Entropy')
    
    # B. Info Rate
    axes[0, 1].plot(df_tradeoff['firing_rate'], df_tradeoff['information_rate'], 'o-')
    axes[0, 1].set_xlabel('Firing Rate (Hz)')
    axes[0, 1].set_ylabel('Bit Rate (bits/s)')
    axes[0, 1].set_title('Information Throughput')
    
    valid_plot_data = df_tradeoff.dropna(subset=['energy_per_bit'])
    
    if not valid_plot_data.empty:
        axes[1, 0].plot(valid_plot_data['I_inj'], valid_plot_data['energy_per_bit'], 'ro-', linewidth=2)
        axes[1, 0].set_xlabel('Input Current (μA/cm²)')
        axes[1, 0].set_ylabel('Energy per Bit (a.u.)')
        axes[1, 0].set_title('Thermodynamic Efficiency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # D. Dual View
        x_indices = range(len(valid_plot_data))
        axes[1, 1].bar(x_indices, valid_plot_data['isi_entropy'], alpha=0.5, color='blue', label='Entropy')
        ax_twin = axes[1, 1].twinx()
        ax_twin.plot(x_indices, valid_plot_data['energy_per_bit'], 'ro-', label='Cost/Bit')
        
        axes[1, 1].set_xticks(x_indices)
        axes[1, 1].set_xticklabels(valid_plot_data['I_inj'].astype(str))
        axes[1, 1].set_xlabel('Input Current')
        axes[1, 1].set_title('Entropy vs Cost')
    
    plt.tight_layout()
    plt.savefig('data/figures/novel_analysis/14_detailed_tradeoff.png', dpi=150)
    print("   Saved: data/figures/novel_analysis/14_detailed_tradeoff.png")
    

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/figures/novel_analysis', exist_ok=True)
    main()