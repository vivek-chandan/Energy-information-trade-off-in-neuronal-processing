
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
    
    I_inj = 8.0
    t, solution = neuron.simulate(I_inj=I_inj, t_max=5000, dt=0.01)
    print(f"   Simulation complete: {len(t)} time points")
    
    scales = [10, 25, 50, 100, 250, 500, 1000]
    multi_scale_results = analyzer.analyze_multiple_scales(t, solution, scales)
    
    print("\n4. Multi-scale efficiency results:")
    print("   " + "-" * 50)
    print(f"   {'Scale (ms)':<12} {'Mean Eff.':<12} {'Std Eff.':<12} {'CV':<12}")
    print("   " + "-" * 50)
    
    for scale in scales:
        result = multi_scale_results['scale_results'][scale]
        print(f"   {scale:<12} {result['mean_efficiency']:<12.4f} "
              f"{result['std_efficiency']:<12.4f} {result['cv_efficiency']:<12.4f}")
    
    print("   " + "-" * 50)
    print(f"   Scale Dependence Factor: {multi_scale_results['scale_dependence_factor']:.4f}")
    
    summary_data = []
    for scale in scales:
        result = multi_scale_results['scale_results'][scale]
        summary_data.append({
            'scale_ms': scale,
            'mean_efficiency': result['mean_efficiency'],
            'std_efficiency': result['std_efficiency'],
            'cv_efficiency': result['cv_efficiency'],
            'n_windows': result['n_windows']
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv('data/processed/multi_scale_results.csv', index=False)
    print("   Saved: data/processed/multi_scale_results.csv")
    
    fig1 = plotter.plot_multi_scale_efficiency(multi_scale_results)
    plt.savefig('data/figures/novel_analysis/09_multi_scale_efficiency.png', 
                dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/novel_analysis/09_multi_scale_efficiency.png")
    
    fig2 = plotter.plot_scale_heatmap(multi_scale_results)
    plt.savefig('data/figures/novel_analysis/10_scale_heatmap.png', 
                dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/novel_analysis/10_scale_heatmap.png")
    
    mean_effs = [multi_scale_results['scale_results'][s]['mean_efficiency'] 
                 for s in scales]
    valid_effs = [(s, e) for s, e in zip(scales, mean_effs) if not np.isnan(e)]
    
    if valid_effs:
        optimal_scale, optimal_eff = min(valid_effs, key=lambda x: x[1])
        print(f"   ★ Optimal temporal scale: {optimal_scale} ms")
        print(f"   ★ Minimum energy per spike: {optimal_eff:.4f} a.u.")
        print(f"   ★ This suggests scale-dependent efficiency!")
    
    
    plt.show()

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/figures/novel_analysis', exist_ok=True)
    main()