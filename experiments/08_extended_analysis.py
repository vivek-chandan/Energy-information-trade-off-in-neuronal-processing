import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bursting_neuron import BurstingNeuron
from src.models.calcium_bursting import CalciumBurstingNeuron
from src.models.stochastic_neuron import StochasticBurstingNeuron
from src.analysis.burst_analyzer import BurstAnalyzer
from src.analysis.information_theory import InformationAnalyzer
from src.analysis.metabolic_cost import MetabolicAnalyzer
from src.analysis.spike_detection import SpikeDetector
from src.analysis.multi_scale_analyzer import MultiScaleAnalyzer
from src.analysis.statistical_validation import StatisticalValidator
from src.analysis.literature_comparision import LiteratureComparator
from src.optimization.parameter_optimizer import ParameterOptimizer
from visualization.advanced_plots import NovelPlotter

def test_stochastic_robustness():
    
    detector = SpikeDetector()
    
    det_model = BurstingNeuron(tau_s=200.0) 
    t_det, sol_det = det_model.simulate(I_inj=8.0, t_max=1000)
    spikes_det, _ = detector.detect_spikes(t_det, sol_det[:, 0])
    
    stoch_results = []
    stoch_spikes = []
    
    for trial in range(3):
        stoch_model = StochasticBurstingNeuron(
            n_channels_Na=1000, n_channels_K=300, noise_strength=0.02, tau_s=200.0
        )
        t_s, sol_s = stoch_model.simulate(I_inj=8.0, t_max=1000, seed=trial)
        stoch_results.append((t_s, sol_s))
        
        spikes, _ = detector.detect_spikes(t_s, sol_s[:, 0])
        stoch_spikes.append(len(spikes))
        print(f"  Trial {trial+1}: {len(spikes)} spikes")

    mean_s = np.mean(stoch_spikes)
    std_s = np.std(stoch_spikes)
    print(f"  Robustness Result: {mean_s:.1f} ± {std_s:.1f} spikes")
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(t_det, sol_det[:, 0], 'k-', linewidth=1)
    axes[0].set_title(f'Deterministic (Baseline)', fontweight='bold')
    
    for i, (t_s, sol_s) in enumerate(stoch_results):
        axes[i+1].plot(t_s, sol_s[:, 0], linewidth=1, alpha=0.8)
        axes[i+1].set_title(f'Stochastic Trial {i+1}')
    
    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig('data/figures/stochastic_comparison.png', dpi=150)
    print("  Saved: data/figures/stochastic_comparison.png")

def test_mechanisms_and_calibration():
    
    detector = SpikeDetector()
    burst_analyzer = BurstAnalyzer()
    info_analyzer = InformationAnalyzer()
    comparator = LiteratureComparator()
    plotter = NovelPlotter()
    
    na_model = BurstingNeuron(tau_s=200.0) 
    ca_model = CalciumBurstingNeuron(g_Ca=1.0, g_K_Ca=2.0)
    
    I_test = 8.0
    t_na, sol_na = na_model.simulate(I_inj=I_test, t_max=2000)
    t_ca, sol_ca = ca_model.simulate(I_inj=I_test, t_max=2000)
    
    spikes_na, _ = detector.detect_spikes(t_na, sol_na[:, 0])
    analyzer_na = MetabolicAnalyzer(na_model)
    ms_analyzer_na = MultiScaleAnalyzer(na_model)
    
    stats_na = analyzer_na.get_energy_statistics(t_na, sol_na, spikes_na)
    bursts_na = burst_analyzer.detect_bursts(spikes_na)
    info_na = info_analyzer.get_information_statistics(spikes_na, bursts_na)
    ms_na = ms_analyzer_na.analyze_multiple_scales(t_na, sol_na, scales=[10, 100, 1000])

    spikes_ca, _ = detector.detect_spikes(t_ca, sol_ca[:, 0])
    analyzer_ca = MetabolicAnalyzer(ca_model)
    ms_analyzer_ca = MultiScaleAnalyzer(ca_model)
    
    stats_ca = analyzer_ca.get_energy_statistics(t_ca, sol_ca, spikes_ca)
    bursts_ca = burst_analyzer.detect_bursts(spikes_ca)
    info_ca = info_analyzer.get_information_statistics(spikes_ca, bursts_ca)
    ms_ca = ms_analyzer_ca.analyze_multiple_scales(t_ca, sol_ca, scales=[10, 100, 1000])
    
    atp_factor = comparator.calibrate_units(stats_na['energy_per_spike'])
    cost_na_atp = stats_na['energy_per_spike'] * atp_factor
    cost_ca_atp = stats_ca['energy_per_spike'] * atp_factor
    
    print(f"\nResults:")
    print(f"  Na+ Cost: {cost_na_atp:.2e} ATP/spike")
    print(f"  Ca2+ Cost: {cost_ca_atp:.2e} ATP/spike")
    print(f"  Result: Ca2+ is {((stats_ca['energy_per_spike'] - stats_na['energy_per_spike'])/stats_na['energy_per_spike'])*100:.1f}% more expensive.")

    burst_data_report = [{'intraburst_freq': b['intraburst_freq'], 'n_spikes': b['n_spikes']} for b in bursts_na]
    validation_data = {
        'burst_patterns': burst_data_report,
        'energy_information': info_na,
        'energy_statistics': stats_na
    }
    report = comparator.generate_report(validation_data, conversion_factor=atp_factor)
    with open('data/processed/final_validation_report.txt', 'w') as f:
        f.write(report)
    
    results_dict = {
        'Na-Driven': {
            'spike_statistics': {'firing_rate': stats_na['n_spikes']/2.0},
            'energy_statistics': {'energy_per_spike': stats_na['energy_per_spike']},
            'burst_statistics': {'n_bursts': len(bursts_na)},
            'energy_information_tradeoff': {'isi_entropy': info_na['isi_entropy']},
            'multi_scale_analysis': ms_na
        },
        'Ca-Driven': {
            'spike_statistics': {'firing_rate': stats_ca['n_spikes']/2.0},
            'energy_statistics': {'energy_per_spike': stats_ca['energy_per_spike']},
            'burst_statistics': {'n_bursts': len(bursts_ca)},
            'energy_information_tradeoff': {'isi_entropy': info_ca['isi_entropy']},
            'multi_scale_analysis': ms_ca
        }
    }
    
    fig_comp = plotter.plot_comparative_analysis(results_dict)
    fig_comp.savefig('data/figures/final_radar_comparison.png', dpi=150)
    print("  Saved: data/figures/final_radar_comparison.png")

def test_optimization():
    
    optimizer = ParameterOptimizer()
    
    bounds = [(0.2, 1.0), (1.0, 4.0), (50.0, 500.0)]
    
    result = optimizer.optimize_bursting(bounds=bounds, maxiter=10)
    
    optimizer.compare_baseline_vs_optimal(
        optimal_params=result.x,
        baseline_params=[0.5, 2.0, 200.0]
    )

if __name__ == "__main__":
    os.makedirs('data/figures', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    test_stochastic_robustness()
    test_mechanisms_and_calibration()
    test_optimization()