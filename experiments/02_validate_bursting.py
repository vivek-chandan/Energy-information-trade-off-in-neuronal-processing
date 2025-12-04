
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.models.bursting_neuron import BurstingNeuron
from src.analysis.spike_detection import SpikeDetector
from src.analysis.burst_analyzer import BurstAnalyzer
from src.visualization.basic_plots import BasicPlotter

def main():

    g_NaP     = 0.7       
    g_K_slow  = 8.0       
    tau_s     = 1000.0   

    I_inj = 0.0 

    neuron = BurstingNeuron(g_NaP=g_NaP, g_K_slow=g_K_slow, tau_s=tau_s)
    
    print(f"   Parameters: g_NaP={g_NaP}, g_K_slow={g_K_slow}, tau_s={tau_s}")
    print(f"   Input Current: {I_inj} uA/cm2")
    
    print("\n2. Simulating...")
    t, solution = neuron.simulate(I_inj=I_inj, t_max=3000, dt=0.01)
    
    spike_detector = SpikeDetector()
    burst_analyzer = BurstAnalyzer()
    
    spike_times, _ = spike_detector.detect_spikes(t, solution[:, 0])
    bursts = burst_analyzer.detect_bursts(spike_times)
    burst_stats = burst_analyzer.get_burst_statistics(spike_times)
    
    
    if burst_stats['n_bursts'] > 2:
        print(f"   Mean spikes/burst: {burst_stats['mean_spikes_per_burst']:.2f}")
        print("   ✓ SUCCESS: Clear rhythmic bursting detected!")
    else:
        print("   ✗ WARNING: Bursting is still weak.")

    print("\n3. Generating plot...")
    plotter = BasicPlotter()
    fig1 = plotter.plot_summary(t, solution, spike_times, burst_stats['bursts'])
    plt.savefig('data/figures/validation/04_bursting_pattern.png', dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/validation/04_bursting_pattern.png")
    
    
    I_values = [-0.8, -0.2, 0.0, 0.5, 1.2]

    fig2, axes = plt.subplots(len(I_values), 1, figsize=(12, 10), sharex=True)

    for idx, I_val in enumerate(I_values):
        neuron_sweep = BurstingNeuron(
            g_NaP=0.8,      
            g_K_slow=7.0,   
            tau_s=1200.0   
        )
        
        t, solution = neuron_sweep.simulate(I_inj=I_val, t_max=5000, dt=0.025)
        
        spike_detector = SpikeDetector()
        burst_analyzer = BurstAnalyzer()
        spikes, _ = spike_detector.detect_spikes(t, solution[:, 0])
        bursts = burst_analyzer.detect_bursts(spikes)
        n_bursts = len(bursts)
        
        axes[idx].plot(t, solution[:, 0], 'k-', linewidth=0.9)
        axes[idx].set_ylabel('V (mV)')
        axes[idx].set_title(f'I = {I_val:+.1f} μA/cm²   |   {n_bursts} bursts')
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)')
    axes[-1].set_xlim(0, 5000)
    plt.tight_layout()
    plt.savefig('data/figures/validation/05_current_dependence_robust.png', dpi=150)
    print("   Saved: data/figures/validation/05_current_dependence_robust.png")

if __name__ == "__main__":
    os.makedirs('data/figures/validation', exist_ok=True)
    main()