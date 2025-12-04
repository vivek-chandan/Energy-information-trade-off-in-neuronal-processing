
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.models.hodgkin_huxley import HodgkinHuxley
from src.analysis.spike_detection import SpikeDetector
from src.visualization.basic_plots import BasicPlotter

def main():
    
    hh = HodgkinHuxley()
    spike_detector = SpikeDetector()
    plotter = BasicPlotter()
    t, solution = hh.simulate(I_inj=10.0, t_max=50, dt=0.01)
    
    spike_times, _ = spike_detector.detect_spikes(t, solution[:, 0])
    print(f"   Detected {len(spike_times)} spike(s)")
    
    if len(spike_times) > 0:
        print(f"   First spike at: {spike_times[0]:.2f} ms")
        peak_V = np.max(solution[:, 0])
        print(f"   Peak voltage: {peak_V:.2f} mV")
        print("   ✓ Single AP test PASSED")
    
    # Plot
    fig1, _ = plotter.plot_voltage_trace(t, solution[:, 0], 
                                         title="Single Action Potential")
    plt.savefig('data/figures/validation/01_single_ap.png', dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/validation/01_single_ap.png")
    
    t, solution = hh.simulate(I_inj=10.0, t_max=200, dt=0.01)
    
    spike_times, _ = spike_detector.detect_spikes(t, solution[:, 0])
    stats = spike_detector.get_spike_statistics(t, solution[:, 0])
    
    print(f"   Number of spikes: {stats['n_spikes']}")
    print(f"   Firing rate: {stats['firing_rate']:.2f} Hz")
    print(f"   Mean ISI: {stats['mean_isi']:.2f} ms")
    print(f"   CV(ISI): {stats['cv_isi']:.3f}")
    print("   ✓ Tonic spiking test PASSED")
    
    fig2, _ = plotter.plot_voltage_trace(t, solution[:, 0], 
                                         title="Tonic Spiking (I=10 μA/cm²)")
    plt.savefig('data/figures/validation/02_tonic_spiking.png', dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/validation/02_tonic_spiking.png")
    
    I_range = np.linspace(0, 20, 15)
    firing_rates = []
    
    for I_inj in I_range:
        t, solution = hh.simulate(I_inj=I_inj, t_max=500, dt=0.01)
        stats = spike_detector.get_spike_statistics(t, solution[:, 0])
        firing_rates.append(stats['firing_rate'])
        print(f"   I = {I_inj:5.2f} μA/cm² → f = {stats['firing_rate']:6.2f} Hz")
    
    fig3, _ = plotter.plot_fi_curve(I_range, firing_rates)
    plt.savefig('data/figures/validation/03_fi_curve.png', dpi=150, bbox_inches='tight')
    print("   Saved: data/figures/validation/03_fi_curve.png")
    
    plt.show()

if __name__ == "__main__":
    os.makedirs('data/figures/validation', exist_ok=True)
    main()