import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bursting_neuron import BurstingNeuron
from src.analysis.sensitivity_analysis import SensitivityAnalyzer

def main():

    base_model = BurstingNeuron(tau_s=200.0)
    sensitivity = SensitivityAnalyzer(base_model)
    
    sensitivity.parameter_sweep_1d(
        param_name='g_NaP', 
        param_range=[0.3, 0.4, 0.5, 0.6, 0.7], 
        metric='energy_per_spike'
    )
    fig1 = sensitivity.plot_sensitivity('g_NaP')
    fig1.savefig('data/figures/sensitivity_gNaP.png', dpi=150)
    print("   Saved: sensitivity_gNaP.png")
    
    sensitivity.parameter_sweep_1d(
        param_name='tau_s', 
        param_range=[100, 200, 300, 400, 500], 
        metric='energy_per_spike'
    )
    fig2 = sensitivity.plot_sensitivity('tau_s')
    fig2.savefig('data/figures/sensitivity_taus.png', dpi=150)
    print("   Saved: sensitivity_taus.png")

    # See how g_NaP and g_K_slow interact
    sensitivity.parameter_sweep_2d(
        param1_name='g_NaP', 
        param1_range=[0.3, 0.4, 0.5, 0.6], 
        param2_name='g_K_slow', 
        param2_range=[1.5, 2.0, 2.5, 3.0]
    )
    fig3 = sensitivity.plot_2d_sensitivity('g_NaP', 'g_K_slow')
    fig3.savefig('data/figures/sensitivity_2d_landscape.png', dpi=150)
    print("   Saved: sensitivity_2d_landscape.png")

    
    # Vary all parameters by +/- 10% randomly
    df_monte_carlo = sensitivity.monte_carlo_robustness(
        n_samples=50, 
        noise_level=0.1
    )
    
    df_monte_carlo.to_csv('data/processed/monte_carlo_results.csv', index=False)
    print("   Saved: monte_carlo_results.csv")
    plt.show()

if __name__ == "__main__":
    os.makedirs('data/figures', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    main()