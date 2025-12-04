"""
Comparison of simulation results with published literature.
Combines biological sanity checks with metabolic unit calibration.

References:
1. Alle et al. (2009). Science.
2. Attwell & Laughlin (2001). J Cerebral Blood Flow & Metab.
3. Wang (1999). Neuroscience.
4. Borst & Theunissen (1999). Nature Neuroscience.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LiteratureComparator:
    """
    Validate findings against experimental literature and calibrate units.
    """
    
    def __init__(self):
        self.literature_data = {
            'alle_2009': {
                'source': 'Alle et al. (2009)',
                'Na_ions_per_AP': 3.5e7, 
                'description': 'Hippocampal mossy fiber APs'
            },
            'attwell_2001': {
                'source': 'Attwell & Laughlin (2001)',
                'ATP_per_AP': 2.4e9,   
                'description': 'Rodent cortical neuron'
            },
            'wang_1999': {
                'source': 'Wang (1999)',
                'burst_freq_range': (100, 300), 
                'spikes_per_burst_range': (2, 10),
                'description': 'Neocortical chattering neurons'
            },
            'borst_1999': {
                'source': 'Borst & Theunissen (1999)',
                'isi_entropy_range': (1.0, 3.5),
                'description': 'Information in spike trains'
            }
        }

    def calibrate_units(self, model_energy_per_spike):

        ref_value = self.literature_data['attwell_2001']['ATP_per_AP']
        
        # Avoid division by zero
        if model_energy_per_spike <= 0:
            return 1.0
            
        conversion_factor = ref_value / model_energy_per_spike
        return conversion_factor

    def compare_efficiency_curve(self, ax, model_rates, model_energies, conversion_factor=1.0):
        """
        Overlay biological efficiency constraints on F-I energy plot.
        """
        # Convert a.u. to ATP
        real_energies = np.array(model_energies) * conversion_factor
        
        # Plot Model Data
        ax.plot(model_rates, real_energies, 'b-o', linewidth=2, label='Simulation Results')
        if len(model_rates) > 0:
            ref_x = np.linspace(min(model_rates), max(model_rates), 100)
            ref_y = real_energies[0] * (ref_x / ref_x[0])**0.8 
            ax.plot(ref_x, ref_y, 'k--', alpha=0.6, label='Theoretical Power Law')

        ax.set_ylabel("Metabolic Cost (ATP molecules)")
        ax.set_xlabel("Firing Rate (Hz)")
        ax.legend()
        return ax

    def compare_burst_patterns(self, your_burst_data):
        """
        Compare burst patterns to Wang (1999).
        """
        df = pd.DataFrame(your_burst_data)
        if df.empty:
            return "No burst data to compare."

        your_freq_mean = df['intraburst_freq'].mean()
        your_spikes_mean = df['n_spikes'].mean()
        
        lit = self.literature_data['wang_1999']
        freq_range = lit['burst_freq_range']
        
        # Check logic
        freq_match = freq_range[0] <= your_freq_mean <= freq_range[1]
        
        comparison = {
            'metric': 'Burst Frequency',
            'your_value': your_freq_mean,
            'lit_range': freq_range,
            'match': freq_match,
            'source': lit['source']
        }
        return comparison

    def compare_information_content(self, your_info_data):
        """
        Compare ISI entropy to Borst & Theunissen (1999).
        """
        # Handle case where input is a dict (single result) or list
        if isinstance(your_info_data, dict):
            your_entropy = your_info_data.get('isi_entropy', 0)
        else:
            df = pd.DataFrame(your_info_data)
            your_entropy = df['isi_entropy'].mean()
            
        lit = self.literature_data['borst_1999']
        r = lit['isi_entropy_range']
        
        match = r[0] <= your_entropy <= r[1]
        
        comparison = {
            'metric': 'ISI Entropy',
            'your_value': your_entropy,
            'lit_range': r,
            'match': match,
            'source': lit['source']
        }
        return comparison

    def convert_au_to_ions(self, energy_au, conversion_factor):
        """
        Convert arbitrary energy units to estimated Na+ ion count.
        """
        # 1 ATP moves 3 Na+ ions
        atp_count = energy_au * conversion_factor
        na_ions = atp_count * 3.0
        return na_ions

    def generate_report(self, results, conversion_factor=1.0):
        """
        Generate comprehensive text report.
        """
        report = "=" * 60 + "\n"
        report += "LITERATURE VALIDATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # 1. Burst Comparison
        if 'burst_patterns' in results and len(results['burst_patterns']) > 0:
            comp = self.compare_burst_patterns(results['burst_patterns'])
            report += f"1. {comp['metric'].upper()}\n"
            report += f"   Model: {comp['your_value']:.1f} Hz\n"
            report += f"   Lit:   {comp['lit_range'][0]}-{comp['lit_range'][1]} Hz ({comp['source']})\n"
            report += f"   Valid: {'YES' if comp['match'] else 'NO (Check tau_s)'}\n\n"

        # 2. Information Comparison
        if 'energy_information' in results:
            comp = self.compare_information_content(results['energy_information'])
            report += f"2. {comp['metric'].upper()}\n"
            report += f"   Model: {comp['your_value']:.2f} bits\n"
            report += f"   Lit:   {comp['lit_range'][0]}-{comp['lit_range'][1]} bits ({comp['source']})\n"
            report += f"   Valid: {'YES' if comp['match'] else 'NO'}\n\n"

        # 3. Metabolic Calibration
        if 'energy_statistics' in results:
             # Get energy per spike
            e_spike = results['energy_statistics']['energy_per_spike']
            na_ions = self.convert_au_to_ions(e_spike, conversion_factor)
            lit_na = self.literature_data['alle_2009']['Na_ions_per_AP']
            
            report += "3. METABOLIC CALIBRATION\n"
            report += f"   Model Cost: {e_spike:.2f} a.u.\n"
            report += f"   Calibrated: {na_ions:.2e} Na+ ions/spike\n"
            report += f"   Reference:  {lit_na:.2e} Na+ ions/spike (Alle et al. 2009)\n"
            
        return report