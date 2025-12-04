

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

class NovelPlotter:
    
    
    def __init__(self):

        sns.set_style("whitegrid")
        self.colors = sns.color_palette("husl", 8)
    
    def plot_multi_scale_efficiency(self, multi_scale_results, figsize=(12, 6)):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        scales = multi_scale_results['scales']
        scale_results = multi_scale_results['scale_results']
        
        mean_effs = [scale_results[s]['mean_efficiency'] for s in scales]
        std_effs = [scale_results[s]['std_efficiency'] for s in scales]
        
        ax1.errorbar(scales, mean_effs, yerr=std_effs, 
                    fmt='o-', linewidth=2, markersize=8, capsize=5,
                    color=self.colors[0])
        ax1.set_xlabel('Temporal Scale (ms)', fontsize=12)
        ax1.set_ylabel('Mean Energy per Spike (a.u.)', fontsize=12)
        ax1.set_title('Multi-Scale Energy Efficiency', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        if not all(np.isnan(mean_effs)):
            valid_effs = [(s, e) for s, e in zip(scales, mean_effs) if not np.isnan(e)]
            if valid_effs:
                optimal_scale, optimal_eff = min(valid_effs, key=lambda x: x[1])
                ax1.axvline(optimal_scale, color='red', linestyle='--', 
                           linewidth=2, alpha=0.7, label=f'Optimal: {optimal_scale} ms')
                ax1.legend(fontsize=10)
        
        cv_effs = [scale_results[s]['cv_efficiency'] for s in scales]
        ax2.plot(scales, cv_effs, 'o-', linewidth=2, markersize=8, 
                color=self.colors[1])
        ax2.set_xlabel('Temporal Scale (ms)', fontsize=12)
        ax2.set_ylabel('CV of Efficiency', fontsize=12)
        ax2.set_title('Efficiency Variability', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_burst_pattern_space(self, bursts, burst_efficiencies, figsize=(10, 8)):
        
        if len(bursts) == 0:
            print("No bursts to plot")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract burst features
        n_spikes = [b['n_spikes'] for b in bursts]
        frequencies = [b['intraburst_freq'] for b in bursts]
        energies = [be['energy_per_spike'] for be in burst_efficiencies]
        
        # Create scatter plot
        scatter = ax.scatter(n_spikes, frequencies, c=energies, 
                            s=100, cmap='RdYlGn_r', edgecolors='black', 
                            linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Spikes per Burst', fontsize=12)
        ax.set_ylabel('Intraburst Frequency (Hz)', fontsize=12)
        ax.set_title('Burst Pattern Space\n(colored by energy efficiency)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Energy per Spike (a.u.)', fontsize=11)
        
        # Mark optimal burst
        if energies:
            optimal_idx = np.argmin(energies)
            ax.scatter(n_spikes[optimal_idx], frequencies[optimal_idx], 
                      s=300, marker='*', color='gold', edgecolors='black', 
                      linewidth=2, label='Most Efficient', zorder=5)
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_energy_information_tradeoff(self, results_list, labels=None, figsize=(10, 8)):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        if labels is None:
            labels = [f'Condition {i+1}' for i in range(len(results_list))]
        
        # Extract data
        energies = [r['energy_information_tradeoff']['total_energy'] for r in results_list]
        entropies = [r['energy_information_tradeoff']['isi_entropy'] for r in results_list]
        info_rates = [r['energy_information_tradeoff']['information_rate'] for r in results_list]
        energy_per_bit = [r['energy_information_tradeoff']['energy_per_bit'] for r in results_list]
        
        # Plot 1: Energy vs Information
        ax1.scatter(entropies, energies, s=100, c=self.colors[:len(results_list)], 
                   edgecolors='black', linewidth=1.5, alpha=0.7)
        
        for i, label in enumerate(labels):
            ax1.annotate(label, (entropies[i], energies[i]), 
                        fontsize=9, ha='right', va='bottom')
        
        ax1.set_xlabel('ISI Entropy (bits)', fontsize=12)
        ax1.set_ylabel('Total Energy (a.u.)', fontsize=12)
        ax1.set_title('Energy-Information Trade-off', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy per bit
        valid_idx = [i for i, e in enumerate(energy_per_bit) if not np.isnan(e)]
        if valid_idx:
            ax2.bar(range(len(valid_idx)), 
                   [energy_per_bit[i] for i in valid_idx],
                   color=[self.colors[i] for i in valid_idx],
                   edgecolor='black', alpha=0.7)
            ax2.set_xticks(range(len(valid_idx)))
            ax2.set_xticklabels([labels[i] for i in valid_idx], rotation=45, ha='right')
            ax2.set_ylabel('Energy per Bit (a.u.)', fontsize=12)
            ax2.set_title('Information Efficiency', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_scale_heatmap(self, multi_scale_results, figsize=(12, 8)):
        
        scales = multi_scale_results['scales']
        scale_results = multi_scale_results['scale_results']
        
        max_windows = max([len(scale_results[s]['windows']) for s in scales])
        efficiency_matrix = np.full((len(scales), max_windows), np.nan)
        
        for i, scale in enumerate(scales):
            windows = scale_results[scale]['windows']
            for j, window in enumerate(windows):
                if not np.isnan(window['efficiency']):
                    efficiency_matrix[i, j] = window['efficiency']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(efficiency_matrix, aspect='auto', cmap='RdYlGn_r', 
                      interpolation='nearest')
        
        ax.set_yticks(range(len(scales)))
        ax.set_yticklabels([f'{s} ms' for s in scales])
        ax.set_ylabel('Temporal Scale', fontsize=12)
        ax.set_xlabel('Time Window Index', fontsize=12)
        ax.set_title('Multi-Scale Efficiency Landscape', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Energy per Spike (a.u.)', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def plot_comparative_analysis(self, results_dict, figsize=(14, 10)):
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        conditions = list(results_dict.keys())
        n_conditions = len(conditions)
        
        # Extract metrics
        firing_rates = [results_dict[c]['spike_statistics']['firing_rate'] for c in conditions]
        energy_per_spike = [results_dict[c]['energy_statistics']['energy_per_spike'] for c in conditions]
        n_bursts = [results_dict[c]['burst_statistics']['n_bursts'] for c in conditions]
        isi_entropy = [results_dict[c]['energy_information_tradeoff']['isi_entropy'] for c in conditions]
        
        # Plot 1: Firing rates
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(range(n_conditions), firing_rates, color=self.colors[:n_conditions], 
               edgecolor='black', alpha=0.7)
        ax1.set_xticks(range(n_conditions))
        ax1.set_xticklabels(conditions, rotation=45, ha='right')
        ax1.set_ylabel('Firing Rate (Hz)', fontsize=11)
        ax1.set_title('Activity Level', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Energy efficiency
        ax2 = fig.add_subplot(gs[0, 1])
        valid_energy = [e if not np.isnan(e) else 0 for e in energy_per_spike]
        ax2.bar(range(n_conditions), valid_energy, color=self.colors[:n_conditions], 
               edgecolor='black', alpha=0.7)
        ax2.set_xticks(range(n_conditions))
        ax2.set_xticklabels(conditions, rotation=45, ha='right')
        ax2.set_ylabel('Energy per Spike (a.u.)', fontsize=11)
        ax2.set_title('Metabolic Efficiency', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Burst count
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(range(n_conditions), n_bursts, color=self.colors[:n_conditions], 
               edgecolor='black', alpha=0.7)
        ax3.set_xticks(range(n_conditions))
        ax3.set_xticklabels(conditions, rotation=45, ha='right')
        ax3.set_ylabel('Number of Bursts', fontsize=11)
        ax3.set_title('Bursting Activity', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Information content
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.bar(range(n_conditions), isi_entropy, color=self.colors[:n_conditions], 
               edgecolor='black', alpha=0.7)
        ax4.set_xticks(range(n_conditions))
        ax4.set_xticklabels(conditions, rotation=45, ha='right')
        ax4.set_ylabel('ISI Entropy (bits)', fontsize=11)
        ax4.set_title('Information Content', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Multi-scale efficiency variability
        ax5 = fig.add_subplot(gs[1, 1])
        scale_deps = [results_dict[c]['multi_scale_analysis']['scale_dependence_factor'] 
                     for c in conditions]
        valid_deps = [d if not np.isnan(d) else 0 for d in scale_deps]
        ax5.bar(range(n_conditions), valid_deps, color=self.colors[:n_conditions], 
               edgecolor='black', alpha=0.7)
        ax5.set_xticks(range(n_conditions))
        ax5.set_xticklabels(conditions, rotation=45, ha='right')
        ax5.set_ylabel('Scale Dependence', fontsize=11)
        ax5.set_title('Multi-Scale Variability', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Summary radar chart
        ax6 = fig.add_subplot(gs[1, 2], projection='polar')
        
        # Normalize metrics for radar chart
        def normalize(values):
            min_val, max_val = min(values), max(values)
            if max_val - min_val == 0:
                return [0.5] * len(values)
            return [(v - min_val) / (max_val - min_val) for v in values]
        
        metrics = ['Firing\nRate', 'Energy\nEfficiency', 'Bursting', 
                  'Information', 'Scale\nVariability']
        
        for i, condition in enumerate(conditions):
            values = [
                firing_rates[i],
                1.0 / (valid_energy[i] + 0.1),
                n_bursts[i],
                isi_entropy[i],
                1.0 / (valid_deps[i] + 0.1)  
            ]
            
            if i == 0:
                all_values = [values]
            else:
                all_values.append(values)
        
        normalized = []
        for metric_idx in range(len(metrics)):
            metric_values = [all_values[cond_idx][metric_idx] for cond_idx in range(n_conditions)]
            norm_values = normalize(metric_values)
            normalized.append(norm_values)
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, condition in enumerate(conditions):
            values = [normalized[j][i] for j in range(len(metrics))]
            values += values[:1]
            ax6.plot(angles, values, 'o-', linewidth=2, label=condition, 
                    color=self.colors[i])
            ax6.fill(angles, values, alpha=0.15, color=self.colors[i])
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics, fontsize=9)
        ax6.set_ylim(0, 1)
        ax6.set_title('Overall Comparison', fontsize=12, fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax6.grid(True)
        
        plt.suptitle('Comparative Analysis Across Conditions', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig