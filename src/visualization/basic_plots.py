
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BasicPlotter:
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):

        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_voltage_trace(self, t, V, title="Membrane Potential", figsize=(12, 4)):
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(t, V, 'k-', linewidth=1)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Membrane Potential (mV)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
        
        plt.tight_layout()
        return fig, ax
    
    def plot_currents(self, t, currents, title="Ion Currents", figsize=(12, 6)):
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for name, current in currents.items():
            if name != 't':
                ax.plot(t, current, label=name, linewidth=1.5)
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Current (μA/cm²)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_spike_raster(self, spike_times, bursts=None, figsize=(12, 3)):
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot spikes
        ax.eventplot([spike_times], colors='black', linewidths=1.5)
        
        if bursts is not None:
            for burst in bursts:
                ax.axvspan(burst['start_time'], burst['end_time'], 
                          alpha=0.3, color='red', label='Burst' if burst == bursts[0] else '')
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_yticks([])
        ax.set_title('Spike Raster', fontsize=14, fontweight='bold')
        if bursts:
            ax.legend()
        
        plt.tight_layout()
        return fig, ax
    
    def plot_fi_curve(self, I_inj_values, firing_rates, figsize=(8, 6)):
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(I_inj_values, firing_rates, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Input Current (μA/cm²)', fontsize=12)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
        ax.set_title('F-I Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_energy_vs_frequency(self, firing_rates, energy_per_spike, figsize=(8, 6)):
        
        fig, ax = plt.subplots(figsize=figsize)
        
        valid_idx = ~np.isnan(energy_per_spike)
        
        ax.plot(np.array(firing_rates)[valid_idx], 
               np.array(energy_per_spike)[valid_idx], 
               'o-', linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Firing Rate (Hz)', fontsize=12)
        ax.set_ylabel('Energy per Spike (a.u.)', fontsize=12)
        ax.set_title('Energy Efficiency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_summary(self, t, solution, spike_times, bursts, figsize=(14, 10)):
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, solution[:, 0], 'k-', linewidth=1)
        ax1.set_ylabel('V (mV)', fontsize=11)
        ax1.set_title('Membrane Potential', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(t, solution[:, 1], label='m', linewidth=1)
        ax2.plot(t, solution[:, 2], label='h', linewidth=1)
        ax2.plot(t, solution[:, 3], label='n', linewidth=1)
        if solution.shape[1] > 4:
            ax2.plot(t, solution[:, 4], label='s', linewidth=1)
        ax2.set_ylabel('Gating Variables', fontsize=11)
        ax2.set_xlabel('Time (ms)', fontsize=11)
        ax2.set_title('Channel Dynamics', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.eventplot([spike_times], colors='black', linewidths=1.5)
        if bursts:
            for burst in bursts:
                ax3.axvspan(burst['start_time'], burst['end_time'], alpha=0.3, color='red')
        ax3.set_xlabel('Time (ms)', fontsize=11)
        ax3.set_yticks([])
        ax3.set_title('Spike Raster', fontsize=12, fontweight='bold')
        
        ax4 = fig.add_subplot(gs[2, 0])
        if len(spike_times) > 1:
            isi = np.diff(spike_times)
            ax4.hist(isi, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('ISI (ms)', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('ISI Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        ax5 = fig.add_subplot(gs[2, 1])
        if bursts and len(bursts) > 0:
            spikes_per_burst = [b['n_spikes'] for b in bursts]
            ax5.bar(range(len(spikes_per_burst)), spikes_per_burst, 
                   color='coral', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Burst Number', fontsize=11)
            ax5.set_ylabel('Spikes per Burst', fontsize=11)
            ax5.set_title('Burst Structure', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, 'No bursts detected', 
                    ha='center', va='center', fontsize=12)
            ax5.set_title('Burst Structure', fontsize=12, fontweight='bold')
        
        plt.suptitle('Neuronal Activity Summary', fontsize=16, fontweight='bold', y=0.995)
        
        return fig