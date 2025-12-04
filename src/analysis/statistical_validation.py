import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.utils import resample

class StatisticalValidator:
    
    def __init__(self):
        self.results = {}
    
    def test_scale_dependence(self, multi_scale_results):

        scales = multi_scale_results['scales']
        scale_results = multi_scale_results['scale_results']
        
        groups = []
        for scale in scales:
            windows = scale_results[scale]['windows']
            efficiencies = [w['efficiency'] for w in windows 
                          if not np.isnan(w['efficiency'])]
            groups.append(efficiencies)
        
        # Kruskal-Wallis H-test (non-parametric ANOVA)
        statistic, p_value = stats.kruskal(*groups)
        
        # Effect size (eta-squared)
        n_total = sum(len(g) for g in groups)
        eta_squared = (statistic - len(groups) + 1) / (n_total - len(groups))
        
        result = {
            'test': 'Kruskal-Wallis H-test',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'eta_squared': eta_squared,
            'interpretation': self._interpret_result(p_value, 'scale-dependence')
        }
        
        # Post-hoc pairwise comparisons (if significant)
        if result['significant']:
            result['pairwise'] = self._pairwise_comparisons(groups, scales)
        
        self.results['scale_dependence'] = result
        return result
    
    def _pairwise_comparisons(self, groups, scales):

        n_comparisons = len(scales) * (len(scales) - 1) // 2
        alpha_corrected = 0.05 / n_comparisons
        
        comparisons = []
        for i in range(len(scales)):
            for j in range(i + 1, len(scales)):
                stat, p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                
                comparisons.append({
                    'scale_1': scales[i],
                    'scale_2': scales[j],
                    'statistic': stat,
                    'p_value': p,
                    'significant': p < alpha_corrected,
                    'mean_diff': np.mean(groups[i]) - np.mean(groups[j])
                })
        
        return comparisons
    
    def test_burst_pattern_differences(self, burst_data):
        
        df = pd.DataFrame(burst_data)
        
        # Split by median
        median_spikes = df['n_spikes'].median()
        median_freq = df['intraburst_freq'].median()
        
        # Small vs large bursts
        small_bursts = df[df['n_spikes'] <= median_spikes]['energy_per_spike']
        large_bursts = df[df['n_spikes'] > median_spikes]['energy_per_spike']
        
        stat_size, p_size = stats.mannwhitneyu(small_bursts, large_bursts)
        cohen_d_size = self._cohen_d(small_bursts, large_bursts)
        
        # Low vs high frequency
        low_freq = df[df['intraburst_freq'] <= median_freq]['energy_per_spike']
        high_freq = df[df['intraburst_freq'] > median_freq]['energy_per_spike']
        
        stat_freq, p_freq = stats.mannwhitneyu(low_freq, high_freq)
        cohen_d_freq = self._cohen_d(low_freq, high_freq)
        
        result = {
            'burst_size': {
                'test': 'Mann-Whitney U',
                'statistic': stat_size,
                'p_value': p_size,
                'significant': p_size < 0.05,
                'cohen_d': cohen_d_size,
                'mean_diff': small_bursts.mean() - large_bursts.mean(),
                'interpretation': f"Large bursts are {abs(cohen_d_size):.2f} std deviations more efficient"
            },
            'burst_frequency': {
                'test': 'Mann-Whitney U',
                'statistic': stat_freq,
                'p_value': p_freq,
                'significant': p_freq < 0.05,
                'cohen_d': cohen_d_freq,
                'mean_diff': low_freq.mean() - high_freq.mean(),
                'interpretation': f"High frequency bursts are {abs(cohen_d_freq):.2f} std deviations more efficient"
            }
        }
        
        self.results['burst_patterns'] = result
        return result
    
    def test_energy_information_correlation(self, energy_info_data):
        
        from scipy.optimize import curve_fit
        
        df = pd.DataFrame(energy_info_data)
        
        # 1. Spearman correlation (Non-parametric, detects any monotonic trend)
        corr_ei, p_ei = stats.spearmanr(df['isi_entropy'], df['total_energy'])
        corr_er, p_er = stats.spearmanr(df['information_rate'], df['total_energy'])
        
        df_clean = df.dropna(subset=['I_inj', 'energy_per_bit'])
        x = df_clean['I_inj'].values
        y = df_clean['energy_per_bit'].values
        
        from scipy.stats import linregress
        slope, intercept, r_value_lin, p_value_lin, std_err = linregress(x, y)
        r2_linear = r_value_lin**2
        
        def exp_decay_model(x_val, a, b, c):
            return a * np.exp(-b * x_val) + c
        
        p0 = [np.max(y) - np.min(y), 0.1, np.min(y)]
        
        try:
            popt, _ = curve_fit(exp_decay_model, x, y, p0=p0, maxfev=5000)
            
            y_pred = exp_decay_model(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_nonlinear = 1 - (ss_res / ss_tot)
            
            best_model = "Exponential" if r2_nonlinear > r2_linear else "Linear"
            
        except RuntimeError:

            r2_nonlinear = 0.0
            best_model = "Linear"
            popt = [0, 0, 0]

        result = {
            'entropy_energy': {
                'correlation': corr_ei,
                'p_value': p_ei,
                'significant': p_ei < 0.05,
                'interpretation': self._interpret_correlation(corr_ei)
            },
            'rate_energy': {
                'correlation': corr_er,
                'p_value': p_er,
                'significant': p_er < 0.05,
                'interpretation': self._interpret_correlation(corr_er)
            },
            'energy_per_bit_trend': {
                'linear_r2': r2_linear,
                'nonlinear_r2': r2_nonlinear,
                'best_fit_model': best_model,
                'slope': slope, 
                'p_value': p_value_lin,
                'significant': p_value_lin < 0.05,
                'interpretation': (
                    f"Energy per bit follows a {best_model} trend "
                    f"(Non-linear R²={r2_nonlinear:.3f} vs Linear R²={r2_linear:.3f})"
                )
            }
        }
        
        self.results['energy_information'] = result
        return result
    def bootstrap_confidence_intervals(self, data, metric_func, n_bootstrap=1000, ci=95):
        
        bootstrap_estimates = []
        
        for _ in range(n_bootstrap):
            sample = resample(data)
            bootstrap_estimates.append(metric_func(sample))
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        alpha = (100 - ci) / 2
        lower = np.percentile(bootstrap_estimates, alpha)
        upper = np.percentile(bootstrap_estimates, 100 - alpha)
        
        return {
            'estimate': metric_func(data),
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_width': upper - lower,
            'std_error': np.std(bootstrap_estimates)
        }
    
    def power_analysis(self, effect_size, alpha=0.05, power=0.8):
        
        from statsmodels.stats.power import ttest_power
        
        
        result = {
            'effect_size': effect_size,
            'alpha': alpha,
            'desired_power': power,
            'note': 'Use statsmodels.stats.power for detailed analysis'
        }
        
        return result
    
    def _cohen_d(self, group1, group2):
    
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _interpret_result(self, p_value, test_name):
    
        if p_value < 0.001:
            return f"Very strong evidence for {test_name} (p < 0.001)"
        elif p_value < 0.01:
            return f"Strong evidence for {test_name} (p < 0.01)"
        elif p_value < 0.05:
            return f"Significant evidence for {test_name} (p < 0.05)"
        else:
            return f"No significant evidence for {test_name} (p = {p_value:.3f})"
    
    def _interpret_correlation(self, r):
    
        r_abs = abs(r)
        if r_abs > 0.7:
            strength = "strong"
        elif r_abs > 0.4:
            strength = "moderate"
        elif r_abs > 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "positive" if r > 0 else "negative"
        return f"{strength.capitalize()} {direction} correlation (r = {r:.3f})"
    
    def generate_report(self):
        
        for test_name, result in self.results.items():
            report += f"\n{test_name.upper().replace('_', ' ')}\n"
            report += "-" * 60 + "\n"
            report += self._format_result(result)
            report += "\n"
        
        return report
    
    def _format_result(self, result, indent=0):

        output = ""
        prefix = "  " * indent
        
        for key, value in result.items():
            if isinstance(value, dict):
                output += f"{prefix}{key}:\n"
                output += self._format_result(value, indent + 1)
            elif isinstance(value, (list, np.ndarray)):
                output += f"{prefix}{key}: [... {len(value)} items ...]\n"
            elif isinstance(value, float):
                output += f"{prefix}{key}: {value:.4f}\n"
            else:
                output += f"{prefix}{key}: {value}\n"
        
        return output
    
    def save_results(self, filename='statistical_validation.json'):
        
        import json
        
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        converted_results = convert(self.results)
        
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"Results saved to {filename}")