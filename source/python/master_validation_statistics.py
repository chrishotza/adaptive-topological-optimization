import os
import sys
import json
import pandas as pd
import numpy as np
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPORT_DIR = "MASTER_VALIDATION_REPORT"

def main():
    runs_path = f"{REPORT_DIR}/detailed_runs.csv"
    if not os.path.exists(runs_path):
        print(f"[!] No se encontró {runs_path}. Ejecute primero los módulos anteriores.")
        sys.exit(1)

    print("[*] Procesando detailed_runs.csv para estadísticas, Wilcoxon y estabilidad por semillas...")
    df = pd.read_csv(runs_path)

    pivot_mean = df.pivot_table(index='Instance', columns='Algorithm', values='Cost')
    pivot_mean.to_csv(f"{REPORT_DIR}/summary_pivot_costs.csv")

    stability_df = df.groupby(['Algorithm', 'Instance'])['Cost'].agg(['mean', 'std']).reset_index()
    stability_df.to_csv(f"{REPORT_DIR}/stability_analysis.csv", index=False)

    algs = pivot_mean.columns.tolist()
    stat_records = []
    target = 'BLOC_RELOC (Proposed)'
    if target in algs:
        for other in algs:
            if other == target: continue
            try:
                w_stat, w_pval = stats.wilcoxon(pivot_mean[target], pivot_mean[other])
            except Exception:
                w_stat, w_pval = 0.0, 1.0
            diff = pivot_mean[target] - pivot_mean[other]
            cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)
            stat_records.append({'Comparison': f"{target} vs {other}", 'Wilcoxon_W': w_stat, 'p_value': w_pval, 'Cohen_d': cohen_d})
    
    df_stats = pd.DataFrame(stat_records)
    df_stats.to_csv(f"{REPORT_DIR}/statistical_comparisons.csv", index=False)

    os.makedirs(f"{REPORT_DIR}/assets", exist_ok=True)
    if not pivot_mean.empty and pivot_mean.shape[1] > 0:
        plt.figure(figsize=(9, 6))
        min_costs = pivot_mean.min(axis=1)
        for col in pivot_mean.columns:
            ratios = pivot_mean[col] / (min_costs + 1e-8)
            sorted_r = np.sort(ratios)
            y_vals = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
            plt.step(sorted_r, y_vals, label=col, where='post', linewidth=2)
        plt.xlabel(r"Performance Ratio $\tau$", fontsize=12)
        plt.ylabel(r"Fraction of Instances $P(r_{i,a} \le \tau)$", fontsize=12)
        plt.title("Dolan-Moré Performance Profiles (Topological Sensitivity)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/assets/dolan_more_profile.png", dpi=300)
        plt.close()

    print(f"\n[✓] MÓDULO ESTADÍSTICO COMPLETADO. Archivos generados en `{REPORT_DIR}/`.")

if __name__ == "__main__":
    main()
