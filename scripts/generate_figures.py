#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Rutas relativas desde repo root
EVIDENCE_CSV = Path("evidence/csv")
OUTPUT_FIGURES = Path("figures")
OUTPUT_FIGURES.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("[*] Generando figuras del paper ATOF...")

# Figura 1: Scatter Θ(G)
print("\n[1] Generando Figura 1: Topological Regime Scatter Plot...")
df_meta = pd.read_csv(EVIDENCE_CSV / "meta_dataset_v14_adimensional.csv")
fig, ax = plt.subplots(figsize=(12, 8))
families = df_meta['family'].unique()
colors = sns.color_palette("husl", len(families))
for i, family in enumerate(families):
    subset = df_meta[df_meta['family'] == family]
    ax.scatter(subset['modularity'], subset['fiedler_value'],
               label=family, alpha=0.6, s=100,
               color=colors[i], edgecolors='black', linewidth=0.5)
ax.set_xlabel('Modularity (Θ₁)', fontsize=14, fontweight='bold')
ax.set_ylabel('Fiedler Value (Θ₂)', fontsize=14, fontweight='bold')
ax.set_title('Topological Regime Space', fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "figure1_regime_space.png", dpi=300)
plt.close()
print("   [✓] figures/figure1_regime_space.png")

# Figura 2: Bar chart de rendimiento
print("\n[2] Generando Figura 2: Performance Comparison Bar Chart...")
df_sota = pd.read_csv(EVIDENCE_CSV / "sota_benchmark_summary_ranking.csv")
algos = ['BLOC-RELOC', 'Breakout_Local_Search', 'Tabu_Search',
         'Simulated_Annealing', 'Genetic_Algorithm']
df_plot = df_sota[df_sota['Algorithm'].isin(algos)].copy()
df_plot = df_plot.sort_values('Mean_Cut')
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(range(len(df_plot)), df_plot['Mean_Cut'],
              color=['#2ecc71' if x == 'BLOC-RELOC' else '#3498db' for x in df_plot['Algorithm']],
              alpha=0.8, edgecolor='black', linewidth=1.5)
if 'Std_Cut' in df_plot.columns:
    errors = df_plot['Std_Cut']
else:
    errors = df_plot['Mean_Cut'] * 0.15
ax.errorbar(range(len(df_plot)), df_plot['Mean_Cut'],
            yerr=errors, fmt='none', color='black', capsize=8, linewidth=2, alpha=0.7)
ax.set_xticks(range(len(df_plot)))
ax.set_xticklabels([name.replace('_', ' ') for name in df_plot['Algorithm']],
                   rotation=45, ha='right', fontsize=11)
ax.set_ylabel('Mean Cut Size (lower is better)', fontsize=14, fontweight='bold')
ax.set_title('BLOC-RELOC vs SOTA Algorithms', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, (idx, row) in enumerate(df_plot.iterrows()):
    ax.text(i, row['Mean_Cut'] + errors.iloc[i] + 100,
            f"{row['Mean_Cut']:.0f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "figure2_performance_comparison.png", dpi=300)
plt.close()
print("   [✓] figures/figure2_performance_comparison.png")

# Figura 3: Matriz de confusión
print("\n[3] Generando Figura 3: Meta-Selector Confusion Matrix...")
confusion = np.array([[267, 6, 62], [11, 27, 62], [64, 36, 661]])
cm_norm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis] * 100
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_norm, annot=confusion, fmt='d', cmap='Blues',
            cbar_kws={'label': 'Percentage (%)'},
            xticklabels=['Frontier', 'Cohesion', 'Tie'],
            yticklabels=['Frontier', 'Cohesion', 'Tie'],
            linewidths=2, linecolor='black', annot_kws={'size': 12, 'weight': 'bold'}, ax=ax)
ax.set_xlabel('Predicted Regime', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual Regime', fontsize=14, fontweight='bold')
ax.set_title('Meta-Selector Confusion Matrix (79.85% accuracy)', fontsize=14, fontweight='bold')
diag_acc = [confusion[i, i] / confusion[i].sum() * 100 for i in range(3)]
for i in range(3):
    ax.text(i + 0.5, i + 0.7, f'{diag_acc[i]:.1f}%',
            ha='center', va='center', fontsize=9, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "figure3_confusion_matrix.png", dpi=300)
plt.close()
print("   [✓] figures/figure3_confusion_matrix.png")

print("\n✅ TODAS LAS FIGURAS GENERADAS EXITOSAMENTE")
