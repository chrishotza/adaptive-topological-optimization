import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

STATIC_FEATURES = [
    'deg_het', 'modularity', 'clustering', 'assortativity', 'fiedler_value', 'N'
]
DYNAMIC_FEATURES = [
    'initial_conductance', 'boundary_node_ratio', 'frontier_gain_mean',
    'frontier_gain_max', 'cohesion_gain_mean', 'cohesion_gain_std', 'dynamic_gain_ratio'
]
ALL_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES

def generate_parameter_varied_dataset(n_samples: int = 1400) -> pd.DataFrame:
    """Recrea el dataset desacoplado de la Fase 9 (H(Y|F) > 1.4 bits)."""
    np.random.seed(2026)
    families = [
        'barabasi_albert', 'erdos_renyi', 'grid_2d', 
        'random_geometric', 'random_regular', 'stochastic_block', 'watts_strogatz'
    ]
    
    data_rows = []
    targets = []
    family_assignments = []
    samples_per_fam = n_samples // len(families)
    
    for fam in families:
        for _ in range(samples_per_fam):
            N = np.random.uniform(50, 1000)
            
            if fam == 'grid_2d':
                diag_ratio = np.random.uniform(0.01, 0.40)
                modularity = np.random.uniform(0.35, 0.75)
                clustering = np.random.uniform(0.01, 0.25)
                fiedler = np.random.uniform(0.05, 0.45)
                
                if diag_ratio > 0.25 and modularity < 0.50:
                    op = 1 # frontier
                elif modularity >= 0.50 and fiedler < 0.20:
                    op = 0 # cohesion
                else:
                    op = 2 # tie
                    
                base_stat = [diag_ratio * 0.5, modularity, clustering, 0.35, fiedler, N]
                base_dyn = [0.15, 0.35, -3.0, 1.2, 3.0, 0.5, 1.2]
                
            elif fam == 'barabasi_albert':
                m_param = np.random.uniform(1, 10)
                deg_het = np.random.uniform(0.30, 0.95)
                clustering = 0.05 + 0.3 * (m_param / 10.0)
                
                if m_param > 6:
                    op = 0
                elif m_param < 3:
                    op = 1
                else:
                    op = 2
                    
                base_stat = [deg_het, 0.25, clustering, -0.10, 1.5, N]
                base_dyn = [0.45, 0.75, -7.0, 3.5, 1.0, 2.0, 0.15]
                
            elif fam == 'erdos_renyi':
                p_prob = np.random.uniform(0.01, 0.30)
                deg_het = np.random.uniform(0.05, 0.45)
                fiedler = p_prob * 10.0
                
                if p_prob < 0.05:
                    op = 0
                elif p_prob > 0.15:
                    op = 1
                else:
                    op = 2
                    
                base_stat = [deg_het, 0.15, p_prob, 0.02, fiedler, N]
                base_dyn = [p_prob * 2, 0.60, -5.0, 1.5, 1.2, 0.9, 0.30]
                
            elif fam == 'random_regular':
                degree = np.random.choice([3, 4, 6, 8, 12])
                fiedler = degree * 0.4
                if degree <= 4:
                    op = 0
                elif degree >= 8:
                    op = 1
                else:
                    op = 2
                base_stat = [0.02, 0.30 - degree*0.02, 0.05, 0.0, fiedler, N]
                base_dyn = [0.35, 0.45, -4.0, 0.5, 1.5, 0.2, 0.40]
                
            elif fam == 'watts_strogatz':
                rewiring_p = np.random.uniform(0.01, 0.80)
                clustering = 0.60 * (1 - rewiring_p)
                if rewiring_p < 0.10:
                    op = 0
                elif rewiring_p > 0.40:
                    op = 1
                else:
                    op = 2
                base_stat = [0.15 * rewiring_p, 0.50, clustering, 0.08, 0.90, N]
                base_dyn = [0.30, 0.50, -3.5, 1.0, 2.5, 0.5, 0.80]
                
            elif fam == 'stochastic_block':
                p_in_out_ratio = np.random.uniform(2.0, 15.0)
                modularity = 0.20 + 0.05 * p_in_out_ratio
                if p_in_out_ratio > 8.0:
                    op = 0
                elif p_in_out_ratio < 4.0:
                    op = 1
                else:
                    op = 2
                base_stat = [0.30, np.clip(modularity, 0.1, 0.8), 0.25, 0.20, 0.60, N]
                base_dyn = [0.25, 0.50, -4.0, 1.8, 2.0, 1.2, 0.50]
                
            else: # random_geometric
                radius = np.random.uniform(0.05, 0.25)
                clustering = 0.55 * (radius / 0.25)
                if radius < 0.10:
                    op = 0
                elif radius > 0.18:
                    op = 1
                else:
                    op = 2
                base_stat = [0.15, 0.45, clustering, 0.30, 0.35, N]
                base_dyn = [0.20, 0.40, -2.8, 1.0, 2.4, 0.6, 0.90]

            noise_stat = np.random.normal(0, 0.03 * np.abs(base_stat) + 0.005)
            noise_dyn = np.random.normal(0, 0.03 * np.abs(base_dyn) + 0.005)
            
            stat_vals = np.abs(np.array(base_stat) + noise_stat)
            dyn_vals = np.array(base_dyn) + noise_dyn
            
            row = np.concatenate([stat_vals, dyn_vals])
            data_rows.append(row)
            targets.append(op)
            family_assignments.append(fam)
            
    df = pd.DataFrame(data_rows, columns=ALL_FEATURES)
    df['family'] = family_assignments
    df['Optimal_Operator'] = targets
    return df

def analyze_mutual_information(df: pd.DataFrame):
    """Calcula I(X_i; Y) para determinar la fuerza de la señal en los descriptores."""
    X = df[ALL_FEATURES].values
    y = df['Optimal_Operator'].values
    
    mi_scores = mutual_info_classif(X, y, random_state=2026)
    mi_df = pd.DataFrame({
        'Feature': ALL_FEATURES,
        'Type': ['Static' if f in STATIC_FEATURES else 'Dynamic' for f in ALL_FEATURES],
        'MI_bits': mi_scores
    }).sort_values(by='MI_bits', ascending=False)
    
    print("========================================================================")
    print("[1] INFORMACIÓN MUTUA DESCRIPTOR-TARGET I(X_i; Y)")
    print("========================================================================")
    print(mi_df.to_string(index=False, float_format=lambda x: f"{x:.4f} bits"))
    
    stat_mi_sum = mi_df[mi_df['Type'] == 'Static']['MI_bits'].sum()
    dyn_mi_sum = mi_df[mi_df['Type'] == 'Dynamic']['MI_bits'].sum()
    print(f"\n  • Suma Informativa Descriptores Estáticos : {stat_mi_sum:.4f} bits")
    print(f"  • Suma Informativa Descriptores Dinámicos : {dyn_mi_sum:.4f} bits")
    print("-" * 75 + "\n")

def evaluate_logo_vs_indomain(df: pd.DataFrame):
    """Compara LOGOCV (Generalización) vs K-Fold Estratificado (Capacidad Expresiva)."""
    X = df[ALL_FEATURES].values
    y = df['Optimal_Operator'].values
    groups = df['family'].values
    
    target_names = ['Clase 0 (Cohesion)', 'Clase 1 (Frontier)', 'Clase 2 (Tie)']
    
    # 1. LOGOCV
    logo = LeaveOneGroupOut()
    oof_logo = np.zeros(len(df), dtype=int) - 1
    
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=2026)
        clf.fit(X[train_idx], y[train_idx])
        oof_logo[test_idx] = clf.predict(X[test_idx])
        
    # 2. In-Domain Stratified K-Fold (5 Folds)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_indomain = np.zeros(len(df), dtype=int) - 1
    
    for train_idx, test_idx in skf.split(X, y):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=2026)
        clf.fit(X[train_idx], y[train_idx])
        oof_indomain[test_idx] = clf.predict(X[test_idx])
        
    print("========================================================================")
    print("[2] MÉTRICAS DESGLOSADAS POR CLASE (PRECISION / RECALL / F1)")
    print("========================================================================\n")
    print("---> LOGOCV (Generalización Inter-Familia Zero-Shot):")
    print(classification_report(y, oof_logo, target_names=target_names, digits=4))
    
    print("\n---> IN-DOMAIN STRATIFIED K-FOLD (Capacidad Expresiva de X):")
    print(classification_report(y, oof_indomain, target_names=target_names, digits=4))
    
    logo_f1 = f1_score(y, oof_logo, average='macro')
    indomain_f1 = f1_score(y, oof_indomain, average='macro')
    
    print("========================================================================")
    print("DIAGNÓSTICO FINAL: ESCENARIO A vs ESCENARIO B")
    print("========================================================================")
    print(f"  • Macro F1 (LOGOCV Inter-Familia)   : {logo_f1:.4f}")
    print(f"  • Macro F1 (In-Domain Stratified)  : {indomain_f1:.4f}")
    print(f"  • Gap de Generalización (Drop-off) : {(indomain_f1 - logo_f1):.4f}")
    
    if indomain_f1 > 0.65 and (indomain_f1 - logo_f1) > 0.20:
        print("\n[✓] CONCLUSIÓN DEFINITIVA: ESCENARIO A CONFIRMADO")
        print("    1. La representación X TIENE alto poder predictivo (In-Domain F1 elevado).")
        print("    2. El cuello de botella NO son los descriptores, sino la DIFICULTAD DEL MODELO")
        print("       para extrapolar las fronteras de decisión a familias no vistas en LOGOCV.")
    else:
        print("\n[!] CONCLUSIÓN DEFINITIVA: ESCENARIO B CONFIRMADO")
        print("    Los descriptores perdieron su señal predictiva intrínseca al desacoplar el dataset.")
    print("========================================================================")

def main():
    print("========================================================================")
    print("FASE 10: DIAGNÓSTICO DE INFORMACIÓN MUTUA Y DESGLOSE DE CLASES (ACTO IV)")
    print("========================================================================\n")
    
    df = generate_parameter_varied_dataset()
    
    # 1. Analizar I(X; Y)
    analyze_mutual_information(df)
    
    # 2. Evaluar LOGOCV vs In-Domain & Desglose por Clase
    evaluate_logo_vs_indomain(df)

if __name__ == '__main__':
    main()
