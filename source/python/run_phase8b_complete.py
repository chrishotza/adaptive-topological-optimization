import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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

def generate_complete_pipeline_dataset(n_samples: int = 1196) -> pd.DataFrame:
    """Simula el dataset completo de Fase 5 con variables estáticas y dinámicas acopladas."""
    print("[INFO] Generando Dataset Completo (Fase 5 Pipeline: Estáticas + Dinámicas)...")
    np.random.seed(2026)
    families = [
        'barabasi_albert', 'erdos_renyi', 'grid_2d', 
        'random_geometric', 'random_regular', 'stochastic_block', 'watts_strogatz'
    ]
    family_assignments = np.random.choice(families, n_samples)
    
    data_rows = []
    targets = []
    
    for fam in family_assignments:
        if fam == 'grid_2d':
            base_stat = [0.02, 0.65, 0.02, 0.40, 0.15, 200.0]
            base_dyn = [0.10, 0.30, -2.0, 0.5, 3.5, 0.2, 1.75]
            op = 'cohesion'
        elif fam == 'barabasi_albert':
            base_stat = [0.75, 0.20, 0.15, -0.15, 1.20, 250.0]
            base_dyn = [0.50, 0.80, -8.0, 4.0, 0.5, 2.5, 0.06]
            op = 'frontier'
        elif fam == 'erdos_renyi':
            base_stat = [0.20, 0.15, 0.05, 0.01, 2.10, 180.0]
            base_dyn = [0.60, 0.70, -6.0, 1.0, 0.8, 0.8, 0.13]
            op = 'tie'
        elif fam == 'random_regular':
            base_stat = [0.01, 0.25, 0.03, 0.00, 3.00, 150.0]
            base_dyn = [0.40, 0.50, -4.0, 0.2, 1.2, 0.1, 0.30]
            op = 'cohesion'
        elif fam == 'watts_strogatz':
            base_stat = [0.10, 0.55, 0.45, 0.10, 0.80, 220.0]
            base_dyn = [0.25, 0.40, -3.0, 0.8, 2.8, 0.4, 0.93]
            op = 'frontier'
        elif fam == 'stochastic_block':
            base_stat = [0.35, 0.58, 0.30, 0.25, 0.50, 300.0]
            base_dyn = [0.20, 0.55, -4.5, 2.0, 2.2, 1.5, 0.48]
            op = 'cohesion'
        else:
            base_stat = [0.18, 0.48, 0.55, 0.35, 0.30, 190.0]
            base_dyn = [0.15, 0.35, -2.5, 1.1, 2.6, 0.6, 1.04]
            op = 'tie'
            
        noise_stat = np.random.normal(0, 0.05 * np.abs(base_stat) + 0.01)
        noise_dyn = np.random.normal(0, 0.08 * np.abs(base_dyn) + 0.01)
        
        stat_vals = np.abs(np.array(base_stat) + noise_stat)
        dyn_vals = np.array(base_dyn) + noise_dyn
        
        row = np.concatenate([stat_vals, dyn_vals])
        data_rows.append(row)
        targets.append(op)
        
    df = pd.DataFrame(data_rows, columns=ALL_FEATURES)
    df['family'] = family_assignments
    df['Optimal_Operator'] = targets
    return df

def compute_residuals(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df_res = df.copy()
    for col in feature_cols:
        fam_means = df.groupby('family')[col].transform('mean')
        df_res[col] = df[col] - fam_means
    return df_res

def evaluate_configuration(df: pd.DataFrame, feature_cols: List[str], name: str) -> Dict:
    X = df[feature_cols].values
    groups = df['family'].values
    y = df['Optimal_Operator'].astype('category').cat.codes.values
    
    fam_codes = df['family'].astype('category').cat.codes.values
    mi_fam = np.mean(mutual_info_classif(X, fam_codes, random_state=2026))
    mi_target = np.mean(mutual_info_classif(X, y, random_state=2026))
    
    logo = LeaveOneGroupOut()
    oof_preds = np.zeros(len(df), dtype=int)
    
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=2026)
        clf.fit(X[train_idx], y[train_idx])
        oof_preds[test_idx] = clf.predict(X[test_idx])
        
    acc = accuracy_score(y, oof_preds)
    f1 = f1_score(y, oof_preds, average='macro')
    
    return {
        'Configuración': name,
        'Accuracy': acc,
        'Macro_F1': f1,
        'I(Z; Familia)': mi_fam,
        'I(Z; Target)': mi_target
    }

def main():
    print("========================================================================")
    print("FASE 8B: EVALUACIÓN RIGUROSA DE REPRESENTACIÓN INVARIANTE (COMPLETA)")
    print("========================================================================\n")
    
    df_raw = generate_complete_pipeline_dataset()
    
    df_res_static = compute_residuals(df_raw, STATIC_FEATURES)
    df_res_full = compute_residuals(df_raw, ALL_FEATURES)
    
    configs = [
        (df_raw, STATIC_FEATURES, "1. Estática Pura (X_stat)"),
        (df_raw, ALL_FEATURES, "2. Completa (Estática + Dinámica)"),
        (df_res_static, STATIC_FEATURES, "3. Residual Estática (ΔX_stat)"),
        (df_res_full, ALL_FEATURES, "4. Residual Completa (ΔX_full) [Invariante]")
    ]
    
    results = []
    for df_subset, cols, cfg_name in configs:
        print(f"[Evaluando] {cfg_name}...")
        res = evaluate_configuration(df_subset, cols, cfg_name)
        results.append(res)
        
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*95)
    print("TABLA COMPARATIVA FINAL - FASE 8B: INVARIANZA VS RENDIMIENTO")
    print("="*95)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*95 + "\n")
    
    f1_full_res = res_df.loc[res_df['Configuración'].str.contains('Residual Completa'), 'Macro_F1'].values[0]
    
    print("====================== CONCLUSIÓN CIENTÍFICA FASE 8B ======================")
    if f1_full_res >= 0.65:
        print("[✓] HIPÓTESIS H8.1 ACEPTADA: Existe una representación invariante robusta.")
        print("    La inclusión de descriptores dinámicos preserva la capacidad predictiva")
        print("    del meta-selector incluso tras purgar la identidad de familia.")
    elif f1_full_res >= 0.40:
        print("[~] HIPÓTESIS H8.2 PARCIAL: Invarianza con degradación moderada.")
        print("    El acoplamiento parcial entre dominio y tarea genera una pérdida de desempeño,")
        print("    pero supera significativamente el azar y el colapso estático puro.")
    else:
        print("[✗] HIPÓTESIS H8.3 RECHAZADA (No-Free-Lunch confirmado con espacio completo):")
        print("    La información de dominio y la tarea están fundamentalmente entrelazadas.")
        print("    Ninguna transformación residual preserva la señal óptima del refinador.")
    print("==========================================================================")

if __name__ == '__main__':
    main()
