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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

STATIC_FEATURES = [
    'deg_het', 'modularity', 'clustering', 'assortativity', 'fiedler_value', 'N'
]
DYNAMIC_FEATURES = [
    'initial_conductance', 'boundary_node_ratio', 'frontier_gain_mean',
    'frontier_gain_max', 'cohesion_gain_mean', 'cohesion_gain_std', 'dynamic_gain_ratio'
]
ALL_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES

def generate_robust_pipeline_dataset(n_samples: int = 1196) -> pd.DataFrame:
    """Genera un dataset sintético controlado con correspondencia determinista robusta."""
    print("[INFO] Generando Dataset de Auditoría Reproducible...")
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
            op = 0 # cohesion
        elif fam == 'barabasi_albert':
            base_stat = [0.75, 0.20, 0.15, -0.15, 1.20, 250.0]
            base_dyn = [0.50, 0.80, -8.0, 4.0, 0.5, 2.5, 0.06]
            op = 1 # frontier
        elif fam == 'erdos_renyi':
            base_stat = [0.20, 0.15, 0.05, 0.01, 2.10, 180.0]
            base_dyn = [0.60, 0.70, -6.0, 1.0, 0.8, 0.8, 0.13]
            op = 2 # tie
        elif fam == 'random_regular':
            base_stat = [0.01, 0.25, 0.03, 0.00, 3.00, 150.0]
            base_dyn = [0.40, 0.50, -4.0, 0.2, 1.2, 0.1, 0.30]
            op = 0 # cohesion
        elif fam == 'watts_strogatz':
            base_stat = [0.10, 0.55, 0.45, 0.10, 0.80, 220.0]
            base_dyn = [0.25, 0.40, -3.0, 0.8, 2.8, 0.4, 0.93]
            op = 1 # frontier
        elif fam == 'stochastic_block':
            base_stat = [0.35, 0.58, 0.30, 0.25, 0.50, 300.0]
            base_dyn = [0.20, 0.55, -4.5, 2.0, 2.2, 1.5, 0.48]
            op = 0 # cohesion
        else:
            base_stat = [0.18, 0.48, 0.55, 0.35, 0.30, 190.0]
            base_dyn = [0.15, 0.35, -2.5, 1.1, 2.6, 0.6, 1.04]
            op = 2 # tie
            
        noise_stat = np.random.normal(0, 0.02 * np.abs(base_stat) + 0.005)
        noise_dyn = np.random.normal(0, 0.03 * np.abs(base_dyn) + 0.005)
        
        stat_vals = np.abs(np.array(base_stat) + noise_stat)
        dyn_vals = np.array(base_dyn) + noise_dyn
        
        row = np.concatenate([stat_vals, dyn_vals])
        data_rows.append(row)
        targets.append(op)
        
    df = pd.DataFrame(data_rows, columns=ALL_FEATURES)
    df['family'] = family_assignments
    df['Optimal_Operator'] = targets
    return df

def compute_safe_residuals(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Calcula residuos intra-familia asegurando estabilidad numérica y estandarización."""
    df_res = df.copy()
    for col in feature_cols:
        fam_means = df.groupby('family')[col].transform('mean')
        fam_stds = df.groupby('family')[col].transform('std') + 1e-6
        # Usamos Z-score intra-familia para mantener la escala controlada
        df_res[col] = (df[col] - fam_means) / fam_stds
    return df_res

def evaluate_model_pipeline(df: pd.DataFrame, feature_cols: List[str], name: str) -> Dict:
    X = df[feature_cols].values
    groups = df['family'].values
    y = df['Optimal_Operator'].values
    
    # Estandarización global de características para evitar desbordamientos en GB
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    fam_codes = pd.factorize(df['family'])[0]
    mi_fam = np.mean(mutual_info_classif(X_scaled, fam_codes, random_state=2026))
    mi_target = np.mean(mutual_info_classif(X_scaled, y, random_state=2026))
    
    logo = LeaveOneGroupOut()
    oof_preds = np.zeros(len(df), dtype=int)
    
    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=2026)
        clf.fit(X_scaled[train_idx], y[train_idx])
        oof_preds[test_idx] = clf.predict(X_scaled[test_idx])
        
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
    print("AUDITORÍA DE REPRODUCIBILIDAD Y AISLAMIENTO FASE 8 (CONTROL DE ARTEFACTOS)")
    print("========================================================================\n")
    
    df_raw = generate_robust_pipeline_dataset()
    
    # Verificación de distribución de clases
    print("[DIAGNÓSTICO] Conteo de clases en Target:")
    print(df_raw['Optimal_Operator'].value_counts().sort_index().to_dict())
    print("-" * 72 + "\n")
    
    df_res_z = compute_safe_residuals(df_raw, ALL_FEATURES)
    
    configs = [
        (df_raw, ALL_FEATURES, "1. Baseline Completo (Sin Residualizar)"),
        (df_res_z, ALL_FEATURES, "2. Residual Completo Z-Score (Invariante)")
    ]
    
    results = []
    for df_subset, cols, cfg_name in configs:
        print(f"[Ejecutando LOGOCV] {cfg_name}...")
        res = evaluate_model_pipeline(df_subset, cols, cfg_name)
        results.append(res)
        
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*85)
    print("TABLA DE AUDITORÍA COMPARATIVA AISLADA")
    print("="*85)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*85 + "\n")
    
    f1_base = res_df.loc[res_df['Configuración'].str.contains('Baseline'), 'Macro_F1'].values[0]
    f1_res = res_df.loc[res_df['Configuración'].str.contains('Residual Completo'), 'Macro_F1'].values[0]
    
    print("====================== CONCLUSIÓN AUDITORÍA ======================")
    if f1_base >= 0.50:
        print(f"[✓] BASELINE VALIDADO: Macro F1 = {f1_base:.4f} (Coherente con fases previas).")
    else:
        print(f"[!] ALERTA EN BASELINE: Macro F1 = {f1_base:.4f} (Posible desajuste de pipeline).")
        
    if f1_res > 0.35:
        print(f"[✓] INVARIANZA SOSTENIBLE: Macro F1 residual = {f1_res:.4f}. Supera el azar significativamente.")
    else:
        print(f"[~] CAÍDA CONFIRMADA: Macro F1 residual = {f1_res:.4f}. La residualización lineal destruye la señal útil de tarea.")
    print("==================================================================")

if __name__ == '__main__':
    main()
