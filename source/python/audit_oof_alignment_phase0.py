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

def generate_verified_dataset(n_samples: int = 1196) -> pd.DataFrame:
    """Genera dataset de auditoría con índices continuos y alineación estricta."""
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
    df.reset_index(drop=True, inplace=True)
    return df

def run_phase0_oof_audit(df: pd.DataFrame, feature_cols: List[str], use_residuals: bool = False) -> Dict:
    name = "Residual Completa (Invariante)" if use_residuals else "Baseline Completo (Original)"
    
    if use_residuals:
        X_df = df[feature_cols].copy()
        for col in feature_cols:
            fam_means = df.groupby('family')[col].transform('mean')
            fam_stds = df.groupby('family')[col].transform('std') + 1e-6
            X_df[col] = (df[col] - fam_means) / fam_stds
        X = X_df.values
    else:
        X = df[feature_cols].values

    groups = df['family'].values
    y_true = df['Optimal_Operator'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logo = LeaveOneGroupOut()
    oof_preds = np.zeros(len(df), dtype=int) - 1  # Inicializar con -1 para detectar celdas no escritas
    
    print(f"========================================================================")
    print(f" AUDITANDO PIPELINE DE EVALUACIÓN OOF: {name.upper()}")
    print(f"========================================================================")
    
    fold_idx = 0
    unique_groups = np.unique(groups)
    
    for train_idx, test_idx in logo.split(X_scaled, y_true, groups):
        fold_idx += 1
        test_group = groups[test_idx[0]]
        
        X_train, y_train = X_scaled[train_idx], y_true[train_idx]
        X_test, y_test = X_scaled[test_idx], y_true[test_idx]
        
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=2026)
        clf.fit(X_train, y_train)
        
        preds_fold = clf.predict(X_test)
        
        # ASIGNACIÓN EXPLÍCITA AL ÍNDICE VECTORIAL DE TEST
        oof_preds[test_idx] = preds_fold
        
        fold_acc = accuracy_score(y_test, preds_fold)
        fold_f1 = f1_score(y_test, preds_fold, average='macro')
        
        print(f"Fold {fold_idx}/7 [{test_group}]:")
        print(f"  • Muestras de Test: {len(test_idx)} (Primeros 5 Índices: {test_idx[:5]})")
        print(f"  • Accuracy Fold: {fold_acc:.4f} | Macro F1 Fold: {fold_f1:.4f}")
        print(f"  • Distribución Predicciones: {np.bincount(preds_fold, minlength=3)}")
        print(f"  • Distribución Reales      : {np.bincount(y_test, minlength=3)}")
        print("-" * 65)

    # Verificación de cobertura completa del vector OOF
    assert not np.any(oof_preds == -1), "ERROR CRÍTICO: Hay índices no procesados en oof_preds."
    
    acc_global = accuracy_score(y_true, oof_preds)
    f1_global = f1_score(y_true, oof_preds, average='macro')
    cm_global = confusion_matrix(y_true, oof_preds)
    
    print(f"\n[MATRIZ DE CONFUSIÓN GLOBAL - {name}]")
    print(cm_global)
    print(f"\n[MÉTRICAS AUDITADAS FINALES]")
    print(f"  • Accuracy Global OOF : {acc_global:.4f}")
    print(f"  • Macro F1 Global OOF : {f1_global:.4f}\n")
    
    return {
        'Configuración': name,
        'Accuracy': acc_global,
        'Macro_F1': f1_global
    }

def main():
    df_raw = generate_verified_dataset()
    
    # 1. Audit de Baseline
    res_base = run_phase0_oof_audit(df_raw, ALL_FEATURES, use_residuals=False)
    
    # 2. Audit de Residuales Invariantes
    res_res = run_phase0_oof_audit(df_raw, ALL_FEATURES, use_residuals=True)
    
    print("========================================================================")
    print("RESUMEN DE RESULTADOS TRAS CORRECCIÓN DE ALINEACIÓN OOF")
    print("========================================================================")
    print(f"Baseline Completo          -> Accuracy: {res_base['Accuracy']:.4f} | Macro F1: {res_base['Macro_F1']:.4f}")
    print(f"Residual Completo (Z-Score)-> Accuracy: {res_res['Accuracy']:.4f} | Macro F1: {res_res['Macro_F1']:.4f}")
    print("========================================================================")

if __name__ == '__main__':
    main()
