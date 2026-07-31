import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

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
                    op = 1
                elif modularity >= 0.50 and fiedler < 0.20:
                    op = 0
                else:
                    op = 2
                    
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

def run_experiment_A(df: pd.DataFrame):
    """EXPERIMENTO A: Comparación Multi-Modelo manteniendo la representación constante."""
    print("========================================================================")
    print("EXPERIMENTO A: BENCHMARKING MULTI-MODELO (REPRESENTACIÓN CONSTANTE)")
    print("========================================================================")
    print("Objetivo: Evaluar si el colapso en LOGOCV es específico de GBDT o generalizado.\n")
    
    X = df[ALL_FEATURES].values
    y = df['Optimal_Operator'].values
    groups = df['family'].values
    
    models = {
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=2026),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=2026),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=2026),
        'SVM (RBF Kernel)': SVC(C=1.0, kernel='rbf', random_state=2026),
        'MLP (Neural Net)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=2026)
    }
    
    results = []
    
    for name, model in models.items():
        # 1. LOGOCV
        logo = LeaveOneGroupOut()
        oof_logo = np.zeros(len(df), dtype=int) - 1
        
        for train_idx, test_idx in logo.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr = y[train_idx]
            
            if 'SVM' in name or 'MLP' in name:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                
            model.fit(X_tr, y_tr)
            oof_logo[test_idx] = model.predict(X_te)
            
        # 2. In-Domain Stratified 5-Fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
        oof_indomain = np.zeros(len(df), dtype=int) - 1
        
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr = y[train_idx]
            
            if 'SVM' in name or 'MLP' in name:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                
            model.fit(X_tr, y_tr)
            oof_indomain[test_idx] = model.predict(X_te)
            
        f1_logo = f1_score(y, oof_logo, average='macro')
        f1_in = f1_score(y, oof_indomain, average='macro')
        gap = f1_in - f1_logo
        
        results.append({
            'Modelo': name,
            'In-Domain F1': f1_in,
            'LOGOCV F1': f1_logo,
            'Gap (Drop-off)': gap
        })
        
    res_df = pd.DataFrame(results).sort_values(by='LOGOCV F1', ascending=False)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 75)
    
    max_logo = res_df['LOGOCV F1'].max()
    min_logo = res_df['LOGOCV F1'].min()
    
    if max_logo <= 0.35:
        print("\n[✓] INTERPRETACIÓN EXPERIMENTO A:")
        print(f"    TODOS los modelos colapsan en LOGOCV (Rango: {min_logo:.4f} - {max_logo:.4f}).")
        print("    Esto DEMUESTRA que la causa principal del gap NO es el modelo específico,")
        print("    sino la falta de invariancia de dominio en la representación X.")
    else:
        print("\n[!] INTERPRETACIÓN EXPERIMENTO A:")
        print("    Hay variabilidad sustancial entre modelos. Algunos logran mitigar parte del gap.")
    print("========================================================================\n")

def run_experiment_B(df: pd.DataFrame):
    """EXPERIMENTO B: Comparación de familias de representaciones."""
    print("========================================================================")
    print("EXPERIMENTO B: VARIACIÓN DE ESPACIOS DE REPRESENTACIÓN")
    print("========================================================================")
    print("Objetivo: Evaluar qué subespacio o transformación de X favorece la invariancia.\n")
    
    y = df['Optimal_Operator'].values
    groups = df['family'].values
    
    # Construcción de espacios de representación
    feature_sets = {}
    
    # 1. Solo Estáticas
    feature_sets['1. Solo Estáticas'] = df[STATIC_FEATURES].values
    
    # 2. Solo Dinámicas
    feature_sets['2. Solo Dinámicas'] = df[DYNAMIC_FEATURES].values
    
    # 3. Combinadas (Base)
    feature_sets['3. Estáticas + Dinámicas'] = df[ALL_FEATURES].values
    
    # 4. Normalizadas por Familia (Intra-Family Z-Score)
    df_norm = df.copy()
    for col in ALL_FEATURES:
        df_norm[col] = df_norm.groupby('family')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    feature_sets['4. Normalizadas Intra-Familia'] = df_norm[ALL_FEATURES].values
    
    # 5. Enriquecidas con Ratios Espectrales
    df_spec = df.copy()
    df_spec['fiedler_modularity_ratio'] = df_spec['fiedler_value'] / (df_spec['modularity'] + 1e-8)
    df_spec['conductance_gain_ratio'] = df_spec['initial_conductance'] / (np.abs(df_spec['frontier_gain_mean']) + 1e-8)
    extended_cols = ALL_FEATURES + ['fiedler_modularity_ratio', 'conductance_gain_ratio']
    feature_sets['5. Enriquecidas (Ratios)'] = df_spec[extended_cols].values

    results = []
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=2026)
    
    for set_name, X_space in feature_sets.items():
        # LOGOCV
        logo = LeaveOneGroupOut()
        oof_logo = np.zeros(len(df), dtype=int) - 1
        
        for train_idx, test_idx in logo.split(X_space, y, groups):
            clf.fit(X_space[train_idx], y[train_idx])
            oof_logo[test_idx] = clf.predict(X_space[test_idx])
            
        # In-Domain
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
        oof_indomain = np.zeros(len(df), dtype=int) - 1
        
        for train_idx, test_idx in skf.split(X_space, y):
            clf.fit(X_space[train_idx], y[train_idx])
            oof_indomain[test_idx] = clf.predict(X_space[test_idx])
            
        f1_logo = f1_score(y, oof_logo, average='macro')
        f1_in = f1_score(y, oof_indomain, average='macro')
        gap = f1_in - f1_logo
        
        results.append({
            'Espacio de Representación': set_name,
            'In-Domain F1': f1_in,
            'LOGOCV F1': f1_logo,
            'Gap (Drop-off)': gap
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 75)
    
    print("\n[✓] INTERPRETACIÓN EXPERIMENTO B:")
    print("    • Compara cómo responde el gap según el tipo de características.")
    print("    • Si las Dinámicas o las Ratios reducen el Gap, ahí se encuentra el camino hacia")
    print("      representaciones verdaderamente invariantes al dominio.")
    print("========================================================================\n")

def main():
    print("========================================================================")
    print("EJECUCIÓN EXPERIMENTAL COMPLETA: EXPERIMENTOS A Y B (FASE 10)")
    print("========================================================================\n")
    
    df = generate_parameter_varied_dataset()
    
    run_experiment_A(df)
    run_experiment_B(df)

if __name__ == '__main__':
    main()
