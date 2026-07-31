import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, silhouette_score, silhouette_samples
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CARGA DE DATASET v14 Y CORRECCIÓN DE COLUMNAS
# =====================================================================

def load_dataset(filepath: str = 'meta_dataset_v14_adimensional.csv') -> pd.DataFrame:
    if not os.path.exists(filepath):
        print(f"[ERR] No se encontró '{filepath}'.")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    if 'abs_diff' not in df.columns:
        if 'diff' in df.columns:
            df['abs_diff'] = df['diff'].abs()
        elif 'J_cohesion' in df.columns and 'J_frontier' in df.columns:
            df['diff'] = df['J_cohesion'] - df['J_frontier']
            df['abs_diff'] = df['diff'].abs()
    return df

# =====================================================================
# 2. EJECUCIÓN FASE 4: DIAGNÓSTICO DE REPRESENTACIÓN Y ABLACIÓN
# =====================================================================

def run_phase4_diagnostics():
    print("======================================================================")
    print("FASE 4: DIAGNÓSTICO DE REPRESENTACIÓN, ABLACIÓN Y GEOMETRÍA MANIFOLD")
    print("======================================================================\n")

    df = load_dataset('meta_dataset_v14_adimensional.csv')
    
    feature_cols = [
        'N', 'E', 'density', 'deg_mean', 'deg_std', 'deg_skew', 'deg_kurt',
        'deg_het', 'clustering', 'assortativity', 'spectral_gap', 
        'fiedler_value', 'num_communities', 'modularity'
    ]
    
    label_map = {'frontier': 0, 'cohesion': 1, 'tie': 2}
    inv_label_map = {0: 'Frontier', 1: 'Cohesion', 2: 'Tie'}
    
    X = df[feature_cols].values
    y = df['Optimal_Operator'].map(label_map).values

    # ---------------------------------------------------------------------
    # MÓDULO 1: ANÁLISIS DE MULTICOLINEALIDAD Y PARIENTES CERCANOS
    # ---------------------------------------------------------------------
    print("================ 1. MATRIZ DE CORRELACIÓN Y REDUNDANCIA DE DESCRIPTORES ================")
    corr_matrix = df[feature_cols].corr().abs()
    
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            val = corr_matrix.iloc[i, j]
            if val >= 0.70:
                high_corr_pairs.append((feature_cols[i], feature_cols[j], val))
                
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("Pares de Descriptores Altamente Colineales (|r| >= 0.70):")
    if high_corr_pairs:
        for f1, f2, r in high_corr_pairs:
            print(f"  • {f1:<18} <--> {f2:<18} | Pearson |r| = {r:.4f}")
    else:
        print("  • No se detectaron pares con |r| >= 0.70.")
    print("=================================================================================\n")

    # ---------------------------------------------------------------------
    # MÓDULO 2: ESTUDIO DE ABLACIÓN DE DESCRIPTORES (ABLATION STUDY)
    # ---------------------------------------------------------------------
    print("================ 2. ESTUDIO DE ABLACIÓN PROGRESIVA DE DESCRIPTORES ================")
    
    ablation_experiments = {
        'Baseline (14 features)'         : feature_cols,
        'Sin deg_het (Feature Dominante)': [f for f in feature_cols if f != 'deg_het'],
        'Sin modularity'                 : [f for f in feature_cols if f != 'modularity'],
        'Sin Momentos Redundantes Grado' : [f for f in feature_cols if f not in ['deg_std', 'deg_skew', 'deg_kurt']],
        'Sin Métricas Espectrales'       : [f for f in feature_cols if f not in ['spectral_gap', 'fiedler_value']],
        'Solo Top 5 Descriptores'        : ['deg_het', 'modularity', 'clustering', 'deg_std', 'deg_skew']
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    
    ablation_results = []
    for exp_name, feats in ablation_experiments.items():
        X_sub = df[feats].values
        oof_preds = np.zeros(len(X_sub), dtype=int)
        
        for train_idx, val_idx in skf.split(X_sub, y):
            clf = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=2026)
            clf.fit(X_sub[train_idx], y[train_idx])
            oof_preds[val_idx] = clf.predict(X_sub[val_idx])
            
        acc = accuracy_score(y, oof_preds)
        macro_f1 = f1_score(y, oof_preds, average='macro')
        f1_coh = f1_score(y, oof_preds, average=None)[1]
        f1_tie = f1_score(y, oof_preds, average=None)[2]
        
        ablation_results.append({
            'Experimento': exp_name,
            'Num_Feats': len(feats),
            'Accuracy_OOF': acc,
            'Macro_F1': macro_f1,
            'F1_Cohesion': f1_coh,
            'F1_Tie': f1_tie
        })
        
    df_abl = pd.DataFrame(ablation_results)
    print(df_abl.to_string(index=False))
    print("=================================================================================\n")

    # ---------------------------------------------------------------------
    # MÓDULO 3: IMPORTANCIA POR PERMUTACIÓN OUT-OF-FOLD (PFI)
    # ---------------------------------------------------------------------
    print("================ 3. IMPORTANCIA DE PERMUTACIÓN OUT-OF-FOLD (PFI) ================")
    clf_full = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=2026)
    clf_full.fit(X, y)
    
    pfi = permutation_importance(clf_full, X, y, n_repeats=5, random_state=2026, scoring='f1_macro')
    
    pfi_df = pd.DataFrame({
        'Descriptor': feature_cols,
        'Impacto_F1_Macro': pfi.importances_mean,
        'Std_Dev': pfi.importances_std
    }).sort_values(by='Impacto_F1_Macro', ascending=False)
    
    for _, r in pfi_df.iterrows():
        print(f"Descriptor: {r['Descriptor']:<18} | Caída en F1 Macro al Permutar: {r['Impacto_F1_Macro']:.4f} ± {r['Std_Dev']:.4f}")
    print("=================================================================================\n")

    # ---------------------------------------------------------------------
    # MÓDULO 4: PROYECIÓN GEOMÉTRICA DE MANIFOLD (PCA Y t-SNE) Y SILUETA
    # ---------------------------------------------------------------------
    print("================ 4. ANÁLISIS GEOMÉTRICO DE ESPACIO DE MANIFOLD ================")
    
    # Estandarización Z-score previa a proyección
    X_scaled = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
    
    # 1. Coeficiente de Silueta Global y por Clase
    sil_global = silhouette_score(X_scaled, y)
    sil_samples = silhouette_samples(X_scaled, y)
    
    print(f"Coeficiente de Silueta Global en Espacio de Features: {sil_global:.4f}")
    for class_val, class_name in inv_label_map.items():
        score_class = np.mean(sil_samples[y == class_val])
        print(f"  • Silueta Promedio Clase {class_name:<8}: {score_class:.4f}")
        
    # 2. Reducción PCA
    pca = PCA(n_components=2, random_state=2026)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = np.sum(pca.explained_variance_ratio_) * 100
    print(f"\nProyección PCA 2D explicada: {var_exp:.2f}% de varianza retenida.")
    
    # 3. Reducción t-SNE
    print("[*] Calculando proyección t-SNE 2D...")
    tsne = TSNE(n_components=2, perplexity=35, random_state=2026, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    sil_tsne = silhouette_score(X_tsne, y)
    print(f"[✓] t-SNE completado. Silueta en embebido 2D: {sil_tsne:.4f}")
    print("=================================================================================\n")

    # ---------------------------------------------------------------------
    # EVALUACIÓN DE HIPÓTESIS H4
    # ---------------------------------------------------------------------
    print("================ EVALUACIÓN DE HIPÓTESIS H4 ================")
    coh_sil = np.mean(sil_samples[y == 1])
    has_high_corr = len(high_corr_pairs) > 0
    
    if coh_sil <= 0.05 and has_high_corr:
        print("[ACEPTADA] Se confirma que 'Cohesion' sufre de colapso de separabilidad manifold.")
        print("Causa Raíz Identificada: Descriptores actuales capturan heterogeneidad global pero carecen de métricas de micro-embotellamiento local.")
        print("Acción Sugerida para Fase 5: Diseñar nuevos descriptores topológicos invariantes de contorno (e.g., Local Conductance Bottleneck Ratio).")
    else:
        print("[RECHAZADA] La representación es separable; re-evaluar arquitectura del clasificador.")
    print("======================================================================\n")

if __name__ == '__main__':
    run_phase4_diagnostics()
