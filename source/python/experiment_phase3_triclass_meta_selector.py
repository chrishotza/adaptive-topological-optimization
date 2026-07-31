import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla
import scipy.stats as stats
import networkx as nx

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CARGA DE DATASET v14 Y CORRECCIÓN DE COLUMNAS DE MARGEN
# =====================================================================

def load_and_fix_dataset(filepath: str = 'meta_dataset_v14_adimensional.csv') -> pd.DataFrame:
    if not os.path.exists(filepath):
        print(f"[ERR] No se encontró '{filepath}'. Ejecute la Fase 1 o 2 previamente.")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    
    # Corrección de ingeniería para abs_diff
    if 'abs_diff' not in df.columns:
        if 'diff' in df.columns:
            df['abs_diff'] = df['diff'].abs()
        elif 'J_cohesion' in df.columns and 'J_frontier' in df.columns:
            df['diff'] = df['J_cohesion'] - df['J_frontier']
            df['abs_diff'] = df['diff'].abs()
        else:
            raise KeyError("No se pudieron reconstruir las columnas 'diff' o 'abs_diff'.")
            
    return df


# =====================================================================
# 2. EJECUCIÓN DE FASE 3: META-LEARNING TRI-CLASE Y COST-AWARE POLICY
# =====================================================================

def run_phase3_experiment():
    print("======================================================================")
    print("FASE 3: CLASIFICACIÓN TRI-CLASE Y EVALUACIÓN DE POLÍTICA COSTO-EFICIENTE")
    print("======================================================================\n")

    df = load_and_fix_dataset('meta_dataset_v14_adimensional.csv')
    
    # Módulo 3 Corregido: Reporte de Margen |diff|
    print("================ 1. ANÁLISIS DEL MARGEN DE RENDIMIENTO (|diff|) ================")
    abs_diff = df['abs_diff'].values
    print(f"Margen Mínimo: {np.min(abs_diff):.6f}")
    print(f"Margen Máximo: {np.max(abs_diff):.6f}")
    print(f"Margen Medio : {np.mean(abs_diff):.6f}")
    print(f"Mediana (P50): {np.median(abs_diff):.6f}")
    print(f"Percentil 75 : {np.percentile(abs_diff, 75):.6f}")
    print("=================================================================================\n")

    feature_cols = [
        'N', 'E', 'density', 'deg_mean', 'deg_std', 'deg_skew', 'deg_kurt',
        'deg_het', 'clustering', 'assortativity', 'spectral_gap', 
        'fiedler_value', 'num_communities', 'modularity'
    ]

    label_map = {'frontier': 0, 'cohesion': 1, 'tie': 2}
    inv_label_map = {0: 'Frontier', 1: 'Cohesion', 2: 'Tie'}
    
    X = df[feature_cols].values
    y = df['Optimal_Operator'].map(label_map).values

    print("================ 2. EVALUACIÓN OOF STRATIFIED CV (3 CLASES) ================")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_preds = np.zeros(len(X), dtype=int)
    oof_probs = np.zeros((len(X), 3))
    
    feature_importances = np.zeros(len(feature_cols))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf = GradientBoostingClassifier(
            n_estimators=120, 
            learning_rate=0.08, 
            max_depth=4, 
            random_state=2026
        )
        clf.fit(X_train, y_train)
        
        oof_preds[val_idx] = clf.predict(X_val)
        oof_probs[val_idx] = clf.predict_proba(X_val)
        feature_importances += clf.feature_importances_ / skf.n_splits

    acc = accuracy_score(y, oof_preds)
    macro_f1 = f1_score(y, oof_preds, average='macro')
    
    print(f"Accuracy OOF Tri-Clase: {acc:.4f}")
    print(f"F1 Macro OOF Tri-Clase: {macro_f1:.4f}\n")
    
    print("Matriz de Confusión Tri-Clase:")
    print(confusion_matrix(y, oof_preds))
    print("\nReporte de Clasificación Detallado:")
    print(classification_report(y, oof_preds, target_names=['Frontier', 'Cohesion', 'Tie'], digits=4))

    print("================ 3. IMPORTANCIA DE DESCRIPTORES TOPOLÓGICOS ================")
    imp_df = pd.DataFrame({
        'Descriptor': feature_cols,
        'Importancia': feature_importances
    }).sort_values(by='Importancia', ascending=False)
    
    for _, r in imp_df.iterrows():
        print(f"Descriptor: {r['Descriptor']:<18} | Importancia Relativa: {r['Importancia']*100:.2f}%")
    print("=================================================================================\n")

    print("================ 4. BENCHMARK DE LA POLÍTICA COSTO-EFICIENTE ================")
    # Regla de Negocio: En caso de 'Tie', ejecutar Frontier por menor costo O(|E_cut|)
    # Compararemos:
    # A) Órculo Teórico (Máximo rendimiento posible)
    # B) Meta-Selector Tri-Clase Adaptativo
    # C) Estrategia Siempre-Frontier
    # D) Estrategia Siempre-Cohesion
    
    j_frontier = df['J_frontier'].values
    j_cohesion = df['J_cohesion'].values
    
    oracle_j = np.maximum(j_frontier, j_cohesion)
    always_f_j = j_frontier
    always_c_j = j_cohesion
    
    # Política Meta-Selector:
    # Predicción 0 (Frontier) -> Frontier
    # Predicción 1 (Cohesion) -> Cohesion
    # Predicción 2 (Tie)      -> Frontier (Menor overhead computacional)
    meta_selected_op = np.where(oof_preds == 1, 1, 0) 
    meta_j = np.where(meta_selected_op == 1, j_cohesion, j_frontier)
    
    print(f"Rendimiento Promedio Objetivo J(O):")
    print(f"  • Órculo Teórico (Límite Superior) : {np.mean(oracle_j):.5f}")
    print(f"  • Meta-Selector Tri-Clase Adaptativo: {np.mean(meta_j):.5f} ({np.mean(meta_j)/np.mean(oracle_j)*100:.2f}% del Órculo)")
    print(f"  • Baseline Fijo (Siempre Frontier) : {np.mean(always_f_j):.5f}")
    print(f"  • Baseline Fijo (Siempre Cohesion) : {np.mean(always_c_j):.5f}")
    
    print("\n================ EVALUACIÓN DE HIPÓTESIS H3 ================")
    if macro_f1 >= 0.75 and (np.mean(meta_j) / np.mean(oracle_j)) >= 0.98:
        print("[ACEPTADA] El régimen 'Tie' es predecible y la política costo-eficiente conserva >98% de la ganancia teórica.")
        print("Siguiente Paso: Benchmarking de Meta-Selector contra METIS y KaHIP en la Fase 4.")
    else:
        print("[RECHAZADA] Ajustar hiperparámetros del modelo o los umbrales de decisión.")
    print("======================================================================\n")

if __name__ == '__main__':
    run_phase3_experiment()
