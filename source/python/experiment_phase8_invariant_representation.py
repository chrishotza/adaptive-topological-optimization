import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS CON ESTRUCTURA INTER-FAMILIA
# =====================================================================

def load_dataset(filepath: str = 'meta_dataset_v14_adimensional.csv') -> pd.DataFrame:
    """Carga el dataset experimental o genera el entorno sintético de prueba Fase 8."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"[✓] Dataset real cargado exitosamente desde '{filepath}'. Total muestras: {len(df)}")
        return df
    
    # Generación sintética idéntica a Fase 6 y 7 si el CSV no existe localmente
    np.random.seed(2026)
    n_samples = 1196
    families = ['barabasi_albert', 'erdos_renyi', 'grid_2d', 'random_geometric', 'random_regular', 'stochastic_block', 'watts_strogatz']
    family_assignments = np.random.choice(families, n_samples)

    feature_cols = [
        'deg_het', 'modularity', 'clustering', 'assortativity', 'fiedler_value', 'N',
        'initial_conductance', 'boundary_node_ratio', 'frontier_gain_mean',
        'frontier_gain_max', 'cohesion_gain_mean', 'cohesion_gain_std', 'dynamic_gain_ratio'
    ]

    X_raw, target_op, j_oracle_list = [], [], []

    for fam in family_assignments:
        if fam == 'grid_2d':
            base = [0.02, 0.65, 0.02, 0.40, 0.15, 200, 0.10, 0.30, -2.0, 0.5, 3.5, 0.2, 1.75]
        elif fam == 'barabasi_albert':
            base = [0.75, 0.20, 0.15, -0.15, 1.20, 250, 0.50, 0.80, -8.0, 4.0, 0.5, 2.5, 0.06]
        elif fam == 'erdos_renyi':
            base = [0.20, 0.15, 0.05, 0.01, 2.10, 180, 0.60, 0.70, -6.0, 1.0, 0.8, 0.8, 0.13]
        elif fam == 'random_regular':
            base = [0.01, 0.25, 0.03, 0.00, 3.00, 150, 0.40, 0.50, -4.0, 0.2, 1.2, 0.1, 0.30]
        elif fam == 'watts_strogatz':
            base = [0.10, 0.55, 0.45, 0.10, 0.80, 220, 0.25, 0.40, -3.0, 0.8, 2.8, 0.4, 0.93]
        elif fam == 'stochastic_block':
            base = [0.35, 0.58, 0.30, 0.25, 0.50, 300, 0.20, 0.55, -4.5, 2.0, 2.2, 1.5, 0.48]
        else:
            base = [0.18, 0.48, 0.55, 0.35, 0.30, 190, 0.15, 0.35, -2.5, 1.1, 2.6, 0.6, 1.04]
        
        noise = np.random.normal(0, np.abs(0.12 * np.array(base)) + 0.01, len(base))
        row = np.abs(np.array(base) + noise)
        X_raw.append(row)

        cg_mean, fg_mean = row[10], abs(row[8])
        diff = cg_mean - fg_mean * 0.4
        if abs(diff) < 0.6:
            target_op.append('tie')
        elif diff > 0.6:
            target_op.append('cohesion')
        else:
            target_op.append('frontier')

        j_oracle_list.append(max(0.01 + fg_mean * 0.003, 0.01 + cg_mean * 0.004))

    df = pd.DataFrame(X_raw, columns=feature_cols)
    df['family'] = family_assignments
    df['Optimal_Operator'] = target_op
    df['J_oracle'] = j_oracle_list
    return df

# =====================================================================
# 2. CONSTRUCCIÓN DE RESIDUOS INTRA-FAMILIA (EXP A & EXP B)
# =====================================================================

def compute_intra_family_residuals(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula el residuo por media (Delta X) y el residuo estandarizado (Z_X) por familia."""
    df_delta = df.copy()
    df_z = df.copy()

    for col in feature_cols:
        fam_means = df.groupby('family')[col].transform('mean')
        fam_stds = df.groupby('family')[col].transform('std') + 1e-6
        
        # Exp A: Δx = x - μ_F
        df_delta[col] = df[col] - fam_means
        
        # Exp B: z = (x - μ_F) / σ_F
        df_z[col] = (df[col] - fam_means) / fam_stds

    return df_delta, df_z

# =====================================================================
# 3. AUDITORÍA INFORMACIONAL (EXP C: FUGA VS RETENCIÓN)
# =====================================================================

def run_information_audit(df_raw: pd.DataFrame, df_delta: pd.DataFrame, df_z: pd.DataFrame, feature_cols: List[str]):
    print("================ EXPERIMENTO C: AUDITORÍA INFORMACIONAL (I(X; Y) vs I(X; F)) ================")
    
    fam_codes = df_raw['family'].astype('category').cat.codes.values
    target_codes = df_raw['Optimal_Operator'].astype('category').cat.codes.values

    # Mutual Info con la Familia F (Fuga de Dominio)
    mi_x_fam = np.mean(mutual_info_classif(df_raw[feature_cols].values, fam_codes, random_state=2026))
    mi_delta_fam = np.mean(mutual_info_classif(df_delta[feature_cols].values, fam_codes, random_state=2026))
    mi_z_fam = np.mean(mutual_info_classif(df_z[feature_cols].values, fam_codes, random_state=2026))

    # Mutual Info con el Target Y (Señales de Tarea)
    mi_x_target = np.mean(mutual_info_classif(df_raw[feature_cols].values, target_codes, random_state=2026))
    mi_delta_target = np.mean(mutual_info_classif(df_delta[feature_cols].values, target_codes, random_state=2026))
    mi_z_target = np.mean(mutual_info_classif(df_z[feature_cols].values, target_codes, random_state=2026))

    red_delta_fam = (1.0 - mi_delta_fam / mi_x_fam) * 100
    red_z_fam = (1.0 - mi_z_fam / mi_x_fam) * 100

    ret_delta_target = (mi_delta_target / mi_x_target) * 100
    ret_z_target = (mi_z_target / mi_x_target) * 100

    print(f"  • I(X; Familia) Original          : {mi_x_fam:.4f}")
    print(f"  • I(ΔX; Familia) (Residuo Media)  : {mi_delta_fam:.4f}  [Reducción de Fuga: {red_delta_fam:.2f}%]")
    print(f"  • I(Z; Familia) (Residuo Z-Score) : {mi_z_fam:.4f}  [Reducción de Fuga: {red_z_fam:.2f}%]\n")

    print(f"  • I(X; Target) Original           : {mi_x_target:.4f}")
    print(f"  • I(ΔX; Target) (Residuo Media)   : {mi_delta_target:.4f}  [Retención de Señal: {ret_delta_target:.2f}%]")
    print(f"  • I(Z; Target) (Residuo Z-Score)  : {mi_z_target:.4f}  [Retención de Señal: {ret_z_target:.2f}%]")
    print("=================================================================================\n")
    return {
        'mi_x_fam': mi_x_fam, 'mi_delta_fam': mi_delta_fam, 'mi_z_fam': mi_z_fam,
        'mi_x_target': mi_x_target, 'mi_delta_target': mi_delta_target, 'mi_z_target': mi_z_target
    }

# =====================================================================
# 4. RE-ENTRENAMIENTO Y LOGOCV (EXP D)
# =====================================================================

def evaluate_logocv_representation(df_data: pd.DataFrame, feature_cols: List[str], rep_name: str) -> Dict:
    logo = LeaveOneGroupOut()
    X = df_data[feature_cols].values
    groups = df_data['family'].values
    
    label_map = {'frontier': 0, 'cohesion': 1, 'tie': 2}
    inv_label_map = {0: 'frontier', 1: 'cohesion', 2: 'tie'}
    y = df_data['Optimal_Operator'].map(label_map).values

    oof_preds = np.zeros(len(df_data), dtype=int)
    family_metrics = []

    for train_idx, test_idx in logo.split(X, y, groups):
        test_fam = groups[test_idx][0]
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=2026)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        oof_preds[test_idx] = preds
        
        acc_fam = accuracy_score(y[test_idx], preds)
        f1_fam = f1_score(y[test_idx], preds, average='macro')
        family_metrics.append({'Familia': test_fam, 'Accuracy': acc_fam, 'Macro_F1': f1_fam})

    acc_global = accuracy_score(y, oof_preds)
    f1_global = f1_score(y, oof_preds, average='macro')
    cm = confusion_matrix(y, oof_preds)
    
    # Reporte por clase
    target_names = ['frontier', 'cohesion', 'tie']
    report_dict = classification_report(y, oof_preds, target_names=target_names, output_dict=True)

    return {
        'rep_name': rep_name,
        'acc': acc_global,
        'f1_macro': f1_global,
        'family_df': pd.DataFrame(family_metrics),
        'cm': cm,
        'class_report': report_dict,
        'oof_preds': oof_preds
    }

# =====================================================================
# 5. EJECUCIÓN PRINCIPAL Y DIAGNÓSTICO DE HIPÓTESIS H8
# =====================================================================

def main():
    print("======================================================================")
    print("FASE 8: EVALUACIÓN DE RESIDUOS INTRA-FAMILIA Y REPRESENTACIÓN INVARIANTE")
    print("======================================================================\n")

    feature_cols = [
        'deg_het', 'modularity', 'clustering', 'assortativity', 'fiedler_value', 'N',
        'initial_conductance', 'boundary_node_ratio', 'frontier_gain_mean',
        'frontier_gain_max', 'cohesion_gain_mean', 'cohesion_gain_std', 'dynamic_gain_ratio'
    ]

    df_raw = load_dataset()
    df_delta, df_z = compute_intra_family_residuals(df_raw, feature_cols)

    # 1. Auditoría Teórico-Informacional
    info_stats = run_information_audit(df_raw, df_delta, df_z, feature_cols)

    # 2. Evaluación LOGOCV para las 3 representaciones
    res_raw = evaluate_logocv_representation(df_raw, feature_cols, "Original (X)")
    res_delta = evaluate_logocv_representation(df_delta, feature_cols, "Residuo Media (ΔX)")
    res_z = evaluate_logocv_representation(df_z, feature_cols, "Residuo Z-Score (Z)")

    # 3. Presentación de Resultados Comparativos
    print("================ BENCHMARK LOGOCV POR REPRESENTACIÓN ================")
    summary_df = pd.DataFrame([
        {'Representación': res_raw['rep_name'], 'Accuracy': res_raw['acc'], 'Macro_F1': res_raw['f1_macro']},
        {'Representación': res_delta['rep_name'], 'Accuracy': res_delta['acc'], 'Macro_F1': res_delta['f1_macro']},
        {'Representación': res_z['rep_name'], 'Accuracy': res_z['acc'], 'Macro_F1': res_z['f1_macro']}
    ])
    print(summary_df.to_string(index=False))
    print("=====================================================================\n")

    print("================ DESGLOSE LOGOCV: RESIDUO Z-SCORE (EXP B) ================")
    print(res_z['family_df'].to_string(index=False))
    print("\nMatriz de Confusión (Filas: Real, Columnas: Predicho) [frontier, cohesion, tie]:")
    print(res_z['cm'])
    
    print("\nF1-Score por Clase:")
    for cls in ['frontier', 'cohesion', 'tie']:
        f1_c = res_z['class_report'][cls]['f1-score']
        print(f"  • {cls:<10}: F1 = {f1_c:.4f}")
    print("=====================================================================\n")

    # 4. Evaluación de la Hipótesis H8
    print("================ EVALUACIÓN DE HIPÓTESIS H8 ================")
    f1_res = res_z['f1_macro']
    mi_target_ret = (info_stats['mi_z_target'] / info_stats['mi_x_target']) * 100

    print(f"Hipótesis H8: I(ΔX; Y) > 0  | Retención de Información de Target: {mi_target_ret:.2f}%")
    
    if f1_res >= 0.70:
        print(f"[ACEPTADA - ESCENARIO 1] Macro F1 = {f1_res:.4f}. Existe una representación universal invariante.")
        print("Recomendación: Proceder con Domain Adversarial Learning (DANN) para optimizar la proyección.")
    elif f1_res >= 0.40:
        print(f"[PARCIAL - ESCENARIO 2] Macro F1 = {f1_res:.4f}. Señal intra-familia presente pero mixta.")
        print("Recomendación: El operador combina reglas locales universales y la escala global de la familia.")
    else:
        print(f"[RECHAZADA - ESCENARIO 3] Macro F1 = {f1_res:.4f}. El rendimiento colapsa al remover la identidad.")
        print("DIAGNÓSTICO TÉCNICO: TEOREMA DE IMPOSIBILIDAD DE TRANSFERENCIA (NO-FREE-LUNCH TOPOLÓGICO).")
        print("La información para elegir el refinador óptimo está intrínsecamente ligada a la familia del grafo.")
    print("======================================================================\n")

if __name__ == '__main__':
    main()
