import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

CSV_FILENAME = 'meta_dataset_v14_adimensional.csv'

# Lista completa de descriptores esperados hasta la Fase 5
EXPECTED_FEATURES = [
    'deg_het', 'modularity', 'clustering', 'assortativity', 'fiedler_value', 'N',
    'initial_conductance', 'boundary_node_ratio', 'frontier_gain_mean',
    'frontier_gain_max', 'cohesion_gain_mean', 'cohesion_gain_std', 'dynamic_gain_ratio'
]

# =====================================================================
# 1. AUDITORÍA DE CONSISTENCIA DEL DATASET (INVENTARIO)
# =====================================================================

def audit_dataset_schema(filepath: str, expected_cols: List[str]) -> Tuple[pd.DataFrame, List[str], bool]:
    """Realiza la auditoría de esquema contra el CSV real reportando OK, RENOMBRADA o FALTA."""
    print("======================================================================")
    print("PASO 1: AUDITORÍA DE CONSISTENCIA DE ESQUEMA DEL DATASET")
    print("======================================================================\n")
    
    if not os.path.exists(filepath):
        print(f"[!] Archivo '{filepath}' NO ENCONTRADO localmente.")
        print("[!] Se procederá con la generación sintética garantizando todas las variables dinámicas de Fase 5.\n")
        return None, expected_cols, False

    df = pd.read_csv(filepath)
    actual_cols = df.columns.tolist()
    print(f"[✓] Archivo '{filepath}' cargado. Total de columnas encontradas: {len(actual_cols)}")

    audit_rows = []
    active_features = []
    has_missing = False

    for exp in expected_cols:
        if exp in actual_cols:
            audit_rows.append({'Esperada': exp, 'Encontrada': exp, 'Estado': 'OK'})
            active_features.append(exp)
        else:
            # Buscar coincidencias parciales / difusas
            candidates = [c for c in actual_cols if exp.replace('_', '') in c.replace('_', '') or c in exp]
            if candidates:
                matched_name = candidates[0]
                audit_rows.append({'Esperada': exp, 'Encontrada': matched_name, 'Estado': 'RENOMBRADA'})
                active_features.append(matched_name)
            else:
                audit_rows.append({'Esperada': exp, 'Encontrada': 'N/A', 'Estado': 'FALTA'})
                has_missing = True

    audit_df = pd.DataFrame(audit_rows)
    print("\n" + audit_df.to_string(index=False) + "\n")

    if has_missing:
        print("[⚠️  ADVERTENCIA DE TRAZABILIDAD]: Faltan descriptores dinámicos clave en el CSV.")
        print("    Para garantizar la validez científica de la Fase 8, las variables faltantes")
        print("    deben reconstruirse desde el pipeline de la Fase 5 antes de tomar conclusiones definitivas.\n")
    else:
        print("[✓] Esquema verificado con éxito. Todos los descriptores esperados están presentes.\n")

    return df, active_features, True

# =====================================================================
# 2. GENERADOR/CARGADOR DE DATOS CON CONTROL DE TRAZABILIDAD
# =====================================================================

def get_prepared_dataset(filepath: str, expected_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df, active_cols, is_real = audit_dataset_schema(filepath, expected_cols)
    
    if is_real and df is not None:
        # Verificar que existen las columnas meta 'family' y 'Optimal_Operator'
        if 'family' in df.columns and 'Optimal_Operator' in df.columns:
            # Filtrar columnas verdaderamente numéricas para las características
            valid_cols = [c for c in active_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            return df, valid_cols

    print("[INFO] Generando Dataset Controlado de Fase 8 con variables dinámicas completas...")
    np.random.seed(2026)
    n_samples = 1196
    families = ['barabasi_albert', 'erdos_renyi', 'grid_2d', 'random_geometric', 'random_regular', 'stochastic_block', 'watts_strogatz']
    family_assignments = np.random.choice(families, n_samples)

    X_raw, target_op = [], []
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

    df_synth = pd.DataFrame(X_raw, columns=expected_cols)
    df_synth['family'] = family_assignments
    df_synth['Optimal_Operator'] = target_op
    return df_synth, expected_cols

# =====================================================================
# 3. CONSTRUCCIÓN DE RESIDUOS INTRA-FAMILIA (EXP A & EXP B)
# =====================================================================

def compute_intra_family_residuals(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_delta = df.copy()
    df_z = df.copy()

    for col in feature_cols:
        fam_means = df.groupby('family')[col].transform('mean')
        fam_stds = df.groupby('family')[col].transform('std') + 1e-6
        df_delta[col] = df[col] - fam_means
        df_z[col] = (df[col] - fam_means) / fam_stds

    return df_delta, df_z

# =====================================================================
# 4. AUDITORÍA INFORMACIONAL (EXP C) Y EVALUACIÓN LOGOCV (EXP D)
# =====================================================================

def run_phase8_experiments(df_raw: pd.DataFrame, feature_cols: List[str]):
    print("======================================================================")
    print("PASO 2: EJECUCIÓN EXPERIMENTAL FASE 8 (HIPÓTESIS H8)")
    print("======================================================================\n")

    df_delta, df_z = compute_intra_family_residuals(df_raw, feature_cols)

    fam_codes = df_raw['family'].astype('category').cat.codes.values
    target_codes = df_raw['Optimal_Operator'].astype('category').cat.codes.values

    # Información Mutua
    mi_x_fam = np.mean(mutual_info_classif(df_raw[feature_cols].values, fam_codes, random_state=2026))
    mi_delta_fam = np.mean(mutual_info_classif(df_delta[feature_cols].values, fam_codes, random_state=2026))
    mi_z_fam = np.mean(mutual_info_classif(df_z[feature_cols].values, fam_codes, random_state=2026))

    mi_x_target = np.mean(mutual_info_classif(df_raw[feature_cols].values, target_codes, random_state=2026))
    mi_delta_target = np.mean(mutual_info_classif(df_delta[feature_cols].values, target_codes, random_state=2026))
    mi_z_target = np.mean(mutual_info_classif(df_z[feature_cols].values, target_codes, random_state=2026))

    print("--- EXPERIMENTO C: AUDITORÍA INFORMACIONAL DE FUGA Y RETENCIÓN ---")
    print(f"  • I(X; Familia) Original          : {mi_x_fam:.4f}")
    print(f"  • I(ΔX; Familia) (Residuo Media)  : {mi_delta_fam:.4f}  [Reducción Fuga: {(1 - mi_delta_fam/mi_x_fam)*100:.2f}%]")
    print(f"  • I(Z; Familia) (Residuo Z-Score) : {mi_z_fam:.4f}  [Reducción Fuga: {(1 - mi_z_fam/mi_x_fam)*100:.2f}%]\n")

    print(f"  • I(X; Target) Original           : {mi_x_target:.4f}")
    print(f"  • I(ΔX; Target) (Residuo Media)   : {mi_delta_target:.4f}  [Retención Señal: {(mi_delta_target/mi_x_target)*100:.2f}%]")
    print(f"  • I(Z; Target) (Residuo Z-Score)  : {mi_z_target:.4f}  [Retención Señal: {(mi_z_target/mi_x_target)*100:.2f}%]")
    print("-------------------------------------------------------------------\n")

    # Evaluación LOGOCV
    def eval_logocv(df_data: pd.DataFrame, rep_name: str):
        logo = LeaveOneGroupOut()
        X = df_data[feature_cols].values
        groups = df_data['family'].values
        y = df_data['Optimal_Operator'].astype('category').cat.codes.values
        
        oof_preds = np.zeros(len(df_data), dtype=int)
        for train_idx, test_idx in logo.split(X, y, groups):
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=2026)
            clf.fit(X[train_idx], y[train_idx])
            oof_preds[test_idx] = clf.predict(X[test_idx])
        
        acc = accuracy_score(y, oof_preds)
        f1 = f1_score(y, oof_preds, average='macro')
        cm = confusion_matrix(y, oof_preds)
        return {'Name': rep_name, 'Accuracy': acc, 'Macro_F1': f1, 'CM': cm}

    res_x = eval_logocv(df_raw, "Original (X)")
    res_delta = eval_logocv(df_delta, "Residuo Media (ΔX)")
    res_z = eval_logocv(df_z, "Residuo Z-Score (Z)")

    print("--- EXPERIMENTO D: RENDIMIENTO EN GENERALIZACIÓN LOGOCV ---")
    res_df = pd.DataFrame([
        {'Representación': res_x['Name'], 'Accuracy': res_x['Accuracy'], 'Macro_F1': res_x['Macro_F1']},
        {'Representación': res_delta['Name'], 'Accuracy': res_delta['Accuracy'], 'Macro_F1': res_delta['Macro_F1']},
        {'Representación': res_z['Name'], 'Accuracy': res_z['Accuracy'], 'Macro_F1': res_z['Macro_F1']}
    ])
    print(res_df.to_string(index=False))
    print("-----------------------------------------------------------\n")

    print("================ EVALUACIÓN DE HIPÓTESIS H8 ================")
    f1_z = res_z['Macro_F1']
    if f1_z >= 0.70:
        print(f"[ACEPTADA - ESCENARIO 1] Macro F1 = {f1_z:.4f}. Representación invariante confirmada.")
    elif f1_z >= 0.40:
        print(f"[PARCIAL - ESCENARIO 2] Macro F1 = {f1_z:.4f}. Representación parcialmente invariante.")
    else:
        print(f"[RECHAZADA - ESCENARIO 3] Macro F1 = {f1_z:.4f}. Colapso al remover dominio.")
        print("DIAGNÓSTICO: Teorema No-Free-Lunch para transferencia de refinadores.")
    print("======================================================================")

def main():
    df, feature_cols = get_prepared_dataset(CSV_FILENAME, EXPECTED_FEATURES)
    run_phase8_experiments(df, feature_cols)

if __name__ == '__main__':
    main()
