# ==============================================================================
# bloc_reloc_ml_analysis_v11.py
# BLOC-RELOC v11.0: Análisis Estadístico Profundo y Meta-Aprendizaje (ML)
# Propietario: Laboratorio TCF
# ==============================================================================

import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    filename = "meta_statistical_dataset_v10.csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"[!] No se encontró el archivo {filename}. Ejecute primero el pipeline v10.")
        return

    print(f"[*] Dataset cargado correctamente: {df.shape[0]} instancias, {df.shape[1]} columnas.\n")

    # 1. Análisis de Magnitud de Diferencia (Delta Cut)
    print("==================== 1. ANÁLISIS DE MAGNITUD (DELTA CUT) ====================")
    df['Delta_Cut'] = np.abs(df['Cut_frontier'] - df['Cut_cohesion'])
    
    summary_delta = df.groupby('Family')['Delta_Cut'].agg(['mean', 'median', 'std', 'min', 'max'])
    print(summary_delta)
    print("===========================================================================\n")

    # 2. Preparación de Features para Machine Learning
    feature_cols = ['N', 'E', 'clustering', 'deg_mean', 'deg_std', 'deg_het', 'density', 'assortativity']
    X = df[feature_cols]
    y = (df['Optimal_Operator'] == 'cohesion').astype(int) # 1 si gana cohesion, 0 si gana frontier

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Modelado de Machine Learning (Validación Cruzada)
    print("==================== 2. EVALUACIÓN DE MODELOS DE ML (CV) ====================")
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        print(f"[*] {name:20s} -> Accuracy Promedio: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print("===========================================================================\n")

    # 4. Importancia de Variables (usando Random Forest)
    print("==================== 3. IMPORTANCIA DE VARIABLES (RANDOM FOREST) =============")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances)
    print("===========================================================================\n")

    # 5. Análisis de Precisión y Matriz de Confusión
    print("==================== 4. RENDIMIENTO DETALLADO ===============================")
    y_pred = rf.predict(X_scaled)
    print("Matriz de Confusión (0 = Frontier, 1 = Cohesion):")
    print(confusion_matrix(y, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y, y_pred, target_names=['Frontier', 'Cohesion']))
    print("===========================================================================\n")

    print("[✓] Análisis v11 completado con éxito.")

if __name__ == "__main__":
    main()
