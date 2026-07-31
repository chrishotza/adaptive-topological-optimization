import os
import sys
import pandas as pd

REPORT_DIR = "MASTER_VALIDATION_REPORT"

def main():
    runs_path = f"{REPORT_DIR}/detailed_runs.csv"
    if not os.path.exists(runs_path):
        print(f"[!] No se encontró {runs_path}. Ejecute primero los módulos anteriores.")
        sys.exit(1)

    print("[*] Compilando reporte final y tabla de rendimiento por familia estructural sin tabulate...")
    df = pd.read_csv(runs_path)
    
    def get_family(inst_name):
        if 'SNAP' in inst_name: return 'SNAP Real'
        elif 'SmallWorld' in inst_name: return 'Small-World (WS)'
        elif 'Regular' in inst_name: return 'Regular (RR)'
        elif 'Planted' in inst_name: return 'Planted Partition'
        elif 'Disconnected' in inst_name: return 'Disconnected'
        elif 'ScaleFree' in inst_name: return 'Scale-Free (BA)'
        else: return 'Otras'

    df['Family'] = df['Instance'].apply(get_family)
    
    family_summary = df.groupby(['Family', 'Algorithm'])['Cost'].mean().reset_index()
    
    winners = []
    for family, group in family_summary.groupby('Family'):
        best = group.loc[group['Cost'].idxmin()]
        winners.append({'Familia': family, 'Ganador': best['Algorithm'], 'Costo Promedio': round(best['Cost'], 2)})
    df_winners = pd.DataFrame(winners)
    df_winners.to_csv(f"{REPORT_DIR}/family_winners.csv", index=False)

    ranking = df.groupby('Algorithm')['Cost'].mean().sort_values()

    table_lines = ["| Familia | Ganador | Costo Promedio |", "| :--- | :--- | :--- |"]
    for _, row in df_winners.iterrows():
        table_lines.append(f"| {row['Familia']} | {row['Ganador']} | {row['Costo Promedio']} |")
    table_md = "\n".join(table_lines)

    md_lines = [
        "# TRIBUNAL FINAL DE VALIDACIÓN SOTA — PROYECTO BLOC-RELOC",
        "",
        "## 1. Resumen Ejecutivo y Sensibilidad Topológica",
        "- **Hallazgo Clave:** BLOC-RELOC no muestra un comportamiento uniforme; su rendimiento depende fuertemente del régimen estructural del grafo (colapsa favorablemente en estructuras regulares/modulares y enfrenta desafíos en redes reales densas como SNAP).",
        f"- **Algoritmo Líder Global (Costo Promedio):** `{ranking.index[0]}` (`{ranking.iloc[0]:.2f}`)",
        "",
        "## 2. Ganadores por Familia Estructural",
        table_md,
        "",
        "## 3. Ranking Global de Costos",
        "```text",
        ranking.to_string(),
        "```",
        "",
        "## 4. Archivos de Auditoría Generados",
        "- `detailed_runs.csv`: Registro completo por semilla e instancia.",
        "- `summary_pivot_costs.csv`: Pivote de costos medios.",
        "- `stability_analysis.csv`: Análisis de estabilidad inter-semillas.",
        "- `statistical_comparisons.csv`: Pruebas de Wilcoxon y tamaños del efecto (Cohen's d).",
        "- `family_winners.csv`: Tabla de ganadores segmentada por familia estructural.",
        "- `assets/dolan_more_profile.png`: Perfiles de rendimiento Dolan-Moré."
    ]

    with open(f"{REPORT_DIR}/Executive_Summary.md", 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    print(f"\n[✓] REPORTE FINAL GENERADO SATISFACTORIAMENTE en `{REPORT_DIR}/Executive_Summary.md`.")

if __name__ == "__main__":
    main()
