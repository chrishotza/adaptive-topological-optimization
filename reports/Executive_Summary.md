# TRIBUNAL FINAL DE VALIDACIÓN SOTA — PROYECTO BLOC-RELOC

## 1. Resumen Ejecutivo y Sensibilidad Topológica
- **Hallazgo Clave:** BLOC-RELOC no muestra un comportamiento uniforme; su rendimiento depende fuertemente del régimen estructural del grafo (colapsa favorablemente en estructuras regulares/modulares y enfrenta desafíos en redes reales densas como SNAP).
- **Algoritmo Líder Global (Costo Promedio):** `BLOC_RELOC (Proposed)` (`1248.11`)

## 2. Ganadores por Familia Estructural
| Familia | Ganador | Costo Promedio |
| :--- | :--- | :--- |
| Disconnected | BLOC_RELOC (Proposed) | 100.0 |
| Planted Partition | BLOC_RELOC (Proposed) | 300.0 |
| Regular (RR) | BLOC_RELOC (Proposed) | 225.0 |
| SNAP Real | Breakout_Local_Search | 3069.2 |
| Scale-Free (BA) | BLOC_RELOC (Proposed) | 300.0 |
| Small-World (WS) | BLOC_RELOC (Proposed) | 300.0 |

## 3. Ranking Global de Costos
```text
Algorithm
BLOC_RELOC (Proposed)      1248.114286
Breakout_Local_Search      1840.139286
Kernighan_Lin              1840.139286
Tabu_Search               13154.175714
KaHyPar_SOTA              13436.315714
Greedy                    13436.315714
Simulated_Annealing       30215.141429
Genetic                   30718.000000
Random                    30893.437143
Spectral                 148219.064286
```

## 4. Archivos de Auditoría Generados
- `detailed_runs.csv`: Registro completo por semilla e instancia.
- `summary_pivot_costs.csv`: Pivote de costos medios.
- `stability_analysis.csv`: Análisis de estabilidad inter-semillas.
- `statistical_comparisons.csv`: Pruebas de Wilcoxon y tamaños del efecto (Cohen's d).
- `family_winners.csv`: Tabla de ganadores segmentada por familia estructural.
- `assets/dolan_more_profile.png`: Perfiles de rendimiento Dolan-Moré.