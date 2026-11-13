# Adaptive Topological Optimization Framework (ATOF)

> **Historical research snapshot / Archivo de investigación.** This repository preserves the earlier 11-phase exploratory graph-optimization research and benchmarks. For the current, Python-facing graph-partitioning toolkit with tests and CI, visit **[ATOF (current framework)](https://github.com/chrishotza/Adaptiv-Topologica-Optimization-Framework)**. Historical numbers below are study-specific, not universal algorithm performance guarantees.

> **Archivo histórico:** este repositorio conserva los experimentos y resultados previos. El framework más reciente, con interfaz para Python y pruebas, está en el [ATOF actual](https://github.com/chrishotza/Adaptiv-Topologica-Optimization-Framework).


## 🚀 Quick Start

```bash
# Lightweight dependencies for re-generating the archived figures
python -m pip install -r requirements-figures.txt

# Works from any working directory
python scripts/generate_figures.py
```

The full `requirements.txt` is a frozen historical notebook environment, **not** a minimal runtime dependency list. Re-running the phase-11 benchmark is a separate, potentially long experimental task that may require native solver packages and environment-specific setup. Figure 2 reads the checked-in CSV and does not invent error bars when standard deviations for `Mean_Cut` are unavailable.

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Max improvement vs SOTA | **10.4×** (Planted Partition) |
| Meta-selector accuracy | **79.85%** |
| Speed vs Breakout | **17× faster** |
| Statistical significance | p < 0.05 vs 7/9 algorithms |
| Families dominated | 5/6 (all except SNAP Real) |

## 📄 Paper & Citation

The original project history describes an arXiv submission, but this README does **not** provide a public arXiv identifier or independently verifiable paper link. The BibTeX entry below is a **provisional manuscript reference**, not a confirmed published preprint. Verify publication status before citing.

```bibtex
@preprint{chrishotza2025atof,
  title={ATOF: A Meta-Framework for Topology-Aware Graph Partitioning via Topological Regime Detection},
  author={Chrishotza},
  year={2025}
}
```

## 📁 Repository Structure

- /source/python/ - Core algorithms and experiments
- /evidence/csv/ - Experimental data, benchmarks, meta-datasets
- /reports/ - Executive summary and statistical comparisons
- /docs/ - Complete research history (11 phases)
- /figures/ - Paper figures
- /scripts/ - Reproducibility and maintenance tools

## 🔬 The Breakthrough

**Traditional view:** Find the single best algorithm for all graphs.

**Our finding:** Different graphs belong to different **topological regimes**. Each regime requires a different strategy. We formalize this via the **Topological Vector Θ(G)** and build ATOF, a meta-framework that predicts the optimal algorithm based on Θ(G).

Repositorio técnico del framework de optimización topológica.

Contiene:

- Código fuente
- Evidencia experimental
- Reportes estadísticos
- Pipelines de validación
- Implementaciones BLOC-RELOC

Estructura:

docs/
source/
evidence/
reports/
benchmarks/

Estado:
Snapshot congelado para auditoría técnica.
