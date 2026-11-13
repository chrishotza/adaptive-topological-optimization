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

### Historical CSV path repair (opt-in)

The archived maintenance tool is **read-only by default**. It previews proposed edits to
simple `pd.read_csv("file.csv")` calls in `source/python/`; it does not run experiments
or modify any historical data. Review every patch before editing the historical scripts:

```bash
python scripts/fix_relative_paths.py --diff
python scripts/fix_relative_paths.py --check  # returns 1 when candidate edits remain
python scripts/fix_relative_paths.py --apply  # explicit write; review the diff first
```

Changes are syntax-checked, affect only actual Python calls (not comments/docstrings),
and are covered by synthetic offline unit tests. This does not guarantee that every
historical script is runnable or that its datasets are available.

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
