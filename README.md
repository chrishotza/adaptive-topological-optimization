# Adaptive Topological Optimization Framework (ATOF)


## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark (optional, ~2h)
python source/python/run_phase11_sota_benchmark.py

# Generate paper figures
python scripts/generate_figures.py
```

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Max improvement vs SOTA | **10.4×** (Planted Partition) |
| Meta-selector accuracy | **79.85%** |
| Speed vs Breakout | **17× faster** |
| Statistical significance | p < 0.05 vs 7/9 algorithms |
| Families dominated | 5/6 (all except SNAP Real) |

## 📄 Paper & Citation

Paper submitted to arXiv. Coming soon.

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
