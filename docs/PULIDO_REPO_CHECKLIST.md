# ✅ CHECKLIST: Pulir repo para publicación en arXiv

## PASO 1: Crear requirements.txt
```bash
cd /ruta/a/tu/repo
pip freeze > requirements.txt
```
**Output esperado:** Archivo `requirements.txt` con todas las librerías y versiones.

---

## PASO 2: Arreglar rutas relativas en scripts
```bash
python fix_relative_paths.py
```
**Qué hace:** Modifica todos los scripts en `source/python/` para usar rutas relativas con `os.path.join()` en lugar de rutas hardcodeadas.

**Verificación:** Abre un script cualquiera (ej: `experiment_phase3_triclass_meta_selector.py`) y verifica que diga:
```python
import os
...
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'evidence', 'csv', 'meta_dataset_v14_adimensional.csv'))
```

---

## PASO 3: Generar las 3 figuras
```bash
python generate_figures.py
```
**Output esperado:** 
```
figures/figure1_regime_space.png
figures/figure2_performance_comparison.png
figures/figure3_confusion_matrix.png
```

**Verificación:** Abre las 3 imágenes en tu carpeta `figures/`. Deben verse claras y con resolución 300 DPI.

---

## PASO 4: Crear un notebook Colab (OPCIONAL pero recomendado)
```python
# Cell 1
!git clone https://github.com/chrishotza/adaptive-topological-optimization.git
%cd adaptive-topological-optimization

# Cell 2
!pip install -r requirements.txt

# Cell 3
import pandas as pd
df = pd.read_csv('evidence/csv/sota_benchmark_summary_ranking.csv')
print(df.head())
# Verificar que los datos cargan sin errores

# Cell 4
!python generate_figures.py
```

Esto permite que cualquiera reproduzca los experimentos sin instalar nada localmente.

---

## PASO 5: Actualizar README.md
Agregar estas secciones:

### 📄 Paper
```markdown
## 📄 Paper
Read the full paper: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

### Abstract
Graph partitioning research has pursued a "universal algorithm" for 40 years. We demonstrate this pursuit is futile. We show that graphs exist in topological regimes, and different algorithms dominate in different regimes. We formalize this via the Topological Vector Θ(G) and introduce ATOF, an adaptive meta-framework.
```

### 🚀 Quick Start
```markdown
## 🚀 Quick Start

### Reproducibility
```bash
# Install dependencies
pip install -r requirements.txt

# Run final benchmark
python source/python/run_phase11_sota_benchmark.py

# Generate paper figures
python generate_figures.py
```

### Expected Output
- `evidence/csv/sota_benchmark_summary_ranking.csv` (if running benchmarks)
- `figures/figure1_regime_space.png`
- `figures/figure2_performance_comparison.png`
- `figures/figure3_confusion_matrix.png`
```

### 📊 Key Figures
```markdown
## 📊 Key Figures

![Topological Regime Space](figures/figure1_regime_space.png)
*Figure 1: Distinct clusters in Θ-space correspond to different optimal algorithms.*

![Performance Comparison](figures/figure2_performance_comparison.png)
*Figure 2: BLOC-RELOC achieves lowest mean cut with narrow variance.*

![Meta-Selector Confusion Matrix](figures/figure3_confusion_matrix.png)
*Figure 3: Meta-selector achieves 79.85% accuracy in regime classification.*
```

### 📋 Citation
```markdown
## 📋 Citation
If you use ATOF or this research, please cite:
```bibtex
@preprint{chrishotza2024atof,
  title={ATOF: A Meta-Framework for Topology-Aware Graph Partitioning via Topological Regime Detection},
  author={Chrishotza, [Your Name]},
  year={2024},
  note={Available at https://arxiv.org/abs/XXXX.XXXXX}
}
```
```

---

## PASO 6: Subir a arXiv
1. Ir a https://arxiv.org/user/login
2. Crear cuenta (si no tienes)
3. Click "Submit"
4. Seleccionar categoría: **cs.DS** (Data Structures and Algorithms) o **cs.LG** (Machine Learning)
5. Subir PDF del paper + código fuente como `.tar.gz`
6. Esperar 24-48 horas para aprobación
7. Recibir arXiv ID (ej: 2401.12345)

**Importante:** En la sección "License" selecciona "CC-BY 4.0" para permitir reutilización

---

## PASO 7: Actualizar repo con link a arXiv
Una vez que tengas el arXiv ID, cambiar el README:

```markdown
# ATOF: Adaptive Topological Optimization Framework

**Paper:** [arXiv:2024.XXXXX](https://arxiv.org/abs/2024.XXXXX)

```

---

## ✅ CHECKLIST FINAL

- [ ] `requirements.txt` creado
- [ ] Rutas relativas en scripts (ejecutar `fix_relative_paths.py`)
- [ ] 3 figuras generadas en `figures/` (ejecutar `generate_figures.py`)
- [ ] README actualizado con instrucciones y figuras
- [ ] Paper PDF listo
- [ ] Código fuente compilado en `.tar.gz`
- [ ] Subido a arXiv
- [ ] README actualizado con arXiv ID
- [ ] Anuncio en Twitter/LinkedIn

---

## 📝 TIEMPO ESTIMADO
- Paso 1: 2 minutos
- Paso 2: 2 minutos
- Paso 3: 3 minutos
- Paso 4: 10 minutos (opcional)
- Paso 5: 10 minutos
- Paso 6: 5 minutos (+ 24-48 horas de espera)
- Paso 7: 2 minutos

**Total: ~35 minutos (excluyendo espera arXiv)**

---

## 🆘 Si algo falla

**Error: "FileNotFoundError: No such file or directory"**
→ Ejecutar scripts desde repo root: `cd /path/to/adaptive-topological-optimization && python fix_relative_paths.py`

**Error: "ModuleNotFoundError: No module named 'pandas'"**
→ Instalar requisitos: `pip install -r requirements.txt`

**Error: Matplotlib backend**
→ Si no ves figuras, ejecutar: `import matplotlib; matplotlib.use('Agg')` al inicio de `generate_figures.py`

---

**¿Preguntas? Todas las herramientas están en `/mnt/user-data/outputs/`**
