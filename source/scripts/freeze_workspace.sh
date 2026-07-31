#!/usr/bin/env bash
set -e

echo "===================================================="
echo "  FREEZE 2 - SNAPSHOT COMPLETO ATOF/BLOC-RELOC"
echo "===================================================="

# Crear estructura destino
mkdir -p source
mkdir -p source/python
mkdir -p source/cpp
mkdir -p source/scripts
mkdir -p reports
mkdir -p docs

echo "[1/5] Copiando código Python..."
cp *.py source/python/ 2>/dev/null || true

echo "[2/5] Copiando C++..."
cp *.cpp source/cpp/ 2>/dev/null || true

echo "[3/5] Copiando scripts de ejecución..."
cp *.sh source/scripts/ 2>/dev/null || true

echo "[4/5] Copiando reportes maestros..."

cp MASTER_VALIDATION_REPORT/*.md reports/ 2>/dev/null || true
cp MASTER_VALIDATION_REPORT/*.csv reports/ 2>/dev/null || true

echo "[5/5] Generando inventario técnico..."

find source reports evidence \
-type f | sort > FULL_PROJECT_INVENTORY.txt


cat << 'README_EOF' > PROJECT_STATE.md
# Adaptive Topological Optimization Framework

## Snapshot State

Freeze 2 - Full Workspace Snapshot

Contenido:

- Código fuente Python
- Implementaciones C++
- Scripts de ejecución
- Evidencia experimental CSV
- Logs
- Reportes estadísticos

## Evidencia

La carpeta evidence contiene:

- benchmark results
- statistical comparisons
- stability analysis
- figures
- execution logs

## Estado

Snapshot congelado para migración a repositorio profesional.
README_EOF


echo ""
echo "===================================================="
echo " SNAPSHOT CREADO"
echo "===================================================="

git add source reports evidence PROJECT_STATE.md FULL_PROJECT_INVENTORY.txt

git commit -m "Freeze 2: complete source code and validation pipeline snapshot"

echo ""
echo "COMMIT FINALIZADO"
git log --oneline -2

