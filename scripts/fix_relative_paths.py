#!/usr/bin/env python3
import os
import re
from pathlib import Path

SOURCE_DIR = Path("source/python")
print(f"[*] Procesando scripts en: {SOURCE_DIR.absolute()}")

PATTERN = r"pd\.read_csv\(['\"]([^'\"]*\.csv)['\"]\)"

def replacement_func(match, filename):
    csv_name = match.group(1)
    if 'evidence' in csv_name:
        return match.group(0)
    return f"pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'evidence', 'csv', '{csv_name}'))"

def ensure_os_import(content):
    if "import os" in content:
        return content
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            lines.insert(i, 'import os')
            return '\n'.join(lines)
    lines.insert(0, 'import os')
    return '\n'.join(lines)

if SOURCE_DIR.exists():
    py_files = sorted(SOURCE_DIR.glob("*.py"))
    modified_count = 0
    for py_file in py_files:
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            original = f.read()
        modified = ensure_os_import(original)
        modified = re.sub(PATTERN, lambda m: replacement_func(m, py_file.name), modified)
        if modified != original:
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(modified)
            modified_count += 1
    print(f"[✓] Procesamiento completado: {modified_count} archivos modificados")
else:
    print("[!] Directorio source/python no encontrado")
