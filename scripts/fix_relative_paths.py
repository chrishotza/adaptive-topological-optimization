#!/usr/bin/env python3
"""Safely preview/repair simple CSV path literals in historical Python scripts.

Default is read-only. Use --apply for an explicit, reversible source edit.
No benchmarks, figures, experiments or raw evidence are executed here.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import os
from pathlib import Path
import re
import stat
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "source" / "python"
SIMPLE_CSV = re.compile(
    r"""pd\.read_csv\(\s*(?P<quote>['"])(?P<name>[A-Za-z0-9_.-]+\.csv)(?P=quote)\s*\)"""
)


def rewrite_source(source: str) -> str:
    """Rewrite only literal CSV basenames; never evaluate source code."""
    def replace(match: re.Match[str]) -> str:
        filename = match.group("name")
        return (
            "pd.read_csv(os.path.join(os.path.dirname(__file__), "
            "'..', '..', 'evidence', 'csv', '" + filename + "'))"
        )

    result, replacements = SIMPLE_CSV.subn(replace, source)
    if not replacements:
        return source
    original_tree = ast.parse(source)
    imports_os = any(
        isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names)
        for node in original_tree.body
    )
    if not imports_os:
        # Future imports must precede ordinary imports. Module docstrings
        # must also remain at the beginning of the Python module.
        insertion = 0
        if (original_tree.body and isinstance(original_tree.body[0], ast.Expr)
                and isinstance(original_tree.body[0].value, ast.Constant)
                and isinstance(original_tree.body[0].value.value, str)):
            insertion = original_tree.body[0].end_lineno
        for node in original_tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                insertion = max(insertion, node.end_lineno)
        lines = result.splitlines(keepends=True)
        if insertion == 0:
            if lines and lines[0].startswith("#!"):
                insertion = 1
            if len(lines) > insertion and re.match(r"#.*coding[:=]", lines[insertion]):
                insertion += 1
        newline = "\r\n" if "\r\n" in source else "\n"
        lines.insert(insertion, "import os" + newline)
        result = "".join(lines)
    ast.parse(result)  # Never generate broken Python without warning.
    return result


def plan_changes(directory: Path) -> list[tuple[Path, str, str]]:
    """Scan without touching files; fail if a historical source is invalid."""
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Source directory missing or symlinked")
    planned: list[tuple[Path, str, str]] = []
    for path in sorted(directory.glob("*.py")):
        if path.is_symlink():
            raise ValueError("Refusing symlinked Python source")
        with path.open("r", encoding="utf-8", newline="") as handle:
            original = handle.read()
        updated = rewrite_source(original)
        if updated != original:
            planned.append((path, original, updated))
    return planned


def atomic_write(path: Path, content: str) -> None:
    """Retain the original mode and replace one opt-in source file atomically."""
    descriptor, temporary = tempfile.mkstemp(prefix=".relative-paths-", dir=path.parent)
    try:
        os.fchmod(descriptor, stat.S_IMODE(path.stat().st_mode))
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true",
                      help="Explicitly write reviewed changes (default: read-only)")
    mode.add_argument("--check", action="store_true",
                      help="Exit nonzero if candidate changes still exist (no writes)")
    parser.add_argument("--diff", action="store_true",
                        help="Show a unified diff of proposed changes")
    args = parser.parse_args(argv)
    try:
        changes = plan_changes(SOURCE_DIR)
    except (OSError, UnicodeError, ValueError, SyntaxError) as exc:
        parser.error("Cannot safely inspect source tree: " + str(exc))
    print("Candidate files:", len(changes))
    for path, before, after in changes:
        print(" -", path.relative_to(ROOT).as_posix())
        if args.diff:
            print("".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=str(path) + " (original)", tofile=str(path) + " (proposed)"
            )))
    if args.apply:
        for path, _, after in changes:
            atomic_write(path, after)
        print("Applied opt-in changes:", len(changes))
    elif changes:
        print("Preview only: no files modified; review with --diff before --apply.")
    return 1 if args.check and changes else 0


if __name__ == "__main__":
    raise SystemExit(main())
