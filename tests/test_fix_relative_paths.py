"""Only synthetic files; no historical experiment sources are modified."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fix_relative_paths.py"
spec = importlib.util.spec_from_file_location("safe_relative_paths", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SafePathMigrationTests(unittest.TestCase):
    def test_preserves_docstring_future_import_and_valid_syntax(self):
        original = (
            '"""Synthetic module documentation."""\n'
            'from __future__ import annotations\n'
            'import pandas as pd\n'
            'frame = pd.read_csv("sample.csv")\n'
        )
        updated = module.rewrite_source(original)
        ast.parse(updated)
        self.assertLess(updated.index("from __future__"), updated.index("import os"))
        self.assertLess(updated.index("import os"), updated.index("frame ="))
        self.assertIn("'evidence', 'csv', 'sample.csv'", updated)
        self.assertEqual(module.rewrite_source(updated), updated)

    def test_no_rewrite_of_nested_or_dynamic_csv_paths(self):
        original = (
            'import pandas as pd\n'
            'data = pd.read_csv("../private/data.csv")\n'
            'data = pd.read_csv(path_from_user)\n'
        )
        self.assertEqual(module.rewrite_source(original), original)

    def test_dry_run_does_not_modify_file_and_apply_is_explicit(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "toy.py"
            original = 'import pandas as pd\nframe = pd.read_csv("toy.csv")\n'
            source.write_text(original, encoding="utf-8")
            proposed = module.plan_changes(folder)
            self.assertEqual(len(proposed), 1)
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            module.atomic_write(source, proposed[0][2])
            self.assertIn("import os", source.read_text(encoding="utf-8"))
            self.assertEqual(module.plan_changes(folder), [])

    def test_rejects_symlinked_input_source(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            target = folder / "reference.txt"
            target.write_text("synthetic", encoding="utf-8")
            alias = folder / "alias.py"
            try:
                alias.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink not available")
            with self.assertRaisesRegex(ValueError, "symlinked"):
                module.plan_changes(folder)
            self.assertEqual(target.read_text(encoding="utf-8"), "synthetic")

    def test_preserves_windows_line_endings_in_synthetic_source(self):
        original = (
            '"""example"""\r\n'
            'from __future__ import annotations\r\n'
            'import pandas as pd\r\n'
            'frame = pd.read_csv("test.csv")\r\n'
        )
        result = module.rewrite_source(original)
        self.assertNotIn("\n", result.replace("\r\n", ""))
        self.assertIn("import os\r\n", result)
        ast.parse(result)


if __name__ == "__main__":
    unittest.main()
