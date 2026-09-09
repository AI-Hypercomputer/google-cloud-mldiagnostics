# Copyright 2025 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AST Presubmit Script to enforce ML framework import rules for google_cloud_mldiagnostics.

This script parses Python source files and verifies that heavy ML frameworks
(such as jax, torch, tensorflow, flax, vllm, etc.) are NOT imported at top-level
module scope in general codebase files.

Exceptions:
1. Framework-specific utility directories (e.g. `jax_utils/`, `torch_utils/`,
   `gpu_utils/`, `libtpu_utils/`) are permitted to import their dedicated ML
   framework.
2. Individual import statements can opt-out using `# lint:
disable=top-level-ml-import`
   or `# ml_diagnostics: allow-top-level-import`.
"""

import argparse
import ast
import os
import sys
from typing import Dict, List, Set, Tuple

# ML framework packages that must be imported lazily inside functions
ML_FRAMEWORKS: Set[str] = {
    "jax",
    "torch",
    "pytorch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "tf",
    "flax",
    "vllm",
    "optax",
    "libtpu",
    "pynvml",
    "triton",
    "tensorboard",
}

# Mapping of framework-specific utility directories to allowed frameworks
DIRECTORY_FRAMEWORK_ALLOWLIST: Dict[str, Set[str]] = {
    "jax_utils": {"jax"},
    "torch_utils": {"torch", "torchvision", "torchaudio", "pytorch"},
    "gpu_utils": {"pynvml"},
    "libtpu_utils": {"libtpu"},
}

OPT_OUT_COMMENT = "# lint: disable=top-level-ml-import"


def _is_opted_out_line(line_content: str) -> bool:
  return OPT_OUT_COMMENT in line_content



def _get_allowed_frameworks_for_file(filename: str) -> Set[str]:
  allowed = set()
  normalized_path = filename.replace("\\", "/")
  path_parts = normalized_path.split("/")
  for dir_name, frameworks in DIRECTORY_FRAMEWORK_ALLOWLIST.items():
    if dir_name in path_parts:
      allowed.update(frameworks)
  return allowed


class MlFrameworkImportVisitor(ast.NodeVisitor):
  """AST Visitor to inspect top-level imports of ML frameworks."""

  def __init__(self, filename: str, file_lines: List[str]):
    self.filename = filename
    self.file_lines = file_lines
    self.allowed_frameworks = _get_allowed_frameworks_for_file(filename)
    self.errors: List[Tuple[int, str]] = []
    self._function_depth = 0

  def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    self._function_depth += 1
    self.generic_visit(node)
    self._function_depth -= 1

  def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
    self._function_depth += 1
    self.generic_visit(node)
    self._function_depth -= 1

  def _is_line_disabled(self, lineno: int) -> bool:
    if 1 <= lineno <= len(self.file_lines):
      return _is_opted_out_line(self.file_lines[lineno - 1])
    return False

  def visit_Import(self, node: ast.Import) -> None:
    if self._function_depth == 0 and not self._is_line_disabled(node.lineno):
      for alias in node.names:
        base_module = alias.name.split(".")[0]
        if (
            base_module in ML_FRAMEWORKS
            and base_module not in self.allowed_frameworks
        ):
          self.errors.append((
              node.lineno,
              (
                  f"Top-level import of ML framework '{alias.name}' is prohibited.\n"
                  f"  -> Rationale: Eagerly importing ML frameworks at module scope pollutes the global import scope and causes initialization failures.\n"
                  f"  -> Fix: Move 'import {alias.name}' lazily inside the function/method where it is used, or add '# lint: disable=top-level-ml-import' if intentional."
              ),
          ))
    self.generic_visit(node)

  def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
    if (
        self._function_depth == 0
        and node.module
        and not self._is_line_disabled(node.lineno)
    ):
      base_module = node.module.split(".")[0]
      if (
          base_module in ML_FRAMEWORKS
          and base_module not in self.allowed_frameworks
      ):
        self.errors.append((
            node.lineno,
            (
                f"Top-level import from ML framework '{node.module}' is prohibited.\n"
                f"  -> Rationale: Eagerly importing ML frameworks at module scope pollutes the global import scope and causes initialization failures.\n"
                f"  -> Fix: Move 'from {node.module} import ...' lazily inside the function/method where it is used, or add '# lint: disable=top-level-ml-import' if intentional."
            ),
        ))
    self.generic_visit(node)



def check_file(filename: str) -> List[str]:
  """Parses a Python file and returns lint errors for top-level ML framework imports."""
  try:
    with open(filename, "r", encoding="utf-8") as f:
      content = f.read()
    lines = content.splitlines()
    tree = ast.parse(content, filename=filename)
  except Exception as e:
    return [f"{filename}: Unable to parse file: {e}"]

  visitor = MlFrameworkImportVisitor(filename=filename, file_lines=lines)
  visitor.visit(tree)
  return [f"{filename}:{lineno}: {msg}" for lineno, msg in visitor.errors]


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Enforce lazy ML framework imports in ml_diagnostics via AST parsing."
      )
  )
  parser.add_argument("files", nargs="+", help="Python source files to check")
  args = parser.parse_args()

  all_errors = []
  for filepath in args.files:
    if filepath.endswith(".py"):
      errors = check_file(filepath)
      all_errors.extend(errors)

  if all_errors:
    print(
        "=== ml_diagnostics ML Framework Import Violations ===", file=sys.stderr
    )
    for error in all_errors:
      print(error, file=sys.stderr)
    sys.exit(1)

  print("All ml_diagnostics ML framework import checks passed successfully.")
  sys.exit(0)


if __name__ == "__main__":
  main()
