"""Pytest path bootstrap for repository-local tests.

The project is intentionally executable both as a repository checkout and as
an installed/module-based package. Tests need both the repository root (for
``industrial_inventory_env``) and ``training_pipelines`` (for legacy local
imports used by older utilities) on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_PIPELINES = _REPO_ROOT / "training_pipelines"

for path in (_REPO_ROOT, _TRAINING_PIPELINES):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
