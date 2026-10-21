"""Put the repo root and per-dataset eval dirs on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent


def ensure_eval_paths() -> Path:
    for path in (
        REPO_ROOT,
        EVAL_ROOT / "dl3dv",
        EVAL_ROOT / "re10k",
        EVAL_ROOT / "acid",
    ):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return REPO_ROOT
