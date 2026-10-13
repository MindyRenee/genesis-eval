"""Shared test setup for the Genesis eval suite.

Lives outside the Genesis checkout — GENESIS_DIR points at the repo
(default: a sibling checkout named "genesis"); the eval root is added
so the flat harness modules (arc_adapter, world_play) import cleanly.
"""

import os
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent
GENESIS_DIR = Path(os.environ.get("GENESIS_DIR", EVAL_ROOT.parent / "genesis"))

sys.path.insert(0, str(GENESIS_DIR / "python"))
sys.path.insert(0, str(EVAL_ROOT))
