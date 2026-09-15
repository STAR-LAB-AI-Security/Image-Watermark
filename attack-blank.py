# -*- coding: utf-8 -*-
"""attack-blank.py -- entry point for the managed 7z submission mode.

The archive root must contain this file.  The heavy lifting lives in the
sibling module wm_attack.py, so this file only adapts the platform contract.

    def attack(sample: dict) -> dict:
        sample["sample_id"]  # str
        sample["image"]      # RGB uint8 H x W x 3
        return {"image": ...}
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from wm_attack import run_attack  # noqa: E402


def attack(sample: dict) -> dict:
    """Remove the hidden watermark while staying inside the fidelity budget."""
    return run_attack(sample)
