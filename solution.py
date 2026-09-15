# -*- coding: utf-8 -*-
"""solution.py -- entry point for the legacy engine mode.

The platform instantiates ``Solution(work_dir)`` and then calls ``attack(...)``.
The heavy lifting lives in the sibling module wm_attack.py, so this file only
adapts the platform contract.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from wm_attack import run_attack  # noqa: E402


class Solution(object):
    """Standard solution class expected by the legacy engine."""

    def __init__(self, work_dir=None):
        self.work_dir = work_dir

    def attack(self, *args, **kwargs):
        return run_attack(*args, **kwargs)
