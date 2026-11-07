"""lsb-vision-pipeline scripts.

The modules in this package import one another by bare module name
(e.g. ``import crop_petris_yolo_seg as cp``). That resolves automatically when a
script is run directly (``python scripts/foo.py`` puts ``scripts/`` on
``sys.path``), but not when the package is imported (``python -m scripts.foo``).
Adding this directory to ``sys.path`` here makes both invocation styles work.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
