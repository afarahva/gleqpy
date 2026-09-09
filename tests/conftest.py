"""Ensure the tests directory is importable so `import _helpers` works
regardless of the directory pytest is invoked from."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
