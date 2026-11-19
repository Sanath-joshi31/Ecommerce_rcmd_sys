"""Pytest configuration and test session initialization."""

import os
import sys
from pathlib import Path

# Ensure project root is at the head of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment flags
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TESTING"] = "1"
