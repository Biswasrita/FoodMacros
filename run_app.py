"""Launcher for the FoodMacros Vision AI Web Application.

Usage:
    python run_app.py [--port 5000]
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from app.web_app import main

if __name__ == "__main__":
    main()

