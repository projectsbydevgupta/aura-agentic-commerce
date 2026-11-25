"""
Root ASGI entrypoint shim for Render Cloud & Uvicorn.
Delegates directly to `backend/server.py` so `uvicorn server:app` works out of the box.
"""

import os
import sys

_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from backend.server import app  # noqa: E402, F401
