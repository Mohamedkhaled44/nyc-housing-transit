"""Vercel serverless entry point.

Vercel's Python runtime looks for a module-level WSGI callable named `app`.
Dash wraps a Flask server, which is exactly that, so we re-export it.

The project root is prepended to sys.path because this file lives in api/ but
app.py sits one level up.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import server as app  # noqa: E402  (path set up above)
