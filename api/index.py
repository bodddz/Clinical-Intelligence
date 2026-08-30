# -*- coding: utf-8 -*-
"""
Vercel Serverless Function ASGI Entrypoint for Clinical RAG Platform.
Exposes FastAPI application `app` for Vercel Python Serverless Runtime.
"""

import os
import sys

# Ensure root directory and backend directory are in PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.main import app

# Vercel ASGI Application Handler
# The `app` object is automatically detected and served by Vercel
