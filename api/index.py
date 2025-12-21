"""
Vercel entrypoint for FastAPI application
"""

import sys
import os

# Add parent directory to path to import api_fastapi
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_fastapi import app

# Export the app for Vercel
__all__ = ['app']
