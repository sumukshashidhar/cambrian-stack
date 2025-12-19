"""Sphinx configuration for Cambrian Stack docs."""
import os
import sys
from datetime import datetime

# Ensure project root is on path (only if needed for autodoc later)
sys.path.insert(0, os.path.abspath(".."))

project = "Cambrian Stack"
author = "Sumuk Shashidhar"
year = datetime.utcnow().year
copyright = f"{year}, {author}"

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
