"""Sphinx Gallery callbacks shared by documentation builds."""

from pathlib import Path


def gallery_arguments(gallery_conf, script_vars):
    """Construct interactive figures without opening browsers during builds."""
    return ["--no-browser"] if Path(script_vars["src_file"]).name == "random_medium.py" else []
