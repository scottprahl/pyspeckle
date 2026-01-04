"""
This file is intended to be the target of a pytest run.

It will recursively find all .ipynb files in the current directory, ignoring
directories that start with . and any files matching patterns found in the file
.testignore
"""

import os.path
import pathlib

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

# Default search path is the current directory
searchpath = pathlib.Path("docs/")  # all notebooks are in here

# Read patterns from .testignore file
ignores = ""
if os.path.exists(".testignore"):
    # Context manager + encoding + meaningful name → fixes pylint W0621, W1514, R1732
    with open(".testignore", "r", encoding="utf-8") as ignore_file:
        ignores = [line.strip() for line in ignore_file if line.strip()]

# Ignore hidden folders (startswith('.')) and files matching ignore patterns
notebooks = [
    notebook
    for notebook in searchpath.glob("*.ipynb")
    if not (
        any(parent.startswith(".") for parent in notebook.parent.parts)
        or any(notebook.match(pattern) for pattern in ignores)
    )
]

notebooks.sort()
ids = [str(n) for n in notebooks]


@pytest.mark.parametrize("notebook", notebooks, ids=ids)
def test_run_notebook(notebook):
    """
    Read and execute notebook.

    The method here is directly from the nbconvert docs.
    """
    # Explicit encoding + descriptive name → fixes W1514 and W0621
    with open(notebook, "r", encoding="utf-8") as nb_file:
        nb = nbformat.read(nb_file, as_version=4)

    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": notebook.parent}})
