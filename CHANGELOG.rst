Changelog
=========

Unreleased ()
-------------
* fix local_contrast_2D variance normalization (contrast was low by sqrt(kernel sum))
* fix create_Rayleigh_3D silently ignoring beta
* correct the alpha/beta docstrings, which described the anisotropy backwards
* seed numpy in each notebook so re-execution reproduces identical figures
* add update-notebooks target to reformat and re-execute notebooks in place
* add coverage target with a terminal-only report
* add lint target that runs ruff, black, pylint, rstcheck, and yamllint
* add black-check target to report formatting drift
* switch Makefile workflows to uv run with dev/docs/lite extras
* remove venv/.ready bootstrap prerequisites from Makefile targets
* add shared Makefile command/file-list variables for consistency
* add RM/RMR Makefile variables and use them for cleanup commands
* require Python >=3.10 and declare support through Python 3.14
* test only Python 3.10 and 3.14 in GitHub Actions
* update default docs/publish automation runtimes to Python 3.14
* relax docs extras to Sphinx>=8.1.3 and nbsphinx>=0.9.7
* update JupyterLite extras to jupyterlite-core/pyodide-kernel >=0.7,<0.8
* add release dates to changelog version titles

0.6.1 (2026-01-05)
------------------
* __init__.py is only source of version
* pyproject.toml is only source of package requirements
* move pngs to docs/images
* create make_readme_images.py
* fix importing of speckle.png
* update .readthedocs.yaml
* update docs/conf.py
* move jupyter_lite_config.json to pyspeckle folder
* update github actions
* use a single source for versioning
* only test ipynb files in first level of docs/
* fix zenodo
* fix badges
* release to pypi upon publishing

0.6.0 (2025-11-16)
------------------
* jupyterlite support
* modernize packaging
* modernize github actions
* use pyproject.toml only
* improve docstrings
* use ruff for linting
* add requirements-dev.txt
* use venv for reproducibility

0.5.1 (2023-09-02)
------------------
* attempt to fix images on pypi.org

0.5.0 (2023-09-02)
------------------
* drop 'v' from version tags
* add testing
* add conda support
* add github automated testing
* improve CITATION.cff
* create zenodo DOI
* add github automated version updates

v0.4.1 (2021-08-07)
-------------------
* add slice_plot() for 3D speckle
* create pure python packaging
* include wheel file
* package as python3 only

v0.4.0 (2021-07-16)
-------------------
* 3D speckle generation
* flake8 testing
* add pypi badge
* automate notebook testing

v0.3.2 (2021-03-25)
-------------------
* add badges for colab and binder
* sphinx-book-theme for docs
* cite SPIE paper
* improve README.rst

v0.3.1 (2020-05-19)
-------------------
* improve api generation using automodapi

v0.3.0 (2020-05-18)
-------------------
* use sphinx for documentation
* implement create_gaussian_1D()
* improve documentation

v0.2.0 (2019-04-29)
-------------------
*  rename functions and doc files

v0.1.0 (2018-05-03)
-------------------
*  initial checkin
