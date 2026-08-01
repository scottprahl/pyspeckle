Changelog
=========

Unreleased ()
-------------
* fix the Documentation URL, which pointed at a 404 (pyspeckle2.readthedocs.io)
* fix the Changelog URL, which pointed at docs/CHANGELOG.rst instead of the root
* declare MIT in pyproject.toml and CITATION.cff, matching LICENSE.txt
* refresh release.txt: CHANGELOG.rst not CHANGELOG.txt, and a version placeholder
* drop wheel from build-system requires; setuptools>=70.1 provides bdist_wheel
* drop the redundant setuptools pin from the dev extra
* bump checkout and setup-python to v7, and build with uv in every workflow
* test on macOS and Windows as well as Linux
* run the linters in CI through the same make lint target used locally
* fix local_contrast_2D variance normalization (contrast was low by sqrt(kernel sum))
* fix create_unpolarized_3D silently ignoring beta
* correct the alpha/beta docstrings, which described the anisotropy backwards
* replace matplotlib.cm.get_cmap, removed in matplotlib 3.11, with plt.get_cmap
* replace Colormap.set_bad, pending deprecation, with Colormap.with_extremes
* correct docstrings for local_contrast_2D, statistics_plot, zvalues, and tvalues
* cite the Duncan & Kirkpatrick copula equations in box_muller/zvalues/tvalues
* local_contrast_2D now returns only valid convolution pixels (shape change)
* statistics_plot now normalizes the PDF so that it integrates to unity
* local_contrast_2D_plot histograms are densities; they were labelled PDF but
  showed raw counts
* seed numpy before every test so CI no longer fails intermittently
* validate pix_per_speckle and image size in create_exponential_2D/create_exponential_3D
* support non-integer pix_per_speckle instead of raising TypeError from numpy
* give create_exponential_3D the polarization and shape checks the 2D version had
* _create_mask_3D now folds case, rejects unknown shapes, and checks its size
* add create_exponential_1D, true 1D speckle with exponential irradiance
* add create_unpolarized_1D so every dimension has both speckle forms
* export box_muller, zvalues, and tvalues, which were unreachable and undocumented
* split the correlated-sequence material into a new 0-Basics notebook
* rewrite the 1D notebook around fully and partially developed speckle
* reorganize the 2D notebook: apertures first, then fully and partially
  developed speckle, then local contrast
* fix 2D notebook headings that named the wrong pix_per_speckle
* stop calling correlated Gaussian noise "speckle" in the notebooks; the old
  text claimed speckle is Gaussian distributed, when it is exponential
* add create_phase_screen_1D and create_phase_screen_2D for partially
  developed speckle: correlated Gaussian phase, gaussian or exponential ACF
* add local_contrast_1D and local_contrast_3D
* add local_contrast_1D_plot, plotting points where the 2D version has images
* add statistics_plot_1D and rename statistics_plot to statistics_plot_2D (breaking)
* fix local_contrast for integer images: squaring a uint8 array wrapped and the
  local contrast came out zero everywhere
* test speckle contrast in every dimension: K=1 polarized, K=1/sqrt(2) unpolarized
* cover the plotting routines and mask branches, reaching 100% test coverage
* force the Agg backend and close figures between tests
* local_contrast_2D now shares one N-dimensional implementation with them
* replace create_exponential_1D/2D/3D with a single create_exponential taking a
  numpy-style shape, like np.ones (breaking)
* likewise create_unpolarized_1D/2D/3D become create_unpolarized (breaking)
* the aperture argument is now called `aperture`, freeing `shape` for the
  array shape (breaking)
* reject arguments that do not apply: alpha/aperture need 2D, beta needs 3D
* non-square shapes now work, e.g. create_exponential((100, 200), 2)
* the aperture masks moved to core as pyspeckle.core._create_mask*
* replace create_phase_screen_1D/2D with create_phase_screen taking a shape,
  which also makes 3D screens available (breaking)
* its autocorrelation argument is now `correlation`, freeing `shape` (breaking)
* replace local_contrast_1D/2D/3D with one local_contrast; it never needed a
  dimension because it infers one from the arrays (breaking)
* reorganize into core.py (generation and analysis), plots.py, and noise.py;
  speckle_1D.py, speckle_2D.py, and speckle_3D.py are gone
* rename create_gaussian_1D to create_gaussian_corr_1D (breaking)
* rename create_Rayleigh to create_unpolarized_2D (breaking)
* rename create_Rayleigh_3D to create_unpolarized_3D (breaking)
* split tests into test_core.py, test_1D.py, test_2D.py, and test_3D.py
* move the numpy seeding fixture into tests/conftest.py
* run the whole test suite in CI instead of only tests/test_basics.py
* rename create_exp_1D to create_exp_corr_1D (breaking)
* rename create_Exponential to create_exponential_2D (breaking)
* split pyspeckle.py into core.py, speckle_1D.py, speckle_2D.py, and speckle_3D.py
* pyspeckle.pyspeckle is gone; the private mask helpers now live at
  pyspeckle.speckle_2D and pyspeckle.speckle_3D
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
