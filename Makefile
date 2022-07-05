SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	open docs/_build/index.html

lintcheck:
	-pylint pyspeckle/pyspeckle.py
	-pylint pyspeckle/__init__.py

doccheck:
	-pydocstyle pyspeckle/pyspeckle.py
	-pydocstyle pyspeckle/__init__.py

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck docs/changelog.rst
	-rstcheck --ignore-directives automodapi docs/pyspeckle.rst

notecheck:
	make clean
	pytest --verbose test_all_notebooks.py
	rm -rf __pycache__

rcheck:
	make notecheck
	make rstcheck
	make lintcheck
	make doccheck
	make html
	flake8 .
	check-manifest
	pyroma -d .
#	tox

clean:
	rm -rf __pycache__
	rm -rf .tox
	rm -rf .ipynb_checkpoints
	rm -rf build
	rm -rf dist
	rm -rf docs/_build
	rm -rf docs/api
	rm -rf docs/.ipynb_checkpoints
	rm -rf pyspeckle.egg-info
	rm -rf pyspeckle/__pycache__
	rm -rf pyspeckle/*.pyc
	

.PHONY: clean check rcheck html doccheck lintcheck rstcheck