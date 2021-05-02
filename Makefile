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
	-rstcheck --ignore-directives automodule docs/pyspeckle.rst

notecheck:
	make clean
	pytest --verbose -n 2 test_all_notebooks.py
	rm -rf __pycache__

rcheck:
	make notecheck
	make rstcheck
	make lintcheck
	make doccheck
	make html
	check-manifest
	pyroma -d .
#	tox

clean:
	rm -rf dist
	rm -rf pyspeckle.egg-info
	rm -rf pyspeckle/__pycache__
	rm -rf docs/_build
	rm -rf docs/api
	rm -rf .tox
	rm -rf __pycache__

.PHONY: clean check rcheck html doccheck lintcheck rstcheck