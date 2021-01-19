SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

check:
	-pylint pyspeckle/pyspeckle.py
	-pydocstyle pyspeckle/pyspeckle.py
	-pylint pyspeckle/__init__.py
	-pydocstyle pyspeckle/__init__.py

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf dist
	rm -rf pyspeckle.egg-info
	rm -rf pyspeckle/__pycache__
	rm -rf docs/_build/*
	rm -rf docs/api/*
	rm -rf docs/_build/.buildinfo
	rm -rf docs/_build/.doctrees
	rm -rf .tox

rcheck:
	make clean
	touch docs/*ipynb
	touch docs/*rst
	make html
	check-manifest
	pyroma -d .
#	tox

.PHONY: clean check rcheck html