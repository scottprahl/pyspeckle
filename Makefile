SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

check:
	-pylint pyspeckle/pyspeckle.py
	-pep257 pyspeckle/pyspeckle.py

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf dist
	rm -rf pyspeckle.egg-info
	rm -rf pyspeckle/__pycache__
	rm -rf docs/_build/*
	rm -rf docs/api/*
	rm -rf .tox

rcheck:
	make clean
	make html
	check-manifest
	pyroma -d .
#	tox

.PHONY: clean check rcheck html