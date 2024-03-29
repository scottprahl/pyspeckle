SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	open docs/_build/index.html

lite:
	jupyter lite build --XeusPythonEnv.packages=numpy,matplotlib,scipy,pyspeckle

lintcheck:
	-pylint pyspeckle/pyspeckle.py
	-pylint pyspeckle/__init__.py

yamlcheck:
	-yamllint .github/workflows/pypi.yaml
	-yamllint .github/workflows/test.yaml
	-yamllint .github/workflows/citation.yaml

doccheck:
	-pydocstyle pyspeckle/pyspeckle.py
	-pydocstyle pyspeckle/__init__.py

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck docs/changelog.rst
	-rstcheck --ignore-directives automodapi docs/pyspeckle.rst

rcheck:
	make test
	make rstcheck
	make lintcheck
	make doccheck
	make html
	flake8 .
	check-manifest
	pyroma -d .

test:
	make clean
	pytest tests/test_basics.py
	pytest tests/test_all_notebooks.py

clean:
	rm -rf __pycache__
	rm -rf _output
	rm -rf .tox
	rm -rf .ipynb_checkpoints
	rm -rf .jupyterlite.doit.db
	rm -rf .pytest_cache
	rm -rf .virtual_documents
	rm -rf build
	rm -rf dist
	rm -rf docs/_build
	rm -rf docs/_contents
	rm -rf docs/_output
	rm -rf docs/api
	rm -rf docs/.ipynb_checkpoints
	rm -rf docs/.jupyterlite.doit.db
	rm -rf tests/__pycache__
	rm -rf pyspeckle.egg-info
	rm -rf pyspeckle/__pycache__
	rm -rf pyspeckle/*.pyc
	

.PHONY: clean check rcheck html doccheck lintcheck rstcheck