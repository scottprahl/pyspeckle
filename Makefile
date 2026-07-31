PACKAGE         := pyspeckle
GITHUB_USER     := scottprahl

PY_VERSION      ?= 3.14
UV              ?= uv
RUN             := $(UV) run --extra dev
RUN_DOCS        := $(UV) run --extra docs
RUN_LITE        := $(UV) run --extra lite
RM              ?= rm -f
RMR             ?= rm -rf

DOCS_DIR        := docs
NOTEBOOKS       := $(DOCS_DIR)/*.ipynb
NB_TIMEOUT      ?= 600
HTML_DIR        := $(DOCS_DIR)/_build/html
OUT_ROOT        := _site
OUT_DIR         := $(OUT_ROOT)/$(PACKAGE)
STAGE_DIR       := .lite_src
DOIT_DB         := .jupyterlite.doit.db
LITE_CONFIG     := $(PACKAGE)/jupyter_lite_config.json

# --- GitHub Pages deploy config ---
PAGES_BRANCH    := gh-pages
WORKTREE        := .gh-pages
REMOTE          := origin

# --- server config (override on CLI if needed) ---
HOST            := 127.0.0.1
PORT            := 8000

PYTEST_OPTS     := -q
UNIT_TESTS      := tests --ignore=tests/test_all_notebooks.py
# terminal report only -- deliberately no --cov-report=html
COV_OPTS        := --cov=$(PACKAGE) --cov-report=term-missing
SPHINX_OPTS     := -T -E -b html -d $(DOCS_DIR)/_build/doctrees -D language=en

# Line length is 120 for ruff/pylint/black (set in pyproject.toml).
# yamllint reads no config here, so it keeps its own 80-column default.
PYLINT_TARGETS  := $(PACKAGE)/*.py tests/*.py .github/scripts/update_citation.py
BLACK_TARGETS   := $(PYLINT_TARGETS)
YAML_TARGETS    := .github/workflows/citation.yaml .github/workflows/pypi.yaml .github/workflows/test.yaml
RST_TARGETS     := README.rst CHANGELOG.rst $(DOCS_DIR)/index.rst $(DOCS_DIR)/changelog.rst
RST_AUTOMODAPI  := $(DOCS_DIR)/$(PACKAGE).rst

.PHONY: help
help:
	@echo "Build Targets:"
	@echo "  dist           - Build sdist+wheel locally"
	@echo "  html           - Build Sphinx HTML documentation"
	@echo "  lab            - Start jupyterlab"
	@echo "  readme         - Create images used in README"
	@echo "  update-notebooks - Re-execute notebooks and refresh committed outputs"
	@echo "  venv           - Install dependencies"
	@echo ""
	@echo "Testing"
	@echo "  test           - Run pytest on python files"
	@echo "  coverage       - Run pytest with a terminal coverage report"
	@echo "  note-test      - Test all notebooks for errors"
	@echo ""
	@echo "Lint/QA"
	@echo "  lint           - Run every linter (ruff, black, pylint, rst, yaml)"
	@echo "  rcheck         - Distribution release checks"
	@echo "  black-check    - Report black formatting drift"
	@echo "  manifest-check - Validate MANIFEST"
	@echo "  pylint-check   - Lint python files"
	@echo "  pyroma-check   - Validate overall packaging"
	@echo "  rst-check      - Validate all RST files"
	@echo "  ruff-check     - Lint all .py files"
	@echo "  yaml-check     - Validate YAML files"
	@echo ""
	@echo "JupyterLite Targets:"
	@echo "  lite           - Build JupyterLite site into $(OUT_DIR)"
	@echo "  lite-serve     - Serve $(OUT_DIR) at http://$(HOST):$(PORT)"
	@echo "  lite-deploy    - Upload to github"
	@echo ""
	@echo "Clean Targets:"
	@echo "  clean          - Remove build caches and docs output"
	@echo "  lite-clean     - Remove JupyterLite outputs"
	@echo "  realclean      - clean + remove .venv"

.PHONY: venv
venv:
	@$(UV) sync --python $(PY_VERSION) --extra dev --extra docs --extra lite

.PHONY: test
test:
	-$(RUN) pytest $(PYTEST_OPTS) $(UNIT_TESTS)

.PHONY: coverage
coverage:
	$(RUN) pytest $(PYTEST_OPTS) $(COV_OPTS) $(UNIT_TESTS)

.PHONY: note-test
note-test:
	$(RUN) pytest --verbose tests/test_all_notebooks.py

.PHONY: dist
dist:
	$(RUN) python -m build
	

.PHONY: html
html:
	@mkdir -p "$(HTML_DIR)"
	$(RUN_DOCS) sphinx-build $(SPHINX_OPTS) "$(DOCS_DIR)" "$(HTML_DIR)"
	@command -v open >/dev/null 2>&1 && open "$(HTML_DIR)/index.html" || true

.PHONY: readme
readme:
	cd "$(DOCS_DIR)/images" && $(RUN) python make_readme_images.py

# Sphinx renders the notebooks with nbsphinx_execute = "never", so the outputs
# committed in the .ipynb files are what readthedocs publishes.  Run this after
# any change that alters a printed value or a plot, then commit the result.
#
# record_timing=False keeps nbclient from stamping every cell with
# metadata.execution wall-clock times.  Those are the only thing that differs
# between two runs of unchanged notebooks, and they alone add ~70 changed lines
# per notebook.  Cell metadata is left otherwise intact, since some cells carry
# tags that nbsphinx acts on.
#
# The notebooks seed numpy in their first cell, so repeated runs regenerate
# byte-identical figures; without that seed every run would rewrite each
# embedded PNG.
#
# black runs first so the committed outputs belong to the formatted source.
# It rewrites cell source only, leaving outputs and execution counts alone,
# and it reads line-length from pyproject.toml like every other black run.
.PHONY: update-notebooks
update-notebooks:
	@echo "==> Formatting $(NOTEBOOKS) with black"
	$(RUN) black $(NOTEBOOKS)
	@echo "==> Executing $(NOTEBOOKS) in place"
	$(RUN) jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=$(NB_TIMEOUT) \
		--ExecutePreprocessor.record_timing=False $(NOTEBOOKS)
	@echo "✅ Outputs refreshed -- review with 'git diff --stat $(DOCS_DIR)'"

.PHONY: lint
lint:
	@echo "Running all linters..."
	@$(MAKE) ruff-check
	@$(MAKE) black-check
	@$(MAKE) pylint-check
	@$(MAKE) rst-check
	@$(MAKE) yaml-check
	@echo "✅ Lint complete"

.PHONY: pylint-check
pylint-check:
	@$(RUN) pylint $(PYLINT_TARGETS)

.PHONY: black-check
black-check:
	@$(RUN) black --check --diff $(BLACK_TARGETS)

.PHONY: yaml-check
yaml-check:
	@$(RUN) yamllint $(YAML_TARGETS)

.PHONY: rst-check
rst-check:
	@$(RUN) rstcheck $(RST_TARGETS)
	@$(RUN) rstcheck --ignore-directives automodapi $(RST_AUTOMODAPI)

.PHONY: ruff-check
ruff-check:
	$(RUN) ruff check

.PHONY: manifest-check
manifest-check:
	$(RUN) check-manifest

.PHONY: pyroma-check
pyroma-check:
	$(RUN) python -m pyroma -d .

.PHONY: rcheck
rcheck:
	@echo "Running all release checks..."
	@$(MAKE) realclean
	@$(MAKE) lint
	@$(MAKE) manifest-check
	@$(MAKE) pyroma-check
	@$(MAKE) html
	@$(MAKE) lite
	@$(MAKE) dist
	@$(MAKE) test
	@$(MAKE) note-test
	@echo "✅ Release checks complete"
	
.PHONY: lite
lite: lite-clean $(LITE_CONFIG) dist
	@echo "==> Staging notebooks from docs -> $(STAGE_DIR)"
	mkdir -p "$(STAGE_DIR)"
	/bin/cp docs/*.ipynb "$(STAGE_DIR)"
	$(RUN) jupyter nbconvert --clear-output --inplace "$(STAGE_DIR)"/*.ipynb
	/bin/mkdir -p "$(STAGE_DIR)/images"
	/bin/cp "docs/images/speckle.png" "$(STAGE_DIR)/images"; \

	@echo "==> Building JupyterLite"
	@$(RUN_LITE) jupyter lite build \
		--config="$(LITE_CONFIG)" \
		--contents="$(STAGE_DIR)" \
		--output-dir="$(OUT_DIR)"
	@touch "$(OUT_DIR)/.nojekyll"  # for github

.PHONY: lite-serve
lite-serve:
	@test -d "$(OUT_DIR)" || { echo "❌ run 'make lite' first"; exit 1; }
	@echo "Serving at"
	@echo "   http://$(HOST):$(PORT)/$(PACKAGE)/?disableCache=1"
	@echo ""
	$(RUN_LITE) python -m http.server -d "$(OUT_ROOT)" --bind $(HOST) $(PORT)

.PHONY: lite-deploy
lite-deploy: 
	@echo "==> Sanity check"
	@test -d "$(OUT_DIR)" || { echo "❌ Run 'make lite' first"; exit 1; }

	@echo "==> Ensure $(PAGES_BRANCH) branch exists"
	@if ! git show-ref --verify --quiet refs/heads/$(PAGES_BRANCH); then \
	  CURRENT=$$(git branch --show-current); \
	  git switch --orphan $(PAGES_BRANCH); \
	  git commit --allow-empty -m "Initialize $(PAGES_BRANCH)"; \
	  git switch $$CURRENT; \
	fi

	@echo "==> Setup deployment worktree"
	@git worktree remove "$(WORKTREE)" --force 2>/dev/null || true
	@git worktree prune || true
	@$(RMR) "$(WORKTREE)"
	@git worktree add "$(WORKTREE)" "$(PAGES_BRANCH)"
	@git -C "$(WORKTREE)" pull "$(REMOTE)" "$(PAGES_BRANCH)" 2>/dev/null || true

	@echo "==> Deploy $(OUT_DIR) -> $(WORKTREE)"
	@rsync -a --delete --exclude ".git*" "$(OUT_DIR)/" "$(WORKTREE)/"
	@touch "$(WORKTREE)/.nojekyll"
	@date -u +"%Y-%m-%d %H:%M:%S UTC" > "$(WORKTREE)/.pages-ping"

	@echo "==> Commit & push"
	@cd "$(WORKTREE)" && \
	  git add -A && \
	  if git diff --quiet --cached; then \
	    echo "✅ No changes to deploy"; \
	  else \
	    git commit -m "Deploy $$(date -u +'%Y-%m-%d %H:%M:%S UTC')" && \
	    git push "$(REMOTE)" "$(PAGES_BRANCH)" && \
	    echo "✅ Deployed to https://$(GITHUB_USER).github.io/$(PACKAGE)/"; \
	  fi

.PHONY: lab
lab:
	@echo "==> Launching JupyterLab"
	$(RUN) jupyter lab --ServerApp.root_dir="$(CURDIR)"

.PHONY: lite-clean
lite-clean:
	@echo "==> Cleaning JupyterLite build artifacts"
	@$(RMR) "$(STAGE_DIR)"
	@$(RMR) "$(OUT_ROOT)"
	@$(RMR) "$(DOIT_DB)"
	@$(RMR) .cache dist $(PACKAGE).egg-info

.PHONY: clean
clean: lite-clean
	@echo "==> Cleaning build artifacts"	
	@find . -name '__pycache__' -type d -exec $(RMR) {} +
	@find . -name '.DS_Store' -type f -exec $(RM) {} +
	@find . -name '.ipynb_checkpoints' -type d -prune -exec $(RMR) {} +
	@find . -name '.pytest_cache' -type d -prune -exec $(RMR) {} +
	$(RMR) .ruff_cache
	$(RM) .coverage
	$(RMR) docs/api
	$(RMR) docs/_build

.PHONY: realclean
realclean: clean
	@echo "==> Deep cleaning: removing venv and deployment worktree"
	@git worktree remove "$(WORKTREE)" --force 2>/dev/null || true
	@git worktree prune || true
	$(RMR) "$(WORKTREE)"
	$(RMR) .venv
	@$(RM) uv.lock
