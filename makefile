.DEFAULT_GOAL := help

PYTHON ?= python3
VERSION ?=
BUILD_DIR ?= build
RELEASE_KIND := $(filter major minor patch,$(MAKECMDGOALS))

.PHONY: help editable install test quality typecheck format check docs build package-check release-check tag release major minor patch clean

help:
	@echo "BornSim development commands"
	@echo "  make editable             Install tests, development and docs dependencies"
	@echo "  make test                 Run tests with coverage"
	@echo "  make quality              Run Ruff and scoped Mypy checks"
	@echo "  make format               Format Python sources"
	@echo "  make check                Run quality and tests"
	@echo "  make docs                 Build Sphinx documentation with warnings as errors"
	@echo "  make build                Build source and wheel distributions"
	@echo "  make package-check        Build and validate package artifacts"
	@echo "  make release-check        Check release metadata consistency"
	@echo "  make tag VERSION=vX.Y.Z   Create a local release commit and annotated tag"
	@echo "  make release patch        Create and push the next release (minor/major also supported)"

editable:
	$(PYTHON) -m pip install -e ".[testing,dev,documentation]"

install:
	$(PYTHON) -m pip install .

typecheck:
	$(PYTHON) -m mypy

quality: typecheck
	$(PYTHON) -m ruff check bornsim tests tools docs/examples
	$(PYTHON) -m ruff format --check bornsim tests tools docs/examples

format:
	$(PYTHON) -m ruff format bornsim tests tools docs/examples

test:
	$(PYTHON) -m pytest --config-file=pytest.ini

check: quality test

docs:
	$(PYTHON) -m sphinx -W --keep-going -b html docs/source docs/build/html

build:
	$(PYTHON) -m build

package-check: build
	$(PYTHON) -m twine check dist/*
	$(PYTHON) tools/check_wheel.py dist

release-check:
	$(PYTHON) tools/check_release.py $(if $(VERSION),--version $(VERSION),)

tag:
	$(PYTHON) tools/release_tag.py "$(VERSION)"

release:
	@test "$(words $(RELEASE_KIND))" -eq 1 || { echo "usage: make release [patch|minor|major]" >&2; exit 2; }
	@set -eu; release_tag="$$($(PYTHON) tools/next_release_version.py $(RELEASE_KIND))"; \
	$(PYTHON) tools/release_tag.py "$$release_tag"; \
	git push origin HEAD "refs/tags/$$release_tag"

major minor patch:
	@:

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('build', 'dist', 'docs/build', 'htmlcov', '.pytest_cache', '.mypy_cache', '.ruff_cache')]"
