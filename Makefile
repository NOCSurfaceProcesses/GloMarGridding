.PHONY: clean clean-build clean-pyc clean-test coverage docs help install lint test dev build
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-docs: ## remove docs artifacts
	rm -rf docs/_build
	rm docs/Documentation.{tex,pdf}

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr .pytest_cache
	rm -fr htmlcov/

install: ## install the library
	python -m pip install .

install-docs: ## install library and dependencies needed for building the docs
	python -m pip install . --quiet --group docs

install-test: ## install library and dependencies needed for standard testing
	python -m pip install . --quiet --group test

install-lint: ## install dependencies needed for linting
	python -m pip install --quiet --group lint

lint: install-lint ## check style
	python -m ruff check
	python -m numpydoc lint glomar_gridding/**.py
	codespell

test: clean-test install-test ## run tests quickly with the default Python
	python -m pytest

coverage: install-test ## check code coverage quickly with the default Python
	python -m coverage run --source glomar_gridding -m pytest
	python -m coverage report -m
	python -m coverage html

docs: clean-docs install-docs ## generate Sphinx PDF documentation
	sphinx-build -M latex ./docs ./docs/_build
	make -C docs/_build/latex
	make -C docs/_build/latex
	cp docs/_build/latex/glomargridding.tex docs/Documentation.tex
	cp docs/_build/latex/glomargridding.pdf docs/Documentation.pdf

dev: clean ## install the package to the active Python's site-packages for development
	python -m pip install --editable . --group dev
	pre-commit install

build: clean dev ## builds source and wheel package
	python -m pip install flit
	python -m flit build
