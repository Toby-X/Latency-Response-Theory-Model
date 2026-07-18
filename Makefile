.PHONY: install test example

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

example:
	python examples/fit_synthetic.py
