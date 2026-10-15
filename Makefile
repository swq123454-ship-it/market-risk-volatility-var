.PHONY: help install pipeline fast docs test lint serve clean

PYTHON ?= python

help:
	@echo "install     install dependencies"
	@echo "pipeline    run the full study end to end (figures, tables, RESULTS.md)"
	@echo "fast        same deliverables on a reduced budget, for a quick check"
	@echo "docs        regenerate README.md and MODEL_DOCUMENTATION.md from metrics.json"
	@echo "test        run the test suite"
	@echo "lint        run ruff"
	@echo "serve       start the risk API on :8000"
	@echo "clean       remove generated artefacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

pipeline:
	$(PYTHON) scripts/run_pipeline.py

fast:
	$(PYTHON) scripts/run_pipeline.py --fast

# The documents are rendered from the run, never edited. Run this after the pipeline
# and commit the result: a README whose numbers were typed once and have not moved
# since is the most confident wrong thing in a repository.
docs:
	$(PYTHON) scripts/build_docs.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src scripts tests

serve:
	$(PYTHON) scripts/serve.py

clean:
	rm -f outputs/figures/*.png outputs/tables/*.csv outputs/metrics.json
	rm -f outputs/RESULTS.md outputs/generator_manifest.json
	rm -rf data/cache .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
