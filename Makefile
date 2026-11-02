# Instinct+ shortcuts. `make help` lists them.
PY ?= .venv/bin/python
STREAMLIT ?= .venv/bin/streamlit

.PHONY: help setup demo data collect train evaluate test app clean-models

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /  -  /'

setup: ## create .venv and install requirements
	python3 -m venv .venv && .venv/bin/pip install -U pip wheel && .venv/bin/pip install -r requirements.txt

demo: data train ## synthetic dataset + trained models, no API keys needed

data: ## generate the synthetic dataset (refuses to overwrite collected data)
	$(PY) -m src.data.synthetic

collect: ## collect the real dataset from the YouTube Data API (needs YOUTUBE_API_KEY)
	$(PY) -m src.collection all

train: ## train the title and thumbnail models from data/processed/
	$(PY) -m src.modelling.train

evaluate: ## run the validation protocol and refresh docs/model_report.md
	$(PY) -m src.modelling.evaluate --out docs/model_report.md

test: ## run the test suite
	$(PY) -m pytest -q

app: ## run the Streamlit app
	$(STREAMLIT) run app/main.py

clean-models: ## delete trained artefacts
	rm -rf models/title models/thumbnail
