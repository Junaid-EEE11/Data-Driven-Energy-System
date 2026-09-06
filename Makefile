.PHONY: setup data test train experiments ablations figures all clean

PYTHON := python

setup:
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) scripts/download_feeder.py
	$(PYTHON) scripts/validate_feeder.py
	$(PYTHON) scripts/generate_dataset.py
	$(PYTHON) scripts/make_splits.py

test:
	pytest -q

train:
	$(PYTHON) scripts/train_baselines.py
	$(PYTHON) scripts/train_gnn.py
	$(PYTHON) scripts/calibrate_conformal.py

experiments:
	$(PYTHON) scripts/run_experiments.py
	$(PYTHON) scripts/run_ood_tests.py

ablations:
	$(PYTHON) scripts/run_ablations.py

figures:
	$(PYTHON) scripts/aggregate_results.py
	$(PYTHON) scripts/make_figures.py

all: setup data test train experiments ablations figures

clean:
	rm -rf data/processed/* data/splits/* results/raw/* results/aggregated/* results/tables/* results/figures/*
