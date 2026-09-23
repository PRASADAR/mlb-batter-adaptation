PYTHON ?= .venv/bin/python
BOOTSTRAP_PYTHON ?= python3.12
export MPLCONFIGDIR := $(CURDIR)/.mplconfig
export JUPYTER_PATH := $(CURDIR)/.venv/share/jupyter
export JUPYTER_RUNTIME_DIR := $(CURDIR)/.jupyter/runtime
export IPYTHONDIR := $(CURDIR)/.jupyter/ipython

.PHONY: setup data audit features models validation exploratory cross-pitcher claim-audit player-qualities rankings submission-notebook submission results abstract pdf notebook test reproduce
setup:
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PYTHON) -m pip install -r requirements.lock.txt
	$(PYTHON) -m ipykernel install --prefix .venv --name deja-swing --display-name "Python (Deja Swing)"
data:
	$(PYTHON) scripts/02_download_statcast.py --full-target --no-combine
	$(PYTHON) scripts/02_download_statcast.py --players-only
	$(PYTHON) scripts/02_download_statcast.py --audit-schedule
audit:
	$(PYTHON) scripts/01_audit_data.py
features:
	$(PYTHON) -m src.pipeline features
models:
	$(PYTHON) -m src.pipeline models
validation:
	$(PYTHON) -m src.pipeline validation
	$(PYTHON) scripts/17_supplementary_validation.py
	$(PYTHON) scripts/18_matched_exposure.py
exploratory:
	$(PYTHON) scripts/19_percentile_stability.py
cross-pitcher:
	$(PYTHON) scripts/20_cross_pitcher_adaptation.py
	$(PYTHON) scripts/21_cross_pitcher_model_search.py
	$(PYTHON) scripts/22_player_curves.py
claim-audit:
	$(PYTHON) scripts/62_abstract_claim_audit.py
player-qualities:
	$(PYTHON) scripts/63_player_quality_outcomes.py
rankings:
	$(PYTHON) scripts/64_adaptability_rankings.py
submission-notebook:
	$(PYTHON) scripts/65_build_submission_notebook.py
submission: claim-audit player-qualities rankings submission-notebook test
results:
	$(PYTHON) scripts/11_pairwise_comparisons.py
	$(PYTHON) -c "from pathlib import Path; from src.visualization.figures import make_figures; make_figures(Path.cwd())"
	$(PYTHON) -m src.pipeline freeze
abstract:
	$(PYTHON) scripts/build_report.py
	$(PYTHON) scripts/15_build_abstract.py
pdf:
	$(PYTHON) scripts/16_export_abstract_pdf.py
notebook:
	$(PYTHON) scripts/build_notebook.py
	$(PYTHON) scripts/execute_notebook.py
test:
	$(PYTHON) -m pytest -q
reproduce: audit features models validation cross-pitcher claim-audit player-qualities rankings submission-notebook test
