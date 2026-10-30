# One command rebuilds the dataset; one command retrains and re-evaluates everything.
PY ?= python
data:     ## rebuild raw data -> cleaned panel -> sentiment -> features
	$(PY) -m src.ingest && $(PY) -m src.clean && $(PY) -m src.sentiment && $(PY) -m src.features
train:    ## retrain + re-evaluate all 8 models, sentiment ablation, portfolio, recommendations
	$(PY) -m src.models_ml && $(PY) -m src.models_dl && $(PY) -m src.compare && $(PY) -m src.portfolio && $(PY) -m src.recommend
report:   ## EDA notebook + written report
	$(PY) -m src.eda_notebook && $(PY) -m src.report
dashboard:
	streamlit run dashboard/app.py
all: data train report
