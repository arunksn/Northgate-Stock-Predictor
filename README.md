# AI-Based Stock Market Prediction & Portfolio Optimization System

Implementation of the Northgate Quantitative Research PRD (v1.0): an end-to-end, leak-free pipeline from raw market data to
forecasts, sentiment, an optimised portfolio, transparent BUY/HOLD/SELL recommendations and a Streamlit dashboard.

> **Educational research and decision-support tool only. Outputs are not financial advice.**

## IMPORTANT: data provenance of the shipped results
The environment this repo was built in had **no access to Yahoo Finance, FRED, a news API or Hugging Face**. The shipped
`data/`, `models/`, `reports/` and dashboard numbers were therefore produced from a clearly labelled **synthetic stand-in market**
(`python -m src.ingest --source synthetic`; stylised facts: market factor, GARCH volatility clustering, fat tails, earnings jumps,
injected data defects, timestamped headlines). They prove that the pipeline runs end to end; they say nothing about real markets, and
FinBERT was replaced by the VADER scorer for the same reason (a logged fallback). **To get real results**, run the commands below on a machine with internet.

## Quick start
```bash
pip install -r requirements.txt            # Python 3.11+ (built and tested on 3.13)
export FINNHUB_API_KEY=...                 # free key, for news headlines (prices/macro need no key)
make data        # ingest -> clean -> sentiment -> features           (one command rebuilds the dataset)
make train       # ML + DL models, comparison, sentiment ablation, portfolio, recommendations   (one command retrains/re-evaluates)
make report      # EDA notebook + written report (reports/report.docx)
make dashboard   # streamlit run dashboard/app.py
make all         # everything
```
Offline demo of the exact shipped results: `python -m src.ingest --source synthetic` in place of the first ingest step
(or set `data.source: synthetic` in `config.yaml`).

Note: free news tiers only return about a year of history, so on real data sentiment features are neutral (0) before that window.

## If you received two zip files
`stock-ai-system-code.zip` (code, notebook, dashboard, report) and `stock-ai-system-data-models.zip` (`data/` and `models/`) are two halves of one project.
Unzip both into the **same** `stock-ai-system/` folder (merge, do not replace). `.parquet`, `.pkl` and `.keras` files are binary and are not meant to be opened in a text
editor: view results through the dashboard (`make dashboard`), `reports/report.pdf`, `notebooks/01_eda.ipynb`, or the readable files in `reports/results/`
(`.json`, `.csv`, `.md`). To peek at a parquet file: `python -c "import pandas as pd; print(pd.read_parquet('data/raw/news.parquet').head())"`.

## Known limitations (see also the report, Section 14)
* Shipped results are synthetic (see above); live yfinance/FRED/Finnhub calls were not testable in the build sandbox (the yfinance reshaping is tested on a mock frame).
* FinBERT needs `torch` (listed unpinned in `requirements.txt` because a matching pinned build could not be verified offline) and a Hugging Face download; otherwise VADER is used.
* Classical models are bit-for-bit reproducible; deep models are seeded with op-determinism and oneDNN is disabled in `src/utils.py`, but results can still differ slightly (~1e-4 in predicted log-return units between fresh runs; the shipped deep-model numbers were produced before oneDNN was disabled, so a retrain will move them slightly).

## Layout (PRD Listing 15.1)
```
config.yaml            universe, dates, split ratios, sequence length, seeds, model grids, portfolio/recommendation settings
src/ingest.py          Sec 3   yfinance + FRED + Finnhub news -> data/raw (parquet); synthetic stand-in generator
src/clean.py           Sec 4   ordered, idempotent cleaning + data-quality report + change log
src/features.py        Sec 5   feature families, make_sequences (Listing 5.2), chronological splits, leak-free dataset builder
src/sentiment.py       Sec 10  text cleaning, FinBERT/VADER scoring, session mapping (no look-ahead), mood index
src/mathfin.py         Sec 7   from-scratch finance maths, manual gradient descent, verification vs libraries, metrics
src/models_ml.py       Sec 8   Ridge, Random Forest, XGBoost, SVR with walk-forward CV
src/models_dl.py       Sec 9   LSTM, GRU, BiLSTM, Transformer (Keras), early stopping, loss-curve diagnostics
src/compare.py         Sec 9.5/10.3  eight-model table vs random walk, best-model selection (validation), sentiment ablation
src/portfolio.py       Sec 11  Ledoit-Wolf, max-Sharpe, efficient frontier, 20k Monte-Carlo check, OOS backtest, bootstrap CIs
src/recommend.py       Sec 12  signal fusion, BUY/HOLD/SELL, rebalancing, backtested hit-rate
src/eda_notebook.py    Sec 6   builds/executes notebooks/01_eda.ipynb
src/report.py          Sec 14  builds reports/report.docx from the saved results
dashboard/app.py       Sec 13  seven-panel Streamlit app (reads cached outputs, never retrains)
data/{raw,processed}   models/  notebooks/  reports/{figures,results}
```
`src/utils.py` (config/seed/paths) and the `Makefile` are small support files; `src/compare.py`, `src/eda_notebook.py`, `src/report.py` host the
comparison, EDA notebook and report steps that the PRD lists as deliverables but not as modules.

## Key design decisions
* **Target**: forward log return of the *adjusted* close over `horizon` days (1 and 5); price forecast = `P_t * exp(pred)`. Fixed in `config.yaml` before modelling.
* **Pooled models**: one model per algorithm trained on all tickers (scale-free features only), metrics computed per ticker in price space and averaged.
* **Leakage controls**: chronological 70/15/15 by date; windows built inside each split; targets inside their split (embargo); scaler fit on train only; target scaled separately;
  walk-forward CV with scaler refit per fold; news after 16:00 ET -> next session; macro publication lags; best model chosen on **validation** error.
* **Random-walk baseline**: price forecast = today's price; direction call = repeat the trailing move.
* **Expected returns** for MPT: 50/50 blend of historical mean and model-implied return, estimated only up to the end of validation (`portfolio.mu_blend`).
* **Current holdings** for the rebalancing table are assumed equal-weight (no real holdings are available).
