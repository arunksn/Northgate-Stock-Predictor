"""Layer 1 - Ingestion (PRD Section 3).

Deterministic, re-runnable download of prices (Yahoo Finance), macro series (FRED) and news headlines.
Raw data is written once to data/raw/ as parquet (long format: one row per date x ticker).

    python -m src.ingest                 # uses data.source from config.yaml (yfinance by default)
    python -m src.ingest --source synthetic   # offline stand-in generator (see README; for sandboxes without internet)
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import pandas as pd
from .utils import RAW, ensure_dirs, get_logger, load_config, seed_everything

log = get_logger("ingest")
PRICE_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# --------------------------------------------------------------------------------------------
# Real sources
# --------------------------------------------------------------------------------------------
def download_prices(cfg: dict) -> pd.DataFrame:
    """Listing 3.1: yfinance download with auto_adjust=False so both Close and Adj Close are kept."""
    import yfinance as yf
    d = cfg["data"]
    tickers = list(d["universe"]) + [d["benchmark"], d["vix"], d["sector_proxy"]]
    px = yf.download(tickers, start=d["start"], end=d["end"], auto_adjust=False, progress=False)
    # yfinance returns MultiIndex columns (field, ticker); reshape to long format for parquet.
    lvl = 0 if set(PRICE_FIELDS) & set(px.columns.get_level_values(0)) else 1
    frames = []
    for t in tickers:
        sub = px.xs(t, axis=1, level=1 - lvl).copy()
        sub = sub[[c for c in PRICE_FIELDS if c in sub.columns]]
        sub["Ticker"] = t
        frames.append(sub.rename_axis("Date").reset_index())
    return pd.concat(frames, ignore_index=True)


def download_macro(cfg: dict) -> pd.DataFrame:
    """FRED macro series (no API key required for pandas-datareader's FRED reader)."""
    import pandas_datareader.data as web
    d = cfg["data"]
    macro = web.DataReader(d["macro_series"], "fred", d["start"], d["end"])
    return macro.rename_axis("Date").reset_index()


def download_news(cfg: dict) -> pd.DataFrame:
    """Headlines + summaries per ticker and for the broad market (Finnhub company-news endpoint).

    Needs FINNHUB_API_KEY in the environment. Each record keeps its exact timestamp (UTC) so that
    src.sentiment can map it to a trading session without look-ahead.
    """
    import requests
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY is not set; export it (free tier) to download news headlines.")
    d = cfg["data"]
    end = pd.Timestamp(d["end"])
    look = d["news"].get("lookback_days")
    start = max(pd.Timestamp(d["start"]), end - pd.Timedelta(days=look)) if look else pd.Timestamp(d["start"])
    symbols = {t: t for t in d["universe"]}
    symbols["MARKET"] = d["news"]["market_query"]
    rows = []
    for tick, sym in symbols.items():
        for s in pd.date_range(start, end, freq="30D"):
            e = min(s + pd.Timedelta(days=29), end)
            r = requests.get("https://finnhub.io/api/v1/company-news",
                             params={"symbol": sym, "from": s.date(), "to": e.date(), "token": key}, timeout=30)
            r.raise_for_status()
            for it in r.json():
                text = " ".join(x for x in (it.get("headline"), it.get("summary")) if x)
                rows.append({"timestamp": pd.to_datetime(it["datetime"], unit="s", utc=True),
                             "ticker": tick, "text": text})
            time.sleep(1.1)  # stay under the free-tier rate limit
    news = pd.DataFrame(rows).drop_duplicates(["timestamp", "ticker", "text"])
    return news.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Offline stand-in generator (used ONLY when data.source == synthetic)
# --------------------------------------------------------------------------------------------
def _trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    return pd.date_range(start, end, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))


def generate_synthetic(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic market with realistic stylised facts: market factor, GARCH volatility clustering,
    fat tails (Student-t), earnings jumps with volume spikes, dividend-style Close/Adj Close gap, VIX tied
    to market volatility, monthly macro releases, and timestamped headlines whose tone reacts to returns.
    A handful of data defects (duplicates, short gaps, an OHLC violation, a bad print) are injected so the
    cleaning pipeline has real work to do. NOT real market data."""
    rng = np.random.default_rng(cfg["seed"])
    d = cfg["data"]
    cal = _trading_calendar(d["start"], d["end"])
    n = len(cal)
    tickers = list(d["universe"])
    syms = tickers + [d["benchmark"], d["sector_proxy"]]

    # market factor with GARCH(1,1) variance, t(5) shocks
    def garch(n, omega, alpha, beta, df=5):
        h = np.empty(n); r = np.empty(n); h[0] = omega / (1 - alpha - beta)
        z = rng.standard_t(df, n) / np.sqrt(df / (df - 2))
        for t in range(n):
            r[t] = np.sqrt(h[t]) * z[t]
            if t + 1 < n:
                h[t + 1] = omega + alpha * r[t] ** 2 + beta * h[t]
        return r, h
    m_ret, m_var = garch(n, 2.0e-6, 0.09, 0.89)
    m_ret += 0.00040
    beta = dict(zip(tickers, rng.uniform(0.6, 1.4, len(tickers))))
    mu_i = dict(zip(tickers, rng.uniform(0.0001, 0.0007, len(tickers))))
    ret = {}
    for t in tickers:
        idio, _ = garch(n, 3.0e-6, 0.06, 0.92)
        ret[t] = beta[t] * m_ret + idio + mu_i[t] - 0.00040 * beta[t]
        # earnings jumps ~ every 63 sessions
        for k in range(40, n - 1, 63):
            k += int(rng.integers(-3, 4))
            ret[t][k] += rng.choice([-1, 1]) * rng.uniform(0.04, 0.09)
    ret[d["benchmark"]] = m_ret
    ret[d["sector_proxy"]] = 1.15 * m_ret + garch(n, 2.0e-6, 0.06, 0.92)[0]

    frames, vols_event = [], {}
    for s in syms:
        adj = np.exp(np.cumsum(ret[s])) * rng.uniform(30, 300)
        div_gap = np.exp(np.linspace(0.12, 0.0, n))        # Adj Close <= Close (dividend adjustment)
        close = adj * div_gap
        rng_day = np.abs(rng.normal(0, 0.006, n)) + np.abs(ret[s]) * 0.6
        opn = close * np.exp(rng.normal(0, 0.003, n) + 0.3 * np.r_[0, ret[s][:-1]] * 0)
        high = np.maximum(opn, close) * (1 + rng_day)
        low = np.minimum(opn, close) * (1 - rng_day)
        vol = np.exp(rng.normal(15.5, 0.25, n) + 25 * np.abs(ret[s])).astype(float)
        f = pd.DataFrame({"Date": cal, "Open": opn, "High": high, "Low": low, "Close": close,
                          "Adj Close": adj, "Volume": vol, "Ticker": s})
        frames.append(f)
    # VIX from conditional market volatility
    vix = 100 * np.sqrt(252 * m_var) * 0.9 + rng.normal(0, 0.8, n) + 4
    frames.append(pd.DataFrame({"Date": cal, "Open": vix, "High": vix * 1.02, "Low": vix * 0.98, "Close": vix,
                                "Adj Close": vix, "Volume": 0.0, "Ticker": d["vix"]}))
    px = pd.concat(frames, ignore_index=True)

    # ---- injected data defects ----
    idx = px.index[px.Ticker == tickers[0]]
    px = px.drop(index=idx[[300, 301, 1200, 1201, 1202]])                      # short gaps (<=3 sessions)
    dup = px[px.Ticker == tickers[1]].iloc[[500, 900]].copy()
    dup[["Close", "Adj Close"]] *= 1.0005                                         # conflicting duplicates
    px = pd.concat([px, dup], ignore_index=True)
    j = px.index[(px.Ticker == tickers[2])][700]
    px.loc[j, "Low"] = px.loc[j, "High"] * 1.05                                  # OHLC invariant violation
    k = px.index[(px.Ticker == tickers[3])][1500]
    for c in ["Open", "High", "Low", "Close", "Adj Close"]:                      # one-day bad print (reverts)
        px.loc[k, c] *= 1.6
    px = px.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # ---- macro (FRED-style: daily yields with holiday NaNs, monthly CPI / unemployment dated 1st of month) ----
    bd = pd.bdate_range(d["start"], d["end"])
    lvl = np.cumsum(rng.normal(0, 0.02, len(bd)))
    dgs10 = np.clip(2.2 + 0.6 * np.sin(np.arange(len(bd)) / 700) + 0.3 * lvl / (1 + np.abs(lvl)), 0.5, 5.0)
    dgs3 = np.clip(dgs10 - 0.8 + 0.5 * np.sin(np.arange(len(bd)) / 500), 0.05, 5.0)
    macro_d = pd.DataFrame({"Date": bd, "DGS10": dgs10, "DGS3MO": dgs3})
    macro_d.loc[~macro_d.Date.isin(cal), ["DGS10", "DGS3MO"]] = np.nan
    mdates = pd.date_range(d["start"], d["end"], freq="MS")
    macro_m = pd.DataFrame({"Date": mdates,
                            "CPIAUCSL": 250 * np.exp(np.cumsum(rng.normal(0.002, 0.0015, len(mdates)))),
                            "UNRATE": np.clip(5 + np.cumsum(rng.normal(0, 0.12, len(mdates))), 3, 9)})
    macro = macro_d.merge(macro_m, on="Date", how="outer").sort_values("Date").reset_index(drop=True)

    # ---- news: timestamped headlines whose tone reacts to same-day return ----
    pos = ["beats expectations", "raises guidance", "record profit", "strong demand", "upgraded by analysts",
           "launches successful product", "buyback announced"]
    neg = ["misses estimates", "cuts guidance", "faces probe", "weak demand", "downgraded by analysts",
           "recalls product", "layoffs announced"]
    neu = ["holds annual meeting", "names new director", "schedules earnings call", "files quarterly report",
           "updates website", "announces conference appearance"]
    rows = []
    r_all = {t: pd.Series(ret[t], index=cal) for t in tickers}
    r_all["MARKET"] = pd.Series(ret[d["benchmark"]], index=cal)
    for t, rs in r_all.items():
        z = (rs / rs.rolling(63, min_periods=10).std().bfill()).to_numpy()
        for i, day in enumerate(cal):
            for _ in range(rng.poisson(0.9 if t != "MARKET" else 1.2)):
                p = rng.normal(0.25 * np.tanh(z[i]), 0.55)
                bank = pos if p > 0.25 else neg if p < -0.25 else neu
                hour = rng.integers(6, 23); minute = rng.integers(0, 60)
                ts = (day + pd.Timedelta(hours=int(hour), minutes=int(minute))).tz_localize("America/New_York")
                name = t if t != "MARKET" else "Wall Street"
                rows.append({"timestamp": ts.tz_convert("UTC"), "ticker": t,
                             "text": f"{name} {bank[rng.integers(len(bank))]} - http://example.com/{t}"})
    news = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return px, macro, news


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yfinance", "synthetic"], default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    src = a.source or cfg["data"]["source"]
    ensure_dirs(); seed_everything(cfg["seed"])
    if src == "synthetic":
        log.warning("SYNTHETIC STAND-IN DATA - not real market data (see README)")
        px, macro, news = generate_synthetic(cfg)
    else:
        px, macro = download_prices(cfg), download_macro(cfg)
        news = download_news(cfg)
    px.to_parquet(RAW / "prices.parquet"); macro.to_parquet(RAW / "macro.parquet"); news.to_parquet(RAW / "news.parquet")
    (RAW / "SOURCE.txt").write_text(src + "\n")
    log.info("wrote prices %s, macro %s, news %s (source=%s)", px.shape, macro.shape, news.shape, src)


if __name__ == "__main__":
    main()
