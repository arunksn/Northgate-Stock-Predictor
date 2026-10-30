"""Layer 2b - Feature engineering, sequence windowing and chronological splits (PRD Section 5).

Rule: every feature at row t uses only information available at or before t (no .shift(-k) in any feature).
Targets (forward log return over `horizon` sessions) are built separately in `make_sequences` from the log-price series.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import PROC, ensure_dirs, get_logger, load_config

log = get_logger("features")

FAMILIES = {
    "returns": ["ret1", "ret5", "ret10", "ret21"],
    "trend": [f"p_sma{w}" for w in (10, 20, 50, 200)] + [f"p_ema{w}" for w in (10, 20, 50, 200)] + ["macd", "macd_signal", "macd_hist"],
    "momentum": ["rsi14", "stoch_k", "stoch_d", "roc10"],
    "volatility": ["vol21", "atr14", "bb_width"],
    "volume": ["vol_z", "obv_z", "vol_price_div"],
    "calendar": ["dow", "month", "turn_of_month"],
    "macro_market": ["vix", "vix_chg", "yield_spread", "index_ret1", "sector_ret1", "cpi_yoy", "unrate"],
    "lagged": [f"ret1_lag{k}" for k in range(1, 6)],
    "sentiment": ["sentiment", "sent_mom3", "news_spike"],
}
# raw price-level columns kept in the panel for charts/EDA but never fed to models (not comparable across tickers)
LEVEL_COLS = ["sma10", "sma20", "sma50", "sma200", "ema10", "ema20", "ema50", "ema200"]


def feature_cols(exclude_families: tuple[str, ...] = ()) -> list[str]:
    return [c for f, cs in FAMILIES.items() if f not in exclude_families for c in cs]


def rsi(p: pd.Series, n: int = 14) -> pd.Series:
    d = p.diff(); up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.rolling(n).mean() / dn.rolling(n).mean()
    return 100 - 100 / (1 + rs)


def add_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Listing 5.1 (extended to every family in Table 5.1) for ONE ticker indexed by Date."""
    p, c = df["Adj Close"], df["Close"]
    lp = np.log(p)
    df["ret1"] = lp.diff()
    for k in (5, 10, 21):
        df[f"ret{k}"] = lp.diff(k)
    for w in (10, 20, 50, 200):
        df[f"sma{w}"] = p.rolling(w).mean(); df[f"p_sma{w}"] = p / df[f"sma{w}"]
        df[f"ema{w}"] = p.ewm(span=w, adjust=False).mean(); df[f"p_ema{w}"] = p / df[f"ema{w}"]
    macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    df["macd"], df["macd_signal"], df["macd_hist"] = macd / p, sig / p, (macd - sig) / p
    df["rsi14"] = rsi(p, 14)
    lo14, hi14 = df["Low"].rolling(14).min(), df["High"].rolling(14).max()
    df["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14); df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["roc10"] = p.pct_change(10) * 100
    df["vol21"] = df["ret1"].rolling(21).std()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(), (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean() / c
    mid, sd = p.rolling(20).mean(), p.rolling(20).std()
    df["bb_width"] = (4 * sd) / mid
    v = df["Volume"].astype(float)
    df["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    obv = (np.sign(df["ret1"].fillna(0)) * v).cumsum()
    df["obv_z"] = (obv - obv.rolling(63).mean()) / obv.rolling(63).std()
    df["vol_price_div"] = np.sign(df["ret5"]) * np.log((v.rolling(5).mean() + 1) / (v.rolling(20).mean() + 1))
    df["dow"], df["month"] = df.index.dayofweek, df.index.month
    df["turn_of_month"] = ((df.index.day <= 3) | (df.index.day >= 28)).astype(int)
    for k in range(1, 6):
        df[f"ret1_lag{k}"] = df["ret1"].shift(k)
    return df


def add_sentiment(df: pd.DataFrame, daily: pd.DataFrame, ticker: str) -> pd.DataFrame:
    s = daily[daily.ticker == ticker].set_index("Date")
    df["sentiment"] = s["sentiment"].reindex(df.index).fillna(0.0)       # no news -> neutral
    n = s["n_news"].reindex(df.index).fillna(0.0)
    m3 = df["sentiment"].rolling(3).mean()
    df["sent_mom3"] = m3 - m3.shift(3)
    df["news_spike"] = (n > (n.rolling(21).mean() + 2 * n.rolling(21).std()).shift(1)).astype(int)
    return df


def build_panel(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config(); ensure_dirs()
    d = cfg["data"]
    px = pd.read_parquet(PROC / "prices_clean.parquet")
    macro = pd.read_parquet(PROC / "macro_clean.parquet").set_index("Date")
    sent = pd.read_parquet(PROC / "sentiment_daily.parquet")
    w = {k: px[px.Ticker == k].set_index("Date") for k in (d["benchmark"], d["vix"], d["sector_proxy"])}
    ctx = pd.DataFrame({
        "vix": w[d["vix"]]["Close"], "vix_chg": w[d["vix"]]["Close"].pct_change(),
        "index_ret1": np.log(w[d["benchmark"]]["Adj Close"]).diff(),
        "sector_ret1": np.log(w[d["sector_proxy"]]["Adj Close"]).diff(),
        "yield_spread": macro["DGS10"] - macro["DGS3MO"],
        "cpi_yoy": macro["CPIAUCSL"].pct_change(252), "unrate": macro["UNRATE"],
    })
    out = []
    for t in d["universe"]:
        df = px[px.Ticker == t].set_index("Date").sort_index().copy()
        df = add_features(df, cfg)
        df = add_sentiment(df, sent, t)
        df = df.join(ctx)
        df["logp"] = np.log(df["Adj Close"])
        df["Ticker"] = t
        out.append(df)
    panel = pd.concat(out).reset_index()
    cols = feature_cols()
    n0 = len(panel)
    panel = panel.dropna(subset=cols).reset_index(drop=True)          # drop warm-up rows from rolling windows
    log.info("panel %s -> %s rows after warm-up drop; %d features", n0, len(panel), len(cols))
    panel.to_parquet(PROC / "features.parquet")
    return panel


# ---------------------------------------------------------------------------------------------
# Windowing and chronological splits
# ---------------------------------------------------------------------------------------------
def make_sequences(X, p, L=60, horizon=1):
    """Listing 5.2. X: (T, F) features; p: (T,) log price. Window i covers rows i..i+L-1; the target is the forward
    log return over `horizon` sessions measured from the window's last row. Returns (samples, L, F), y, end index."""
    Xs, ys, ends = [], [], []
    for i in range(len(X) - L - horizon + 1):
        Xs.append(X[i:i + L])
        ys.append(p[i + L + horizon - 1] - p[i + L - 1])
        ends.append(i + L - 1)
    return np.array(Xs), np.array(ys), np.array(ends)


def split_dates(panel: pd.DataFrame, cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = np.sort(panel["Date"].unique())
    n = len(dates); s = cfg["split"]
    return pd.Timestamp(dates[int(n * s["train"]) - 1]), pd.Timestamp(dates[int(n * (s["train"] + s["val"])) - 1])


def build_dataset(panel: pd.DataFrame, cfg: dict, horizon: int, cols: list[str]) -> dict:
    """Chronological 70/15/15 split by DATE (never shuffled). Windows are built inside each split so none straddles a
    split boundary, and each target lies inside its own split (the last `horizon` rows of a split have no in-split target,
    which acts as an embargo). ML and DL models both consume exactly these samples."""
    L = cfg["features"]["seq_len"]
    d_tr, d_va = split_dates(panel, cfg)
    parts = {"train": [], "val": [], "test": []}
    latest = []
    for t, g in panel.groupby("Ticker"):
        g = g.sort_values("Date").reset_index(drop=True)
        masks = {"train": g.Date <= d_tr, "val": (g.Date > d_tr) & (g.Date <= d_va), "test": g.Date > d_va}
        for name, m in masks.items():
            h = g[m].reset_index(drop=True)
            X3, y, ends = make_sequences(h[cols].to_numpy(np.float32), h["logp"].to_numpy(), L, horizon)
            if len(y) == 0:
                continue
            e = h.iloc[ends]
            fut = h["logp"].to_numpy()[ends + horizon]
            parts[name].append({
                "X3": X3, "X2": X3[:, -1, :], "y": y, "date": e["Date"].to_numpy(), "ticker": np.full(len(y), t),
                "price_t": np.exp(e["logp"].to_numpy()), "price_true": np.exp(fut),
                "trail": e["logp"].to_numpy() - h["logp"].to_numpy()[np.maximum(ends - horizon, 0)]})
        last = g.iloc[-L:]
        latest.append({"X3": last[cols].to_numpy(np.float32)[None], "date": last["Date"].iloc[-1], "ticker": t,
                       "price_t": float(np.exp(last["logp"].iloc[-1])),
                       "trail": float(last["logp"].iloc[-1] - g["logp"].iloc[-1 - horizon])})
    out = {}
    for name, lst in parts.items():
        cat = {k: np.concatenate([x[k] for x in lst]) for k in lst[0]}
        order = np.lexsort((cat["ticker"], cat["date"]))              # date-major so batches stay chronological
        out[name] = {k: v[order] for k, v in cat.items()}
    out["latest"] = {"X3": np.concatenate([x["X3"] for x in latest]), "X2": np.concatenate([x["X3"] for x in latest])[:, -1, :],
                     "date": np.array([x["date"] for x in latest]), "ticker": np.array([x["ticker"] for x in latest]),
                     "price_t": np.array([x["price_t"] for x in latest]), "trail": np.array([x["trail"] for x in latest])}
    out["meta"] = {"train_end": d_tr, "val_end": d_va, "cols": cols, "horizon": horizon, "L": L}
    return out


if __name__ == "__main__":
    build_panel()
