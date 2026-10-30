"""Layer 4 - Sentiment (PRD Section 10): headlines -> daily per-asset sentiment aligned to trading sessions.

Look-ahead guard: a headline stamped after the 16:00 (New York) close, or on a non-trading day, maps to the NEXT session.
"""
from __future__ import annotations
import json, re
import numpy as np
import pandas as pd
from .utils import PROC, RAW, RES, ensure_dirs, get_logger, load_config

log = get_logger("sentiment")
ABBREV = {r"\bq([1-4])\b": r"quarter \1", r"\beps\b": "earnings per share", r"\bipo\b": "initial public offering",
          r"\bceo\b": "chief executive", r"\byoy\b": "year over year", r"\bfy\b": "fiscal year"}


def clean_text(s: str, tickers: set[str]) -> str:
    """Lowercase, strip URLs and tickers, expand common finance abbreviations, remove boilerplate."""
    s = re.sub(r"https?://\S+", " ", s.lower())
    s = re.sub(r"\b(" + "|".join(re.escape(t.lower()) for t in tickers) + r")\b", " ", s)
    for k, v in ABBREV.items():
        s = re.sub(k, v, s)
    s = re.sub(r"\b(click here|read more|subscribe|advertisement)\b", " ", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%$.,' -]", " ", s)).strip()


def score_vader(texts: list[str]) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer()
    return np.array([a.polarity_scores(t)["compound"] for t in texts])


def score_finbert(texts: list[str]) -> np.ndarray:
    """Listing 10.1: polarity = {positive:+1, neutral:0, negative:-1} x class confidence."""
    from transformers import pipeline
    fb = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)
    m = {"positive": 1, "neutral": 0, "negative": -1}
    out = fb(texts, batch_size=32)
    return np.array([m[o["label"].lower()] * o["score"] for o in out])


def map_to_session(ts: pd.Series, cal: pd.DatetimeIndex, close_hour: int) -> pd.Series:
    local = ts.dt.tz_convert("America/New_York")
    base = local.dt.tz_localize(None).dt.normalize() + pd.to_timedelta((local.dt.hour >= close_hour).astype(int), unit="D")
    pos = cal.searchsorted(base.to_numpy(), side="left")
    return pd.Series(np.where(pos < len(cal), cal[np.minimum(pos, len(cal) - 1)], pd.NaT), index=ts.index)


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config(); ensure_dirs()
    news = pd.read_parquet(RAW / "news.parquet")
    px = pd.read_parquet(PROC / "prices_clean.parquet")
    cal = pd.DatetimeIndex(sorted(px.loc[px.Ticker == cfg["data"]["benchmark"], "Date"].unique()))
    s = cfg["sentiment"]
    news["clean"] = [clean_text(t, set(cfg["data"]["universe"])) for t in news["text"]]
    news["session"] = map_to_session(news["timestamp"], cal, s["market_close_hour"])
    news = news.dropna(subset=["session"])
    texts = news["clean"].tolist()
    news["vader"] = score_vader(texts)
    used = "vader"
    if s["primary"] == "finbert":
        try:
            news["finbert"] = score_finbert(texts); used = "finbert"
        except Exception as e:  # weights unavailable offline: fall back, loudly
            log.warning("FinBERT unavailable (%s: %s) -> using VADER as the primary scorer", type(e).__name__, str(e)[:120])
    news["polarity"] = news[used]
    g = news.groupby(["ticker", "session"])
    daily = g.agg(sentiment=("polarity", "mean"), sentiment_vader=("vader", "mean"), n_news=("polarity", "size")).reset_index()
    daily = daily.rename(columns={"session": "Date"})
    daily.to_parquet(PROC / "sentiment_daily.parquet")
    # market-mood index: universe-wide daily gauge (PRD 10.3)
    uni = daily[daily.ticker.isin(cfg["data"]["universe"] + ["MARKET"])]
    mood = uni.groupby("Date")["sentiment"].mean().rename("mood").reset_index()
    t = s["mood_threshold"]
    mood["label"] = np.where(mood.mood > t, "bullish", np.where(mood.mood < -t, "bearish", "neutral"))
    mood.to_parquet(PROC / "market_mood.parquet")
    info = {"scorer_used": used, "requested": s["primary"], "n_headlines": int(len(news)),
            "corr_finbert_vader": float(news[["finbert", "vader"]].corr().iloc[0, 1]) if "finbert" in news else None}
    (RES / "sentiment_info.json").write_text(json.dumps(info, indent=2))
    log.info("sentiment done: %s", info)


if __name__ == "__main__":
    run()
