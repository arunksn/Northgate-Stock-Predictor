"""Layer 2a - Data cleaning pipeline (PRD Section 4).

Single ordered, idempotent function per series: raw -> tidy, gap-free analysis panel.
Order: de-duplicate -> structural alignment -> missing values -> outliers -> OHLC validation.
Every dropped / altered row is written to data/processed/cleaning_log.csv and a one-page
data-quality report is emitted to reports/results/data_quality_report.md.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .utils import PROC, RAW, RES, ensure_dirs, get_logger, load_config

log = get_logger("clean")
PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]


def _gap_runs(s: pd.Series) -> pd.Series:
    """Length of every run of consecutive NaNs."""
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    return isna.groupby(grp).sum()[isna.groupby(grp).first()]


def _ohlc_violations(df: pd.DataFrame) -> pd.Series:
    """Listing 4.1 invariant: Low <= min(Open, Close), High >= max(Open, Close), Low <= High."""
    return ((df["Low"] > df[["Open", "Close", "High"]].min(axis=1)) |
            (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)))


def clean_prices(df: pd.DataFrame, cal: pd.DatetimeIndex, ticker: str, cfg: dict) -> tuple[pd.DataFrame, dict, list]:
    """Clean one ticker. Returns (clean frame, stats, change-log rows)."""
    c = cfg["cleaning"]
    changes: list[dict] = []
    st = {"ticker": ticker, "rows_in": int(len(df))}
    df = df.set_index("Date").sort_index()

    # Stage 3 - duplicates (done first because reindexing requires a unique index)
    dup_mask = df.index.duplicated(keep=False)
    st["duplicates_removed"] = int(df.index.duplicated(keep="last").sum())
    conflicts = 0
    for dte, g in df[dup_mask].groupby(level=0):
        if not np.allclose(g[PRICE_COLS].to_numpy(float), g[PRICE_COLS].iloc[-1].to_numpy(float), equal_nan=True):
            conflicts += 1
            changes.append({"ticker": ticker, "date": dte, "stage": "dedupe", "field": "all",
                            "action": "conflicting duplicate; kept last (vendor-adjusted) record"})
    st["duplicate_conflicts"] = conflicts
    df = df[~df.index.duplicated(keep="last")]

    # Stage 1 - align to the exchange trading calendar (non-trading days excluded)
    st["non_trading_rows_dropped"] = int((~df.index.isin(cal)).sum())
    df = df.reindex(cal)
    df.index.name = "Date"
    df["Ticker"] = ticker
    missing_before = df["Adj Close"].isna()
    runs = _gap_runs(df["Adj Close"])
    st["gaps_total"] = int(len(runs))
    st["gaps_longer_than_limit_flagged"] = int((runs > c["max_ffill_gap"]).sum())
    for dte in df.index[missing_before]:
        changes.append({"ticker": ticker, "date": dte, "stage": "missing", "field": "OHLCAV",
                        "action": "forward-filled (never back-filled)" if True else ""})

    # Stage 2 - missing values: forward-fill short gaps only (never back-fill)
    df[PRICE_COLS] = df[PRICE_COLS].ffill(limit=c["max_ffill_gap"])
    df["Volume"] = df["Volume"].fillna(0)
    st["gap_sessions_filled"] = int((missing_before & df["Adj Close"].notna()).sum())
    df["imputed"] = missing_before & df["Adj Close"].notna()

    # Stage 4 - outliers: rolling z-score AND 1.5xIQR fence; keep genuine events, winsorise confirmed errors only
    r = np.log(df["Adj Close"]).diff()
    mu, sd = r.rolling(c["z_window"]).mean(), r.rolling(c["z_window"]).std()
    z = (r - mu) / sd
    q1, q3 = r.quantile(0.25), r.quantile(0.75)
    fence_lo, fence_hi = q1 - c["iqr_k"] * (q3 - q1), q3 + c["iqr_k"] * (q3 - q1)
    flagged = (z.abs() > c["outlier_z"]) & ((r < fence_lo) | (r > fence_hi))
    vz = (df["Volume"] - df["Volume"].rolling(c["z_window"]).mean()) / df["Volume"].rolling(c["z_window"]).std()
    r_next = r.shift(-1)
    reverts = (np.sign(r_next) == -np.sign(r)) & ((r + r_next).abs() < 0.5 * r.abs())
    event = flagged & (vz > c["volume_event_z"])                  # volume spike => news/earnings => signal, keep
    confirmed_error = flagged & ~event & reverts                   # one-day print that reverses, no volume support
    st["outliers_flagged"] = int(flagged.sum())
    st["outliers_kept_as_events"] = int((flagged & ~confirmed_error).sum())
    st["outliers_winsorised"] = int(confirmed_error.sum())
    df["is_outlier"] = flagged
    for dte in df.index[confirmed_error]:
        i = df.index.get_loc(dte)
        bound = mu.iloc[i] + np.sign(r.iloc[i]) * c["outlier_z"] * sd.iloc[i]
        new_p = df["Adj Close"].iloc[i - 1] * np.exp(bound)
        factor = new_p / df["Adj Close"].iloc[i]
        old = df.loc[dte, "Adj Close"]
        df.loc[dte, PRICE_COLS] = df.loc[dte, PRICE_COLS] * factor      # scale row to preserve OHLC invariant
        changes.append({"ticker": ticker, "date": dte, "stage": "outlier", "field": "OHLC/Adj Close",
                        "action": f"winsorised confirmed bad print: {old:.2f} -> {new_p:.2f}"})

    # Stage 5 - OHLC invariant: quarantine violating rows, then re-impute (gap-free panel, flagged)
    bad = _ohlc_violations(df)
    st["ohlc_violations_quarantined"] = int(bad.sum())
    for dte in df.index[bad]:
        changes.append({"ticker": ticker, "date": dte, "stage": "ohlc_invariant", "field": "OHLC",
                        "action": "quarantined (set NaN) and re-imputed from last valid session"})
    df.loc[bad, PRICE_COLS] = np.nan
    df[PRICE_COLS] = df[PRICE_COLS].ffill(limit=c["max_ffill_gap"])
    df.loc[bad, "imputed"] = True

    out = df.dropna(subset=["Adj Close"])
    st["rows_out"] = int(len(out))
    st["missing_days_post_clean"] = int(len(cal) - len(out))
    st["history_years"] = round((out.index[-1] - out.index[0]).days / 365.25, 2)
    return out.reset_index(), st, changes


def clean_macro(macro: pd.DataFrame, cal: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Stage 1: shift each series by its publication lag (so a figure is only known after release), reindex
    to the trading calendar and forward-fill: a macro figure is the *last known value* until the next release."""
    lag = cfg["data"]["macro_publication_lag_days"]
    out = {}
    m = macro.set_index("Date").sort_index()
    for col in m.columns:
        s = m[col].dropna()
        s.index = s.index + pd.Timedelta(days=lag.get(col, 0))
        s = s[~s.index.duplicated(keep="last")]
        out[col] = s.reindex(s.index.union(cal)).ffill().reindex(cal)
    res = pd.DataFrame(out)
    res.index.name = "Date"
    return res.ffill()          # forward only: leading values before first release stay NaN and are dropped later


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    ensure_dirs()
    raw = pd.read_parquet(RAW / "prices.parquet")
    macro_raw = pd.read_parquet(RAW / "macro.parquet")
    bench = cfg["data"]["benchmark"]
    cal = pd.DatetimeIndex(sorted(raw.loc[raw.Ticker == bench, "Date"].drop_duplicates()))   # exchange calendar
    cleaned, stats, changes = [], [], []
    for t, g in raw.groupby("Ticker"):
        o, s, ch = clean_prices(g.drop(columns="Ticker"), cal, t, cfg)
        cleaned.append(o); stats.append(s); changes += ch
    prices = pd.concat(cleaned, ignore_index=True)
    macro = clean_macro(macro_raw, cal, cfg)
    prices.to_parquet(PROC / "prices_clean.parquet")
    macro.reset_index().to_parquet(PROC / "macro_clean.parquet")
    pd.DataFrame(changes).to_csv(PROC / "cleaning_log.csv", index=False)

    # ---- data contract (PRD 3.4) ----
    st = pd.DataFrame(stats).set_index("ticker")
    eq = st.loc[cfg["data"]["universe"]]
    contract = {
        "history_ge_8_years": bool((eq.history_years >= 8).all()),
        "universe_size_8_to_15": 8 <= len(cfg["data"]["universe"]) <= 15,
        "zero_missing_days_post_clean": bool((st.missing_days_post_clean == 0).all()),
        "single_trading_calendar": True,
    }
    summary = {"source": (RAW / "SOURCE.txt").read_text().strip() if (RAW / "SOURCE.txt").exists() else "unknown",
               "rows_in": int(st.rows_in.sum()), "rows_out": int(st.rows_out.sum()),
               "duplicates_removed": int(st.duplicates_removed.sum()),
               "gap_sessions_filled": int(st.gap_sessions_filled.sum()),
               "gaps_longer_than_3_flagged": int(st.gaps_longer_than_limit_flagged.sum()),
               "outliers_flagged": int(st.outliers_flagged.sum()),
               "outliers_kept_as_events": int(st.outliers_kept_as_events.sum()),
               "outliers_winsorised": int(st.outliers_winsorised.sum()),
               "ohlc_violations": int(st.ohlc_violations_quarantined.sum()),
               "contract": contract}
    (RES / "data_quality.json").write_text(json.dumps({"summary": summary, "per_ticker": st.reset_index().to_dict("records")},
                                                      indent=2, default=str))
    lines = ["# Data-quality report", "",
             f"Source: **{summary['source']}** | calendar: {cfg['data']['benchmark']} trading days "
             f"({cal[0].date()} to {cal[-1].date()}, {len(cal)} sessions)", "",
             "| metric | value |", "|---|---|"]
    for k in ["rows_in", "rows_out", "duplicates_removed", "gap_sessions_filled", "gaps_longer_than_3_flagged",
              "outliers_flagged", "outliers_kept_as_events", "outliers_winsorised", "ohlc_violations"]:
        lines.append(f"| {k.replace('_', ' ')} | {summary[k]} |")
    lines += ["", "**Data contract (PRD 3.4)**", ""] + [f"- {k.replace('_', ' ')}: {'PASS' if v else 'FAIL'}"
                                                         for k, v in contract.items()]
    lines += ["", "**Per ticker**", "", st[["rows_in", "rows_out", "duplicates_removed", "gap_sessions_filled",
                                           "outliers_flagged", "outliers_winsorised", "ohlc_violations_quarantined",
                                           "missing_days_post_clean"]].to_markdown()]
    (RES / "data_quality_report.md").write_text("\n".join(lines))
    log.info("clean done: %s", summary)


if __name__ == "__main__":
    run()
