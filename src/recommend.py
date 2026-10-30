"""Layer 6 - Transparent rule-based recommendations (PRD Section 12).

Composite = w1*forecast + w2*sentiment - w3*risk, each sub-signal cross-sectionally z-scored per date (clipped at +-3).
BUY if composite >= high, SELL if <= low, else HOLD (a no-trade band). Rebalancing compares current vs target weights.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .utils import PROC, RES, ensure_dirs, get_logger, load_config

log = get_logger("recommend")


def recommend(score: float, low: float, high: float) -> str:          # Listing 12.1
    if score >= high: return "BUY"
    if score <= low: return "SELL"
    return "HOLD"


def rebalance(current_w: dict, target_w: dict, band: float = 0.05) -> dict:   # Listing 12.1
    actions = {}
    for a in target_w:
        drift = target_w[a] - current_w.get(a, 0)
        if abs(drift) > band:
            actions[a] = ("INCREASE" if drift > 0 else "REDUCE", round(drift, 3))
    return actions


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return ((s - s.mean()) / sd).clip(-3, 3) if sd > 0 else s * 0.0


def composite(inputs: pd.DataFrame, w1: float, w2: float, w3: float) -> pd.DataFrame:
    """inputs: date, ticker, y_pred, sentiment, risk_raw. Adds sub-scores and the composite score."""
    g = inputs.groupby("date")
    out = inputs.copy()
    out["forecast_sig"] = g["y_pred"].transform(_z)
    out["sentiment_sig"] = g["sentiment"].transform(_z)
    out["risk_sig"] = g["risk_raw"].transform(_z)
    out["score"] = w1 * out.forecast_sig + w2 * out.sentiment_sig - w3 * out.risk_sig
    return out


def risk_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Risk signal from trailing information only: 21d volatility, 252d Beta, current drawdown from the 252d high."""
    rows = []
    for t, g in panel.sort_values("Date").groupby("Ticker"):
        g = g.set_index("Date")
        beta = g["ret1"].rolling(252, min_periods=126).cov(g["index_ret1"]) / g["index_ret1"].rolling(252, min_periods=126).var()
        dd = g["Adj Close"] / g["Adj Close"].rolling(252, min_periods=60).max() - 1
        rows.append(pd.DataFrame({"date": g.index, "ticker": t, "vol21": g["vol21"].to_numpy(), "beta": beta.to_numpy(),
                                  "drawdown": dd.to_numpy(), "sentiment": g["sentiment"].to_numpy()}))
    r = pd.concat(rows).dropna(subset=["beta", "drawdown"])
    z = lambda c: r.groupby("date")[c].transform(_z)
    r["risk_raw"] = (z("vol21") + z("beta") + z("drawdown").mul(-1)) / 3        # deeper drawdown = higher risk
    return r


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config(); ensure_dirs()
    rc = cfg["recommend"]
    panel = pd.read_parquet(PROC / "features.parquet")
    pr = pd.read_parquet(RES / "predictions.parquet")
    best = json.loads((RES / "best_models.json").read_text())
    port = json.loads((RES / "portfolio.json").read_text())
    rk = risk_table(panel)
    all_inputs, bt = [], {}
    for h in cfg["features"]["horizons"]:
        m = best[str(h)]["model"]
        p = pr[(pr.model == m) & (pr.horizon == h) & (pr.split.isin(["test", "latest"]))].rename(columns={"date": "date"})
        x = p.merge(rk, on=["date", "ticker"], how="inner")[["date", "ticker", "split", "y_pred", "y_true", "sentiment", "risk_raw",
                                                             "vol21", "beta", "drawdown"]]
        x["horizon"] = h
        all_inputs.append(x)
        t = composite(x[x.split == "test"], rc["w_forecast"], rc["w_sentiment"], rc["w_risk"])
        t["rec"] = [recommend(s, rc["low"], rc["high"]) for s in t.score]
        buy, sell = t[t.rec == "BUY"], t[t.rec == "SELL"]
        up = (t.y_true > 0)
        bt[h] = {"model": m, "n_buy": int(len(buy)), "n_sell": int(len(sell)), "n_hold": int((t.rec == "HOLD").sum()),
                 "hit_rate_buy": float((buy.y_true > 0).mean()) if len(buy) else None,
                 "hit_rate_sell": float((sell.y_true < 0).mean()) if len(sell) else None,
                 "buy_and_hold_up_rate": float(up.mean()),
                 "avg_fwd_return_buy": float(buy.y_true.mean()) if len(buy) else None,
                 "avg_fwd_return_all": float(t.y_true.mean()),
                 "beats_buy_and_hold_hit_rate": bool(len(buy) > 0 and (buy.y_true > 0).mean() > up.mean())}
    inputs = pd.concat(all_inputs, ignore_index=True)
    inputs.to_parquet(RES / "rec_inputs.parquet")
    h0 = cfg["features"]["horizons"][0]
    lt = composite(inputs[(inputs.horizon == h0) & (inputs.split == "latest")], rc["w_forecast"], rc["w_sentiment"], rc["w_risk"])
    lt["recommendation"] = [recommend(s, rc["low"], rc["high"]) for s in lt.score]
    lt.to_csv(RES / "recommendations_latest.csv", index=False)
    tw = port["weights"]; cw = {k: 1 / len(tw) for k in tw}              # current holdings assumed equal-weight (documented)
    rb = pd.DataFrame([{"ticker": a, "current_weight": cw[a], "target_weight": tw[a], "drift": tw[a] - cw[a],
                        "action": rebalance(cw, tw, rc["rebalance_band"]).get(a, ("HOLD", 0))[0]} for a in tw])
    rb.to_csv(RES / "rebalance_latest.csv", index=False)
    (RES / "recommend_backtest.json").write_text(json.dumps(bt, indent=2))
    log.info("recommend backtest: %s", bt)


if __name__ == "__main__":
    run()
