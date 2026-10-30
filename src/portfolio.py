"""Layer 5 - Modern Portfolio Theory (PRD Section 11).

Expected returns mu = blend of (a) historical mean return and (b) model-implied return from the best next-day forecaster,
both estimated ONLY on data up to the end of the validation window. Covariance = Ledoit-Wolf shrinkage. Long-only with a
per-asset cap. Weights are then frozen and backtested on the untouched test window against equal-weight and benchmark.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from . import mathfin as mf
from .features import split_dates
from .utils import PROC, RES, ensure_dirs, get_logger, load_config

log = get_logger("portfolio")


def optimise(mu: pd.Series, S: pd.DataFrame, rf: float, cap: float):
    """Max-Sharpe (tangency) weights; falls back to minimum variance if no asset beats the risk-free rate."""
    ef = EfficientFrontier(mu, S, weight_bounds=(0, cap))
    fallback = False
    try:
        ef.max_sharpe(risk_free_rate=rf)
    except Exception as e:  # infeasible when every mu <= rf
        log.warning("max_sharpe infeasible (%s); falling back to min-volatility", str(e)[:80])
        ef = EfficientFrontier(mu, S, weight_bounds=(0, cap)); ef.min_volatility(); fallback = True
    w = pd.Series(ef.clean_weights())
    return w, ef.portfolio_performance(risk_free_rate=rf), fallback


def frontier(mu, S, cap, n=40):
    ef0 = EfficientFrontier(mu, S, weight_bounds=(0, cap)); ef0.min_volatility()
    lo = ef0.portfolio_performance()[0]
    hi = min(mu.max() * 0.999, lo + 3 * (mu.max() - lo))
    pts = []
    for tr in np.linspace(lo, hi, n):
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, cap)); ef.efficient_return(tr)
            r, v, _ = ef.portfolio_performance(); pts.append((v, r))
        except Exception:
            continue
    return pd.DataFrame(pts, columns=["vol", "ret"])


def monte_carlo(mu, S, rf, cap, n, seed):
    """n random long-only portfolios (Dirichlet weights, rejection-sampled to respect the per-asset cap)."""
    rng = np.random.default_rng(seed)
    parts, have = [], 0
    while have < n:
        W = rng.dirichlet(np.ones(len(mu)), size=n)
        W = W[(W <= cap).all(1)]
        parts.append(W); have += len(W)
    W = np.concatenate(parts)[:n]
    r = W @ mu.to_numpy(); v = np.sqrt(np.einsum("ij,jk,ik->i", W, S.to_numpy(), W))
    return pd.DataFrame({"vol": v, "ret": r, "sharpe": (r - rf) / v})


def perf(r: np.ndarray, bench: np.ndarray, rf: float, cfg: dict) -> dict:
    p = cfg["portfolio"]
    lo, hi = mf.block_bootstrap_sharpe_ci(r, rf, p["bootstrap_block"], p["bootstrap_reps"], cfg["seed"])
    return {"ann_return": float(np.prod(1 + r) ** (252 / len(r)) - 1), "ann_vol": float(mf.volatility(r) * np.sqrt(252)),
            "sharpe": float(mf.sharpe(r, rf)), "sharpe_ci95": [lo, hi], "sortino": float(mf.sortino(r, rf)),
            "max_drawdown": mf.max_drawdown(r), "calmar": float(mf.calmar(r)), "beta": float(mf.beta(r, bench)),
            "var95_hist": mf.var_hist(r), "var95_t": mf.var_t(r), "total_return": float(np.prod(1 + r) - 1)}


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config(); ensure_dirs()
    p, d = cfg["portfolio"], cfg["data"]
    rf, cap = p["risk_free_rate"], p["max_weight"]
    px = pd.read_parquet(PROC / "prices_clean.parquet")
    panel = pd.read_parquet(PROC / "features.parquet")
    _, val_end = split_dates(panel, cfg)
    wide = px[px.Ticker.isin(d["universe"] + [d["benchmark"]])].pivot(index="Date", columns="Ticker", values="Adj Close")
    uni = d["universe"]
    est, test = wide[wide.index <= val_end], wide[wide.index > val_end]

    mu_hist = expected_returns.mean_historical_return(est[uni])
    best = json.loads((RES / "best_models.json").read_text())[str(cfg["features"]["horizons"][0])]["model"]
    pr = pd.read_parquet(RES / "predictions.parquet")
    pv = pr[(pr.model == best) & (pr.horizon == cfg["features"]["horizons"][0]) & (pr.split == "val")]
    last = np.sort(pv.date.unique())[-63:]
    mu_model = pv[pv.date.isin(last)].groupby("ticker")["y_pred"].mean().reindex(uni) * 252
    mu = (1 - p["mu_blend"]) * mu_hist + p["mu_blend"] * mu_model
    S = risk_models.CovarianceShrinkage(est[uni]).ledoit_wolf()
    w, (mu_p, sig_p, sh), fb = optimise(mu, S, rf, cap)
    ef_mv = EfficientFrontier(mu, S, weight_bounds=(0, cap)); ef_mv.min_volatility()
    w_mv = pd.Series(ef_mv.clean_weights()); mv_perf = ef_mv.portfolio_performance(risk_free_rate=rf)
    fr = frontier(mu, S, cap); mc = monte_carlo(mu, S, rf, cap, p["mc_portfolios"], cfg["seed"])
    # MC cross-check: frontier should sit on/above the cloud's upper-left edge
    mc_best = mc.sharpe.max()
    check = {"mc_max_sharpe": float(mc_best), "optimiser_sharpe": float(sh), "optimiser_ge_mc": bool(sh >= mc_best - 1e-6),
             "n_mc": int(len(mc))}

    # ---- out-of-sample backtest with frozen weights ----
    rets = test.pct_change().dropna()
    rb = rets[d["benchmark"]].to_numpy()
    series = {"Optimised (max-Sharpe)": rets[uni] @ w.reindex(uni).fillna(0),
              "Equal-weight": rets[uni].mean(axis=1), "Benchmark buy-and-hold": rets[d["benchmark"]]}
    bt = {k: perf(v.to_numpy(), rb, rf, cfg) for k, v in series.items()}
    pd.DataFrame(series).rename_axis("Date").reset_index().to_parquet(RES / "portfolio_returns.parquet")
    fr.to_parquet(RES / "frontier.parquet"); mc.to_parquet(RES / "monte_carlo.parquet")
    pd.concat([mu.rename("mu"), mu_hist.rename("mu_hist"), mu_model.rename("mu_model"), S], axis=1).to_parquet(RES / "mu_cov.parquet")
    res = {"best_forecaster": best, "val_end": str(val_end.date()), "test_start": str(test.index[1].date()),
           "mu_method": f"{1 - p['mu_blend']:.0%} historical mean + {p['mu_blend']:.0%} model-implied (last 63 validation days), "
                        "estimated up to the validation end only",
           "weights": w.to_dict(), "weights_min_var": w_mv.to_dict(), "fallback_min_vol": fb,
           "in_sample": {"ret": mu_p, "vol": sig_p, "sharpe": sh}, "min_var_in_sample": {"ret": mv_perf[0], "vol": mv_perf[1], "sharpe": mv_perf[2]},
           "mc_check": check, "backtest": bt, "risk_free_rate": rf, "max_weight": cap,
           "acceptance_sharpe_gt_equal_weight": bool(bt["Optimised (max-Sharpe)"]["sharpe"] > bt["Equal-weight"]["sharpe"])}
    (RES / "portfolio.json").write_text(json.dumps(res, indent=2))
    log.info("portfolio: weights=%s | OOS sharpe opt=%.2f ew=%.2f bench=%.2f", {k: v for k, v in w.items() if v > 0},
             bt["Optimised (max-Sharpe)"]["sharpe"], bt["Equal-weight"]["sharpe"], bt["Benchmark buy-and-hold"]["sharpe"])


if __name__ == "__main__":
    run()
