"""Section 7 - financial mathematics implemented from first principles, plus verification against libraries,
plus the manual gradient-descent regression (Listing 7.1) checked against the analytical OLS solution.
Also holds the evaluation metrics (Section 8.3) and risk-adjusted metrics (Section 7.4)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---- returns & risk (7.1) ----
def simple_returns(p): p = np.asarray(p, float); return p[1:] / p[:-1] - 1
def log_returns(p): p = np.asarray(p, float); return np.log(p[1:] / p[:-1])
def variance(r, ddof=1): r = np.asarray(r, float); return ((r - r.mean()) ** 2).sum() / (len(r) - ddof)
def volatility(r, ddof=1): return np.sqrt(variance(r, ddof))
def annualise(mu_d, sig_d): return mu_d * TRADING_DAYS, sig_d * np.sqrt(TRADING_DAYS)


# ---- covariance, correlation, beta (7.2) ----
def covariance(a, b, ddof=1):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return ((a - a.mean()) * (b - b.mean())).sum() / (len(a) - ddof)
def correlation(a, b): return covariance(a, b) / (volatility(a) * volatility(b))
def beta(r_i, r_m): return covariance(r_i, r_m) / variance(r_m)
def cov_matrix(R):
    R = np.asarray(R, float); X = R - R.mean(0); return X.T @ X / (len(R) - 1)


# ---- portfolio maths (7.3) ----
def portfolio_return(w, mu): return float(np.asarray(w) @ np.asarray(mu))
def portfolio_vol(w, S): w = np.asarray(w); return float(np.sqrt(w @ np.asarray(S) @ w))


# ---- risk-adjusted metrics (7.4) ----
def sharpe(r, rf=0.0, periods=TRADING_DAYS):
    r = np.asarray(r, float); ex = r.mean() * periods - rf
    return ex / (volatility(r) * np.sqrt(periods))
def sortino(r, rf=0.0, periods=TRADING_DAYS):
    r = np.asarray(r, float); thr = rf / periods
    dd = np.sqrt(np.mean(np.minimum(r - thr, 0) ** 2)) * np.sqrt(periods)
    return (r.mean() * periods - rf) / dd
def drawdown_curve(r):
    v = np.cumprod(1 + np.asarray(r, float)); return v / np.maximum.accumulate(v) - 1
def max_drawdown(r): return float(drawdown_curve(r).min())
def calmar(r, periods=TRADING_DAYS):
    r = np.asarray(r, float); ann = np.prod(1 + r) ** (periods / len(r)) - 1
    return ann / abs(max_drawdown(r))
def var_hist(r, level=0.95): return float(np.percentile(np.asarray(r, float), 100 * (1 - level)))
def var_t(r, level=0.95):
    """Parametric VaR with a fitted Student-t (fat tails, 7.6)."""
    from scipy import stats
    df, loc, scale = stats.t.fit(np.asarray(r, float))
    return float(stats.t.ppf(1 - level, df, loc, scale))
def block_bootstrap_sharpe_ci(r, rf=0.0, block=21, reps=2000, seed=42, alpha=0.05):
    """Circular block bootstrap CI for the Sharpe ratio (7.6)."""
    r = np.asarray(r, float); n = len(r); rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); out = np.empty(reps)
    for k in range(reps):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n] % n
        out[k] = sharpe(r[idx], rf)
    return float(np.percentile(out, 100 * alpha / 2)), float(np.percentile(out, 100 * (1 - alpha / 2)))


# ---- forecasting evaluation (8.3) ----
def rmse(y, p): y, p = np.asarray(y, float), np.asarray(p, float); return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))
def mape(y, p): y, p = np.asarray(y, float), np.asarray(p, float); return float(100 * np.mean(np.abs((y - p) / y)))
def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def directional_accuracy(actual_move, pred_move):
    return float(np.mean(np.sign(actual_move) == np.sign(pred_move)))


# ---- Listing 7.1: manual gradient descent ----
def gd_linreg(X, y, lr=1e-2, epochs=2000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        err = X @ w + b - y
        w -= lr * (2 / n) * (X.T @ err)
        b -= lr * (2 / n) * err.sum()
    return w, b


def verify(path=None) -> dict:
    """Check every from-scratch function against numpy / pandas / scipy / sklearn / PyPortfolioOpt."""
    from sklearn.linear_model import LinearRegression
    from scipy import stats
    rng = np.random.default_rng(0)
    m = rng.normal(0.0004, 0.01, 1500)
    R = np.column_stack([1.1 * m + rng.normal(0.0002, 0.008, 1500), 0.7 * m + rng.normal(0.0003, 0.006, 1500),
                         rng.normal(0.0005, 0.012, 1500)])
    p = 100 * np.cumprod(1 + R[:, 0])
    r = R[:, 0]
    chk = {}
    def close(a, b, tol=1e-9): return bool(np.allclose(a, b, rtol=tol, atol=tol))
    chk["simple_return"] = close(simple_returns(p), pd.Series(p).pct_change().dropna())
    chk["log_return"] = close(log_returns(p), np.diff(np.log(p)))
    chk["variance"] = close(variance(r), np.var(r, ddof=1)); chk["volatility"] = close(volatility(r), np.std(r, ddof=1))
    chk["covariance"] = close(covariance(R[:, 0], m * 0 + R[:, 1]), np.cov(R[:, 0], R[:, 1])[0, 1])
    chk["correlation"] = close(correlation(R[:, 0], R[:, 1]), np.corrcoef(R[:, 0], R[:, 1])[0, 1])
    chk["beta"] = close(beta(R[:, 0], m), stats.linregress(m, R[:, 0]).slope)
    chk["cov_matrix"] = close(cov_matrix(R), np.cov(R.T))
    w = np.array([0.5, 0.3, 0.2]); S = np.cov(R.T)
    chk["portfolio_vol"] = close(portfolio_vol(w, S), np.std(R @ w, ddof=1))
    ex = (r.mean() * 252 - 0.02) / (r.std(ddof=1) * np.sqrt(252)); chk["sharpe"] = close(sharpe(r, 0.02), ex)
    v = pd.Series(np.cumprod(1 + r)); chk["max_drawdown"] = close(max_drawdown(r), (v / v.cummax() - 1).min())
    chk["var_hist"] = close(var_hist(r), np.quantile(r, 0.05))
    Xf = rng.normal(size=(800, 3)); yf = Xf @ np.array([1.5, -2.0, 0.5]) + 0.7 + rng.normal(0, 0.1, 800)
    wg, bg = gd_linreg(Xf, yf, lr=0.05, epochs=4000); ols = LinearRegression().fit(Xf, yf)
    chk["gd_matches_ols_weights"] = close(wg, ols.coef_, 1e-4); chk["gd_matches_ols_bias"] = close(bg, ols.intercept_, 1e-4)
    from pypfopt import risk_models
    chk["cov_vs_pypfopt_sample"] = close(cov_matrix(R) * 252, risk_models.sample_cov(pd.DataFrame(R), returns_data=True), 1e-7)
    out = {"checks": chk, "all_pass": all(chk.values()),
           "gd_vs_ols": {"gd_w": wg.round(5).tolist(), "ols_w": ols.coef_.round(5).tolist(),
                         "gd_b": round(float(bg), 5), "ols_b": round(float(ols.intercept_), 5)}}
    if path: open(path, "w").write(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    from .utils import RES, ensure_dirs
    ensure_dirs(); o = verify(RES / "mathfin_verification.json"); print(json.dumps(o, indent=2))
