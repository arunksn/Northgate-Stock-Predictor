"""Layer 7 - Streamlit dashboard (PRD Section 13). Reads pre-computed outputs only; it never retrains models.

    streamlit run dashboard/app.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pypfopt import EfficientFrontier
from src import mathfin as mf
from src.recommend import composite, rebalance, recommend
from src.utils import DISCLAIMER, PROC, RAW, RES, load_config

cfg = load_config()
U = cfg["data"]["universe"]
st.set_page_config(page_title="AI Stock Prediction & Portfolio Optimisation", layout="wide")


@st.cache_data
def load():
    d = {"panel": pd.read_parquet(PROC / "features.parquet"), "prices": pd.read_parquet(PROC / "prices_clean.parquet"),
         "pred": pd.read_parquet(RES / "predictions.parquet"), "mood": pd.read_parquet(PROC / "market_mood.parquet"),
         "frontier": pd.read_parquet(RES / "frontier.parquet"), "mc": pd.read_parquet(RES / "monte_carlo.parquet"),
         "pret": pd.read_parquet(RES / "portfolio_returns.parquet"), "mucov": pd.read_parquet(RES / "mu_cov.parquet"),
         "rec": pd.read_parquet(RES / "rec_inputs.parquet")}
    for k in ("best_models", "portfolio", "sentiment_ablation", "recommend_backtest", "sentiment_info", "dl_diagnostics"):
        d[k] = json.loads((RES / f"{k}.json").read_text())
    d["comp"] = {h: pd.read_csv(RES / f"comparison_h{h}.csv", index_col=0) for h in cfg["features"]["horizons"]}
    d["source"] = (RAW / "SOURCE.txt").read_text().strip()
    return d


D = load()
st.title("AI-Based Stock Market Prediction & Portfolio Optimisation")
st.warning(DISCLAIMER + " Past performance, model forecasts and back-tests do not guarantee future results.")   # on every page
if D["source"] == "synthetic":
    st.error("This build was produced from SYNTHETIC stand-in data (no internet in the build environment). "
             "Figures illustrate the system, not real markets. Rebuild with data.source: yfinance for real results.")

PANELS = ["Overview", "Price & prediction", "Model comparison", "Portfolio analytics", "Risk dashboard", "Sentiment", "Recommendations"]
page = st.sidebar.radio("Panel", PANELS)
ticker = st.sidebar.selectbox("Ticker", U)
horizon = st.sidebar.selectbox("Forecast horizon (trading days)", cfg["features"]["horizons"])
rf = st.sidebar.slider("Risk-free rate", 0.0, 0.08, float(cfg["portfolio"]["risk_free_rate"]), 0.005, format="%.3f")
best = D["best_models"][str(horizon)]["model"]
px_t = D["prices"][D["prices"].Ticker == ticker].set_index("Date")["Adj Close"]
pred = D["pred"]
port = D["portfolio"]


def fwd_dates(idx_dates, h):
    cal = pd.DatetimeIndex(sorted(D["prices"][D["prices"].Ticker == cfg["data"]["benchmark"]].Date.unique()))
    pos = cal.searchsorted(pd.DatetimeIndex(idx_dates))
    ext = cal.append(pd.bdate_range(cal[-1] + pd.Timedelta(days=1), periods=h + 5))
    return ext[np.minimum(pos + h, len(ext) - 1)]


if page == "Overview":
    c = st.columns(4)
    te = D["comp"][horizon]
    rw = te.loc["Naive random walk (baseline)"]; bm = te.loc[best]
    c[0].metric("Best forecaster (validation MAE)", best)
    c[1].metric("Test MAE vs random walk", f"{bm.MAE:.3f}", f"{bm.MAE - rw.MAE:+.3f}", delta_color="inverse")
    c[2].metric("Test directional accuracy", f"{bm.DirAcc:.1f}%", f"{bm.DirAcc - rw.DirAcc:+.1f} pts vs RW")
    bt = port["backtest"]
    c[3].metric("OOS Sharpe: optimised vs equal-weight", f"{bt['Optimised (max-Sharpe)']['sharpe']:.2f}",
                f"{bt['Optimised (max-Sharpe)']['sharpe'] - bt['Equal-weight']['sharpe']:+.2f} vs EW")
    mood = D["mood"].iloc[-1]
    g = go.Figure(go.Indicator(mode="gauge+number", value=float(mood.mood), title={"text": f"Market mood: {mood.label}"},
                               gauge={"axis": {"range": [-0.5, 0.5]}, "steps": [{"range": [-0.5, -0.05], "color": "#f4b6b6"},
                                      {"range": [-0.05, 0.05], "color": "#e6e6e6"}, {"range": [0.05, 0.5], "color": "#b6e3b6"}]}))
    g.update_layout(height=280)
    left, right = st.columns([1, 2])
    left.plotly_chart(g, width="stretch")
    snap = []
    for t in U:
        p = D["prices"][D["prices"].Ticker == t].set_index("Date")["Adj Close"]
        pn = D["panel"][D["panel"].Ticker == t].iloc[-1]
        snap.append({"Ticker": t, "Last price": round(p.iloc[-1], 2), "1d %": round(100 * p.pct_change().iloc[-1], 2),
                     "21d vol (ann. %)": round(100 * pn.vol21 * np.sqrt(252), 1), "RSI14": round(pn.rsi14, 1), "Sentiment": round(pn.sentiment, 3)})
    right.subheader("Universe snapshot"); right.dataframe(pd.DataFrame(snap).set_index("Ticker"), width="stretch")

elif page == "Price & prediction":
    st.subheader(f"{ticker}: price and {horizon}-day forecast ({best})")
    p = pred[(pred.model == best) & (pred.horizon == horizon) & (pred.ticker == ticker)]
    va, te, lt = (p[p.split == s] for s in ("val", "test", "latest"))
    sig = float((va.y_true - va.y_pred).std())
    te = te.assign(pp=te.price_t * np.exp(te.y_pred), x=fwd_dates(te.date, horizon))
    te["lo"], te["hi"] = te.price_t * np.exp(te.y_pred - 1.96 * sig), te.price_t * np.exp(te.y_pred + 1.96 * sig)
    hist = px_t[px_t.index >= te.date.min() - pd.Timedelta(days=120)]
    f = go.Figure()
    f.add_scatter(x=hist.index, y=hist, name="Adj Close")
    f.add_scatter(x=te.x, y=te.hi, line=dict(width=0), showlegend=False, hoverinfo="skip")
    f.add_scatter(x=te.x, y=te.lo, fill="tonexty", line=dict(width=0), name="95% band (validation residuals)", fillcolor="rgba(255,127,14,.2)")
    f.add_scatter(x=te.x, y=te.pp, name=f"Forecast ({best})", line=dict(color="#ff7f0e", width=1))
    if len(lt):
        r = lt.iloc[0]; xd = fwd_dates([r.date], horizon)[0]
        f.add_scatter(x=[xd], y=[r.price_t * np.exp(r.y_pred)], mode="markers", marker=dict(size=11, symbol="star", color="red"),
                      name="Next forecast", error_y=dict(type="data", symmetric=False, array=[r.price_t * (np.exp(1.96 * sig) - 1)],
                                                         arrayminus=[r.price_t * (1 - np.exp(-1.96 * sig))]))
        st.metric(f"Latest {horizon}-day forecast for {xd.date()}", f"{r.price_t * np.exp(r.y_pred):.2f}",
                  f"{100 * (np.exp(r.y_pred) - 1):+.2f}% vs {r.price_t:.2f}")
    f.update_layout(height=520, hovermode="x unified", yaxis_title="Adjusted close"); st.plotly_chart(f, width="stretch")

elif page == "Model comparison":
    st.subheader(f"Eight-model comparison on the held-out test window (horizon = {horizon} day)")
    t = D["comp"][horizon]
    st.dataframe(t.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "MAPE": "{:.2f}", "R2": "{:.4f}", "DirAcc": "{:.1f}", "n": "{:.0f}"}), width="stretch")
    long = t.reset_index(names="Model").melt(id_vars="Model", value_vars=["RMSE", "MAE"])
    st.plotly_chart(px.bar(long, x="Model", y="value", color="variable", barmode="group", height=420), width="stretch")
    b = D["best_models"][str(horizon)]
    st.info(f"Best on validation MAE: **{b['model']}**. Beats random walk on MAE: **{b['beats_rw_mae']}**; on directional accuracy: "
            f"**{b['beats_rw_dir']}** -> acceptance bar {'MET' if b['acceptance_met'] else 'NOT MET (reported honestly)'}.")
    a = D["sentiment_ablation"]
    st.markdown(f"**Sentiment experiment ({a['model']}, h={a['horizon']}):** delta MAE = {a['delta_MAE']:+.4f}, delta directional accuracy = "
                f"{a['delta_DirAcc']:+.2f} pts -> sentiment {'helped' if a['sentiment_helped'] else 'did not clearly help'} on this dataset "
                f"(scorer: {D['sentiment_info']['scorer_used']}).")

elif page == "Portfolio analytics":
    st.subheader("Efficient frontier and max-Sharpe allocation")
    mc = D["mucov"]; mu = mc["mu"]; S = mc[U].loc[U]
    try:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, port["max_weight"])); ef.max_sharpe(risk_free_rate=rf)
        w = pd.Series(ef.clean_weights()); r_, v_, s_ = ef.portfolio_performance(risk_free_rate=rf)
    except Exception:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, port["max_weight"])); ef.min_volatility()
        w = pd.Series(ef.clean_weights()); r_, v_, s_ = ef.portfolio_performance(risk_free_rate=rf)
        st.caption("No asset beats the chosen risk-free rate; showing the minimum-variance portfolio.")
    f = go.Figure()
    f.add_scatter(x=D["mc"].vol, y=D["mc"].ret, mode="markers", marker=dict(size=3, color=D["mc"].sharpe, colorscale="Viridis", opacity=.4),
                  name="20,000 random long-only portfolios")
    f.add_scatter(x=D["frontier"].vol, y=D["frontier"].ret, name="Efficient frontier", line=dict(color="black"))
    mv = port["min_var_in_sample"]
    f.add_scatter(x=[mv["vol"]], y=[mv["ret"]], mode="markers", marker=dict(size=13, symbol="diamond", color="blue"), name="Min-variance")
    f.add_scatter(x=[v_], y=[r_], mode="markers", marker=dict(size=15, symbol="star", color="red"), name=f"Max-Sharpe (rf={rf:.3f})")
    f.update_layout(height=480, xaxis_title="Annualised volatility", yaxis_title="Expected annual return")
    st.plotly_chart(f, width="stretch")
    c1, c2 = st.columns([1, 2])
    wp = w[w > 0.0005]; c1.plotly_chart(px.pie(values=wp.values, names=wp.index, height=330, title="Chosen weights"), width="stretch")
    bt = pd.DataFrame(port["backtest"]).T[["ann_return", "ann_vol", "sharpe", "sortino", "beta", "max_drawdown", "calmar"]]
    c2.markdown(f"In-sample (estimation window): return {r_:.1%}, volatility {v_:.1%}, Sharpe {s_:.2f}. "
                f"**Out-of-sample backtest** (frozen weights at rf={port['risk_free_rate']:.3f}):")
    c2.dataframe(bt.style.format("{:.3f}"), width="stretch")
    st.caption(f"mu = {port['mu_method']}. Sharpe 95% block-bootstrap CI, optimised: "
               f"[{port['backtest']['Optimised (max-Sharpe)']['sharpe_ci95'][0]:.2f}, {port['backtest']['Optimised (max-Sharpe)']['sharpe_ci95'][1]:.2f}]")
    cum = (1 + D["pret"].set_index("Date")).cumprod()
    st.plotly_chart(px.line(cum, height=360, title="Out-of-sample growth of 1 unit"), width="stretch")

elif page == "Risk dashboard":
    pr_ = D["pret"].set_index("Date")
    dd = pr_.apply(lambda s: pd.Series(mf.drawdown_curve(s.to_numpy()), index=s.index))
    st.plotly_chart(px.line(dd, height=330, title="Drawdown curve (out-of-sample)"), width="stretch")
    rets = D["panel"].pivot(index="Date", columns="Ticker", values="ret1")
    c1, c2 = st.columns(2)
    rv = (rets[ticker].rolling(21).std() * np.sqrt(252)).dropna()
    c1.plotly_chart(px.line(rv, title=f"{ticker} rolling 21d volatility (annualised)", height=330), width="stretch")
    c2.plotly_chart(px.imshow(rets.corr(), zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Return correlation", height=330), width="stretch")
    v = pd.DataFrame({k: {"VaR 95% (empirical)": mf.var_hist(pr_[k].to_numpy()), "VaR 95% (Student-t)": mf.var_t(pr_[k].to_numpy())} for k in pr_}).T
    st.markdown("**One-day Value-at-Risk (out-of-sample daily returns)**"); st.dataframe(v.style.format("{:.2%}"), width="stretch")

elif page == "Sentiment":
    m = D["mood"].iloc[-1]
    st.metric("Current market mood", m.label.title(), f"index {m.mood:+.3f}")
    s = D["panel"][D["panel"].Ticker == ticker].set_index("Date")["sentiment"]
    st.plotly_chart(px.line(s.rolling(21).mean(), title=f"{ticker}: 21-day average sentiment", height=340), width="stretch")
    st.plotly_chart(px.line(D["mood"].set_index("Date")["mood"].rolling(21).mean(), title="Market-mood index (21-day average)", height=300), width="stretch")
    st.caption(f"Scorer used: {D['sentiment_info']['scorer_used']} (requested: {D['sentiment_info']['requested']}). "
               "News after the 16:00 close is mapped to the next session (no look-ahead).")

elif page == "Recommendations":
    st.subheader(f"BUY / HOLD / SELL - {horizon}-day horizon ({best})")
    c = st.columns(5)
    w1 = c[0].number_input("w1 forecast", 0.0, 3.0, cfg["recommend"]["w_forecast"], 0.1)
    w2 = c[1].number_input("w2 sentiment", 0.0, 3.0, cfg["recommend"]["w_sentiment"], 0.1)
    w3 = c[2].number_input("w3 risk", 0.0, 3.0, cfg["recommend"]["w_risk"], 0.1)
    lo = c[3].number_input("SELL if score <=", -3.0, 0.0, cfg["recommend"]["low"], 0.1)
    hi = c[4].number_input("BUY if score >=", 0.0, 3.0, cfg["recommend"]["high"], 0.1)
    x = D["rec"][(D["rec"].horizon == horizon) & (D["rec"].split == "latest")]
    t = composite(x, w1, w2, w3); t["Recommendation"] = [recommend(s, lo, hi) for s in t.score]
    show = t[["ticker", "Recommendation", "score", "forecast_sig", "sentiment_sig", "risk_sig", "y_pred"]].rename(
        columns={"score": "Composite", "forecast_sig": "Forecast signal", "sentiment_sig": "Sentiment signal", "risk_sig": "Risk signal", "y_pred": f"Predicted {horizon}d return"})
    st.dataframe(show.set_index("ticker").style.format("{:.3f}", subset=show.columns[2:]), width="stretch")
    st.markdown("**Composite = w1 x forecast + w2 x sentiment - w3 x risk** (each cross-sectionally z-scored).")
    st.subheader("Rebalancing plan (current holdings assumed equal-weight)")
    tw = {k: v for k, v in port["weights"].items()}; cw = {k: 1 / len(tw) for k in tw}
    band = st.slider("No-trade band", 0.0, 0.15, cfg["recommend"]["rebalance_band"], 0.01)
    acts = rebalance(cw, tw, band)
    rb = pd.DataFrame([{"Ticker": a, "Current": cw[a], "Target": tw[a], "Drift": tw[a] - cw[a],
                        "Action": acts.get(a, ("HOLD", 0))[0]} for a in tw]).set_index("Ticker")
    st.dataframe(rb.style.format({"Current": "{:.1%}", "Target": "{:.1%}", "Drift": "{:+.1%}"}), width="stretch")
    b = D["recommend_backtest"][str(horizon)]
    st.markdown(f"**Backtest on the test window:** BUY hit-rate {b['hit_rate_buy'] if b['hit_rate_buy'] is None else f'{b['hit_rate_buy']:.1%}'} "
                f"({b['n_buy']} calls) vs buy-and-hold up-rate {b['buy_and_hold_up_rate']:.1%}; SELL hit-rate "
                f"{b['hit_rate_sell'] if b['hit_rate_sell'] is None else f'{b['hit_rate_sell']:.1%}'} ({b['n_sell']} calls).")

st.divider(); st.caption(DISCLAIMER)
