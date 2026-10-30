"""Section 9.5 / 10.3 / 14 - the eight-model comparison on the identical held-out test window, the random-walk baseline,
best-model selection (on VALIDATION error, never on test) and the with/without-sentiment experiment."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import mathfin as mf
from .features import build_dataset, feature_cols
from .models_dl import BUILDERS, fit_predict
from .models_dl import NAMES as DL_NAMES
from .models_ml import NAMES as ML_NAMES, pred_frame, run as run_ml
from .utils import PROC, RES, ensure_dirs, get_logger, load_config

log = get_logger("compare")
RW = "Naive random walk (baseline)"


def metrics(df: pd.DataFrame, rw: bool = False) -> dict:
    """RMSE / MAE / MAPE / R2 in price space (mean over tickers); directional accuracy pooled over all samples.
    Random walk: price forecast = today's price; its direction call is persistence (repeat the trailing move)."""
    per = []
    for _, g in df.groupby("ticker"):
        pp = g.price_t.to_numpy() if rw else g.price_t.to_numpy() * np.exp(g.y_pred.to_numpy())
        y = g.price_true.to_numpy()
        per.append([mf.rmse(y, pp), mf.mae(y, pp), mf.mape(y, pp), mf.r2(y, pp)])
    m = np.mean(per, axis=0)
    pred_move = df.trail if rw else df.y_pred
    return {"RMSE": m[0], "MAE": m[1], "MAPE": m[2], "R2": m[3],
            "DirAcc": mf.directional_accuracy(df.y_true, pred_move) * 100, "n": int(len(df))}


def table(pred: pd.DataFrame, h: int, split: str) -> pd.DataFrame:
    p = pred[(pred.horizon == h) & (pred.split == split)]
    rows = {RW: metrics(p[p.model == list(ML_NAMES.values())[0]], rw=True)}
    for name in list(ML_NAMES.values()) + list(DL_NAMES.values()):
        rows[name] = metrics(p[p.model == name])
    return pd.DataFrame(rows).T


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config(); ensure_dirs()
    pred = pd.concat([pd.read_parquet(RES / "pred_ml.parquet"), pd.read_parquet(RES / "pred_dl.parquet")], ignore_index=True)
    pred.to_parquet(RES / "predictions.parquet")
    out, best = {}, {}
    for h in cfg["features"]["horizons"]:
        te, va = table(pred, h, "test"), table(pred, h, "val")
        te.to_csv(RES / f"comparison_h{h}.csv"); va.to_csv(RES / f"comparison_val_h{h}.csv")
        b = va.drop(index=RW)["MAE"].idxmin()                       # model selection on validation MAE only
        out[h] = {"test": te.to_dict("index"), "val": va.to_dict("index")}
        best[h] = {"model": b, "beats_rw_mae": bool(te.loc[b, "MAE"] < te.loc[RW, "MAE"]),
                   "beats_rw_dir": bool(te.loc[b, "DirAcc"] > te.loc[RW, "DirAcc"]),
                   "test_mae": float(te.loc[b, "MAE"]), "rw_mae": float(te.loc[RW, "MAE"]),
                   "test_dir": float(te.loc[b, "DirAcc"]), "rw_dir": float(te.loc[RW, "DirAcc"])}
        best[h]["acceptance_met"] = best[h]["beats_rw_mae"] and best[h]["beats_rw_dir"]
        log.info("h=%d best(val)=%s | %s", h, b, best[h])
    (RES / "comparison.json").write_text(json.dumps(out, indent=2))
    (RES / "best_models.json").write_text(json.dumps(best, indent=2))

    # ---- sentiment ablation on the best next-day model (PRD 10.3) ----
    h = cfg["features"]["horizons"][0]
    name = best[h]["model"]
    panel = pd.read_parquet(PROC / "features.parquet")
    ds_no = build_dataset(panel, cfg, h, feature_cols(("sentiment",)))
    inv_ml = {v: k for k, v in ML_NAMES.items()}; inv_dl = {v: k for k, v in DL_NAMES.items()}
    if name in inv_ml:
        p_no = run_ml(cfg, exclude_families=("sentiment",), tag="_nosent", only=[inv_ml[name]], horizons=[h])
        p_no = p_no[(p_no.split == "test")]
    else:
        p_no, _ = fit_predict(inv_dl[name], ds_no, cfg, h, tag="_nosent", save=False)
        p_no = p_no[p_no.split == "test"]
    with_s = metrics(pred[(pred.model == name) & (pred.horizon == h) & (pred.split == "test")])
    without = metrics(p_no[p_no.model == name])
    abl = {"model": name, "horizon": h, "with_sentiment": with_s, "without_sentiment": without,
           "delta_MAE": with_s["MAE"] - without["MAE"], "delta_DirAcc": with_s["DirAcc"] - without["DirAcc"],
           "sentiment_helped": bool(with_s["MAE"] < without["MAE"] and with_s["DirAcc"] > without["DirAcc"])}
    (RES / "sentiment_ablation.json").write_text(json.dumps(abl, indent=2))
    log.info("sentiment ablation: %s", abl)


if __name__ == "__main__":
    run()
