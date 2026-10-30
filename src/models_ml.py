"""Layer 3a - Classical ML regressors (PRD Section 8): Ridge/Linear, Random Forest, XGBoost, SVR.

Identical features and identical chronological samples as the deep models. Hyper-parameters are tuned with an
expanding-window walk-forward CV (date-based TimeSeriesSplit, `horizon`-day embargo); the scaler is refit inside every fold.
The target (forward log return) is scaled separately and inverted for reporting.
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from .features import build_dataset, feature_cols
from .utils import MODELS, PROC, RES, ensure_dirs, get_logger, load_config, seed_everything

log = get_logger("models_ml")
NAMES = {"ridge": "Linear / Ridge Regression", "rf": "Random Forest", "xgb": "XGBoost", "svr": "SVR"}


def panel_cv(dates: np.ndarray, n_splits: int, gap: int):
    """Expanding-window splits over unique DATES so every ticker's rows for a date land in the same fold."""
    ud = np.unique(dates)
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(ud):
        yield np.where(np.isin(dates, ud[tr]))[0], np.where(np.isin(dates, ud[te]))[0]


def make_model(name: str, cfg: dict):
    s = cfg["seed"]
    base = {"ridge": Ridge(random_state=s),
            "rf": RandomForestRegressor(random_state=s, n_jobs=2),
            "xgb": XGBRegressor(random_state=s, n_jobs=2, tree_method="hist", verbosity=0),
            "svr": SVR()}[name]
    return TransformedTargetRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]),
                                      transformer=StandardScaler())


def pred_frame(model_name: str, horizon: int, split: str, d: dict, yhat: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"model": model_name, "horizon": horizon, "split": split, "date": d["date"], "ticker": d["ticker"],
                         "price_t": d["price_t"], "price_true": d.get("price_true", np.nan), "y_true": d.get("y", np.nan),
                         "y_pred": yhat, "trail": d["trail"]})


def run(cfg: dict | None = None, exclude_families: tuple[str, ...] = (), tag: str = "", only: list[str] | None = None,
        horizons: list[int] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config(); ensure_dirs(); seed_everything(cfg["seed"])
    panel = pd.read_parquet(PROC / "features.parquet")
    cols = feature_cols(exclude_families)
    allp, info = [], {}
    for h in (horizons or cfg["features"]["horizons"]):
        ds = build_dataset(panel, cfg, h, cols)
        tr = ds["train"]
        cv = list(panel_cv(tr["date"], cfg["split"]["cv_splits"], gap=h))
        for key in (only or NAMES):
            grid = {f"regressor__model__{k}": v for k, v in cfg["ml"][key].items()}
            gs = GridSearchCV(make_model(key, cfg), grid, cv=cv, scoring="neg_mean_absolute_error", refit=True, n_jobs=1)
            gs.fit(tr["X2"], tr["y"])
            log.info("h=%d %-6s best=%s cv_mae=%.5f", h, key, gs.best_params_, -gs.best_score_)
            info[f"{key}_h{h}"] = {"best_params": {k.split("__")[-1]: v for k, v in gs.best_params_.items()},
                                   "cv_mae": float(-gs.best_score_)}
            joblib.dump({"model": gs.best_estimator_, "cols": cols}, MODELS / f"{key}_h{h}{tag}.pkl")
            for split in ("val", "test", "latest"):
                d = ds[split]
                allp.append(pred_frame(NAMES[key], h, split, d, gs.best_estimator_.predict(d["X2"])))
    out = pd.concat(allp, ignore_index=True)
    out.to_parquet(RES / f"pred_ml{tag}.parquet")
    (RES / f"ml_tuning{tag}.json").write_text(json.dumps(info, indent=2, default=str))
    return out


if __name__ == "__main__":
    run()
