"""Layer 3b - Deep sequence models (PRD Section 9): LSTM, GRU, Bidirectional LSTM, Transformer encoder.

Inputs are the (samples, L, n_features) windows of Listing 5.2 on the same chronological samples the ML models use.
Huber loss, Adam, early stopping on the chronological validation split, shuffle=False, all seeds fixed.
"""
from __future__ import annotations
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .features import build_dataset, feature_cols
from .models_ml import pred_frame
from .utils import FIGS, MODELS, PROC, RES, ensure_dirs, get_logger, load_config, seed_everything

log = get_logger("models_dl")
NAMES = {"lstm": "LSTM", "gru": "GRU", "bilstm": "Bidirectional LSTM", "transformer": "Transformer"}


def _layers():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    return tf, layers, models


def build_lstm(L, n_features, lr=1e-3):                         # Listing 9.1
    tf, layers, models = _layers()
    m = models.Sequential([layers.Input((L, n_features)), layers.LSTM(64, return_sequences=True), layers.Dropout(0.2),
                           layers.LSTM(32), layers.Dropout(0.2), layers.Dense(16, activation="relu"), layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="huber", metrics=["mae"])
    return m


def build_gru(L, n_features, lr=1e-3):
    tf, layers, models = _layers()
    m = models.Sequential([layers.Input((L, n_features)), layers.GRU(64, return_sequences=True), layers.Dropout(0.2),
                           layers.GRU(32), layers.Dropout(0.2), layers.Dense(16, activation="relu"), layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="huber", metrics=["mae"])
    return m


def build_bilstm(L, n_features, lr=1e-3):
    tf, layers, models = _layers()
    m = models.Sequential([layers.Input((L, n_features)), layers.Bidirectional(layers.LSTM(48)), layers.Dropout(0.2),
                           layers.Dense(16, activation="relu"), layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="huber", metrics=["mae"])
    return m


def sinusoidal_positions(L: int, d: int) -> np.ndarray:
    pos, i = np.arange(L)[:, None], np.arange(d)[None, :]
    ang = pos / np.power(10000, (2 * (i // 2)) / d)
    pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang))
    return pe[None].astype("float32")


def transformer_block(x, heads=4, key_dim=32, ff=64, drop=0.1):   # Listing 9.2
    _, layers, _ = _layers()
    a = layers.MultiHeadAttention(heads, key_dim)(x, x)
    a = layers.Dropout(drop)(a)
    x = layers.LayerNormalization()(x + a)
    f = layers.Dense(ff, activation="relu")(x)
    f = layers.Dense(x.shape[-1])(f)
    return layers.LayerNormalization()(x + f)


def build_transformer(L, n_features, lr=1e-3):
    tf, layers, models = _layers()
    inp = layers.Input((L, n_features))
    x = inp + tf.constant(sinusoidal_positions(L, n_features))   # positional encoding
    for _ in range(2):
        x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    m = models.Model(inp, layers.Dense(1)(x))
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="huber", metrics=["mae"])
    return m


BUILDERS = {"lstm": build_lstm, "gru": build_gru, "bilstm": build_bilstm, "transformer": build_transformer}


def diagnose(hist: dict, patience: int) -> dict:
    tl, vl = np.array(hist["loss"]), np.array(hist["val_loss"])
    b = int(vl.argmin())
    gap = float(vl[b] / tl[b])
    widening = bool(len(vl) - 1 > b and vl[-1] > vl[b] * 1.02 and tl[-1] < tl[b])
    verdict = ("overfitting (validation loss rose while training loss kept falling)" if widening or gap > 1.3 else
               "underfitting / no usable signal (train and validation loss both flat and close)" if gap < 1.05 and abs(tl[0] - tl[b]) / tl[0] < 0.1
               else "reasonable fit (small train-validation gap)")
    return {"epochs_run": len(tl), "best_epoch": b + 1, "train_loss_at_best": float(tl[b]), "val_loss_at_best": float(vl[b]),
            "val_to_train_ratio": gap, "verdict": verdict}


def loss_figure(hist: dict, name: str, h: int, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(hist["loss"], label="train"); ax.plot(hist["val_loss"], label="validation")
    ax.set(title=f"{name} (h={h}) Huber loss", xlabel="epoch", ylabel="loss"); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def fit_predict(name: str, ds: dict, cfg: dict, h: int, tag: str = "", save: bool = True):
    import tensorflow as tf
    seed_everything(cfg["seed"])
    d = cfg["dl"]; L = ds["meta"]["L"]; F = len(ds["meta"]["cols"])
    xs = StandardScaler().fit(ds["train"]["X2"])                 # FIT ON TRAIN ONLY
    ys = StandardScaler().fit(ds["train"]["y"][:, None])
    tx = lambda X: xs.transform(X.reshape(-1, F)).reshape(X.shape).astype("float32")
    ty = lambda y: ys.transform(y[:, None])[:, 0].astype("float32")
    model = BUILDERS[name](L, F, d["learning_rate"])
    es = tf.keras.callbacks.EarlyStopping(patience=d["patience"], restore_best_weights=True)
    hist = model.fit(tx(ds["train"]["X3"]), ty(ds["train"]["y"]), validation_data=(tx(ds["val"]["X3"]), ty(ds["val"]["y"])),
                     epochs=d["epochs"], batch_size=d["batch_size"], callbacks=[es], shuffle=False, verbose=0)
    hh = {k: [float(x) for x in v] for k, v in hist.history.items()}
    frames = []
    for split in ("val", "test", "latest"):
        yhat = ys.inverse_transform(model.predict(tx(ds[split]["X3"]), verbose=0, batch_size=512))[:, 0]
        frames.append(pred_frame(NAMES[name], h, split, ds[split], yhat))
    if save:
        model.save(MODELS / f"{name}_h{h}{tag}.keras")
        joblib.dump({"x_scaler": xs, "y_scaler": ys, "cols": ds["meta"]["cols"]}, MODELS / f"{name}_h{h}{tag}_scalers.pkl")
    return pd.concat(frames, ignore_index=True), hh


def run(cfg: dict | None = None, only: list[str] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config(); ensure_dirs()
    panel = pd.read_parquet(PROC / "features.parquet")
    cols = feature_cols()
    allp, diag, hists = [], {}, {}
    for h in cfg["features"]["horizons"]:
        ds = build_dataset(panel, cfg, h, cols)
        for name in (only or BUILDERS):
            p, hh = fit_predict(name, ds, cfg, h)
            allp.append(p); hists[f"{name}_h{h}"] = hh
            diag[f"{name}_h{h}"] = diagnose(hh, cfg["dl"]["patience"])
            loss_figure(hh, NAMES[name], h, FIGS / f"loss_{name}_h{h}.png")
            log.info("h=%d %-11s %s", h, name, diag[f"{name}_h{h}"])
    out = pd.concat(allp, ignore_index=True)
    out.to_parquet(RES / "pred_dl.parquet")
    (RES / "dl_diagnostics.json").write_text(json.dumps(diag, indent=2))
    (RES / "dl_history.json").write_text(json.dumps(hists))
    return out


if __name__ == "__main__":
    run()
