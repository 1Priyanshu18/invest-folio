import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, mean_absolute_error,mean_squared_error, r2_score)
from xgboost import XGBClassifier, XGBRegressor
import joblib

import config as cfg
import features as ft

# 1. train/test split
def chrono_split(frame: pd.DataFrame, test_size: float = cfg.TEST_SIZE):
    n = len(frame)
    split_idx = int(n * (1 - test_size))
    train = frame.iloc[:split_idx]
    test = frame.iloc[split_idx:]
    return train, test


def split_xy(frame: pd.DataFrame, target: str):
    feature_cols = [c for c in frame.columns if not c.startswith("target_")]
    X = frame[feature_cols]
    y = frame[target]
    return X, y, feature_cols

# 2. Direction classifier
def train_classifier(frame: pd.DataFrame, verbose: bool = True):
    train, test = chrono_split(frame)
    X_tr, y_tr, feat_cols = split_xy(train, "target_dir")
    X_te, y_te, _ = split_xy(test, "target_dir")

    y_tr = y_tr.astype(int)
    y_te = y_te.astype(int)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=cfg.RANDOM_SEED,
        eval_metric="logloss", n_jobs=-1,
    )
    model.fit(X_tr_s, y_tr)

    pred = model.predict(X_te_s)
    acc = accuracy_score(y_te, pred)

    # Baseline: always predict the majority class from training
    majority = int(y_tr.mode().iloc[0])
    base_acc = accuracy_score(y_te, np.full_like(y_te, majority))

    if verbose:
        print(f"Direction → Accuracy {acc:.1%}  "
              f"(baseline {base_acc:.1%}, edge {acc - base_acc:+.1%})")

    return {"model": model, "scaler": scaler, "features": feat_cols,
            "accuracy": acc, "baseline": base_acc}


# 3. Return regressor
def train_regressor(frame: pd.DataFrame, verbose: bool = True):
    train, test = chrono_split(frame)
    X_tr, y_tr, feat_cols = split_xy(train, "target_ret")
    X_te, y_te, _ = split_xy(test, "target_ret")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=cfg.RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_tr_s, y_tr)

    pred = model.predict(X_te_s)
    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)

    # Baseline: predict 0 (no change)
    base_mae = mean_absolute_error(y_te, np.zeros_like(y_te))

    # Directional accuracy DERIVED from the regressor (sign of prediction)
    dir_acc = accuracy_score((y_te > 0).astype(int), (pred > 0).astype(int))

    if verbose:
        print(f"Return → MAE {mae:.5f}  RMSE {rmse:.5f}  R² {r2:+.4f}")
        print(f"(baseline MAE {base_mae:.5f}; "
              f"derived dir-acc {dir_acc:.1%})")

    return {"model": model, "scaler": scaler, "features": feat_cols,
            "mae": mae, "rmse": rmse, "r2": r2,
            "baseline_mae": base_mae, "dir_acc": dir_acc}

# 4. Time-series cross-validation
def cv_classifier(frame: pd.DataFrame, n_splits: int = 5):
    X, y, _ = split_xy(frame, "target_dir")
    y = y.astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    accs = []
    for tr_idx, te_idx in tscv.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X.iloc[tr_idx])
        X_te = scaler.transform(X.iloc[te_idx])
        m = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=cfg.RANDOM_SEED, eval_metric="logloss", n_jobs=-1)
        m.fit(X_tr, y.iloc[tr_idx])
        accs.append(accuracy_score(y.iloc[te_idx], m.predict(X_te)))

    accs = np.array(accs)
    print(f"CV accuracy: {accs.mean():.1%} ± {accs.std():.1%}  "
          f"(folds: {', '.join(f'{a:.1%}' for a in accs)})")
    return accs

def train_and_save(ticker: str, horizon: int = 1) -> dict:
    frame = ft.build_modeling_frame(ticker, horizon=horizon, dropna=True)
    if len(frame) < 500:
        print(f"  [skip] {ticker}: only {len(frame)} rows")
        return None

    clf = train_classifier(frame, verbose=False)
    reg = train_regressor(frame, verbose=False)

    bundle = {
        "ticker": ticker, "horizon": horizon,
        "clf_model": clf["model"], "clf_scaler": clf["scaler"],
        "reg_model": reg["model"], "reg_scaler": reg["scaler"],
        "features": clf["features"],
    }
    path = cfg.MODELS / f"{ticker}_predictor.pkl"
    joblib.dump(bundle, path)

    row = {
        "ticker": ticker,
        "dir_acc": clf["accuracy"],
        "baseline": clf["baseline"],
        "edge": clf["accuracy"] - clf["baseline"],
        "mae": reg["mae"],
        "rmse": reg["rmse"],
        "r2": reg["r2"],
        "reg_dir_acc": reg["dir_acc"],
    }
    return row


def train_universe(tickers: list[str], horizon: int = 1) -> pd.DataFrame:
    rows = []
    for t in tickers:
        print(f"Training {t} ...")
        r = train_and_save(t, horizon=horizon)
        if r:
            rows.append(r)
    results = pd.DataFrame(rows).set_index("ticker")
    return results

import sys
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "train":
    # Full valid universe — same stocks the portfolio module uses
    import portfolio as pf
    rets, dropped = pf.build_universe_returns()
    universe = list(rets.columns)
    print(f"── Training predictor models for {len(universe)} stocks ──")
    print(f"   (full valid universe; dropped gap-heavy: {dropped})\n")
    results = train_universe(universe, horizon=1)

    if results.empty:
        print("\n[!] No results — check [ERROR]/[warn] lines above.")
    else:
        print("\nRESULTS")
        show_cols = ["dir_acc", "baseline", "edge", "r2", "reg_dir_acc"]
        if "ic" in results.columns:
            show_cols.append("ic")
        print(results[show_cols].sort_values("edge", ascending=False)
              .to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"\nMean directional accuracy: {results['dir_acc'].mean():.1%}")
        print(f"Mean edge over baseline  : {results['edge'].mean():+.2%}")
        print(f"Stocks beating baseline  : {(results['edge'] > 0).sum()}/{len(results)}")

        # Save the results table for the report
        results.to_csv(cfg.OUTPUTS / "predictor_results.csv")
        print(f"\nResults table saved to: {cfg.OUTPUTS / 'predictor_results.csv'}")
        print(f"Models saved to: {cfg.MODELS}")