import numpy as np
import pandas as pd
import shap
import joblib

import config as cfg
import features as ft
import predictor as pred

# 1. Load a saved model bundle
def load_bundle(ticker: str) -> dict:
    path = cfg.MODELS / f"{ticker}_predictor.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model for {ticker}. Run: python predictor.py train")
    return joblib.load(path)


# 2. Build a SHAP explainer for the classifier
def build_explainer(ticker: str, sample_size: int = 500):
    bundle = load_bundle(ticker)
    model = bundle["clf_model"]
    scaler = bundle["clf_scaler"]
    feat_cols = bundle["features"]

    # Rebuild the same modeling frame, take the TEST portion
    frame = ft.build_modeling_frame(ticker, horizon=bundle["horizon"],
                                    dropna=True)
    _, test = pred.chrono_split(frame)
    X_test = test[feat_cols]

    # Sample for speed (SHAP on thousands of rows is slow; 500 is plenty)
    if len(X_test) > sample_size:
        X_sample = X_test.sample(sample_size, random_state=cfg.RANDOM_SEED)
    else:
        X_sample = X_test
    X_sample = X_sample.sort_index()

    X_scaled = scaler.transform(X_sample)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feat_cols,
                               index=X_sample.index)

    # TreeExplainer:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)

    return explainer, shap_values, X_scaled_df, feat_cols


# 3. Global feature importance (mean |SHAP|)
def global_importance(shap_values, feat_cols) -> pd.Series:
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.Series(mean_abs, index=feat_cols).sort_values(ascending=False)
    return imp

# 4. Local explanation:
def explain_prediction(ticker: str, date=None) -> dict:
    bundle = load_bundle(ticker)
    model = bundle["clf_model"]
    scaler = bundle["clf_scaler"]
    feat_cols = bundle["features"]

    frame = ft.build_modeling_frame(ticker, horizon=bundle["horizon"],dropna=True)
    _, test = pred.chrono_split(frame)
    X_test = test[feat_cols]

    # pick the row to explain
    if date is None:
        row = X_test.iloc[[-1]]
    else:
        row = X_test.loc[[pd.Timestamp(date)]]

    X_scaled = scaler.transform(row)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_scaled)
    sv = np.array(sv).reshape(-1)

    proba = model.predict_proba(X_scaled)[0, 1]   # P(up)
    pred_dir = "UP" if proba > 0.5 else "DOWN"

    contrib = pd.Series(sv, index=feat_cols).sort_values()
    pushers_down = contrib.head(4)
    pushers_up = contrib.tail(4).iloc[::-1]

    return {
        "ticker": ticker,
        "date": row.index[0],
        "prediction": pred_dir,
        "prob_up": proba,
        "raw_values": row.iloc[0],
        "push_up": pushers_up,
        "push_down": pushers_down,
    }


def plain_english(explanation: dict) -> str:
    e = explanation
    raw = e["raw_values"]

    def describe(feat):
        v = raw.get(feat, np.nan)
        if feat == "rsi_14":
            if v > 70:   return f"RSI is overbought ({v:.0f})"
            if v < 30:   return f"RSI is oversold ({v:.0f})"
            return f"RSI is neutral ({v:.0f})"
        if feat.startswith("deliv"):
            return f"delivery conviction is {'high' if v > 0 else 'low'}"
        if feat.startswith("vol_"):
            return f"recent volatility ({feat})"
        if feat == "sma_10_50_ratio":
            return f"trend is {'up' if v > 0 else 'down'} (short vs long MA)"
        if feat.startswith("ret_lag"):
            return f"past return {feat.split('_')[-1]}d ago"
        return feat

    up_terms = ", ".join(describe(f) for f in e["push_up"].index[:3])
    down_terms = ", ".join(describe(f) for f in e["push_down"].index[:3])

    return (
        f"{e['ticker']} — predicted {e['prediction']} for the next day "
        f"(P(up) = {e['prob_up']:.0%}).\n"
        f"  Pushing UP   : {up_terms}\n"
        f"  Pushing DOWN : {down_terms}"
    )

if __name__ == "__main__":
    ticker = "SBIN"   # our best-edge stock from Phase 5
    print(f"── SHAP explainability: {ticker} ──\n")

    explainer, shap_values, X_sample, feat_cols = build_explainer(ticker)
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    print(f"Sample explained : {X_sample.shape[0]} predictions\n")

    imp = global_importance(shap_values, feat_cols)
    print("Top 12 features by mean |SHAP| (global importance):")
    for feat, val in imp.head(12).items():
        bar = "█" * int(val / imp.max() * 30)
        print(f"  {feat:16s} {val:.4f}  {bar}")

    print(f"\nLeast important 5:")
    for feat, val in imp.tail(5).items():
        print(f"  {feat:16s} {val:.4f}")

    print("\n── Local explanation (most recent test day) ──")
    exp = explain_prediction("SBIN")
    print(plain_english(exp))
    print(f"\nTop features pushing UP:")
    for f, v in exp["push_up"].items():
        print(f"  {f:16s} {v:+.4f}")
    print(f"Top features pushing DOWN:")
    for f, v in exp["push_down"].items():
        print(f"  {f:16s} {v:+.4f}")