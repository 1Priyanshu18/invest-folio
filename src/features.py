import numpy as np
import pandas as pd

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

import config as cfg
import data_loader as dl

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    close = df["Close"]
    high, low = df["High"], df["Low"]
    volume = df["Volume"]

    # 1. Returns
    # log returns
    log_ret = np.log(close / close.shift(1))
    if "split_flag" in df.columns:
        log_ret[df["split_flag"]] = np.nan
    out["log_ret"] = log_ret

    # Lagged returns
    for lag in (1, 2, 3, 5, 10):
        out[f"ret_lag_{lag}"] = log_ret.shift(lag)

    # Multi-day cumulative momentum
    for win in (5, 10, 21):
        out[f"ret_sum_{win}"] = log_ret.rolling(win).sum()

    # 2. Trends
    for win in (10, 20, 50):
        sma = SMAIndicator(close, window=win).sma_indicator()
        ema = EMAIndicator(close, window=win).ema_indicator()
        # ratio of price to its MA
        out[f"price_sma_{win}"] = close / sma - 1.0
        out[f"price_ema_{win}"] = close / ema - 1.0

    # MA crossover signal (short vs long)
    sma10 = SMAIndicator(close, window=10).sma_indicator()
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    out["sma_10_50_ratio"] = sma10 / sma50 - 1.0

    # 3. Momentum: RSI, MACD
    out["rsi_14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close) # 12/26/9 defaults
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # 4. Volatility
    for win in (5, 10, 21):
        out[f"vol_{win}"] = log_ret.rolling(win).std()

    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    out["atr_14"] = atr / close # normalized ATR

    bb = BollingerBands(close, window=20, window_dev=2)
    out["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    out["bb_pctb"] = bb.bollinger_pband() # where price sits in the band (0..1)

    # 5. Volume
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    out["volume_z"] = (volume - vol_mean) / vol_std   # volume spike detector

    # %Deliverble = fraction of trades taken to delivery = CONVICTION.
    # High delivery + price up = genuine accumulation, not day-trading froth.
    if cfg.DELIV_COL in df.columns:
        deliv = pd.to_numeric(df[cfg.DELIV_COL], errors="coerce")
        out["deliv_pct"] = deliv
        out["deliv_z"] = (deliv - deliv.rolling(20).mean()) / deliv.rolling(20).std()

    # Calendar (mild seasonality)
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month

    return out



# Targets (kept separate from features)
def add_targets(features: pd.DataFrame, df: pd.DataFrame,
                horizon: int = 1) -> pd.DataFrame:
    out = features.copy()
    close = df["Close"]

    fwd_ret = np.log(close.shift(-horizon) / close)
    # neutralize splits in the target window too
    if "split_flag" in df.columns:
        # if a split occurs within the forward window, blank the label
        split_ahead = df["split_flag"].shift(-horizon).fillna(False)
        fwd_ret[split_ahead] = np.nan

    out["target_ret"] = fwd_ret
    out["target_dir"] = (fwd_ret > 0).astype("Int64")  # nullable int
    out.loc[fwd_ret.isna(), "target_dir"] = pd.NA
    return out


def build_modeling_frame(ticker: str, horizon: int = 1,
                         dropna: bool = True) -> pd.DataFrame:
    df = dl.load_stock(ticker)
    feats = make_features(df)
    full = add_targets(feats, df, horizon=horizon)
    if dropna:
        full = full.dropna()
    return full

if __name__ == "__main__":
    frame = build_modeling_frame("RELIANCE", horizon=1, dropna=False)
    feature_cols = [c for c in frame.columns if not c.startswith("target_")]

    print(f"RELIANCE modeling frame: {frame.shape}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"\nFeatures:\n{feature_cols}")

    clean = frame.dropna()
    print(f"\nRows after dropna: {clean.shape[0]} "
          f"(warm-up lost {frame.shape[0] - clean.shape[0]} rows)")

    print(f"\nTarget balance (direction):")
    print(clean["target_dir"].value_counts(normalize=True).round(3))

    print(f"\nSample (last 3 rows, few cols):")
    print(clean[["log_ret", "rsi_14", "macd_diff", "vol_21",
                 "deliv_pct", "target_ret", "target_dir"]].tail(3))