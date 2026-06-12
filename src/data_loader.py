from pathlib import Path
import warnings
import numpy as np
import pandas as pd

import config as cfg

# 1. Listing all available stock tickers
def list_tickers() -> list[str]:
    csvs = cfg.DATA_RAW.glob('*.csv')
    tickers = [p.stem for p in csvs if p.name not in cfg.NON_STOCK_FILES]
    return sorted(tickers)

# 2. Load a single cleaned stock
def load_stock(ticker:str, detect_splits:bool=True) -> pd.DataFrame:
    path = cfg.DATA_RAW/f"{ticker}.csv"

    if not path.exists():
        raise FileNotFoundError(f"No csv for {ticker} at {path}")
    
    df = pd.read_csv(path, parse_dates=['Date'])

    if 'Series' in df.columns:
        df = df[df['Series'] == 'EQ'].copy()

    df = df.sort_values('Date').reset_index(drop=True)

    df['ticker'] = ticker

    df = df.drop_duplicates(subset='Date', keep='last').reset_index(drop=True)

    if detect_splits:
        df = flag_splits(df)

    df = df.set_index('Date')
    return df

# 3. Split/anomaly detection 
def flag_splits(df : pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:

    df = df.copy()
    prev_close_actual = df['Close'].shift(1)
    overnight_ret = (df['Close']/prev_close_actual) - 1.0
    pc_mismatch = (df['Prev Close'] - prev_close_actual).abs()/prev_close_actual

    df['split_flag'] = ((overnight_ret.abs() > threshold) | (pc_mismatch > threshold)).fillna(False)

    return df

# 4. Loading metadata file
def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(cfg.METADATA_FILE)
    meta.columns = [c.strip() for c in meta.columns]
    meta = meta.set_index('Symbol')
    return meta

def get_sector_map() -> dict[str, str]:
    meta = load_metadata()
    sector = meta['Industry'].to_dict()
    return {t: sector.get(t, 'UNKNOWN') for t in list_tickers()}

# 5. Combining closing prices of all the stocks in one single dataframe
def load_price_panel(tickers: list[str] | None = None, price_col: str = cfg.PRICE_COL) -> pd.DataFrame:
    if tickers is None:
        tickers = list_tickers()

    series = {}
    for t in tickers:
        try:
            s = load_stock(t, detect_splits=False)[price_col]
            series[t] = s
        except Exception as e:
            warnings.warn(f"Skipping {t}: {e}")

    panel = pd.DataFrame(series).sort_index()
    return panel

# 5b. Alignment helpers (robust to late listings + internal gaps)
def align_on_common_dates(panel: pd.DataFrame, ffill_gaps: bool = True, max_ffill: int = 5,)-> pd.DataFrame:
    
    if panel.shape[1] == 0:
        return panel
    
    starts = panel.apply(lambda c: c.first_valid_index())
    ends   = panel.apply(lambda c: c.last_valid_index())
    window_start = starts.max()
    window_end   = ends.min()

    if pd.isna(window_start) or pd.isna(window_end) or window_start > window_end:
        return panel.iloc[0:0]

    aligned = panel.loc[window_start:window_end].copy()

    # Forward-fill only short internal gaps
    if ffill_gaps:
        aligned = aligned.ffill(limit=max_ffill)

    # 3. Drop whatever still has holes
    aligned = aligned.dropna(axis=0, how="any")
    return aligned

def align_drop_sparse(panel: pd.DataFrame, max_internal_gap: int = 30,
                      ffill_gaps: bool = True, max_ffill: int = 5,
                      ) -> tuple[pd.DataFrame, list[str]]:
    keep, dropped = [], []
    for t in panel.columns:
        s = panel[t]
        fv = s.first_valid_index()
        if fv is None:
            dropped.append(t)
            continue
        life = s.loc[fv:]
        is_na = life.isna().astype(int)
        longest_gap = is_na.groupby((is_na != is_na.shift()).cumsum()).cumsum().max()
        if longest_gap > max_internal_gap:
            dropped.append(t)
        else:
            keep.append(t)
    aligned = align_on_common_dates(panel[keep], ffill_gaps=ffill_gaps, max_ffill=max_ffill)
    return aligned, dropped