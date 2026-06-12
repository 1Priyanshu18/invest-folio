import numpy as np
import pandas as pd

import config as cfg

# Return/volatility basics
def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return r.mean() * cfg.TRADING_DAYS


def annualized_volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return r.std() * np.sqrt(cfg.TRADING_DAYS)


def cagr(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    # Treat as log returns: total growth = exp(sum)
    total_growth = np.exp(r.sum())
    years = len(r) / cfg.TRADING_DAYS
    return total_growth ** (1 / years) - 1

# Risk-adjusted ratios
def sharpe_ratio(returns: pd.Series, rf: float = cfg.RISK_FREE_RATE) -> float:
    r = returns.dropna()
    vol = annualized_volatility(r)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (annualized_return(r) - rf) / vol


def sortino_ratio(returns: pd.Series, rf: float = cfg.RISK_FREE_RATE) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if len(downside) == 0:
        return np.nan
    downside_dev = downside.std() * np.sqrt(cfg.TRADING_DAYS)
    if downside_dev == 0:
        return np.nan
    return (annualized_return(r) - rf) / downside_dev


# Drawdown
def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    cum = np.exp(r.cumsum())
    running_peak = cum.cummax()
    drawdown = cum / running_peak - 1.0
    return drawdown.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    cum = np.exp(r.cumsum())
    return cum / cum.cummax() - 1.0


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return cagr(returns) / abs(mdd)

# Market-relative: Beta
def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    df = pd.concat([returns, market_returns], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    s, m = df.iloc[:, 0], df.iloc[:, 1]
    var_m = m.var()
    if var_m == 0:
        return np.nan
    return s.cov(m) / var_m

def risk_summary(returns: pd.Series,
                 market_returns: pd.Series | None = None,
                 rf: float = cfg.RISK_FREE_RATE) -> dict:
    out = {
        "ann_return": annualized_return(returns),
        "cagr": cagr(returns),
        "ann_vol": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, rf),
        "sortino": sortino_ratio(returns, rf),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
    }
    if market_returns is not None:
        out["beta"] = beta(returns, market_returns)
    return out

if __name__ == "__main__":
    import data_loader as dl

    # Reliance returns (split-neutralized, same as EDA Cell 4)
    df = dl.load_stock("RELIANCE")
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    log_ret[df["split_flag"]] = np.nan

    # Build a market proxy: equal-weight average of all stocks' returns
    panel = dl.load_price_panel().apply(pd.to_numeric, errors='coerce')
    mkt_ret = np.log(panel / panel.shift(1)).mean(axis=1)

    print("── RELIANCE risk summary ──")
    summary = risk_summary(log_ret, market_returns=mkt_ret)
    for k, v in summary.items():
        print(f"  {k:14s}: {v:.4f}")

    print("\nCross-check vs EDA Cell 4:")
    print(f"  ann_vol should be ~0.349 → got {summary['ann_vol']:.3f}")
    print(f"  ann_return should be ~0.166 → got {summary['ann_return']:.3f}")

    # Sanity: a few stocks ranked by Sharpe
    print("\n── Sharpe across 5 stocks ──")
    for t in ["ASIANPAINT", "RELIANCE", "COALINDIA", "ZEEL", "HDFCBANK"]:
        d = dl.load_stock(t)
        lr = np.log(d["Close"] / d["Close"].shift(1))
        lr[d["split_flag"]] = np.nan
        print(f"  {t:12s}: Sharpe {sharpe_ratio(lr):+.3f}  "
              f"Sortino {sortino_ratio(lr):+.3f}  "
              f"MaxDD {max_drawdown(lr):+.2%}")