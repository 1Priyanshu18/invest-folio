import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
import data_loader as dl
import risk
import features as ft
import portfolio as pf

sns.set_style("whitegrid")
FIG = cfg.FIGURES


def _save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def split_neutralized_returns(ticker):
    d = dl.load_stock(ticker)
    close = pd.to_numeric(d["Close"], errors="coerce")
    lr = np.log(close / close.shift(1))
    lr[d["split_flag"]] = np.nan
    return d, lr


def fig_price_splits(ticker="RELIANCE"):
    d, _ = split_neutralized_returns(ticker)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    close = pd.to_numeric(d["Close"], errors="coerce")
    ax.plot(d.index, close, lw=0.8, label="Close")
    flagged = d[d["split_flag"]]
    ax.scatter(flagged.index, pd.to_numeric(flagged["Close"], errors="coerce"),
               color="red", s=35, zorder=5, label=f"Split/anomaly ({len(flagged)})")
    ax.set_title(f"{ticker} — Close price with flagged corporate actions")
    ax.set_ylabel("Price (₹)"); ax.legend()
    _save(fig, "01_price_splits.png")


def fig_return_dist(ticker="RELIANCE"):
    from scipy import stats
    _, lr = split_neutralized_returns(ticker)
    r = lr.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(r, bins=100, color="steelblue")
    axes[0].set_title(f"{ticker} — daily log-return distribution")
    stats.probplot(r, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q vs normal (kurtosis {r.kurtosis():.1f})")
    _save(fig, "02_return_distribution.png")


def fig_risk_return_scatter():
    cols = {}
    for t in dl.list_tickers():
        _, lr = split_neutralized_returns(t)
        cols[t] = lr
    panel = pd.DataFrame(cols)
    annual = pd.DataFrame({
        "ret": panel.mean() * cfg.TRADING_DAYS,
        "vol": panel.std() * np.sqrt(cfg.TRADING_DAYS),
    }).dropna()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(annual["vol"], annual["ret"], s=45, alpha=0.7)
    for t in annual.index:
        ax.annotate(t, (annual.loc[t, "vol"], annual.loc[t, "ret"]),
                    fontsize=6, alpha=0.7, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Annualized volatility"); ax.set_ylabel("Annualized return")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Risk vs return — all NIFTY-50 stocks")
    _save(fig, "03_risk_return_scatter.png")
    return panel


def fig_correlation(panel):
    corr = panel.corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, cmap="RdYlBu_r", center=0, vmin=0, vmax=1,
                square=True, linewidths=0.2, cbar_kws={"shrink": 0.6}, ax=ax)
    ax.set_title("Daily-return correlation — all NIFTY-50 stocks")
    _save(fig, "04_correlation_heatmap.png")


def fig_crashes():
    panel = dl.load_price_panel().apply(pd.to_numeric, errors="coerce").ffill().dropna(how="all")
    market = panel.div(panel.iloc[0]).mean(axis=1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(market.index, market.values, lw=1.2, color="navy")
    ax.axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2009-03-31"),
               color="red", alpha=0.15, label="2008 GFC")
    ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"),
               color="orange", alpha=0.20, label="COVID-19")
    ax.set_title("Equal-weighted market proxy — major crashes")
    ax.set_ylabel("Growth of ₹1"); ax.legend()
    _save(fig, "05_crashes.png")


def fig_efficient_frontier():
    rets, _ = pf.build_universe_returns()
    mc = pf.monte_carlo_frontier(rets, n_portfolios=20000)
    opt = pf.optimal_portfolios(rets)
    profiles = pf.build_investor_portfolios(rets)
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(mc["vol"], mc["ret"], c=mc["sharpe"], cmap="viridis", s=6, alpha=0.5)
    plt.colorbar(sc, label="Sharpe")
    ax.scatter(opt["max_sharpe"]["vol"], opt["max_sharpe"]["ret"], marker="*",
               s=450, color="red", edgecolor="black", label="Max Sharpe (Balanced)", zorder=5)
    ax.scatter(opt["min_variance"]["vol"], opt["min_variance"]["ret"], marker="*",
               s=450, color="deepskyblue", edgecolor="black", label="Min Variance (Conservative)", zorder=5)
    agg = profiles["Aggressive"]["metrics"]
    ax.scatter(agg["ann_vol"], agg["ann_return"], marker="D", s=150,
               color="orange", edgecolor="black", label="Aggressive", zorder=5)
    ax.set_xlabel("Annualized volatility"); ax.set_ylabel("Annualized return")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Efficient frontier — 20,000 simulated portfolios + optimal allocations")
    ax.legend(loc="lower right")
    _save(fig, "06_efficient_frontier.png")


def fig_drawdown(ticker="RELIANCE"):
    _, lr = split_neutralized_returns(ticker)
    dd = risk.drawdown_series(lr)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.6)
    ax.set_title(f"{ticker} — drawdown (underwater) curve")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    _save(fig, "07_drawdown.png")


if __name__ == "__main__":
    print("Generating report figures...")
    fig_price_splits()
    fig_return_dist()
    panel = fig_risk_return_scatter()
    fig_correlation(panel)
    fig_crashes()
    fig_efficient_frontier()
    fig_drawdown()