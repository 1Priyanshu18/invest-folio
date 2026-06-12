import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import config as cfg
import data_loader as dl
import risk
import features as ft
import portfolio as pf
import predictor as pred
import explain as ex

st.set_page_config(page_title="invest-folio", layout="wide")


@st.cache_data
def get_tickers():
    return dl.list_tickers()

@st.cache_data
def get_stock(ticker: str):
    """Load one stock with split flags + split-neutralized log returns."""
    df = dl.load_stock(ticker)
    close = pd.to_numeric(df["Close"], errors="coerce")
    lr = np.log(close / close.shift(1))
    lr[df["split_flag"]] = np.nan
    df["log_ret"] = lr
    return df

@st.cache_data
def get_market_returns():
    """Equal-weight market proxy for beta."""
    panel = dl.load_price_panel().apply(pd.to_numeric, errors="coerce")
    return np.log(panel / panel.shift(1)).mean(axis=1)

@st.cache_data
def get_sector_map():
    return dl.get_sector_map()


# Header
st.title("📈 invest-folio")
st.caption("Data-Driven Investment Intelligence on NIFTY-50 — "
           "prediction · portfolios · risk · explainability")

tickers = get_tickers()
sector_map = get_sector_map()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Stock Explorer",
    "🤖 Prediction",
    "💼 Portfolio Builder",
    "🌐 Market Overview",
])

# TAB 1 — STOCK EXPLORER
with tab1:
    st.header("Stock Explorer")

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        ticker = st.selectbox("Select a stock", tickers,
                              index=tickers.index("RELIANCE")
                              if "RELIANCE" in tickers else 0)
        sector = sector_map.get(ticker, "UNKNOWN")
        st.metric("Sector", sector)

    df = get_stock(ticker)
    mkt = get_market_returns()
    metrics = risk.risk_summary(df["log_ret"], market_returns=mkt)

    with col_info:
        st.subheader(f"{ticker} — key risk metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ann. Return", f"{metrics['ann_return']:.1%}")
        m2.metric("Ann. Volatility", f"{metrics['ann_vol']:.1%}")
        m3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
        m4.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Sortino", f"{metrics['sortino']:.2f}")
        m6.metric("Calmar", f"{metrics['calmar']:.2f}")
        m7.metric("Beta", f"{metrics.get('beta', float('nan')):.2f}")
        m8.metric("CAGR", f"{metrics['cagr']:.1%}")

    st.divider()

    # Price chart with moving averages
    st.subheader("Price history")
    price_df = pd.DataFrame({
        "Close": pd.to_numeric(df["Close"], errors="coerce"),
        "SMA50": pd.to_numeric(df["Close"], errors="coerce").rolling(50).mean(),
        "SMA200": pd.to_numeric(df["Close"], errors="coerce").rolling(200).mean(),
    })
    st.line_chart(price_df, height=350)

    # Drawdown (underwater) chart
    st.subheader("Drawdown (underwater curve)")
    dd = risk.drawdown_series(df["log_ret"])
    st.area_chart(dd, height=220, color="#d62728")

    # Recent technical snapshot
    with st.expander("Recent technical indicators (last 5 days)"):
        feats = ft.make_features(df)
        cols_show = ["log_ret", "rsi_14", "macd_diff", "vol_21",
                     "bb_pctb", "deliv_pct"]
        st.dataframe(feats[cols_show].tail(5).style.format("{:.4f}"))

# TAB 2 — PREDICTION + EXPLANATION
@st.cache_data
def list_trained_models():
    """Tickers that have a saved predictor bundle."""
    return sorted(p.stem.replace("_predictor", "")
                  for p in cfg.MODELS.glob("*_predictor.pkl"))

with tab2:
    st.header("Direction Prediction + Explanation")
    st.caption("Next-day direction forecast with a transparent, "
               "feature-level explanation of *why*.")

    trained = list_trained_models()
    if not trained:
        st.warning("No trained models found. Run `python predictor.py train` "
                   "from the src/ folder first.")
    else:
        pcol1, pcol2 = st.columns([1, 2])
        with pcol1:
            ptick = st.selectbox("Stock", trained, key="pred_ticker")

        try:
            exp = ex.explain_prediction(ptick)

            with pcol1:
                direction = exp["prediction"]
                prob = exp["prob_up"]
                color = "🟢" if direction == "UP" else "🔴"
                st.metric("Predicted direction", f"{color} {direction}")
                st.metric("P(up next day)", f"{prob:.0%}")
                st.progress(float(prob))

            with pcol2:
                st.subheader("Why this prediction?")
                st.code(ex.plain_english(exp), language=None)

            st.divider()

            # Feature contribution chart (SHAP push up vs down)
            st.subheader("Feature contributions (SHAP)")
            push = pd.concat([exp["push_up"], exp["push_down"]]).sort_values()
            fig, ax = plt.subplots(figsize=(9, 4))
            colors = ["#2ca02c" if v > 0 else "#d62728" for v in push.values]
            ax.barh(push.index, push.values, color=colors)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("SHAP value  (← pushes DOWN | pushes UP →)")
            ax.set_title(f"{ptick} — top feature contributions")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


        except Exception as e:
            st.error(f"Could not explain {ptick}: {e}")


# TAB 3 — PORTFOLIO BUILDER
@st.cache_data
def get_universe_returns():
    return pf.build_universe_returns()

@st.cache_data
def get_investor_portfolios():
    rets, dropped = get_universe_returns()
    return pf.build_investor_portfolios(rets), dropped

with tab3:
    st.header("Portfolio Builder")
    st.caption("Optimized allocations for three investor profiles, with "
               "quantitative justification.")

    with st.spinner("Optimizing portfolios..."):
        profiles, dropped = get_investor_portfolios()

    profile_name = st.radio(
        "Investor profile",
        ["Conservative", "Balanced", "Aggressive"],
        horizontal=True,
        captions=["Capital protection", "Best risk-adjusted return",
                  "Maximum growth"],
    )

    prof = profiles[profile_name]
    m = prof["metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exp. Return", f"{m['ann_return']:.1%}")
    c2.metric("Volatility", f"{m['ann_vol']:.1%}")
    c3.metric("Sharpe", f"{m['sharpe']:.2f}")
    c4.metric("Max Drawdown", f"{m['max_drawdown']:.1%}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sortino", f"{m['sortino']:.2f}")
    c6.metric("CAGR", f"{m['cagr']:.1%}")
    c7.metric("Beta", f"{m.get('beta', float('nan')):.2f}")
    c8.metric("Holdings", f"{prof['n_holdings']}")

    st.divider()

    acol1, acol2 = st.columns([1, 1])
    with acol1:
        st.subheader("Allocation")
        weights = pd.Series(prof["weights"]).sort_values(ascending=False)
        st.dataframe(
            weights.rename("weight").to_frame()
            .style.format("{:.1%}").background_gradient(cmap="Blues"),
            height=380,
        )
    with acol2:
        st.subheader("Allocation chart")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(weights.values, labels=weights.index, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 8})
        ax.set_title(f"{profile_name} portfolio")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Justification")
    st.code(pf.justify_portfolio(profile_name, prof), language=None)


# TAB 4 — MARKET OVERVIEW
@st.cache_data
def get_returns_panel():
    """Split-neutralized returns for all stocks (for correlation/scatter)."""
    cols = {}
    for t in tickers:
        d = dl.load_stock(t)
        close = pd.to_numeric(d["Close"], errors="coerce")
        lr = np.log(close / close.shift(1))
        lr[d["split_flag"]] = np.nan
        cols[t] = lr
    return pd.DataFrame(cols).sort_index()

with tab4:
    st.header("Market Overview")

    ret_panel = get_returns_panel()

    # Risk/return scatter
    st.subheader("Risk vs Return — all stocks")
    annual = pd.DataFrame({
        "ann_return": ret_panel.mean() * cfg.TRADING_DAYS,
        "ann_vol": ret_panel.std() * np.sqrt(cfg.TRADING_DAYS),
    }).dropna()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(annual["ann_vol"], annual["ann_return"], s=50, alpha=0.7)
    for t in annual.index:
        ax.annotate(t, (annual.loc[t, "ann_vol"], annual.loc[t, "ann_return"]),
                    fontsize=6, alpha=0.7, xytext=(2, 2),
                    textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Correlation heatmap
    st.subheader("Return correlation heatmap")
    import seaborn as sns
    corr = ret_panel.corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, cmap="RdYlBu_r", center=0, vmin=0, vmax=1,
                square=True, linewidths=0.2, cbar_kws={"shrink": 0.6}, ax=ax)
    ax.set_title("Daily return correlation — all NIFTY-50 stocks")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Least/most correlated pairs
    c = corr.where(~np.eye(len(corr), dtype=bool))
    pairs = c.unstack().dropna().sort_values()
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown("**Best diversifiers** (least correlated)")
        st.dataframe(pairs.head(6).rename("corr").to_frame()
                     .style.format("{:.3f}"))
    with oc2:
        st.markdown("**Move together** (most correlated)")
        st.dataframe(pairs.tail(6).iloc[::-1].rename("corr").to_frame()
                     .style.format("{:.3f}"))