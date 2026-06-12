import numpy as np
import pandas as pd

import config as cfg
import data_loader as dl
import risk

# 1. Building returns matrix for all the stocks
def build_universe_returns(max_internal_gap: int = 30):
    panel = dl.load_price_panel().apply(pd.to_numeric, errors="coerce")
    _, dropped = dl.align_drop_sparse(panel, max_internal_gap=max_internal_gap)
    keep = [c for c in panel.columns if c not in dropped]

    # Per-stock split-neutralized log returns
    ret_cols = {}
    for t in keep:
        d = dl.load_stock(t)
        close = pd.to_numeric(d["Close"], errors="coerce")
        lr = np.log(close / close.shift(1))
        lr[d["split_flag"]] = np.nan
        ret_cols[t] = lr

    rets = pd.DataFrame(ret_cols).sort_index()

    # Common window: from the latest stock-start to earliest stock-end
    starts = rets.apply(lambda c: c.first_valid_index())
    ends   = rets.apply(lambda c: c.last_valid_index())
    rets = rets.loc[starts.max():ends.min()]
    rets = rets.fillna(0.0)

    return rets, dropped

# 2. Portfolio performance for a given weight vector
def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series,
                          cov_matrix: pd.DataFrame,
                          rf: float = cfg.RISK_FREE_RATE):
    ann_ret = np.dot(weights, mean_returns) * cfg.TRADING_DAYS
    ann_var = np.dot(weights, np.dot(cov_matrix, weights)) * cfg.TRADING_DAYS
    ann_vol = np.sqrt(ann_var)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe


# ─────────────────────────────────────────────────────────────
# 3. Monte Carlo frontier
# ─────────────────────────────────────────────────────────────
def monte_carlo_frontier(rets: pd.DataFrame, n_portfolios: int = 20000,
                         max_weight: float = 0.25,
                         seed: int = cfg.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mean_returns = rets.mean()
    cov_matrix = rets.cov()
    n_assets = len(rets.columns)

    results = []
    weights_store = []
    # minimum assets needed so the cap is satisfiable: n*max_weight >= 1
    attempts = 0
    while len(results) < n_portfolios and attempts < n_portfolios * 3:
        attempts += 1
        w = rng.random(n_assets)
        w /= w.sum()
        if w.max() > max_weight:
            # push down over-cap weights by redrawing dirichlet-like
            w = rng.dirichlet(np.ones(n_assets))
            if w.max() > max_weight:
                continue
        ann_ret, ann_vol, sharpe = portfolio_performance(w, mean_returns, cov_matrix)
        results.append((ann_ret, ann_vol, sharpe))
        weights_store.append(w)

    df = pd.DataFrame(results, columns=["ret", "vol", "sharpe"])
    df["weights"] = weights_store
    return df


# 4. Exact optimizers (PyPortfolioOpt) with Monte Carlo fallback
def max_return_portfolio(rets: pd.DataFrame, max_weight: float = 0.40,
                         rf: float = cfg.RISK_FREE_RATE) -> dict:
    mean_ann = rets.mean()*cfg.TRADING_DAYS
    cov_ann = rets.cov()*cfg.TRADING_DAYS

    ranked = mean_ann.sort_values(ascending=False)
    weights = {}
    remaining = 1.0
    for tk in ranked.index:
        w = min(max_weight, remaining)
        if w <= 0:
            break
        weights[tk] = w
        remaining -= w

    w_vec = pd.Series(weights).reindex(rets.columns).fillna(0.0)
    ann_ret, ann_vol, sharpe = portfolio_performance(
        w_vec.values, rets.mean(), rets.cov(), rf)
    return {
        "weights": {k: v for k, v in weights.items() if v > 0},
        "ret": ann_ret, "vol": ann_vol, "sharpe": sharpe,
    }

def optimal_portfolios(rets: pd.DataFrame, max_weight: float = 0.25,
                       rf: float = cfg.RISK_FREE_RATE) -> dict:
    try:
        from pypfopt import EfficientFrontier, expected_returns, risk_models

        mu = expected_returns.mean_historical_return(
            rets, returns_data=True, frequency=cfg.TRADING_DAYS)
        S = risk_models.sample_cov(rets, returns_data=True,
                                   frequency=cfg.TRADING_DAYS)

        out = {}
        # Max Sharpe
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        ef.max_sharpe(risk_free_rate=rf)
        w = ef.clean_weights()
        r, v, s = ef.portfolio_performance(risk_free_rate=rf)
        out["max_sharpe"] = {"weights": w, "ret": r, "vol": v, "sharpe": s}

        # Min Variance
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        ef.min_volatility()
        w = ef.clean_weights()
        r, v, s = ef.portfolio_performance(risk_free_rate=rf)
        out["min_variance"] = {"weights": w, "ret": r, "vol": v, "sharpe": s}

        return out

    except Exception as e:  # noqa: BLE001
        print(f"[PyPortfolioOpt unavailable → Monte Carlo fallback] {e}")
        mc = monte_carlo_frontier(rets, max_weight=max_weight)
        cols = list(rets.columns)
        best_sharpe = mc.loc[mc["sharpe"].idxmax()]
        min_vol = mc.loc[mc["vol"].idxmin()]
        return {
            "max_sharpe": {
                "weights": dict(zip(cols, best_sharpe["weights"])),
                "ret": best_sharpe["ret"], "vol": best_sharpe["vol"],
                "sharpe": best_sharpe["sharpe"]},
            "min_variance": {
                "weights": dict(zip(cols, min_vol["weights"])),
                "ret": min_vol["ret"], "vol": min_vol["vol"],
                "sharpe": min_vol["sharpe"]},
        }

# 5. Investor-profile portfolios + justification
def build_investor_portfolios(rets: pd.DataFrame, max_weight: float = 0.25,
                              rf: float = cfg.RISK_FREE_RATE) -> dict:
    opt = optimal_portfolios(rets, max_weight=max_weight, rf=rf)

    # Aggressive: allow more concentration (higher cap) → chases return
    agg = max_return_portfolio(rets, max_weight=0.40, rf=rf)

    profiles = {
        "Conservative": opt["min_variance"],
        "Balanced": opt["max_sharpe"],
        "Aggressive": agg,
    }

    # Attach realized risk metrics by reconstructing each portfolio's historical return series and running it through risk.py
    market = rets.mean(axis=1)   # equal-weight proxy for beta
    enriched = {}
    for name, p in profiles.items():
        w = pd.Series(p["weights"]).reindex(rets.columns).fillna(0.0)
        port_ret = (rets * w).sum(axis=1)        # daily portfolio returns
        metrics = risk.risk_summary(port_ret, market_returns=market, rf=rf)
        # top holdings (non-trivial weights)
        holdings = {k: v for k, v in sorted(p["weights"].items(),
                    key=lambda x: -x[1]) if v > 0.005}
        enriched[name] = {
            "weights": holdings,
            "metrics": metrics,
            "n_holdings": len(holdings),
        }
    return enriched


def justify_portfolio(name: str, profile: dict) -> str:
    m = profile["metrics"]
    top = list(profile["weights"].items())[:4]
    top_str = ", ".join(f"{k} ({v:.0%})" for k, v in top)

    blurb = {
        "Conservative":
            "prioritizes capital protection. It minimizes portfolio "
            "volatility, accepting lower returns for a smoother ride and "
            "shallower drawdowns. Dominated by defensive sectors (FMCG, "
            "utilities, stable IT).",
        "Balanced":
            "maximizes risk-adjusted return (Sharpe). It seeks the best "
            "return per unit of risk, diversified across sectors with a "
            "25% per-stock cap to avoid concentration.",
        "Aggressive":
            "chases higher absolute returns, tolerating greater volatility "
            "and deeper drawdowns. A looser 40% cap allows more "
            "concentration in high-growth names.",
    }[name]

    return (
        f"The {name} portfolio {blurb}\n"
        f"  • Holdings   : {profile['n_holdings']} stocks — top: {top_str}\n"
        f"  • Exp. return: {m['ann_return']:.1%} (CAGR {m['cagr']:.1%})\n"
        f"  • Volatility : {m['ann_vol']:.1%}\n"
        f"  • Sharpe     : {m['sharpe']:.2f}   Sortino: {m['sortino']:.2f}\n"
        f"  • Max DD     : {m['max_drawdown']:.1%}   Beta: {m.get('beta', float('nan')):.2f}"
    )