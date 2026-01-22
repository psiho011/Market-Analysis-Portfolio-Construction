# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:42:37 2026

@author: mpsih
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# FF5 data (Ken French)
try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except Exception:
    HAS_PDR = False

# ----------------------------
# 1) Policy configuration (edit to match your report)
# ----------------------------

SLEEVE_BOUNDS = {
    "US_EQUITY": (0.15, 0.25),
    "IG_CREDIT": (0.25, 0.40),
    "TRANSITION_EQUITY": (0.15, 0.30),
    "PRIVATE_CLIMATE": (0.10, 0.20),
    "EM_HC_DEBT": (0.05, 0.15),
    "CASH": (0.03, 0.07),
}

SLEEVE_CMA_NOMINAL = {
    "US_EQUITY": 0.065,
    "IG_CREDIT": 0.035,
    "TRANSITION_EQUITY": 0.065,
    "PRIVATE_CLIMATE": 0.070,
    "EM_HC_DEBT": 0.060,
    "CASH": 0.030,
}

# Default benchmark proxies (change later if you want)
# Idea: each sleeve has a tradable benchmark series so we can compute tracking error.
SLEEVE_BENCHMARK_TICKER = {
    "US_EQUITY": "SPY",          # S&P 500 proxy
    "IG_CREDIT": "LQD",          # US IG credit proxy
    "TRANSITION_EQUITY": "ICLN", # clean energy proxy (can swap)
    "PRIVATE_CLIMATE": "PSP",    # listed private equity proxy (placeholder)
    "EM_HC_DEBT": "EMB",         # EM USD debt
    "CASH": "BIL",               # t-bills
}

# ----------------------------
# 2) Helpers
# ----------------------------

def parse_list(raw: str):
    return [t.strip().upper() for t in raw.split(",") if t.strip()]

def download_adjclose(tickers, start):
    tickers = list(dict.fromkeys([t.upper() for t in tickers if t]))
    data = yf.download(tickers, start=start, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        px = data["Adj Close"].copy()
    else:
        px = data[["Adj Close"]].copy()
        px.columns = tickers

    px = px.dropna(how="all").dropna()
    return px

def returns_from_prices(prices):
    return prices.pct_change().dropna()

def annualize_mu(mu_daily, periods=252):
    return mu_daily * periods

def annualize_cov(cov_daily, periods=252):
    return cov_daily * periods

def max_drawdown(eq):
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())

def build_sleeve_matrix(all_tickers, sleeve_of):
    sleeves = sorted(set(sleeve_of[t] for t in all_tickers))
    A = np.zeros((len(sleeves), len(all_tickers)))
    for j, s in enumerate(sleeves):
        for i, t in enumerate(all_tickers):
            if sleeve_of[t] == s:
                A[j, i] = 1.0
    return sleeves, A

def sleeve_constraints(A, sleeves):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for j, s in enumerate(sleeves):
        lb, ub = SLEEVE_BOUNDS[s]
        cons.append({"type": "ineq", "fun": lambda w, row=A[j, :], lb=lb: (row @ w) - lb})
        cons.append({"type": "ineq", "fun": lambda w, row=A[j, :], ub=ub: ub - (row @ w)})
    return cons

def neg_sharpe(w, mu, cov, rf):
    pret = float(w @ mu)
    pvol = float(np.sqrt(w.T @ cov @ w))
    if pvol <= 1e-12:
        return 1e9
    return -((pret - rf) / pvol)

def plot_equity_curve(r, title):
    eq = (1 + r).cumprod()
    plt.figure()
    plt.plot(eq.index, eq.values)
    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drawdown(r, title):
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    plt.figure()
    plt.plot(dd.index, dd.values)
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_weights(w, tickers, title):
    s = pd.Series(w, index=tickers).sort_values(ascending=False)
    plt.figure()
    plt.bar(s.index, s.values)
    plt.title(title)
    plt.ylabel("Weight")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

# ----------------------------
# 3) Benchmark tracking per sleeve
# ----------------------------

def tracking_stats(active_daily, periods=252):
    """
    active_daily: portfolio - benchmark (daily)
    Returns annualized active return, tracking error, info ratio.
    """
    ar = active_daily.mean() * periods
    te = active_daily.std() * np.sqrt(periods)
    ir = ar / te if te > 1e-12 else np.nan
    return float(ar), float(te), float(ir)

def sleeve_tracking_report(rets, w, tickers, sleeve_of, sleeve_bmk_rets):
    """
    rets: asset daily returns DataFrame (tickers)
    w: optimized weights array aligned to tickers
    sleeve_bmk_rets: DataFrame of benchmark daily returns (columns are sleeves)
    """
    w_ser = pd.Series(w, index=tickers)

    rows = []
    for sleeve, bmk_col in sleeve_bmk_rets.items():
        pass

    # Build sleeve returns (weighted within sleeve, normalized to sleeve weight)
    sleeves = sorted(set(sleeve_of[t] for t in tickers))
    out = []

    for s in sleeves:
        idx = [t for t in tickers if sleeve_of[t] == s]
        if not idx:
            continue

        w_s = w_ser.loc[idx]
        sleeve_weight = float(w_s.sum())

        # If sleeve weight ~0 (shouldn't happen due to policy mins), skip safely
        if sleeve_weight <= 1e-12:
            continue

        # Sleeve implementation return: weights inside sleeve normalized to 1
        w_norm = (w_s / sleeve_weight).values
        sleeve_ret = pd.Series(rets[idx].values @ w_norm, index=rets.index, name=s)

        # Benchmark return series for sleeve
        if s not in sleeve_bmk_rets.columns:
            continue
        bmk_ret = sleeve_bmk_rets[s].reindex(rets.index).dropna()
        sleeve_ret, bmk_ret = sleeve_ret.align(bmk_ret, join="inner")

        active = sleeve_ret - bmk_ret
        ar, te, ir = tracking_stats(active)

        corr = float(sleeve_ret.corr(bmk_ret)) if len(sleeve_ret) > 5 else np.nan

        out.append({
            "Sleeve": s,
            "SleeveWeight": sleeve_weight,
            "ActiveReturn": ar,
            "TrackingError": te,
            "InfoRatio": ir,
            "CorrToBench": corr
        })

    return pd.DataFrame(out).set_index("Sleeve").sort_values("SleeveWeight", ascending=False)

# ----------------------------
# 4) FF5 exposure for final portfolio
# ----------------------------

def fetch_ff5_daily(start_date):
    """
    Returns a DataFrame with columns:
    MKT_RF, SMB, HML, RMW, CMA, RF  (all in decimal, daily)
    """
    if not HAS_PDR:
        raise ImportError("pandas_datareader not installed. Run: pip install pandas_datareader")

    ds = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench")
    ff = ds[0].copy()  # percent values
    ff.index = pd.to_datetime(ff.index)
    ff = ff.loc[ff.index >= pd.to_datetime(start_date)]
    ff = ff.rename(columns={
        "Mkt-RF": "MKT_RF"
    })
    # convert percent to decimal
    ff = ff / 100.0
    return ff[["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]]

def ols_betas(y, X):
    """
    OLS with intercept using numpy.
    y: (n,) array
    X: (n,k) array  (already excludes intercept)
    Returns: alpha, betas (k,), r2
    """
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X)
    X1 = np.column_stack([np.ones(len(X)), X])  # intercept

    # beta = (X'X)^-1 X'y
    b = np.linalg.lstsq(X1, y, rcond=None)[0].flatten()

    yhat = X1 @ b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    alpha = float(b[0])
    betas = b[1:].astype(float)
    return alpha, betas, float(r2)

def ff5_exposure_report(port_daily_returns, start_date):
    """
    port_daily_returns: Series of daily simple returns
    """
    ff = fetch_ff5_daily(start_date)

    # Align to common dates
    port = port_daily_returns.copy()
    port.index = pd.to_datetime(port.index)
    port = port.loc[port.index >= pd.to_datetime(start_date)]

    df = pd.concat([port.rename("PORT"), ff], axis=1).dropna()

    # Excess return regression: (Rp - Rf) = alpha + b' * factors + eps
    y = (df["PORT"] - df["RF"]).values
    X = df[["MKT_RF", "SMB", "HML", "RMW", "CMA"]].values

    alpha, betas, r2 = ols_betas(y, X)

    out = pd.Series(
        [alpha] + list(betas) + [r2],
        index=["Alpha_daily", "Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_RMW", "Beta_CMA", "R2"]
    )
    return out

# ----------------------------
# 5) Main
# ----------------------------

def main():
    print("\n=== Policy-Constrained Portfolio Builder (Sleeves + Bench + FF5) ===\n")

    start = input("Start date YYYY-MM-DD (default 2018-01-01): ").strip() or "2018-01-01"
    rf = float(input("Annual risk-free rate (default 0.04): ").strip() or "0.04")
    inflation = float(input("Inflation assumption (default 0.025): ").strip() or "0.025")
    max_w = float(input("Max weight per security (default 0.05): ").strip() or "0.05")

    print("\nEnter STOCK tickers for equity sleeves, and ETF tickers for credit/EM/cash sleeves.\n")

    us_eq = parse_list(input("US_EQUITY stocks (comma-separated): "))
    tr_eq = parse_list(input("TRANSITION_EQUITY stocks (comma-separated): "))
    ig = parse_list(input("IG_CREDIT ETF(s) (comma-separated): "))
    em = parse_list(input("EM_HC_DEBT ETF(s) (comma-separated): "))
    cash = parse_list(input("CASH ETF(s) (comma-separated): "))

    priv_raw = input("PRIVATE_CLIMATE proxy ticker (optional, blank to skip): ").strip().upper()
    priv = [priv_raw] if priv_raw else []

    sleeve_of = {}
    for t in us_eq: sleeve_of[t] = "US_EQUITY"
    for t in tr_eq: sleeve_of[t] = "TRANSITION_EQUITY"
    for t in ig:    sleeve_of[t] = "IG_CREDIT"
    for t in em:    sleeve_of[t] = "EM_HC_DEBT"
    for t in cash:  sleeve_of[t] = "CASH"
    for t in priv:  sleeve_of[t] = "PRIVATE_CLIMATE"

    tickers = list(dict.fromkeys(us_eq + tr_eq + ig + em + cash + priv))
    if len(tickers) < 5:
        print("Not enough tickers to build a portfolio.")
        return

    # Download assets
    print("\nDownloading prices (assets)...")
    prices = download_adjclose(tickers, start=start)

    # Benchmarks per sleeve (download only the ones we actually need)
    sleeves_needed = sorted(set(sleeve_of[t] for t in tickers))
    bench_tickers = []
    for s in sleeves_needed:
        bt = SLEEVE_BENCHMARK_TICKER.get(s, None)
        if bt:
            bench_tickers.append(bt)

    bench_tickers = list(dict.fromkeys(bench_tickers))

    print("Downloading prices (sleeve benchmarks)...")
    bench_prices = download_adjclose(bench_tickers, start=start)

    # Clean assets: keep tickers with decent coverage
    good = prices.columns[prices.notna().sum() > 50].tolist()
    prices = prices[good].dropna()
    tickers = good

    # Rebuild sleeve map after filtering
    sleeve_of = {t: sleeve_of[t] for t in tickers if t in sleeve_of}
    sleeves_needed = sorted(set(sleeve_of[t] for t in tickers))

    if len(tickers) < 5:
        print("Not enough usable tickers after cleaning.")
        return

    rets = returns_from_prices(prices)

    # Benchmark returns per sleeve (map sleeve -> its benchmark ticker return series)
    bench_rets_all = returns_from_prices(bench_prices)

    sleeve_bmk_rets = {}
    for s in sleeves_needed:
        bt = SLEEVE_BENCHMARK_TICKER.get(s, None)
        if bt and bt in bench_rets_all.columns:
            sleeve_bmk_rets[s] = bench_rets_all[bt]

    sleeve_bmk_rets = pd.DataFrame(sleeve_bmk_rets).dropna(how="all")

    # Risk model from realized returns
    mu_hist = annualize_mu(rets.mean()).values
    cov = annualize_cov(rets.cov()).values

    # Expected returns anchored to sleeve CMAs + small historical tilt
    mu_cma = np.array([SLEEVE_CMA_NOMINAL[sleeve_of[t]] for t in tickers])
    mu = 0.75 * mu_cma + 0.25 * mu_hist

    # Constraints + optimize
    sleeves, A = build_sleeve_matrix(tickers, sleeve_of)
    cons = sleeve_constraints(A, sleeves)
    bounds = [(0.0, max_w) for _ in tickers]

    w0 = np.ones(len(tickers)) / len(tickers)
    res = minimize(neg_sharpe, w0, args=(mu, cov, rf), method="SLSQP",
                   bounds=bounds, constraints=cons, options={"maxiter": 500})

    if not res.success:
        print("\nOptimization failed:", res.message)
        return

    w = np.clip(res.x, 0, None)
    w = w / w.sum()

    # Portfolio expected stats
    port_ret_nom = float(w @ mu)
    port_vol = float(np.sqrt(w.T @ cov @ w))
    sharpe = (port_ret_nom - rf) / port_vol if port_vol > 0 else np.nan
    port_ret_real = port_ret_nom - inflation

    # Sleeve weights
    sleeve_w = {s: float(A[j, :] @ w) for j, s in enumerate(sleeves)}

    print("\n" + "=" * 70)
    print("Final Portfolio Weights (Top 30)")
    print("=" * 70)
    print(pd.Series(w, index=tickers).sort_values(ascending=False).head(30).to_string())

    print("\nSleeve weights (policy constraints):")
    for s in sleeves:
        lb, ub = SLEEVE_BOUNDS[s]
        print(f"{s:18s} {sleeve_w[s]:.4f}   (range {lb:.2f}–{ub:.2f})")

    print("\nExpected return / risk (annualized):")
    print(f"Nominal E[R] : {port_ret_nom:.4f}")
    print(f"Real E[R]    : {port_ret_real:.4f}   (inflation {inflation:.3f})")
    print(f"Vol          : {port_vol:.4f}")
    print(f"Sharpe       : {sharpe:.4f}")

    # Realized backtest with fixed weights (research view)
    port_daily = pd.Series(rets.values @ w, index=rets.index, name="PORT")
    eq = (1 + port_daily).cumprod()
    print(f"\nRealized Max Drawdown (fixed weights): {max_drawdown(eq):.4f}")

    # --- NEW: Sleeve benchmark tracking report ---
    if not sleeve_bmk_rets.empty:
        aligned_rets, aligned_bmks = rets.align(sleeve_bmk_rets, join="inner", axis=0)
        sleeve_track = sleeve_tracking_report(aligned_rets, w, tickers, sleeve_of, aligned_bmks)

        print("\n" + "=" * 70)
        print("Sleeve Benchmark Tracking (implementation vs benchmark)")
        print("=" * 70)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
        print(sleeve_track)

    else:
        print("\n(No sleeve benchmark returns available — check benchmark tickers.)")

    # --- NEW: FF5 exposure for final portfolio ---
    if HAS_PDR:
        try:
            ff5 = ff5_exposure_report(port_daily, start_date=start)
            print("\n" + "=" * 70)
            print("Fama-French 5-Factor Exposure (daily regression, excess returns)")
            print("=" * 70)
            print(ff5.to_string(float_format=lambda x: f"{x:,.6f}"))

            # optional: convert daily alpha to annual (approx)
            alpha_ann = (1 + ff5["Alpha_daily"])**252 - 1
            print(f"\nAlpha_annualized (approx): {alpha_ann:.4f}")

        except Exception as e:
            print("\nFF5 regression failed:", e)
    else:
        print("\nFF5 skipped: pandas_datareader not installed.")
        print("Install with: pip install pandas_datareader")

    # Plots
    plot_equity_curve(port_daily, "Policy-Constrained Portfolio Equity Curve (Fixed Weights)")
    plot_drawdown(port_daily, "Policy-Constrained Portfolio Drawdown (Fixed Weights)")
    plot_weights(w, tickers, "Final Security Weights")

if __name__ == "__main__":
    main()
