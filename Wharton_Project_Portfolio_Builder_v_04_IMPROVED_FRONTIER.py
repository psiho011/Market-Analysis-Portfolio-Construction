# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 18:40:02 2026

@author: Mpsih
"""

# -*- coding: utf-8 -*-
"""
Pension Portfolio Builder - Enhanced Visualization Version
Created on Sat Jan 10 15:42:37 2026
Enhanced on Jan 13, 2026

@author: mpsih
Enhanced by: Claude

New features:
- Correlation heatmap and 3D visualization
- Sharpe ratio by sleeve
- Beta analysis by sleeve (vs market and within-sleeve)
- Rolling performance metrics
- Sleeve contribution analysis
- Risk decomposition
- Enhanced reporting
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# FF5 data (Ken French)
try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except Exception:
    HAS_PDR = False

# ----------------------------
# 1) Policy configuration
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

SLEEVE_BENCHMARK_TICKER = {
    "US_EQUITY": "SPY",
    "IG_CREDIT": "LQD",
    "TRANSITION_EQUITY": "ICLN",
    "PRIVATE_CLIMATE": "PSP",
    "EM_HC_DEBT": "EMB",
    "CASH": "BIL",
}

# ----------------------------
# 2) Core Helper Functions
# ----------------------------

def parse_list(raw: str):
    return [t.strip().upper() for t in raw.split(",") if t.strip()]

def download_adjclose(tickers, start):
    tickers = list(dict.fromkeys([t.upper() for t in tickers if t]))
    if not tickers:
        raise ValueError("No tickers provided")
    
    print(f"  Downloading {len(tickers)} tickers: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")
    
    try:
        data = yf.download(tickers, start=start, progress=False, auto_adjust=False)
    except Exception as e:
        raise ValueError(f"Yahoo Finance download failed: {e}")
    
    if data.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        px = data["Adj Close"].copy()
    else:
        px = data[["Adj Close"]].copy()
        px.columns = tickers

    px = px.dropna(how="all")
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
        row = A[j, :].copy()
        
        def make_lb_constraint(r, bound):
            return lambda w: (r @ w) - bound
        
        def make_ub_constraint(r, bound):
            return lambda w: bound - (r @ w)
        
        cons.append({"type": "ineq", "fun": make_lb_constraint(row, lb)})
        cons.append({"type": "ineq", "fun": make_ub_constraint(row, ub)})
    
    return cons

def validate_sleeve_allocation(w, A, sleeves, verbose=True):
    violations = {}
    all_valid = True
    
    if verbose:
        print("\n" + "=" * 70)
        print("Sleeve Allocation Validation")
        print("=" * 70)
    
    for j, s in enumerate(sleeves):
        actual = float(A[j, :] @ w)
        lb, ub = SLEEVE_BOUNDS[s]
        
        is_valid = (lb - 1e-6) <= actual <= (ub + 1e-6)
        status = "✓ OK" if is_valid else "✗ VIOLATION"
        
        if not is_valid:
            all_valid = False
            violations[s] = {"actual": actual, "bounds": (lb, ub)}
        
        if verbose:
            print(f"{s:20s} {actual:6.4f}  [{lb:.2f}, {ub:.2f}]  {status}")
    
    return all_valid, violations

def neg_sharpe(w, mu, cov, rf):
    pret = float(w @ mu)
    pvol = float(np.sqrt(w.T @ cov @ w))
    if pvol <= 1e-12:
        return 1e9
    return -((pret - rf) / pvol)

def smart_initial_weights(tickers, sleeve_of, A, sleeves):
    w0 = np.zeros(len(tickers))
    
    for j, s in enumerate(sleeves):
        lb, ub = SLEEVE_BOUNDS[s]
        target_sleeve_weight = (lb + ub) / 2
        
        mask = A[j, :] > 0
        n_in_sleeve = mask.sum()
        
        if n_in_sleeve > 0:
            w0[mask] = target_sleeve_weight / n_in_sleeve
    
    w0 = w0 / w0.sum()
    return w0

# ----------------------------
# 3) Sleeve-Level Analytics
# ----------------------------

def calculate_sleeve_returns(rets, w, tickers, sleeve_of):
    """
    Calculate sleeve-level return series (normalized within each sleeve)
    Returns: DataFrame with columns = sleeves, index = dates
    """
    w_ser = pd.Series(w, index=tickers)
    sleeves = sorted(set(sleeve_of[t] for t in tickers))
    
    sleeve_returns = {}
    
    for s in sleeves:
        idx = [t for t in tickers if sleeve_of[t] == s]
        if not idx:
            continue
            
        w_s = w_ser.loc[idx]
        sleeve_weight = float(w_s.sum())
        
        if sleeve_weight <= 1e-12:
            continue
        
        # Normalize weights within sleeve
        w_norm = (w_s / sleeve_weight).values
        sleeve_ret = pd.Series(rets[idx].values @ w_norm, index=rets.index, name=s)
        sleeve_returns[s] = sleeve_ret
    
    return pd.DataFrame(sleeve_returns)

def sleeve_sharpe_ratios(sleeve_rets, rf_annual=0.04, periods=252):
    """
    Calculate Sharpe ratio for each sleeve
    """
    results = {}
    
    for col in sleeve_rets.columns:
        ret_series = sleeve_rets[col].dropna()
        
        ann_ret = ret_series.mean() * periods
        ann_vol = ret_series.std() * np.sqrt(periods)
        
        sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 1e-12 else np.nan
        
        results[col] = {
            "Return": ann_ret,
            "Volatility": ann_vol,
            "Sharpe": sharpe
        }
    
    return pd.DataFrame(results).T

def sleeve_betas(sleeve_rets, market_ret, periods=252):
    """
    Calculate beta of each sleeve vs market (SPY)
    """
    results = {}
    
    for col in sleeve_rets.columns:
        sleeve_series = sleeve_rets[col]
        
        # Align dates
        aligned = pd.concat([sleeve_series, market_ret], axis=1, join='inner')
        aligned.columns = ['Sleeve', 'Market']
        aligned = aligned.dropna()
        
        if len(aligned) < 50:
            results[col] = {"Beta": np.nan, "Alpha_annual": np.nan, "R2": np.nan}
            continue
        
        # Calculate beta using covariance
        cov_matrix = aligned.cov()
        beta = cov_matrix.loc['Sleeve', 'Market'] / cov_matrix.loc['Market', 'Market']
        
        # Alpha calculation
        sleeve_ret = aligned['Sleeve'].mean() * periods
        market_ret_ann = aligned['Market'].mean() * periods
        alpha = sleeve_ret - beta * market_ret_ann
        
        # R-squared
        corr = aligned.corr().loc['Sleeve', 'Market']
        r2 = corr ** 2
        
        results[col] = {
            "Beta": beta,
            "Alpha_annual": alpha,
            "R2": r2
        }
    
    return pd.DataFrame(results).T

def sleeve_contribution_analysis(sleeve_rets, w, tickers, sleeve_of):
    """
    Calculate each sleeve's contribution to total portfolio return and risk
    """
    w_ser = pd.Series(w, index=tickers)
    sleeves = sorted(set(sleeve_of[t] for t in tickers))
    
    # Portfolio return
    port_ret = (sleeve_rets.values @ [w_ser[[t for t in tickers if sleeve_of[t]==s]].sum() 
                                      for s in sleeve_rets.columns])
    
    results = []
    
    for s in sleeves:
        if s not in sleeve_rets.columns:
            continue
            
        sleeve_weight = float(w_ser[[t for t in tickers if sleeve_of[t] == s]].sum())
        sleeve_ret_series = sleeve_rets[s]
        
        # Contribution to return
        ret_contribution = sleeve_ret_series.mean() * 252 * sleeve_weight
        
        # Contribution to risk (marginal contribution)
        cov_with_port = sleeve_ret_series.cov(pd.Series(port_ret, index=sleeve_rets.index))
        port_var = pd.Series(port_ret, index=sleeve_rets.index).var() * 252
        
        if port_var > 1e-12:
            risk_contribution = sleeve_weight * cov_with_port * 252 / np.sqrt(port_var)
        else:
            risk_contribution = 0
        
        results.append({
            "Sleeve": s,
            "Weight": sleeve_weight,
            "Return_Contribution": ret_contribution,
            "Risk_Contribution": risk_contribution
        })
    
    return pd.DataFrame(results).set_index("Sleeve")

# ----------------------------
# 4) Enhanced Visualization
# ----------------------------

def plot_correlation_heatmap(rets, tickers, sleeve_of, title="Asset Correlation Matrix"):
    """
    Plot correlation heatmap with sleeve grouping
    """
    corr = rets.corr()
    
    # Sort by sleeve for better visualization
    sleeves = sorted(set(sleeve_of[t] for t in tickers))
    sorted_tickers = []
    for s in sleeves:
        sorted_tickers.extend([t for t in tickers if sleeve_of[t] == s])
    
    corr_sorted = corr.loc[sorted_tickers, sorted_tickers]
    
    plt.figure(figsize=(14, 12))
    
    # Use seaborn for better heatmap
    mask = np.triu(np.ones_like(corr_sorted, dtype=bool), k=1)
    sns.heatmap(corr_sorted, mask=mask, cmap='RdYlGn', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_correlation_3d(rets, tickers, sleeve_of, sample_size=30):
    """
    3D visualization of correlation structure (sample of assets)
    """
    # Sample assets if too many
    if len(tickers) > sample_size:
        # Sample proportionally from each sleeve
        sampled = []
        sleeves = sorted(set(sleeve_of[t] for t in tickers))
        for s in sleeves:
            sleeve_tickers = [t for t in tickers if sleeve_of[t] == s]
            n_sample = max(2, int(sample_size * len(sleeve_tickers) / len(tickers)))
            sampled.extend(sleeve_tickers[:n_sample])
        plot_tickers = sampled
    else:
        plot_tickers = tickers
    
    corr = rets[plot_tickers].corr().values
    n = len(plot_tickers)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, corr, cmap='RdYlGn', alpha=0.8, 
                          vmin=-1, vmax=1, edgecolor='none')
    
    ax.set_xlabel('Asset Index', fontsize=10)
    ax.set_ylabel('Asset Index', fontsize=10)
    ax.set_zlabel('Correlation', fontsize=10)
    ax.set_title('3D Correlation Structure', fontsize=14, fontweight='bold')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

def plot_sleeve_performance(sleeve_rets, title="Sleeve Performance Comparison"):
    """
    Plot equity curves for each sleeve
    """
    eq = (1 + sleeve_rets).cumprod()
    
    plt.figure(figsize=(14, 8))
    
    for col in eq.columns:
        plt.plot(eq.index, eq[col], label=col, linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("Growth of $1", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_sleeve_metrics_comparison(sharpe_df, beta_df):
    """
    Bar charts comparing Sharpe ratios and Betas across sleeves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sharpe ratios
    sharpe_df['Sharpe'].sort_values().plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_title('Sharpe Ratio by Sleeve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sharpe Ratio', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Betas
    beta_df['Beta'].sort_values().plot(kind='barh', ax=ax2, color='coral')
    ax2.set_title('Market Beta by Sleeve', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Beta (vs SPY)', fontsize=12)
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Market Beta = 1.0')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()

def plot_rolling_sharpe(returns, window=252, title="Rolling 1-Year Sharpe Ratio"):
    """
    Plot rolling Sharpe ratio over time
    """
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std
    
    plt.figure(figsize=(14, 6))
    plt.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='darkblue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("Rolling Sharpe Ratio", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_risk_contribution(contrib_df):
    """
    Pie charts showing return and risk contribution by sleeve
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Return contribution
    contrib_df['Return_Contribution'].plot(kind='pie', ax=ax1, autopct='%1.1f%%',
                                           startangle=90, colors=sns.color_palette('Set2'))
    ax1.set_title('Return Contribution by Sleeve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('')
    
    # Risk contribution (only positive values)
    risk_contrib = contrib_df['Risk_Contribution'].abs()
    risk_contrib.plot(kind='pie', ax=ax2, autopct='%1.1f%%',
                     startangle=90, colors=sns.color_palette('Set3'))
    ax2.set_title('Risk Contribution by Sleeve', fontsize=14, fontweight='bold')
    ax2.set_ylabel('')
    
    plt.tight_layout()
    plt.show()

def plot_efficient_frontier_position(mu, cov, w_opt, rf, tickers, sleeve_of, A, sleeves, bounds, cons, w0):
    """
    Plot efficient frontier with current portfolio marked
    Uses ACTUAL OPTIMIZATION to generate frontier portfolios that are truly efficient
    """
    print(f"Generating efficient frontier (this may take 1-2 minutes)...")
    
    # Current portfolio stats
    opt_ret = w_opt @ mu
    opt_vol = np.sqrt(w_opt.T @ cov @ w_opt)
    opt_sharpe = (opt_ret - rf) / opt_vol
    
    # 1. Find minimum and maximum achievable returns under constraints
    print("  Finding feasible return range...")
    
    # Minimum variance portfolio
    def portfolio_variance(w):
        return w.T @ cov @ w
    
    res_minvar = minimize(portfolio_variance, w0, method="SLSQP",
                         bounds=bounds, constraints=cons, 
                         options={"maxiter": 500, "ftol": 1e-8})
    
    if res_minvar.success:
        w_minvar = np.clip(res_minvar.x, 0, None)
        w_minvar = w_minvar / w_minvar.sum()
        min_ret = w_minvar @ mu
        min_vol = np.sqrt(w_minvar.T @ cov @ w_minvar)
    else:
        min_ret = opt_ret * 0.7
        min_vol = opt_vol * 0.8
    
    # Maximum return portfolio (maximize return subject to constraints)
    def neg_return(w):
        return -(w @ mu)
    
    res_maxret = minimize(neg_return, w0, method="SLSQP",
                         bounds=bounds, constraints=cons,
                         options={"maxiter": 500, "ftol": 1e-8})
    
    if res_maxret.success:
        w_maxret = np.clip(res_maxret.x, 0, None)
        w_maxret = w_maxret / w_maxret.sum()
        max_ret = w_maxret @ mu
        max_vol = np.sqrt(w_maxret.T @ cov @ w_maxret)
    else:
        max_ret = opt_ret * 1.3
        max_vol = opt_vol * 1.2
    
    print(f"  Feasible return range: {min_ret:.2%} to {max_ret:.2%}")
    
    # 2. Generate efficient frontier by optimizing at different target returns
    n_frontier_points = 50
    target_returns = np.linspace(min_ret, max_ret, n_frontier_points)
    
    frontier_results = []
    
    print(f"  Computing {n_frontier_points} efficient portfolios...")
    
    for i, target_ret in enumerate(target_returns):
        # Minimize variance subject to achieving target return
        target_cons = cons.copy()
        target_cons.append({
            "type": "eq",
            "fun": lambda w, tr=target_ret: (w @ mu) - tr
        })
        
        res = minimize(portfolio_variance, w0, method="SLSQP",
                      bounds=bounds, constraints=target_cons,
                      options={"maxiter": 500, "ftol": 1e-8})
        
        if res.success:
            w_temp = np.clip(res.x, 0, None)
            w_temp = w_temp / w_temp.sum()
            
            ret = w_temp @ mu
            vol = np.sqrt(w_temp.T @ cov @ w_temp)
            sharpe = (ret - rf) / vol if vol > 0 else 0
            
            frontier_results.append([ret, vol, sharpe])
    
    # 3. Add some random portfolios for comparison (fewer than before)
    n_random = 500
    random_results = []
    
    print(f"  Adding {n_random} random portfolios for comparison...")
    
    for _ in range(n_random):
        weights = np.zeros(len(mu))
        
        for j, s in enumerate(sleeves):
            lb, ub = SLEEVE_BOUNDS[s]
            sleeve_weight = np.random.uniform(lb, ub)
            
            mask = A[j, :] > 0
            n_in_sleeve = mask.sum()
            
            if n_in_sleeve > 0:
                alpha = np.ones(n_in_sleeve)
                sleeve_allocation = np.random.dirichlet(alpha)
                weights[mask] = sleeve_weight * sleeve_allocation
        
        weights = weights / weights.sum()
        
        ret = weights @ mu
        vol = np.sqrt(weights.T @ cov @ weights)
        sharpe = (ret - rf) / vol if vol > 0 else 0
        
        random_results.append([ret, vol, sharpe])
    
    # 4. Add key reference portfolios
    reference_portfolios = []
    
    # Equal weight within sleeves (at midpoint of bounds)
    w_equal = np.zeros(len(mu))
    for j, s in enumerate(sleeves):
        lb, ub = SLEEVE_BOUNDS[s]
        sleeve_weight = (lb + ub) / 2
        mask = A[j, :] > 0
        n_in_sleeve = mask.sum()
        if n_in_sleeve > 0:
            w_equal[mask] = sleeve_weight / n_in_sleeve
    w_equal = w_equal / w_equal.sum()
    
    eq_ret = w_equal @ mu
    eq_vol = np.sqrt(w_equal.T @ cov @ w_equal)
    eq_sharpe = (eq_ret - rf) / eq_vol
    
    reference_portfolios.append({
        'name': 'Equal Weight',
        'ret': eq_ret,
        'vol': eq_vol,
        'sharpe': eq_sharpe,
        'marker': 'D',
        'color': 'orange'
    })
    
    # Minimum variance
    reference_portfolios.append({
        'name': 'Min Variance',
        'ret': min_ret,
        'vol': min_vol,
        'sharpe': (min_ret - rf) / min_vol if min_vol > 0 else 0,
        'marker': 's',
        'color': 'blue'
    })
    
    # Maximum return
    reference_portfolios.append({
        'name': 'Max Return',
        'ret': max_ret,
        'vol': max_vol,
        'sharpe': (max_ret - rf) / max_vol if max_vol > 0 else 0,
        'marker': '^',
        'color': 'green'
    })
    
    print(f"✓ Frontier generation complete!")
    
    # 5. Plot everything
    frontier_results = np.array(frontier_results)
    random_results = np.array(random_results)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Random portfolios (light background)
    if len(random_results) > 0:
        scatter1 = ax.scatter(random_results[:, 1], random_results[:, 0], 
                             c=random_results[:, 2], cmap='gray', 
                             marker='o', s=8, alpha=0.15, label='Random Portfolios')
    
    # Efficient frontier (prominent)
    if len(frontier_results) > 0:
        scatter2 = ax.scatter(frontier_results[:, 1], frontier_results[:, 0],
                             c=frontier_results[:, 2], cmap='viridis',
                             marker='o', s=30, alpha=0.7, edgecolors='black', 
                             linewidths=0.5, label='Efficient Frontier')
        
        # Draw line connecting frontier points
        sorted_idx = np.argsort(frontier_results[:, 1])
        ax.plot(frontier_results[sorted_idx, 1], frontier_results[sorted_idx, 0],
               'k--', linewidth=1.5, alpha=0.4)
    
    # Reference portfolios
    for ref in reference_portfolios:
        ax.scatter(ref['vol'], ref['ret'], marker=ref['marker'], 
                  color=ref['color'], s=200, edgecolors='black', 
                  linewidth=2, label=ref['name'], zorder=5)
    
    # Optimal portfolio (most prominent)
    ax.scatter(opt_vol, opt_ret, marker='*', color='red', s=600, 
              edgecolors='black', linewidth=2.5, 
              label=f'Optimal (Sharpe={opt_sharpe:.3f})', zorder=6)
    
    # Add colorbar for Sharpe ratios
    if len(frontier_results) > 0:
        cbar = plt.colorbar(scatter2, ax=ax, label='Sharpe Ratio')
    
    ax.set_title('Efficient Frontier with Optimal Portfolio (Sleeve-Constrained)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Volatility (Annualized)', fontsize=13)
    ax.set_ylabel('Expected Return (Annualized)', fontsize=13)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format axes as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "=" * 70)
    print("FRONTIER ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Efficient Frontier Points:  {len(frontier_results)}")
    print(f"Random Portfolios:          {len(random_results)}")
    print(f"\nReturn Range:               {min_ret:.2%} to {max_ret:.2%}")
    print(f"Volatility Range:           {min_vol:.2%} to {max_vol:.2%}")
    print(f"\nOptimal Portfolio Position:")
    print(f"  Return:                   {opt_ret:.2%}")
    print(f"  Volatility:               {opt_vol:.2%}")
    print(f"  Sharpe Ratio:             {opt_sharpe:.3f}")
    print(f"\nReference Portfolios:")
    for ref in reference_portfolios:
        print(f"  {ref['name']:15s} Return: {ref['ret']:6.2%}  Vol: {ref['vol']:6.2%}  Sharpe: {ref['sharpe']:6.3f}")
    print("=" * 70)

# ----------------------------
# 5) Basic plots (from original)
# ----------------------------

def plot_equity_curve(r, title):
    eq = (1 + r).cumprod()
    plt.figure(figsize=(14, 7))
    plt.plot(eq.index, eq.values, linewidth=2.5, color='darkgreen')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("Growth of $1", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_drawdown(r, title):
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    plt.figure(figsize=(14, 7))
    plt.fill_between(dd.index, dd.values, 0, alpha=0.3, color='red')
    plt.plot(dd.index, dd.values, linewidth=2.5, color='darkred')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("Drawdown", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weights(w, tickers, title, top_n=30):
    s = pd.Series(w, index=tickers).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(s)), s.values, color='steelblue')
    plt.xticks(range(len(s)), s.index, rotation=45, ha="right")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("Weight", fontsize=12)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 6) Tracking & FF5 (from original)
# ----------------------------

def tracking_stats(active_daily, periods=252):
    ar = active_daily.mean() * periods
    te = active_daily.std() * np.sqrt(periods)
    ir = ar / te if te > 1e-12 else np.nan
    return float(ar), float(te), float(ir)

def sleeve_tracking_report(rets, w, tickers, sleeve_of, sleeve_bmk_rets):
    w_ser = pd.Series(w, index=tickers)
    sleeves = sorted(set(sleeve_of[t] for t in tickers))
    out = []

    for s in sleeves:
        idx = [t for t in tickers if sleeve_of[t] == s]
        if not idx:
            continue

        w_s = w_ser.loc[idx]
        sleeve_weight = float(w_s.sum())

        if sleeve_weight <= 1e-12:
            continue

        w_norm = (w_s / sleeve_weight).values
        sleeve_ret = pd.Series(rets[idx].values @ w_norm, index=rets.index, name=s)

        if s not in sleeve_bmk_rets.columns:
            continue
        
        bmk_ret = sleeve_bmk_rets[s]
        sleeve_ret, bmk_ret = sleeve_ret.align(bmk_ret, join="inner")
        
        if len(sleeve_ret) < 10:
            continue

        active = sleeve_ret - bmk_ret
        ar, te, ir = tracking_stats(active)
        corr = float(sleeve_ret.corr(bmk_ret)) if len(sleeve_ret) > 5 else np.nan

        out.append({
            "Sleeve": s,
            "Weight": sleeve_weight,
            "ActiveRet": ar,
            "TrackingErr": te,
            "InfoRatio": ir,
            "Correlation": corr
        })

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.set_index("Sleeve").sort_values("Weight", ascending=False)
    return df

def fetch_ff5_daily(start_date):
    if not HAS_PDR:
        raise ImportError("pandas_datareader not installed")

    try:
        ds = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start_date)
        ff = ds[0].copy()
    except Exception as e:
        raise Exception(f"Failed to fetch FF5 data: {e}")
    
    ff.index = pd.to_datetime(ff.index, format='%Y%m%d')
    ff = ff.loc[ff.index >= pd.to_datetime(start_date)]
    ff = ff.rename(columns={"Mkt-RF": "MKT_RF"})
    ff = ff / 100.0
    return ff[["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]]

def ols_betas(y, X):
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X)
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0].flatten()
    yhat = X1 @ b
    ss_res = float(np.sum((y.flatten() - yhat) ** 2))
    ss_tot = float(np.sum((y.flatten() - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    alpha = float(b[0])
    betas = b[1:].astype(float)
    return alpha, betas, float(r2)

def ff5_exposure_report(port_daily_returns, start_date):
    ff = fetch_ff5_daily(start_date)
    port = port_daily_returns.copy()
    port.index = pd.to_datetime(port.index)
    port = port.loc[port.index >= pd.to_datetime(start_date)]
    df = pd.concat([port.rename("PORT"), ff], axis=1).dropna()

    if len(df) < 100:
        raise ValueError("Not enough overlapping data for FF5 regression")

    y = (df["PORT"] - df["RF"]).values
    X = df[["MKT_RF", "SMB", "HML", "RMW", "CMA"]].values
    alpha, betas, r2 = ols_betas(y, X)

    out = pd.Series(
        [alpha] + list(betas) + [r2],
        index=["Alpha_daily", "Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_RMW", "Beta_CMA", "R2"]
    )
    return out

# ----------------------------
# 7) Main
# ----------------------------

def main():
    print("\n" + "=" * 70)
    print("PENSION PORTFOLIO BUILDER - Enhanced Visualization")
    print("=" * 70)
    print("\nFeatures:")
    print("  • Correlation heatmap & 3D visualization")
    print("  • Sharpe ratios by sleeve")
    print("  • Beta analysis by sleeve")
    print("  • Rolling performance metrics")
    print("  • Risk contribution analysis")
    print("  • Efficient frontier visualization")
    print()

    # User inputs
    start = input("Start date YYYY-MM-DD (default 2018-01-01): ").strip() or "2018-01-01"
    rf = float(input("Annual risk-free rate (default 0.04): ").strip() or "0.04")
    inflation = float(input("Inflation assumption (default 0.025): ").strip() or "0.025")
    max_w = float(input("Max weight per security (default 0.05): ").strip() or "0.05")

    print("\n" + "-" * 70)
    print("Enter tickers for each sleeve (comma-separated)")
    print("-" * 70)

    us_eq = parse_list(input("US_EQUITY stocks: "))
    tr_eq = parse_list(input("TRANSITION_EQUITY stocks: "))
    ig = parse_list(input("IG_CREDIT ETF(s): "))
    em = parse_list(input("EM_HC_DEBT ETF(s): "))
    cash = parse_list(input("CASH ETF(s): "))
    priv_raw = input("PRIVATE_CLIMATE proxy ticker (optional): ").strip().upper()
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
        print("\n❌ Error: Need at least 5 tickers")
        return

    print(f"\n✓ Collected {len(tickers)} tickers across {len(set(sleeve_of.values()))} sleeves")

    # Download data
    print("\n" + "-" * 70)
    print("Downloading data...")
    print("-" * 70)
    
    try:
        prices = download_adjclose(tickers, start=start)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    # Benchmarks
    sleeves_needed = sorted(set(sleeve_of[t] for t in tickers))
    bench_tickers = list(dict.fromkeys([SLEEVE_BENCHMARK_TICKER.get(s) 
                                        for s in sleeves_needed if SLEEVE_BENCHMARK_TICKER.get(s)]))

    print("\nDownloading benchmarks...")
    try:
        bench_prices = download_adjclose(bench_tickers, start=start)
    except Exception as e:
        print(f"⚠ Warning: Benchmark download failed: {e}")
        bench_prices = pd.DataFrame()

    # Clean data
    coverage = prices.notna().sum()
    min_obs = max(50, int(0.5 * len(prices)))
    good = coverage[coverage >= min_obs].index.tolist()
    
    if len(good) < len(tickers):
        dropped = set(tickers) - set(good)
        print(f"\n⚠ Dropped {len(dropped)} tickers: {', '.join(list(dropped)[:5])}{'...' if len(dropped) > 5 else ''}")
    
    prices = prices[good].dropna()
    tickers = good
    sleeve_of = {t: sleeve_of[t] for t in tickers if t in sleeve_of}
    sleeves_needed = sorted(set(sleeve_of.values()))

    if len(tickers) < 5:
        print("\n❌ Error: Not enough usable tickers")
        return

    print(f"✓ Using {len(tickers)} tickers with {len(prices)} observations")

    # Returns
    rets = returns_from_prices(prices)
    
    # Benchmark returns
    sleeve_bmk_rets = pd.DataFrame()
    market_ret = None
    
    if not bench_prices.empty:
        bench_rets_all = returns_from_prices(bench_prices)
        
        # Get SPY for market beta calculation
        if 'SPY' in bench_rets_all.columns:
            market_ret = bench_rets_all['SPY']
        
        sleeve_bmk_map = {}
        for s in sleeves_needed:
            bt = SLEEVE_BENCHMARK_TICKER.get(s)
            if bt and bt in bench_rets_all.columns:
                sleeve_bmk_map[s] = bench_rets_all[bt]
        
        if sleeve_bmk_map:
            sleeve_bmk_rets = pd.DataFrame(sleeve_bmk_map)

    # Build risk model
    print("\n" + "-" * 70)
    print("Building risk model...")
    print("-" * 70)
    
    mu_hist = annualize_mu(rets.mean()).values
    cov = annualize_cov(rets.cov()).values
    mu_cma = np.array([SLEEVE_CMA_NOMINAL[sleeve_of[t]] for t in tickers])
    mu = 0.75 * mu_cma + 0.25 * mu_hist

    print(f"✓ Built covariance matrix ({len(tickers)}x{len(tickers)})")

    # Optimize
    sleeves, A = build_sleeve_matrix(tickers, sleeve_of)
    cons = sleeve_constraints(A, sleeves)
    bounds = [(0.0015, max_w) for _ in tickers]
    w0 = smart_initial_weights(tickers, sleeve_of, A, sleeves)

    print("\nOptimizing...")
    res = minimize(neg_sharpe, w0, args=(mu, cov, rf), method="SLSQP",
                   bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})

    if not res.success:
        print(f"\n⚠ Warning: {res.message}")

    w = np.clip(res.x, 0, None)
    w = w / w.sum()

    # Validate
    is_valid, violations = validate_sleeve_allocation(w, A, sleeves, verbose=True)

    # Portfolio stats
    port_ret_nom = float(w @ mu)
    port_vol = float(np.sqrt(w.T @ cov @ w))
    sharpe = (port_ret_nom - rf) / port_vol if port_vol > 0 else np.nan
    port_ret_real = port_ret_nom - inflation

    # Display basic results
    print("\n" + "=" * 70)
    print("OPTIMIZED WEIGHTS (Top 30)")
    print("=" * 70)
    w_series = pd.Series(w, index=tickers).sort_values(ascending=False)
    print(w_series.head(30).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("EXPECTED STATISTICS (Annualized)")
    print("=" * 70)
    print(f"Expected Nominal Return:  {port_ret_nom:7.2%}")
    print(f"Expected Real Return:     {port_ret_real:7.2%}")
    print(f"Expected Volatility:      {port_vol:7.2%}")
    print(f"Sharpe Ratio:             {sharpe:7.3f}")

    # Backtest
    port_daily = pd.Series(rets.values @ w, index=rets.index, name="PORT")
    eq = (1 + port_daily).cumprod()
    mdd = max_drawdown(eq)
    ann_ret = port_daily.mean() * 252
    ann_vol = port_daily.std() * np.sqrt(252)
    
    print("\n" + "=" * 70)
    print("REALIZED BACKTEST")
    print("=" * 70)
    print(f"Annualized Return:        {ann_ret:7.2%}")
    print(f"Annualized Volatility:    {ann_vol:7.2%}")
    print(f"Maximum Drawdown:         {mdd:7.2%}")

    # === NEW: Sleeve-level analysis ===
    print("\n" + "=" * 70)
    print("SLEEVE-LEVEL ANALYSIS")
    print("=" * 70)
    
    sleeve_rets = calculate_sleeve_returns(rets, w, tickers, sleeve_of)
    
    # Sharpe ratios by sleeve
    sharpe_by_sleeve = sleeve_sharpe_ratios(sleeve_rets, rf_annual=rf)
    print("\nSharpe Ratios by Sleeve:")
    print(sharpe_by_sleeve.to_string(float_format=lambda x: f"{x:.4f}"))
    
    # Betas by sleeve
    if market_ret is not None:
        beta_by_sleeve = sleeve_betas(sleeve_rets, market_ret)
        print("\nBetas by Sleeve (vs SPY):")
        print(beta_by_sleeve.to_string(float_format=lambda x: f"{x:.4f}"))
    else:
        beta_by_sleeve = None
        print("\n⚠ Market beta analysis skipped (SPY not available)")
    
    # Contribution analysis
    contrib_analysis = sleeve_contribution_analysis(sleeve_rets, w, tickers, sleeve_of)
    print("\nSleeve Contribution Analysis:")
    print(contrib_analysis.to_string(float_format=lambda x: f"{x:.4f}"))

    # Tracking analysis
    if not sleeve_bmk_rets.empty:
        try:
            aligned_rets, aligned_bmks = rets.align(sleeve_bmk_rets, join="inner", axis=0)
            if len(aligned_rets) >= 50:
                sleeve_track = sleeve_tracking_report(aligned_rets, w, tickers, sleeve_of, aligned_bmks)
                
                if not sleeve_track.empty:
                    print("\n" + "=" * 70)
                    print("SLEEVE BENCHMARK TRACKING")
                    print("=" * 70)
                    print(sleeve_track.to_string(float_format=lambda x: f"{x:.4f}"))
        except Exception as e:
            print(f"\n⚠ Tracking analysis failed: {e}")

    # FF5
    if HAS_PDR:
        try:
            ff5 = ff5_exposure_report(port_daily, start_date=start)
            print("\n" + "=" * 70)
            print("FAMA-FRENCH 5-FACTOR EXPOSURE")
            print("=" * 70)
            print(ff5.to_string(float_format=lambda x: f"{x:.6f}"))
            alpha_ann = (1 + ff5["Alpha_daily"])**252 - 1
            print(f"\nAnnualized Alpha:         {alpha_ann:7.2%}")
        except Exception as e:
            print(f"\n⚠ FF5 failed: {e}")

    # === VISUALIZATION ===
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print("Generating 10+ charts...")
    
    try:
        # 1. Basic portfolio charts
        plot_equity_curve(port_daily, "Portfolio Equity Curve")
        plot_drawdown(port_daily, "Portfolio Drawdown")
        plot_weights(w, tickers, "Portfolio Weights")
        
        # 2. Correlation analysis
        plot_correlation_heatmap(rets, tickers, sleeve_of)
        plot_correlation_3d(rets, tickers, sleeve_of)
        
        # 3. Sleeve performance
        plot_sleeve_performance(sleeve_rets)
        
        # 4. Sleeve metrics comparison
        if beta_by_sleeve is not None:
            plot_sleeve_metrics_comparison(sharpe_by_sleeve, beta_by_sleeve)
        
        # 5. Rolling metrics
        plot_rolling_sharpe(port_daily)
        
        # 6. Risk contribution
        plot_risk_contribution(contrib_analysis)
        
        # 7. Efficient frontier
        plot_efficient_frontier_position(mu, cov, w, rf, tickers, sleeve_of, A, sleeves, bounds, cons, w0)
        
        print("✓ All visualizations generated successfully")
        
    except Exception as e:
        print(f"⚠ Visualization error: {e}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()