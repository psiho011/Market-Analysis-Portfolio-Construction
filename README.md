# 💼 Pension Portfolio Builder - Quick Start Guide

A Python tool that builds and analyzes institutional pension portfolios with policy constraints, risk analysis, and beautiful visualizations.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Python Packages

```bash
pip install numpy pandas yfinance matplotlib scipy
```

**Optional (but recommended):**
```bash
pip install seaborn pandas-datareader
```

### Step 2: Run the Script

```bash
python pension_portfolio_auto_default.py
```

### Step 3: Choose Your Path

**Option A - Use Pre-Built Portfolio (EASIEST)**
```
Enter profile: AGGRESSIVE
Start date: [press Enter]
Risk-free rate: [press Enter]
Inflation: [press Enter]
```

**Option B - Build Custom Portfolio**
```
Enter profile: [press Enter to skip]
Start date: 2018-01-01
Risk-free rate: 0.04
Inflation: 0.025
Max weight: 0.10

US_EQUITY stocks: JPM,MSFT,AAPL,GOOGL
TRANSITION_EQUITY: ICLN,TAN,FAN,NEE
IG_CREDIT ETFs: LQD,VCIT,AGG,BND
EM_HC_DEBT ETFs: EMB,VWOB,CEMB
CASH ETFs: BIL,SGOV,MINT
PRIVATE_CLIMATE: PSP
```

Done! The script will generate 10+ charts and detailed analysis.

---

## 📊 What You Get

### **10+ Visualizations**
1. Portfolio equity curve (growth over time)
2. Drawdown chart (worst losses)
3. Portfolio weight bar chart
4. **Correlation heatmap** (how assets move together)
5. **3D correlation surface** (cool 3D view)
6. Sleeve performance comparison (each asset class)
7. **Sharpe ratio comparison** (risk-adjusted returns by sleeve)
8. **Beta comparison** (market exposure by sleeve)
9. Rolling 1-year Sharpe ratio
10. Return & risk contribution pie charts
11. Efficient frontier with your portfolio marked

### **Detailed Analytics**
- Expected return, volatility, Sharpe ratio
- Realized backtest performance
- **Sharpe ratios by asset class** ✨
- **Market betas by asset class** ✨
- Sleeve contribution to return & risk
- Tracking error vs benchmarks
- Fama-French factor exposure (if available)

---

## 🎯 Pre-Built Portfolios

### **CONSERVATIVE** → 5-6% Nominal Returns
- **Goal**: Steady growth, low volatility
- **For**: Risk-averse pensions, mature funds
- **Allocation**: 45% bonds, 30% equity, 25% alternatives
- **Real Return**: 2.5-3.5% (after 2.5% inflation)

### **MODERATE** → 6.5-7.5% Nominal Returns
- **Goal**: Balanced growth and stability
- **For**: Most traditional pensions
- **Allocation**: 35% bonds, 45% equity, 20% alternatives
- **Real Return**: 4-5% (after 2.5% inflation)

### **AGGRESSIVE** → 8-9% Nominal Returns
- **Goal**: High growth, accept volatility
- **For**: Young funds, long time horizon
- **Allocation**: 25% bonds, 60% equity, 15% alternatives
- **Real Return**: 5.5-6.5% (after 2.5% inflation) ✅

---

## 🔧 Common Issues & Fixes

### Issue 1: "All weights capped at 5%"

**Problem**: You don't have enough securities for the sleeve minimums.

**Fix**: Either increase max weight or add more tickers.

```python
# When prompted:
Max weight per security: 0.10  # Change from 0.05 to 0.10
```

**Or** add more tickers to each sleeve (recommended):
```
US_EQUITY: JPM,JNJ,PG,MSFT,UNH,BRK.B,V,WMT,XOM,CVX,AAPL,GOOGL,HD,MA,KO
# At least 10-15 per equity sleeve
```

### Issue 2: "ModuleNotFoundError: No module named 'seaborn'"

**Fix**: Either install it or ignore (script works without it):
```bash
pip install seaborn
```

If you don't install it, you'll get slightly less pretty heatmaps (but everything still works).

### Issue 3: "Optimization failed" or "Constraint violation"

**Problem**: Your constraints are infeasible (can't be satisfied).

**Fixes**:
1. **Use wider max weight**: 0.10 instead of 0.05
2. **Add more securities** to sleeves that are failing
3. **Use a pre-built profile** (they're guaranteed to work)

### Issue 4: Downloads fail / "No data returned"

**Problem**: Yahoo Finance is down or tickers are wrong.

**Fixes**:
1. Check your internet connection
2. Verify ticker symbols are correct (use ALL CAPS)
3. Try a more recent start date: `2020-01-01`
4. Remove problematic tickers and re-run

### Issue 5: "Not enough overlapping data for FF5"

**Fix**: This is optional analysis. If it fails, everything else still works.

To fix it:
```bash
pip install pandas-datareader
```

---

## 📝 Customization Examples

### Example 1: $20 Billion Pension (Original Goal)

```
Profile: AGGRESSIVE
Max weight: 0.05  (5% = $1 billion per position)

Why? At $20B scale:
- 5% = $1B per security (reasonable)
- Need 60-80 total positions
- Use ETFs for credit/EM debt (instant diversification)
```

### Example 2: Target 7.5% Nominal (5% Real) Returns

```
Profile: AGGRESSIVE
Inflation: 0.025

Expected result:
- Nominal: 8-9%
- Real: 5.5-6.5%
- Sharpe: ~1.0
```

### Example 3: Conservative with Custom Stocks

```
Profile: [press Enter]
Max weight: 0.08

US_EQUITY: JPM,JNJ,PG,KO,WMT,XOM,CVX
TRANSITION_EQUITY: NEE,BEP,ETN
IG_CREDIT: LQD,VCIT,IGIB,USIG,AGG,BND,VCSH
PRIVATE_CLIMATE: PSP
EM_HC_DEBT: EMB,VWOB,PCY
CASH: BIL,SGOV,SHV,MINT
```

---

## 💡 Pro Tips

### Tip 1: Start with Defaults
Your first run should be:
```
Profile: MODERATE
[Press Enter for all other prompts]
```
This lets you see how everything works before customizing.

### Tip 2: Position Sizing Math
```
Minimum positions needed per sleeve = Sleeve minimum ÷ Max weight

Example:
- IG_CREDIT minimum = 25%
- Max weight = 5%
- Need at least: 25% ÷ 5% = 5 positions

For clean allocation, use 8-10 positions per sleeve.
```

### Tip 3: ETFs vs Individual Stocks

**Use ETFs for:**
- IG Credit (LQD, VCIT, AGG)
- EM Debt (EMB, VWOB)
- Cash (BIL, SGOV)
- Small sleeves (<$2B allocation)

**Use Individual Stocks for:**
- US Equity (JPM, MSFT, AAPL)
- Transition Equity (if you want specific exposure)
- Large sleeves (>$4B allocation)

### Tip 4: Realistic Backtest Period

```
Start date: 2018-01-01  ← Good (includes COVID crash)
Start date: 2015-01-01  ← Better (more history)
Start date: 2020-01-01  ← Too short, misleading
```

### Tip 5: Reading the Output

**Look for these key metrics:**

✅ **Expected Nominal Return**: Should match your target (5%, 7%, 9%)

✅ **Sharpe Ratio**: 
- \> 1.0 = Excellent
- 0.7-1.0 = Good
- < 0.7 = Needs work

✅ **Sleeve Validation**: All should show "✓ OK"

✅ **Max Drawdown**: How much you'd lose in worst case
- Conservative: -15% to -25%
- Moderate: -25% to -35%
- Aggressive: -35% to -45%

---

## 🎓 Understanding the Output

### Section 1: Sleeve Allocation Validation
```
US_EQUITY          0.2450  [0.15, 0.25]  ✓ OK
IG_CREDIT          0.3200  [0.25, 0.40]  ✓ OK
```
**What it means**: Your portfolio meets all policy constraints.

### Section 2: Expected Statistics
```
Expected Nominal Return:   7.85%
Expected Real Return:      5.35%
Expected Volatility:      12.50%
Sharpe Ratio:              0.987
```
**What it means**: Your portfolio should return 7.85% per year with 12.5% volatility.

### Section 3: Sharpe Ratios by Sleeve
```
              Return  Volatility  Sharpe
US_EQUITY      0.085       0.180   1.250
IG_CREDIT      0.038       0.055   0.145
TRANSITION     0.092       0.220   1.145
```
**What it means**: US_EQUITY has the best risk-adjusted returns (Sharpe = 1.25).

### Section 4: Betas by Sleeve
```
                Beta  Alpha_annual     R2
US_EQUITY      0.98         0.015  0.920
TRANSITION     1.15        -0.005  0.650
IG_CREDIT      0.12         0.008  0.085
```
**What it means**: 
- US_EQUITY moves with the market (Beta ≈ 1.0)
- TRANSITION_EQUITY is more volatile (Beta = 1.15)
- IG_CREDIT barely moves with stocks (Beta = 0.12)

---

## 📚 File Outputs

The script generates all visualizations as pop-up windows. 

**To save them**:
1. Click the save icon (💾) in each plot window
2. Or modify the script to auto-save:

```python
# Add at the end of each plot function:
plt.savefig('chart_name.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 🆘 Still Stuck?

### Quick Checklist
- [ ] Python 3.8+ installed?
- [ ] All packages installed? (`pip install numpy pandas yfinance matplotlib scipy`)
- [ ] Using correct ticker symbols? (ALL CAPS, valid symbols)
- [ ] Start date not too old? (2015+ recommended)
- [ ] Enough tickers per sleeve? (5+ minimum, 10+ recommended)
- [ ] Max weight large enough? (0.08-0.10 for custom, or use defaults)

### Common Error Messages

**"ValueError: No data returned"**
→ Check tickers, try recent start date

**"Optimization did not converge"**
→ Use default profile OR increase max weight OR add more tickers

**"Not enough usable tickers"**
→ Some tickers failed to download, add more to each sleeve

**"KeyError" or "IndexError"**
→ Ticker mismatch, verify all tickers are valid

---

## 🌟 Success Criteria

Your portfolio is ready when:

✅ All sleeve constraints show "✓ OK"

✅ Expected return matches your target (±0.5%)

✅ No optimization warnings

✅ Realized backtest Sharpe > 0.7

✅ No single position > max weight

✅ At least 30 total positions

✅ Each sleeve has 5+ positions

---

## 📖 Real-World Example

Let's build a **$20B aggressive pension** targeting **8% nominal returns**:

```bash
$ python pension_portfolio_auto_default.py

Enter profile: AGGRESSIVE
Start date: 2018-01-01
Risk-free rate: 0.04
Inflation: 0.025

✓ Generated portfolio with 31 ETFs across 6 sleeves

OPTIMIZED WEIGHTS (Top 30)
QQQ     0.0500
SPY     0.0500
ICLN    0.0500
BX      0.0500
...

EXPECTED STATISTICS
Expected Nominal Return:   8.23%  ✓ Target achieved
Expected Real Return:      5.73%  ✓ Above 5% target
Expected Volatility:      13.80%
Sharpe Ratio:              1.051  ✓ Excellent

REALIZED BACKTEST
Annualized Return:         9.12%
Maximum Drawdown:         -32.4%  ← Expect in worst case

[10 charts generated...]

ANALYSIS COMPLETE
```

**Result**: 8.23% nominal, 5.73% real, 1.05 Sharpe. Perfect! 🎉

---

## 🔗 Quick Reference

| Goal | Profile | Expected Nominal | Expected Real | Risk Level |
|------|---------|------------------|---------------|------------|
| Safety First | CONSERVATIVE | 5.0-6.0% | 2.5-3.5% | Low |
| Balanced | MODERATE | 6.5-7.5% | 4.0-5.0% | Medium |
| High Growth | AGGRESSIVE | 8.0-9.0% | 5.5-6.5% | High |

---

**Questions?** 
- Review the output tables carefully
- Check that all sleeve allocations are ✓ OK  
- Verify your tickers are valid (google each one)
- Try a default profile first before customizing

**Happy portfolio building!** 💼📈

