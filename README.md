# Portfolio Risk & Allocation Analyzer

**A wealth management tool for multi-asset portfolio analysis, risk assessment, and capital allocation optimization.**

Built as part of [The Meridian Playbook](https://themeridianplaybook.com) — a research project on capital allocation, portfolio strategy and global financial systems.

---

## What It Does

This tool downloads real market data and produces a full portfolio analysis report covering:

- **Cumulative performance** across asset classes from a $1M starting capital
- **Efficient Frontier** via Monte Carlo simulation (5,000 portfolios)
- **Maximum Sharpe Ratio** portfolio optimization
- **Minimum Volatility** portfolio optimization
- **Risk metrics** — annualised return, volatility, Sharpe ratio, max drawdown
- **Correlation matrix** across assets
- **Capital allocation breakdown** for each optimised portfolio

---

## Asset Universe (Default)

| Asset | Ticker | Description |
|-------|--------|-------------|
| US Equities | SPY | S&P 500 ETF |
| European Equities | EZU | MSCI Eurozone ETF |
| Emerging Markets | EEM | MSCI EM ETF |
| US Bonds | TLT | 20+ Year Treasury ETF |
| Gold | GLD | Gold ETF |
| Real Estate | VNQ | US REIT ETF |

You can modify the `PORTFOLIO` dictionary in the config section to use any tickers.

---

## Output

The tool generates a single high-resolution report (`outputs/portfolio_report.png`) with 9 panels:

1. Cumulative performance chart
2. Efficient Frontier scatter plot
3. Return vs. Volatility comparison
4. Sharpe Ratio by asset
5. Maximum Drawdown by asset
6. Correlation heatmap
7. Max Sharpe portfolio allocation
8. Min Volatility portfolio allocation
9. Summary statistics

---

## Installation

```bash
git clone https://github.com/your-username/portfolio-analyzer.git
cd portfolio-analyzer
pip install -r requirements.txt
```

## Usage

```bash
python src/portfolio_analyzer.py
```

To customize the analysis, edit the configuration block at the top of `portfolio_analyzer.py`:

```python
PORTFOLIO = {
    "US Equities (SPY)":       "SPY",
    "European Equities (EZU)": "EZU",
    # add or remove assets here
}

START_DATE       = "2019-01-01"
END_DATE         = "2024-12-31"
RISK_FREE        = 0.04        # annual risk-free rate
N_SIMULATIONS    = 5000        # Monte Carlo portfolios
INITIAL_CAPITAL  = 1_000_000   # USD
```

---

## Methodology

**Efficient Frontier (Monte Carlo)**
Random portfolio weights are sampled using a Dirichlet distribution. For each portfolio, annualised return, volatility, and Sharpe ratio are computed.

**Portfolio Optimization (scipy)**
The Maximum Sharpe Ratio and Minimum Volatility portfolios are found using SLSQP optimization with long-only constraints (1%–60% per asset).

**Risk Metrics**
- Annualised return: daily mean × 252
- Annualised volatility: daily std × √252
- Sharpe ratio: (return − risk-free rate) / volatility
- Max drawdown: max peak-to-trough decline over the period

---

## Context

This project is part of my broader research on **capital allocation and portfolio strategy** at [The Meridian Playbook](https://themeridianplaybook.com).

The tool is designed to reflect how a portfolio manager or wealth advisor thinks about risk-adjusted returns and asset allocation — not just returns in isolation.

---

*Federico Bonessi — MSc Finance, IÉSEG School of Management*  
*[LinkedIn](https://www.linkedin.com/in/federico-bonessi/) | [The Meridian Playbook](https://themeridianplaybook.com)*
