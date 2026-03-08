"""
Portfolio Risk & Allocation Analyzer
=====================================
A wealth management tool for multi-asset portfolio analysis,
risk assessment, and capital allocation optimization.

Author: Federico Bonessi | The Meridian Playbook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PORTFOLIO = {
    "US Equities (SPY)":     "SPY",
    "European Equities (EZU)": "EZU",
    "Emerging Markets (EEM)": "EEM",
    "US Bonds (TLT)":        "TLT",
    "Gold (GLD)":            "GLD",
    "Real Estate (VNQ)":     "VNQ",
}

START_DATE  = "2019-01-01"
END_DATE    = "2024-12-31"
RISK_FREE   = 0.04          # Annual risk-free rate
N_SIMULATIONS = 5000        # Monte Carlo portfolios
INITIAL_CAPITAL = 1_000_000 # USD


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def fetch_data(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for all assets."""
    print("📥  Fetching market data...")
    raw = yf.download(list(tickers.values()), start=start, end=end, auto_adjust=True)["Close"]
    raw.columns = list(tickers.keys())
    raw.dropna(inplace=True)
    print(f"    ✓ {len(raw)} trading days | {raw.shape[1]} assets\n")
    return raw


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(prices: pd.DataFrame, rf: float = RISK_FREE) -> dict:
    """Compute annualised portfolio statistics."""
    returns = prices.pct_change().dropna()
    ann_returns = returns.mean() * 252
    ann_vol     = returns.std() * np.sqrt(252)
    sharpe      = (ann_returns - rf) / ann_vol

    # Max drawdown per asset
    drawdowns = {}
    for col in prices.columns:
        roll_max = prices[col].cummax()
        dd = (prices[col] - roll_max) / roll_max
        drawdowns[col] = dd.min()

    corr = returns.corr()

    return {
        "returns":   ann_returns,
        "volatility": ann_vol,
        "sharpe":    sharpe,
        "drawdowns": pd.Series(drawdowns),
        "corr":      corr,
        "daily_ret": returns,
    }


# ─────────────────────────────────────────────
# MODERN PORTFOLIO THEORY — EFFICIENT FRONTIER
# ─────────────────────────────────────────────

def run_monte_carlo(daily_ret: pd.DataFrame, n: int = N_SIMULATIONS, rf: float = RISK_FREE):
    """Simulate random portfolios to map the efficient frontier."""
    print("🎲  Running Monte Carlo simulation...")
    mu  = daily_ret.mean() * 252
    cov = daily_ret.cov()  * 252
    n_assets = len(mu)

    results = np.zeros((3, n))
    weights_store = []

    for i in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        p_ret = w @ mu.values
        p_vol = np.sqrt(w @ cov.values @ w)
        results[0, i] = p_ret
        results[1, i] = p_vol
        results[2, i] = (p_ret - rf) / p_vol
        weights_store.append(w)

    print(f"    ✓ {n:,} portfolios simulated\n")
    return results, weights_store, mu, cov


def optimize_portfolio(mu, cov, rf=RISK_FREE, target="sharpe"):
    """Find the Maximum Sharpe Ratio portfolio via scipy."""
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.01, 0.60) for _ in range(n))
    w0 = np.ones(n) / n

    if target == "sharpe":
        def neg_sharpe(w):
            r = w @ mu.values
            v = np.sqrt(w @ cov.values @ w)
            return -(r - rf) / v
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    elif target == "minvol":
        def port_vol(w):
            return np.sqrt(w @ cov.values @ w)
        res = minimize(port_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    return res.x


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

DARK_BG   = "#0d1117"
GOLD      = "#c9a84c"
WHITE     = "#e6edf3"
GREY      = "#30363d"
RED       = "#f85149"
GREEN     = "#3fb950"
BLUE      = "#58a6ff"

def make_report(prices, metrics, mc_results, weights_store, mu, cov,
                w_sharpe, w_minvol, assets, output_path="outputs/portfolio_report.png"):

    print("📊  Generating report...")
    daily_ret = metrics["daily_ret"]

    # ── Compute optimised portfolio stats
    def port_stats(w):
        r = w @ mu.values
        v = np.sqrt(w @ cov.values @ w)
        s = (r - RISK_FREE) / v
        return r, v, s

    sh_r, sh_v, sh_s = port_stats(w_sharpe)
    mv_r, mv_v, mv_s = port_stats(w_minvol)

    fig = plt.figure(figsize=(20, 24), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.spines[:].set_color(GREY)
        if title:
            ax.set_title(title, color=GOLD, fontsize=11, fontweight="bold", pad=10)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(WHITE)

    pct_fmt  = FuncFormatter(lambda x, _: f"{x:.0%}")
    usd_fmt  = FuncFormatter(lambda x, _: f"${x:,.0f}")

    # ── 1. TITLE
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(DARK_BG)
    ax_title.axis("off")
    ax_title.text(0.5, 0.75, "PORTFOLIO RISK & ALLOCATION ANALYZER",
                  ha="center", va="center", color=GOLD,
                  fontsize=22, fontweight="bold", transform=ax_title.transAxes)
    ax_title.text(0.5, 0.35, f"Multi-Asset Wealth Management Analysis  |  {START_DATE} – {END_DATE}",
                  ha="center", va="center", color=WHITE,
                  fontsize=12, transform=ax_title.transAxes)
    ax_title.axhline(0.1, color=GOLD, linewidth=0.8, xmin=0.1, xmax=0.9)

    # ── 2. CUMULATIVE RETURNS
    ax1 = fig.add_subplot(gs[1, :2])
    cum = (1 + daily_ret).cumprod() * INITIAL_CAPITAL
    colors = [GOLD, BLUE, GREEN, RED, "#a371f7", "#ffa657"]
    for i, col in enumerate(cum.columns):
        ax1.plot(cum.index, cum[col], label=col, linewidth=1.5, color=colors[i % len(colors)])
    ax1.yaxis.set_major_formatter(usd_fmt)
    ax1.legend(fontsize=8, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY, ncol=2)
    ax1.set_xlabel("Date", color=WHITE, fontsize=9)
    ax1.set_ylabel("Portfolio Value (USD)", color=WHITE, fontsize=9)
    style_ax(ax1, "Cumulative Performance — $1,000,000 Initial Capital")

    # ── 3. EFFICIENT FRONTIER
    ax2 = fig.add_subplot(gs[1, 2])
    sc = ax2.scatter(mc_results[1], mc_results[0], c=mc_results[2],
                     cmap="plasma", alpha=0.4, s=4)
    ax2.scatter(sh_v, sh_r, color=GOLD,  s=120, zorder=5, label=f"Max Sharpe ({sh_s:.2f})", marker="*")
    ax2.scatter(mv_v, mv_r, color=GREEN, s=80,  zorder=5, label=f"Min Vol ({mv_v:.1%})",    marker="D")
    cb = plt.colorbar(sc, ax=ax2)
    cb.set_label("Sharpe Ratio", color=WHITE, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=WHITE)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)
    ax2.xaxis.set_major_formatter(pct_fmt)
    ax2.yaxis.set_major_formatter(pct_fmt)
    ax2.set_xlabel("Annual Volatility", color=WHITE, fontsize=9)
    ax2.set_ylabel("Annual Return",     color=WHITE, fontsize=9)
    ax2.legend(fontsize=7, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY)
    style_ax(ax2, "Efficient Frontier")

    # ── 4. RISK / RETURN BAR
    ax3 = fig.add_subplot(gs[2, 0])
    x = np.arange(len(assets))
    w = 0.35
    ax3.bar(x - w/2, metrics["returns"],    width=w, color=GOLD,  label="Return",     alpha=0.9)
    ax3.bar(x + w/2, metrics["volatility"], width=w, color=BLUE,  label="Volatility", alpha=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([a.split("(")[0].strip() for a in assets], rotation=30, ha="right", fontsize=8)
    ax3.yaxis.set_major_formatter(pct_fmt)
    ax3.legend(fontsize=8, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY)
    style_ax(ax3, "Annualised Return vs. Volatility")

    # ── 5. SHARPE RATIOS
    ax4 = fig.add_subplot(gs[2, 1])
    sharpe_vals = metrics["sharpe"].values
    bar_colors  = [GREEN if s > 0 else RED for s in sharpe_vals]
    ax4.barh([a.split("(")[0].strip() for a in assets], sharpe_vals, color=bar_colors, alpha=0.9)
    ax4.axvline(0, color=WHITE, linewidth=0.8, linestyle="--")
    ax4.set_xlabel("Sharpe Ratio", color=WHITE, fontsize=9)
    style_ax(ax4, "Sharpe Ratio by Asset")

    # ── 6. MAX DRAWDOWN
    ax5 = fig.add_subplot(gs[2, 2])
    dd_vals = metrics["drawdowns"].values
    ax5.barh([a.split("(")[0].strip() for a in assets], dd_vals, color=RED, alpha=0.8)
    ax5.xaxis.set_major_formatter(pct_fmt)
    ax5.set_xlabel("Max Drawdown", color=WHITE, fontsize=9)
    style_ax(ax5, "Maximum Drawdown by Asset")

    # ── 7. CORRELATION HEATMAP
    ax6 = fig.add_subplot(gs[3, 0])
    corr = metrics["corr"].values
    im   = ax6.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    short = [a.split("(")[0].strip()[:8] for a in assets]
    ax6.set_xticks(range(len(assets))); ax6.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax6.set_yticks(range(len(assets))); ax6.set_yticklabels(short, fontsize=7)
    for i in range(len(assets)):
        for j in range(len(assets)):
            ax6.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                     fontsize=6, color="black" if abs(corr[i,j]) > 0.5 else WHITE)
    plt.colorbar(im, ax=ax6)
    style_ax(ax6, "Correlation Matrix")

    # ── 8. MAX SHARPE ALLOCATION PIE
    ax7 = fig.add_subplot(gs[3, 1])
    wedge_colors = [GOLD, BLUE, GREEN, RED, "#a371f7", "#ffa657"]
    wedges, texts, autotexts = ax7.pie(
        w_sharpe, labels=[a.split("(")[0].strip() for a in assets],
        autopct="%1.1f%%", colors=wedge_colors,
        textprops={"color": WHITE, "fontsize": 7},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5}
    )
    for at in autotexts: at.set_color(DARK_BG)
    style_ax(ax7, f"Max Sharpe Allocation\nSharpe: {sh_s:.2f} | Return: {sh_r:.1%} | Vol: {sh_v:.1%}")

    # ── 9. MIN VOL ALLOCATION PIE
    ax8 = fig.add_subplot(gs[3, 2])
    wedges2, texts2, autotexts2 = ax8.pie(
        w_minvol, labels=[a.split("(")[0].strip() for a in assets],
        autopct="%1.1f%%", colors=wedge_colors,
        textprops={"color": WHITE, "fontsize": 7},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5}
    )
    for at in autotexts2: at.set_color(DARK_BG)
    style_ax(ax8, f"Min Volatility Allocation\nSharpe: {mv_s:.2f} | Return: {mv_r:.1%} | Vol: {mv_v:.1%}")

    # ── Footer
    fig.text(0.5, 0.01,
             "The Meridian Playbook  |  Research on Capital Allocation & Financial Systems  |  themeridianplaybook.com",
             ha="center", color=GREY, fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"    ✓ Report saved → {output_path}\n")
    plt.close()


# ─────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary(metrics, w_sharpe, w_minvol, assets, mu, cov):
    print("=" * 65)
    print("  PORTFOLIO SUMMARY — INDIVIDUAL ASSETS")
    print("=" * 65)
    df = pd.DataFrame({
        "Ann. Return":    metrics["returns"].map("{:.2%}".format),
        "Ann. Vol":       metrics["volatility"].map("{:.2%}".format),
        "Sharpe Ratio":   metrics["sharpe"].map("{:.2f}".format),
        "Max Drawdown":   metrics["drawdowns"].map("{:.2%}".format),
    })
    df.index = [a.split("(")[0].strip() for a in assets]
    print(df.to_string())

    print("\n" + "=" * 65)
    print("  OPTIMISED PORTFOLIOS")
    print("=" * 65)
    for label, w in [("Max Sharpe Ratio", w_sharpe), ("Min Volatility", w_minvol)]:
        r = w @ mu.values
        v = np.sqrt(w @ cov.values @ w)
        s = (r - RISK_FREE) / v
        print(f"\n  [{label}]")
        print(f"  Return: {r:.2%}  |  Volatility: {v:.2%}  |  Sharpe: {s:.2f}")
        print("  Weights:")
        for asset, weight in zip(assets, w):
            bar = "█" * int(weight * 40)
            print(f"    {asset.split('(')[0].strip():<25} {weight:.1%}  {bar}")
    print("=" * 65)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   PORTFOLIO RISK & ALLOCATION ANALYZER       ║")
    print("║   The Meridian Playbook                      ║")
    print("╚══════════════════════════════════════════════╝\n")

    assets = list(PORTFOLIO.keys())

    prices   = fetch_data(PORTFOLIO, START_DATE, END_DATE)
    metrics  = compute_metrics(prices)

    mc_results, weights_store, mu, cov = run_monte_carlo(metrics["daily_ret"])

    w_sharpe = optimize_portfolio(mu, cov, target="sharpe")
    w_minvol = optimize_portfolio(mu, cov, target="minvol")

    print_summary(metrics, w_sharpe, w_minvol, assets, mu, cov)

    make_report(prices, metrics, mc_results, weights_store,
                mu, cov, w_sharpe, w_minvol, assets)

    print("✅  Analysis complete.")
    print("    Open outputs/portfolio_report.png to view the full report.\n")


if __name__ == "__main__":
    main()
