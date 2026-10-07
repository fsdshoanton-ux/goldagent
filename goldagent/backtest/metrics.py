"""Performance metrics computed from closed trades and the equity curve."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction of the peak."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((peak - equity) / peak).max())


def longest_losing_streak(pnl: pd.Series) -> int:
    best = cur = 0
    for x in pnl:
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return best


def compute_metrics(trades: pd.DataFrame, equity: pd.Series, initial_balance: float) -> dict:
    n = len(trades)
    final = float(equity.iloc[-1]) if len(equity) else initial_balance
    out: dict = {
        "trades": n,
        "net_profit": round(final - initial_balance, 2),
        "return_pct": round((final / initial_balance - 1) * 100, 2),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
    }
    if n == 0:
        return out
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    daily = equity.resample("1D").last().dropna()
    daily_ret = daily.pct_change().dropna()
    sharpe = (
        daily_ret.mean() / daily_ret.std() * math.sqrt(252)
        if len(daily_ret) > 1 and daily_ret.std() > 0
        else float("nan")
    )
    out.update(
        {
            "win_rate_pct": round(len(wins) / n * 100, 2),
            "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
            "expectancy_usd": round(pnl.mean(), 3),
            "expectancy_r": round(trades["r_multiple"].mean(), 3),
            "avg_win_r": round(trades.loc[pnl > 0, "r_multiple"].mean(), 3) if len(wins) else 0.0,
            "avg_loss_r": round(trades.loc[pnl < 0, "r_multiple"].mean(), 3) if len(losses) else 0.0,
            "total_commission": round(trades["commission"].sum(), 2),
            "sharpe_daily": round(float(sharpe), 3) if not np.isnan(sharpe) else None,
            "longest_losing_streak": longest_losing_streak(pnl),
            "avg_bars_held": round(trades["bars_held"].mean(), 1),
            "long_trades": int((trades["side"] > 0).sum()),
            "short_trades": int((trades["side"] < 0).sum()),
            "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
        }
    )
    return out


# Gate from the plan: a strategy must pass these out-of-sample before paper trading.
PROMOTION_GATE = {"profit_factor": 1.3, "trades": 200, "max_drawdown_pct": 10.0}


def check_gate(metrics: dict) -> list[str]:
    """Return a list of failed gate conditions (empty = passed)."""
    failures = []
    if metrics.get("trades", 0) < PROMOTION_GATE["trades"]:
        failures.append(f"trades {metrics.get('trades', 0)} < {PROMOTION_GATE['trades']}")
    if metrics.get("profit_factor", 0) < PROMOTION_GATE["profit_factor"]:
        failures.append(f"profit factor {metrics.get('profit_factor', 0)} < {PROMOTION_GATE['profit_factor']}")
    if metrics.get("max_drawdown_pct", 100) > PROMOTION_GATE["max_drawdown_pct"]:
        failures.append(f"max drawdown {metrics.get('max_drawdown_pct')}% > {PROMOTION_GATE['max_drawdown_pct']}%")
    return failures
