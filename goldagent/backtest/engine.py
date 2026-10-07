"""Event-driven bar-by-bar backtester.

Timeline for each bar i:
  1. fill the signal generated at the close of bar i-1 at bar i's open;
  2. check stop / target / time exits inside bar i;
  3. force-flat outside trading hours or when the kill-switch trips;
  4. at bar i's close, ask the strategy for a new signal (only when flat and allowed).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from goldagent.backtest.metrics import compute_metrics
from goldagent.config import Config
from goldagent.execution.sim_broker import Bar, Position, SimBroker, Trade
from goldagent.risk.manager import RiskManager
from goldagent.strategies import Strategy, build_strategy
from goldagent.strategies.base import Signal


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    risk_events: list = field(default_factory=list)
    skipped_signals: int = 0


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[f for f in Trade.__dataclass_fields__])
    rows = []
    for t in trades:
        d = asdict(t)
        ctx = d.pop("context")
        d.update({f"ctx_{k}": v for k, v in ctx.items()})
        rows.append(d)
    return pd.DataFrame(rows)


def _is_flat_time(ts: pd.Timestamp, cfg: Config) -> bool:
    bt = cfg.backtest
    if ts.weekday() == 4 and ts.hour >= bt.friday_flat_hour:
        return True
    return ts.hour >= bt.flat_hour


def run_backtest(bars: pd.DataFrame, cfg: Config, strategy: Strategy | None = None) -> BacktestResult:
    strategy = strategy or build_strategy(cfg.strategy.name, cfg.strategy.params)
    features = strategy.prepare(bars)
    rows = features.to_dict("records")
    broker = SimBroker(cfg.instrument, cfg.costs)
    balance = cfg.backtest.initial_balance
    risk = RiskManager(cfg.risk, cfg.instrument, balance)

    spread = bars["spread"] if "spread" in bars.columns else pd.Series(cfg.costs.default_spread, index=bars.index)
    spread = spread.fillna(cfg.costs.default_spread)
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    sp = spread.to_numpy()
    index = bars.index

    pos: Position | None = None
    pending: tuple[pd.Timestamp, Signal] | None = None
    trades: list[Trade] = []
    equity = []
    skipped = 0

    def book(trade: Trade) -> None:
        nonlocal balance
        balance += trade.pnl
        trades.append(trade)
        risk.on_trade_closed(trade.exit_time, trade.pnl, balance)

    for i, ts in enumerate(index):
        bar = Bar(ts, o[i], h[i], l[i], c[i], sp[i])

        if pending is not None:
            sig_time, sig = pending
            pending = None
            lots = risk.position_size(balance, sig.stop_distance)
            if lots > 0:
                pos = broker.open(bar, sig, lots, sig_time)
                strategy.on_entry(ts, sig)
            else:
                skipped += 1

        if pos is not None:
            trade = broker.check_exit(pos, bar)
            if trade is not None:
                book(trade)
                pos = None

        flat_time = _is_flat_time(ts, cfg)
        if pos is not None and flat_time:
            book(broker.close_at_market(pos, bar, "session_close"))
            pos = None

        eq = balance + (broker.unrealized(pos, bar) if pos is not None else 0.0)
        risk.update_equity(ts, eq)
        if pos is not None and risk.state.halted:
            book(broker.close_at_market(pos, bar, "kill_switch"))
            pos = None
            eq = balance
        equity.append(eq)

        if pos is None and not flat_time and i + 1 < len(index):
            allowed, _ = risk.can_open(ts, 0)
            if allowed:
                sig = strategy.on_bar(ts, rows[i])
                if sig is not None:
                    pending = (ts, sig)

    if pos is not None:
        book(broker.close_at_market(pos, bar, "end_of_data"))
        equity[-1] = balance

    trades_df = trades_to_frame(trades)
    equity_s = pd.Series(equity, index=index, name="equity")
    metrics = compute_metrics(trades_df, equity_s, cfg.backtest.initial_balance)
    metrics["skipped_signals_min_lot"] = skipped
    return BacktestResult(trades_df, equity_s, metrics, risk.state.events, skipped)
