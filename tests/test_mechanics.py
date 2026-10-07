"""Unit tests for arithmetic and execution mechanics.

These use a handful of hand-written bars to check exact numbers (fills, stops,
lot sizes). They say nothing about strategy quality: that is judged on real
market data only.
"""

import lzma
import struct
from datetime import date

import pandas as pd
import pytest

from goldagent.config import Config
from goldagent.data.loader import decode_dukascopy_candles
from goldagent.execution.sim_broker import Bar, SimBroker
from goldagent.indicators import align_higher_timeframe
from goldagent.risk.manager import RiskManager
from goldagent.strategies.base import Signal

T0 = pd.Timestamp("2024-03-05 08:00", tz="UTC")


def bar(minutes, o, h, l, c, spread=0.2):
    return Bar(T0 + pd.Timedelta(minutes=minutes), o, h, l, c, spread)


@pytest.fixture
def cfg():
    c = Config()
    c.costs.slippage = 0.0
    c.costs.commission_per_lot = 7.0
    return c


def test_position_size_rounds_down_and_respects_min_lot(cfg):
    rm = RiskManager(cfg.risk, cfg.instrument, 1000)
    # 0.5% of 1000 = $5; stop $2 -> 5 / (2*100) = 0.025 lots -> 0.02
    assert rm.position_size(1000, 2.0) == pytest.approx(0.02)
    # stop $6 -> 0.0083 lots < min lot -> skip trade
    assert rm.position_size(1000, 6.0) == 0.0


def test_long_entry_pays_spread_and_stop_exit(cfg):
    broker = SimBroker(cfg.instrument, cfg.costs)
    sig = Signal(side=1, stop_distance=2.0, tp_distance=4.0)
    pos = broker.open(bar(0, 2000.0, 2001, 1999, 2000.5), sig, lots=0.01)
    assert pos.entry_price == pytest.approx(2000.2)  # ask = bid + spread
    trade = broker.check_exit(pos, bar(5, 2000.0, 2000.5, 1998.0, 1998.5))
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(1998.2)
    # -2.0 * 1 oz - $0.07 commission
    assert trade.pnl == pytest.approx(-2.07)


def test_same_bar_stop_and_target_counts_as_stop(cfg):
    broker = SimBroker(cfg.instrument, cfg.costs)
    pos = broker.open(bar(0, 2000.0, 2000, 2000, 2000), Signal(1, 2.0, 4.0), lots=0.01)
    trade = broker.check_exit(pos, bar(5, 2000.0, 2010.0, 1990.0, 2000.0))
    assert trade.exit_reason == "stop"


def test_short_exits_at_ask(cfg):
    broker = SimBroker(cfg.instrument, cfg.costs)
    pos = broker.open(bar(0, 2000.0, 2000, 2000, 2000), Signal(-1, 2.0, 3.0), lots=0.01)
    assert pos.entry_price == pytest.approx(2000.0)
    # bid low 1996.8 -> ask low 1997.0 == target
    trade = broker.check_exit(pos, bar(5, 1999.0, 1999.5, 1996.8, 1998.0))
    assert trade.exit_reason == "target"
    assert trade.gross_pnl == pytest.approx(3.0)


def test_breakeven_moves_stop_after_bar(cfg):
    broker = SimBroker(cfg.instrument, cfg.costs)
    pos = broker.open(bar(0, 2000.0, 2000, 2000, 2000), Signal(1, 2.0, 10.0, breakeven_r=1.0), lots=0.01)
    assert broker.check_exit(pos, bar(5, 2000.0, 2002.5, 2000.0, 2002.0)) is None
    assert pos.stop == pytest.approx(pos.entry_price)
    trade = broker.check_exit(pos, bar(10, 2001.0, 2001.0, 2000.0, 2000.0))
    assert trade.exit_reason == "breakeven"


def test_daily_limits_and_kill_switch(cfg):
    cfg.risk.max_consecutive_losses = 2
    rm = RiskManager(cfg.risk, cfg.instrument, 1000)
    rm.update_equity(T0, 1000)
    rm.on_trade_closed(T0, -5, 995)
    rm.on_trade_closed(T0, -5, 990)
    assert rm.can_open(T0, 0)[0] is False
    next_day = T0 + pd.Timedelta(days=1)
    assert rm.can_open(next_day, 0)[0] is True
    rm.update_equity(next_day, 690)  # -31% from peak 1000
    allowed, reason = rm.can_open(next_day, 0)
    assert not allowed and "kill-switch" in reason


def test_htf_alignment_has_no_lookahead():
    h1_index = pd.date_range("2024-03-05 08:00", periods=3, freq="1h", tz="UTC")
    h1 = pd.Series([1.0, 2.0, 3.0], index=h1_index)
    m5_index = pd.date_range("2024-03-05 08:00", periods=36, freq="5min", tz="UTC")
    aligned = align_higher_timeframe(h1, "1h", m5_index, "5min")
    # The 08:00 H1 bar closes at 09:00; the M5 bar 08:55 closes at 09:00 and may see it.
    assert pd.isna(aligned[pd.Timestamp("2024-03-05 08:50", tz="UTC")])
    assert aligned[pd.Timestamp("2024-03-05 08:55", tz="UTC")] == 1.0
    assert aligned[pd.Timestamp("2024-03-05 09:55", tz="UTC")] == 2.0


def test_dukascopy_decoder():
    rec = struct.Struct(">5if")
    raw = rec.pack(60, 2030120, 2030450, 2029900, 2030600, 12.5) + rec.pack(120, 2030450, 2030000, 2029800, 2030500, 0.0)
    df = decode_dukascopy_candles(lzma.compress(raw, format=lzma.FORMAT_ALONE), date(2024, 3, 5), 1000.0)
    assert len(df) == 1  # zero-volume padding candle dropped
    row = df.iloc[0]
    assert df.index[0] == pd.Timestamp("2024-03-05 00:01", tz="UTC")
    assert (row.open, row.high, row.low, row.close) == pytest.approx((2030.12, 2030.6, 2029.9, 2030.45))
