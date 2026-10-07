"""Checks that need real XAUUSD history. Skipped until data/XAUUSD_M1.parquet exists.

Get the data with:  python -m goldagent download --from 2020-01-01 --to 2025-12-31
"""

from pathlib import Path

import pandas as pd
import pytest

from goldagent.backtest.engine import run_backtest
from goldagent.config import load_config
from goldagent.data.loader import load_bars
from goldagent.data.resample import resample_ohlc
from goldagent.strategies import build_strategy

DATA = Path(__file__).resolve().parent.parent / "data" / "XAUUSD_M1.parquet"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="real XAUUSD data not downloaded")


@pytest.fixture(scope="module")
def m5():
    bars = load_bars(DATA)
    start = bars.index[0] + pd.Timedelta(days=30)
    return resample_ohlc(bars[(bars.index >= start) & (bars.index < start + pd.Timedelta(days=90))], "5min")


def test_features_have_no_lookahead(m5):
    cfg = load_config()
    full = build_strategy(cfg.strategy.name, cfg.strategy.params).prepare(m5)
    for cut in (len(m5) // 3, len(m5) // 2, 2 * len(m5) // 3):
        part = build_strategy(cfg.strategy.name, cfg.strategy.params).prepare(m5.iloc[:cut])
        pd.testing.assert_frame_equal(part, full.iloc[:cut], check_freq=False)


def test_backtest_is_deterministic(m5):
    cfg = load_config()
    a, b = run_backtest(m5, cfg), run_backtest(m5, cfg)
    assert a.metrics == b.metrics
