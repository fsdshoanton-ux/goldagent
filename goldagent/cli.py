"""Command line interface: goldagent <command> ..."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from goldagent.backtest.engine import run_backtest
from goldagent.backtest.metrics import check_gate
from goldagent.config import load_config
from goldagent.data.loader import download_dukascopy, load_bars, load_mt5_csv, save_bars
from goldagent.data.resample import resample_ohlc


def cmd_download(args: argparse.Namespace) -> None:
    df = download_dukascopy(
        args.symbol, date.fromisoformat(args.start), date.fromisoformat(args.end), Path(args.cache), args.workers
    )
    save_bars(df, args.out)
    print(f"saved {len(df):,} M1 bars {df.index[0]} .. {df.index[-1]} -> {args.out}")


def cmd_import_mt5(args: argparse.Namespace) -> None:
    df = load_mt5_csv(args.csv, args.utc_offset, args.point)
    save_bars(df, args.out)
    print(f"saved {len(df):,} bars {df.index[0]} .. {df.index[-1]} -> {args.out}")


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.spread_mult != 1.0:
        cfg.costs.default_spread *= args.spread_mult
    bars = load_bars(args.data, args.start, args.end)
    bars = resample_ohlc(bars, cfg.backtest.timeframe)
    if args.spread_mult != 1.0 and "spread" in bars.columns:
        bars["spread"] *= args.spread_mult
    result = run_backtest(bars, cfg)
    m = result.metrics
    print(json.dumps(m, indent=2, default=str))
    for ts, event in result.risk_events:
        print(f"RISK EVENT {ts}: {event}")
    failures = check_gate(m)
    print("GATE: PASSED" if not failures else "GATE: FAILED -> " + "; ".join(failures))
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.strategy.name}_{bars.index[0]:%Y%m%d}_{bars.index[-1]:%Y%m%d}"
    result.trades.to_csv(out / f"{tag}_trades.csv", index=False)
    result.equity.resample("1h").last().dropna().to_csv(out / f"{tag}_equity.csv")
    print(f"reports -> {out}/{tag}_*.csv")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="goldagent")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download XAUUSD M1 bars from Dukascopy")
    d.add_argument("--symbol", default="XAUUSD")
    d.add_argument("--from", dest="start", required=True)
    d.add_argument("--to", dest="end", required=True)
    d.add_argument("--out", default="data/XAUUSD_M1.parquet")
    d.add_argument("--cache", default="data/cache")
    d.add_argument("--workers", type=int, default=2)
    d.set_defaults(func=cmd_download)

    m = sub.add_parser("import-mt5", help="import a MetaTrader 5 'Export bars' CSV")
    m.add_argument("csv")
    m.add_argument("--utc-offset", type=float, default=0.0, help="broker server time offset from UTC, hours")
    m.add_argument("--point", type=float, default=0.01, help="USD per MT5 spread point")
    m.add_argument("--out", default="data/XAUUSD_M1.parquet")
    m.set_defaults(func=cmd_import_mt5)

    b = sub.add_parser("backtest", help="run a backtest")
    b.add_argument("--data", default="data/XAUUSD_M1.parquet")
    b.add_argument("--from", dest="start")
    b.add_argument("--to", dest="end")
    b.add_argument("--config")
    b.add_argument("--spread-mult", type=float, default=1.0, help="stress test: multiply spreads")
    b.add_argument("--report-dir", default="reports")
    b.set_defaults(func=cmd_backtest)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
