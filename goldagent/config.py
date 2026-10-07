"""Typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class InstrumentConfig(BaseModel):
    symbol: str = "XAUUSD"
    contract_size: float = 100.0
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 1.0


class CostsConfig(BaseModel):
    default_spread: float = 0.30
    commission_per_lot: float = 7.0
    slippage: float = 0.05


class RiskConfig(BaseModel):
    risk_per_trade: float = Field(0.005, gt=0, le=0.05)
    daily_loss_limit: float = Field(0.10, gt=0, le=1)
    max_consecutive_losses: int = Field(6, ge=1)
    weekly_loss_limit: float = Field(0.15, gt=0, le=1)
    kill_switch_drawdown: float = Field(0.30, gt=0, le=1)
    max_open_positions: int = Field(1, ge=1)


class BacktestConfig(BaseModel):
    initial_balance: float = 1000.0
    timeframe: str = "5min"
    flat_hour: int = 20
    friday_flat_hour: int = 19


class StrategyConfig(BaseModel):
    name: str = "session_breakout"
    params: dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel):
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
