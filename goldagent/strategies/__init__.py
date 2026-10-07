from goldagent.strategies.base import Signal, Strategy
from goldagent.strategies.session_breakout import SessionBreakout

STRATEGIES: dict[str, type[Strategy]] = {
    SessionBreakout.name: SessionBreakout,
}


def build_strategy(name: str, params: dict) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"Unknown strategy {name!r}; available: {sorted(STRATEGIES)}") from None
    return cls(params)


__all__ = ["Signal", "Strategy", "STRATEGIES", "build_strategy"]
