from collections import defaultdict

from kasegu.core.position import Trade


def _stats(trades: list[Trade]) -> dict:
    net = [t.net for t in trades]
    wins = [x for x in net if x > 0]
    loss = [x for x in net if x <= 0]
    eq, peak, dd = 0.0, 0.0, 0.0
    for x in net:
        eq += x
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    n = len(net)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = -sum(loss) / len(loss) if loss else 0.0
    return {
        "trades": n,
        "net": eq,
        "max_drawdown": dd,
        "profit_factor": sum(wins) / -sum(loss) if sum(loss) < 0 else None,
        "win_rate": len(wins) / n if n else None,
        "payoff": avg_win / avg_loss if avg_loss else None,
        "avg_mfe_pct": sum(t.mfe_pct for t in trades) / n if n else None,
        "avg_mae_pct": sum(t.mae_pct for t in trades) / n if n else None,
        "avg_hold_min": sum(t.hold_minutes for t in trades) / n if n else None,
    }


def summarize(trades: list[Trade], entries: int, filled: int, slip_pct: float) -> dict:
    """Experiment report; all money values in quote asset."""
    out = _stats(trades)
    out.update(
        entries=entries,
        fill_rate=filled / entries if entries else None,
        fees=sum(t.fees for t in trades),
        funding=sum(t.funding for t in trades),
        # exits are taker orders filled with the configured slippage
        slippage=sum(t.qty * t.exit_price * slip_pct / 100 for t in trades),
        gross=sum(t.pnl for t in trades),
    )
    groups: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        groups[f"{t.symbol} {t.side}"].append(t)
    out["by_group"] = {k: _stats(v) for k, v in sorted(groups.items())}
    reasons: dict[str, int] = defaultdict(int)
    for t in trades:
        reasons[t.reason] += 1
    out["exit_reasons"] = dict(reasons)
    return out
