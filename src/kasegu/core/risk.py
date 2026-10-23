from dataclasses import dataclass

from kasegu.config import RiskConfig
from kasegu.core.types import Quote, Side, Signal
from kasegu.exchange.rules import SymbolRules


@dataclass(frozen=True, slots=True)
class Plan:
    signal: Signal
    price: float
    qty: float

    @property
    def notional(self) -> float:
        return self.price * self.qty


class RiskManager:
    def __init__(self, cfg: RiskConfig, rules: dict[str, SymbolRules]):
        self.cfg = cfg
        self.rules = rules
        self.stop_loss_pct = 0.0  # set by the engine from the exit config

    def size(self, symbol: str, price: float) -> float:
        c = self.cfg
        notional = c.margin_budget * c.leverage * (1 - c.cost_reserve_pct / 100)
        return self.rules[symbol].qty(notional / price)

    def check(
        self,
        sig: Signal,
        quote: Quote | None,
        open_count: int,
        exposure: float,
        stale: bool = False,
        unknown: bool = False,
    ) -> tuple[Plan | None, str]:
        c, r = self.cfg, self.rules[sig.symbol]
        if stale or quote is None:
            return None, "stale_data"
        if unknown:
            return None, "unknown_order_state"
        if quote.spread_pct > c.max_spread_pct:
            return None, f"spread {quote.spread_pct:.4f}%"
        if min(quote.bid_qty, quote.ask_qty) < c.min_quote_qty:
            return None, "liquidity"
        if open_count >= c.max_positions:
            return None, "max_positions"

        if sig.side is Side.BUY:
            price = r.price_down(sig.limit_price)
            if price >= quote.ask:
                return None, "would_take"
        else:
            price = r.price_up(sig.limit_price)
            if price <= quote.bid:
                return None, "would_take"
        if price <= 0:
            return None, "bad_price"

        qty = self.size(sig.symbol, price)
        err = r.valid(price, qty)
        if err:
            return None, err
        if exposure + price * qty > c.max_notional:
            return None, "max_notional"
        if c.max_loss_per_trade and price * qty * self.stop_loss_pct / 100 > c.max_loss_per_trade:
            # more leverage means more notional, so it cannot get around this
            return None, "max_loss_per_trade"
        return Plan(sig, price, qty), "ok"
