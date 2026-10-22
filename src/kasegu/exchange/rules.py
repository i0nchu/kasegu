from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal


def _step(value: float, step: float, rounding: str) -> float:
    d, s = Decimal(str(value)), Decimal(str(step))
    return float((d / s).to_integral_value(rounding=rounding) * s)


@dataclass(frozen=True, slots=True)
class SymbolRules:
    symbol: str
    tick: float
    step: float
    min_qty: float
    max_qty: float
    min_notional: float

    def price_down(self, price: float) -> float:
        return _step(price, self.tick, ROUND_FLOOR)

    def price_up(self, price: float) -> float:
        return _step(price, self.tick, ROUND_CEILING)

    def qty(self, qty: float) -> float:
        return min(_step(qty, self.step, ROUND_FLOOR), self.max_qty)

    def valid(self, price: float, qty: float) -> str | None:
        if qty < self.min_qty:
            return f"qty {qty} < min {self.min_qty}"
        if price * qty < self.min_notional:
            return f"notional {price * qty:.4f} < min {self.min_notional}"
        return None


def parse_rules(info: dict) -> dict[str, SymbolRules]:
    out = {}
    for s in info.get("symbols", []):
        f = {x["filterType"]: x for x in s.get("filters", [])}
        lot = f.get("LOT_SIZE", {})
        out[s["symbol"]] = SymbolRules(
            symbol=s["symbol"],
            tick=float(f["PRICE_FILTER"]["tickSize"]),
            step=float(lot["stepSize"]),
            min_qty=float(lot["minQty"]),
            max_qty=float(lot["maxQty"]),
            min_notional=float(f.get("MIN_NOTIONAL", {}).get("notional", 0)),
        )
    return out
