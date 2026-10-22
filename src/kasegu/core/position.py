from dataclasses import asdict, dataclass, field
from enum import StrEnum

from kasegu.core.types import Fill, Purpose, Side

EPS = 1e-12


class PosState(StrEnum):
    FLAT = "FLAT"
    PENDING_ENTRY = "PENDING_ENTRY"
    POSITION_OPEN = "POSITION_OPEN"
    EXIT_PENDING = "EXIT_PENDING"


@dataclass(slots=True)
class Trade:
    symbol: str
    side: Side
    qty: float
    entry_price: float
    exit_price: float
    open_time: int
    close_time: int
    pnl: float  # gross, quote asset
    fees: float
    funding: float
    mfe_pct: float
    mae_pct: float
    reason: str
    meta: dict = field(default_factory=dict)

    @property
    def net(self) -> float:
        return self.pnl - self.fees + self.funding

    @property
    def hold_minutes(self) -> float:
        return (self.close_time - self.open_time) / 60_000


@dataclass(slots=True)
class Position:
    symbol: str
    state: PosState = PosState.FLAT
    side: Side | None = None
    qty: float = 0.0  # open quantity, always >= 0
    entry_price: float = 0.0
    open_time: int | None = None  # first fill
    peak_pct: float = 0.0  # H
    trough_pct: float = 0.0
    tier: int = -1
    stop_pct: float | None = None  # effective trailing stop
    entry_qty: float = 0.0
    exit_qty: float = 0.0
    exit_value: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    exit_reason: str = ""
    meta: dict = field(default_factory=dict)

    def profit_pct(self, price: float) -> float:
        if self.side is None or self.entry_price <= 0:
            return 0.0
        return (price - self.entry_price) / self.entry_price * 100 * self.side.sign

    def notional(self, price: float) -> float:
        return self.qty * price

    @property
    def has_qty(self) -> bool:
        return self.qty > EPS

    def apply(self, f: Fill, purpose: Purpose) -> Trade | None:
        """Apply a fill. Returns a Trade when the position is closed."""
        self.fees += f.fee
        if purpose is Purpose.ENTRY:
            total = self.qty + f.qty
            self.entry_price = (self.entry_price * self.qty + f.price * f.qty) / total
            self.qty = total
            self.entry_qty += f.qty
            self.side = f.side
            if self.open_time is None:
                self.open_time = f.time
            if self.state in (PosState.FLAT, PosState.PENDING_ENTRY):
                self.state = PosState.POSITION_OPEN
            return None

        qty = min(f.qty, self.qty)
        self.qty -= qty
        self.exit_qty += qty
        self.exit_value += qty * f.price
        if self.has_qty:
            return None
        return self._close(f.time)

    def _close(self, ts: int) -> Trade:
        exit_px = self.exit_value / self.exit_qty if self.exit_qty else self.entry_price
        pnl = (exit_px - self.entry_price) * self.exit_qty * self.side.sign
        t = Trade(
            symbol=self.symbol,
            side=self.side,
            qty=self.entry_qty,
            entry_price=self.entry_price,
            exit_price=exit_px,
            open_time=self.open_time or ts,
            close_time=ts,
            pnl=pnl,
            fees=self.fees,
            funding=self.funding,
            mfe_pct=self.peak_pct,
            mae_pct=self.trough_pct,
            reason=self.exit_reason,
            meta=dict(self.meta),
        )
        self.reset()
        return t

    def reset(self):
        fresh = Position(self.symbol)
        for k in self.__slots__:
            setattr(self, k, getattr(fresh, k))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        d = dict(d)
        d["state"] = PosState(d["state"])
        d["side"] = Side(d["side"]) if d.get("side") else None
        return cls(**d)


class PositionManager:
    def __init__(self, symbols: list[str]):
        self.pos = {s: Position(s) for s in symbols}

    def __getitem__(self, symbol: str) -> Position:
        return self.pos[symbol]

    def open_count(self) -> int:
        return sum(1 for p in self.pos.values() if p.state is not PosState.FLAT)

    def exposure(self, prices: dict[str, float]) -> float:
        return sum(p.notional(prices.get(s, p.entry_price)) for s, p in self.pos.items())
