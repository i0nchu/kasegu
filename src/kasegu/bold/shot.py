"""One leveraged bet, bar by bar. Leave when equity reaches `target` (take profit) or falls to
`bust` (stop before liquidation), or at the time limit.

Entry "market": at the open of the start bar (taker + slippage). Entry "maker": a limit at the open
of the start bar, filled when price trades `pen` beyond it within `entry_wait` bars, else market at
the next bar open. With `tp_maker` the take profit is a resting limit (maker fee, needs `pen`
penetration). The stop is always market (taker + slippage).

Equity is a multiple of the stake. Fees are charged up front, funding is added as it is paid.
Within a bar the adverse extreme is checked first (research/alpha/bold-entry)."""

from dataclasses import asdict, dataclass

BAR_MS = 300_000


@dataclass
class ShotParams:
    leverage: float = 10.0
    target: float = 2.0  # equity multiple to take profit
    bust: float = 0.1  # equity multiple to stop out
    taker: float = 0.0005
    slip: float = 0.0005
    max_bars: int = 30 * 288  # 30 days of 5m bars
    entry: str = "market"  # market or maker
    tp_maker: bool = False
    maker: float = 0.0002
    pen: float = 0.0005  # a limit fills only when price trades this far through it
    entry_wait: int = 12  # bars to wait for the maker entry

    @property
    def stop_fee(self) -> float:
        return self.taker + self.slip

    @property
    def tp_fee(self) -> float:
        return self.maker if self.tp_maker else self.taker + self.slip


@dataclass
class Shot:
    symbol: str
    side: int  # +1 long, -1 short
    start: int  # open time (ms) of the entry bar
    params: ShotParams
    stake: float = 100.0
    p0: float | None = None
    limit: float | None = None
    entry_fee: float | None = None
    fill_bar: int | None = None
    funding: float = 0.0  # equity fraction paid so far (positive = paid)
    last_bar: int | None = None
    last_close: float | None = None
    exit_time: int | None = None
    exit_price: float | None = None
    reason: str | None = None  # target, bust, time
    equity: float | None = None  # final equity multiple

    @property
    def open(self) -> bool:
        return self.reason is None

    def up_move(self) -> float:
        p = self.params
        return (p.target - 1 + p.leverage * (self.entry_fee + p.tp_fee) + self.funding) / p.leverage

    def down_move(self) -> float:
        p = self.params
        return (1 - p.bust - p.leverage * (self.entry_fee + p.stop_fee) - self.funding) / p.leverage

    def mark(self, price: float, fee: float | None = None) -> float:
        """Equity multiple if closed at price (default: by market)."""
        p = self.params
        fee = p.stop_fee if fee is None else fee
        move = self.side * p.leverage * (price / self.p0 - 1)
        return 1 + move - p.leverage * (self.entry_fee + fee) - self.funding

    def _exit(self, t, price, reason, equity):
        self.exit_time, self.exit_price, self.reason = t, price, reason
        self.equity = max(equity, 0.0)

    def on_bar(self, t: int, o: float, h: float, low: float, c: float, funding_rate: float = 0.0):
        if not self.open or t < self.start or (self.last_bar is not None and t <= self.last_bar):
            return
        p = self.params
        s = self.side
        self.last_bar, self.last_close = t, c
        if self.p0 is None:
            waited = (t - self.start) // BAR_MS
            if p.entry == "maker" and waited < p.entry_wait:
                lim = self.limit_price(o)
                if (low <= lim * (1 - p.pen)) if s == 1 else (h >= lim * (1 + p.pen)):
                    self.p0, self.entry_fee, self.fill_bar = lim, p.maker, t
                return  # exits are checked from the bar after a maker fill
            self.p0, self.entry_fee, self.fill_bar = o * (1 + s * p.slip), p.taker + p.slip, t
        bust_px = self.p0 * (1 - s * self.down_move())
        tp_px = self.p0 * (1 + s * self.up_move())
        adv, fav = (low, h) if s == 1 else (h, low)
        if (adv <= bust_px) if s == 1 else (adv >= bust_px):
            gap = (o < bust_px) if s == 1 else (o > bust_px)
            px = o if gap else bust_px
            self._exit(t, px, "bust", self.mark(px) if gap else p.bust)
            return
        need = tp_px * (1 + s * p.pen) if p.tp_maker else tp_px
        if (fav >= need) if s == 1 else (fav <= need):
            gap = (o > tp_px) if s == 1 else (o < tp_px)
            px = o if gap else tp_px
            self._exit(t, px, "target", self.mark(px, p.tp_fee) if gap else p.target)
            return
        if funding_rate:
            self.funding += s * funding_rate * p.leverage
        if (t - self.start) // BAR_MS >= p.max_bars - 1:
            self._exit(t, c, "time", self.mark(c))

    def limit_price(self, o: float) -> float:
        """The maker entry limit: the open of the start bar (kept once seen)."""
        if self.limit is None:
            self.limit = o
        return self.limit

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Shot":
        d = dict(d)
        d["params"] = ShotParams(**d["params"])
        return cls(**d)
