"""One leveraged bet, bar by bar. Enter at the open of the start bar; leave when equity reaches
`target` (take profit) or falls to `bust` (stop before liquidation), or at the time limit.

Equity is a multiple of the stake. Costs: taker + slippage on both legs are charged up front as in
the research, funding is added as it is paid. Within a bar the adverse extreme is checked first."""

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

    @property
    def cost(self) -> float:
        return 2 * self.leverage * (self.taker + self.slip)


@dataclass
class Shot:
    symbol: str
    side: int  # +1 long, -1 short
    start: int  # open time (ms) of the entry bar
    params: ShotParams
    stake: float = 100.0
    p0: float | None = None
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
        return (p.target - 1 + p.cost + self.funding) / p.leverage

    def down_move(self) -> float:
        p = self.params
        return (1 - p.bust - p.cost - self.funding) / p.leverage

    def mark(self, price: float) -> float:
        """Equity multiple if closed at price."""
        p = self.params
        return 1 + self.side * p.leverage * (price / self.p0 - 1) - p.cost - self.funding

    def _exit(self, t, price, reason, equity):
        self.exit_time, self.exit_price, self.reason = t, price, reason
        self.equity = max(equity, 0.0)

    def on_bar(self, t: int, o: float, h: float, low: float, c: float, funding_rate: float = 0.0):
        if not self.open or t < self.start or (self.last_bar is not None and t <= self.last_bar):
            return
        p = self.params
        s = self.side
        if self.p0 is None:
            self.p0 = o * (1 + s * p.slip)
        self.last_bar, self.last_close = t, c
        bust_px = self.p0 * (1 - s * self.down_move())
        tp_px = self.p0 * (1 + s * self.up_move())
        adv, fav = (low, h) if s == 1 else (h, low)
        if (adv <= bust_px) if s == 1 else (adv >= bust_px):
            gap = (o < bust_px) if s == 1 else (o > bust_px)
            px = o if gap else bust_px
            self._exit(t, px, "bust", self.mark(px) if gap else p.bust)
            return
        if (fav >= tp_px) if s == 1 else (fav <= tp_px):
            gap = (o > tp_px) if s == 1 else (o < tp_px)
            px = o if gap else tp_px
            self._exit(t, px, "target", self.mark(px) if gap else p.target)
            return
        if funding_rate:
            self.funding += s * funding_rate * p.leverage
        if (t - self.start) // BAR_MS >= p.max_bars - 1:
            self._exit(t, c, "time", self.mark(c))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Shot":
        d = dict(d)
        d["params"] = ShotParams(**d["params"])
        return cls(**d)
