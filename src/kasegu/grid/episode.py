"""One martingale grid episode, bar by bar, with the same fill rules as the research backtest.

Long grid (side +1): a market buy at the open of the start bar, then limit buys every `step` below
the start price, each 1.5x the previous one. Take profit: limit sell of everything at the average
entry + one step. Stop: market sell one step below the last level. Time exit after 72h.
Short grid (side -1) is the mirror image.

Bar order is conservative: adverse extreme first (fills, liquidation, stop), then take profit with
the order that rested before the bar. If the bar also filled new levels, no take profit in that bar.
"""

from dataclasses import asdict, dataclass, field

BAR_MS = 300_000
HOLD_BARS = 864  # 72h of 5m bars


@dataclass
class GridParams:
    levels: int = 6
    mult: float = 1.5
    notional: float = 104.0  # USDT, all orders together
    margin: float = 100 / 3  # USDT isolated margin of the slot
    maker: float = 0.0002
    taker: float = 0.0005
    slip_in: float = 0.0002
    slip_out: float = 0.0005
    pen: float = 0.0005  # a limit fills only when the price trades this far through it
    hold_bars: int = HOLD_BARS

    def weights(self) -> list[float]:
        w = [self.mult**i for i in range(self.levels)]
        s = sum(w)
        return [x / s for x in w]


@dataclass
class Fill:
    time: int
    level: int
    price: float
    qty: float
    fee: float


@dataclass
class GridEpisode:
    symbol: str
    side: int  # +1 long, -1 short
    step: float  # fraction, e.g. 0.1
    start: int  # open time (ms) of the first bar
    params: GridParams = field(default_factory=GridParams)
    info: dict = field(default_factory=dict)  # signal values for the ledger
    p0: float | None = None
    qty: float = 0.0
    cost: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    filled: int = 0
    last_bar: int | None = None  # open time of the last processed bar
    last_close: float | None = None
    fills: list[Fill] = field(default_factory=list)
    exit_time: int | None = None
    exit_price: float | None = None
    reason: str | None = None  # tp, stop, liquidation, time
    pnl: float | None = None  # USDT, net of fees and funding

    @property
    def open(self) -> bool:
        return self.reason is None

    @property
    def end_bar(self) -> int:
        return self.start + (self.params.hold_bars - 1) * BAR_MS

    def level_price(self, i: int) -> float:
        return self.p0 * (1 - self.side * i * self.step)

    @property
    def stop_price(self) -> float:
        return self.p0 * (1 - self.side * self.params.levels * self.step)

    def avg(self) -> float:
        return self.cost / self.qty

    def upnl(self, price: float) -> float:
        """Unrealized PnL at price, net of fees paid and funding so far."""
        if not self.qty:
            return 0.0
        return self.side * (self.qty * price - self.cost) - self.fees + self.funding

    def _close(self, t, price, fee_rate, reason, qty=None, cost=None, fees=None):
        qty = self.qty if qty is None else qty
        cost = self.cost if cost is None else cost
        fees = self.fees if fees is None else fees
        self.exit_time, self.exit_price, self.reason = t, price, reason
        self.pnl = self.side * (qty * price - cost) - fees - qty * price * fee_rate + self.funding

    def on_bar(self, t: int, o: float, h: float, low: float, c: float, funding_rate: float = 0.0):
        """Feed one closed 5m bar (open time t). funding_rate: funding settled inside this bar."""
        if not self.open or t < self.start or (self.last_bar is not None and t <= self.last_bar):
            return
        p = self.params
        w = p.weights()
        side = self.side
        if self.p0 is None:
            self.p0 = o
            px = o * (1 + side * p.slip_in)
            q = w[0] * p.notional / px
            self.qty, self.cost = q, q * px
            self.fees = w[0] * p.notional * p.taker
            self.filled = 1
            self.fills.append(Fill(t, 0, px, q, self.fees))
        self.last_bar, self.last_close = t, c
        q_pre, c_pre, f_pre = self.qty, self.cost, self.fees
        tp_pre = (c_pre / q_pre) * (1 + side * self.step)
        new_fill = False
        ext = low if side == 1 else h
        while self.filled < p.levels:
            lv = self.level_price(self.filled)
            hit = ext <= lv * (1 - p.pen) if side == 1 else ext >= lv * (1 + p.pen)
            if not hit:
                break
            n = w[self.filled] * p.notional
            q = n / lv
            self.qty += q
            self.cost += q * lv
            self.fees += n * p.maker
            self.fills.append(Fill(t, self.filled, lv, q, n * p.maker))
            self.filled += 1
            new_fill = True
        stop = self.stop_price
        stop_hit = ext <= stop if side == 1 else ext >= stop
        gap = o < stop if side == 1 else o > stop
        stop_px = (o if gap else stop) * (1 - side * p.slip_out)
        if self.upnl(ext) <= -p.margin:
            if stop_hit:
                self._close(t, stop_px, p.taker, "stop")
                if self.pnl > -p.margin:
                    return
            self.exit_time, self.exit_price, self.reason = t, ext, "liquidation"
            self.pnl = -p.margin - 0.001 * p.notional
            return
        if stop_hit:
            self._close(t, stop_px, p.taker, "stop")
            return
        if t > self.start and not new_fill:
            fav = h if side == 1 else low
            if (fav >= tp_pre * (1 + p.pen)) if side == 1 else (fav <= tp_pre * (1 - p.pen)):
                self._close(t, tp_pre, p.maker, "tp", q_pre, c_pre, f_pre)
                return
        if funding_rate:
            self.funding -= side * funding_rate * self.qty * c
        if t >= self.end_bar:
            self._close(t, c * (1 - side * p.slip_out), p.taker, "time")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GridEpisode":
        d = dict(d)
        d["params"] = GridParams(**d["params"])
        d["fills"] = [Fill(**f) for f in d["fills"]]
        return cls(**d)
