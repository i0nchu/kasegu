import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import aiohttp
from binance_common.configuration import ConfigurationRestAPI
from binance_common.errors import BadRequestError
from binance_sdk_derivatives_trading_usds_futures import DerivativesTradingUsdsFutures

from kasegu.config import ExchangeConfig, FeeConfig, Secrets
from kasegu.core.execution import EPS, Broker, purpose_of
from kasegu.core.types import Fill, Order, OrderStatus, OrderType, Purpose, Side, TimeInForce
from kasegu.exchange.binance import REST_PROD, REST_TEST, WS_PROD, WS_TEST, plain

log = logging.getLogger(__name__)

# request reached the exchange but the outcome is unknown, query before retrying
UNKNOWN_CODES = {-1000, -1001, -1006, -1007, -1021}
MISSING_CODES = {-2011, -2013}  # unknown order
POST_ONLY = -5022
USER_EVENTS = ["ORDER_TRADE_UPDATE", "ALGO_UPDATE", "ACCOUNT_UPDATE"]

_ALGO_STATUS = {
    "NEW": OrderStatus.NEW,
    "TRIGGERING": OrderStatus.NEW,
    "TRIGGERED": OrderStatus.NEW,  # child order is working
    "CANCELED": OrderStatus.CANCELED,
    "EXPIRED": OrderStatus.EXPIRED,
    "REJECTED": OrderStatus.REJECTED,
}


def _f(v: Any) -> float:
    return float(v) if v not in (None, "") else 0.0


def num(x: float) -> float:
    # drop float noise such as 0.30000000000000004 before sending
    return float(f"{x:.12g}")


def order_from_rest(d: dict, purpose: Purpose) -> Order:
    return Order(
        client_id=d["clientOrderId"],
        symbol=d["symbol"],
        side=Side(d["side"]),
        type=OrderType(d.get("origType") or d["type"]),
        qty=_f(d.get("origQty")),
        purpose=purpose,
        price=_f(d.get("price")) or None,
        stop_price=_f(d.get("stopPrice")) or None,
        tif=TimeInForce(d.get("timeInForce") or "GTC"),
        reduce_only=bool(d.get("reduceOnly")),
        status=OrderStatus(d["status"]),
        filled_qty=_f(d.get("executedQty")),
        avg_price=_f(d.get("avgPrice")),
        time=int(d.get("updateTime") or d.get("time") or 0),
    )


def order_from_algo(d: dict) -> Order:
    """Conditional (algo) order from REST, used for exchange side stops."""
    filled = _f(d.get("actualQty"))
    st = d.get("algoStatus", "NEW")
    status = _ALGO_STATUS.get(st)
    if status is None:  # FINISHED: child order done
        status = OrderStatus.FILLED if filled > EPS else OrderStatus.EXPIRED
    return Order(
        client_id=d["clientAlgoId"],
        symbol=d["symbol"],
        side=Side(d["side"]),
        type=OrderType.STOP_MARKET,
        qty=_f(d.get("quantity")),
        purpose=Purpose.STOP,
        stop_price=_f(d.get("triggerPrice")) or None,
        reduce_only=bool(d.get("reduceOnly")),
        status=status,
        filled_qty=filled,
        avg_price=_f(d.get("actualPrice")),
        time=int(d.get("updateTime") or d.get("createTime") or 0),
    )


def algo_from_ws(o: dict, ts: int) -> Order:
    return order_from_algo(
        {
            "clientAlgoId": o["caid"],
            "symbol": o["s"],
            "side": o["S"],
            "quantity": o.get("q"),
            "triggerPrice": o.get("tp"),
            "reduceOnly": o.get("R"),
            "algoStatus": o.get("X"),
            "actualQty": o.get("aq"),
            "actualPrice": o.get("ap"),
            "updateTime": ts,
        }
    )


def ws_status(x: str) -> OrderStatus:
    if x in OrderStatus.__members__:
        return OrderStatus(x)
    return OrderStatus.EXPIRED if x.startswith("EXPIRED") else OrderStatus.NEW


def order_from_ws(o: dict, client_id: str, purpose: Purpose) -> Order:
    """ORDER_TRADE_UPDATE payload; stop children are reported under the stop's id."""
    stop = purpose is Purpose.STOP
    return Order(
        client_id=client_id,
        symbol=o["s"],
        side=Side(o["S"]),
        type=OrderType.STOP_MARKET if stop else OrderType(o.get("ot") or o["o"]),
        qty=_f(o.get("q")),
        purpose=purpose,
        price=_f(o.get("p")) or None,
        stop_price=_f(o.get("sp")) or None,
        tif=TimeInForce(o["f"]) if o.get("f") in TimeInForce.__members__ else TimeInForce.GTC,
        reduce_only=bool(o.get("R")),
        status=ws_status(o.get("X", "")),
        filled_qty=_f(o.get("z")),
        avg_price=_f(o.get("ap")),
        reason=o.get("er") or "",
        time=int(o.get("T") or 0),
    )


class LiveBroker(Broker):
    """Binance USDⓈ-M orders over REST, fills from the user data stream.

    Fills are emitted once per client id from the cumulative filled qty, so REST
    responses, stream events and reconcile after a reconnect never double count.
    Stops are conditional (algo) orders; their triggered child order is mapped back
    to the stop's client id.
    """

    def __init__(self, cfg: ExchangeConfig, fees: FeeConfig, secrets: Secrets, client=None):
        super().__init__()
        key = secrets.api_key.get_secret_value() if secrets.api_key else None
        sec = secrets.api_secret.get_secret_value() if secrets.api_secret else None
        if client is None and not (key and sec):
            raise ValueError("live mode needs BINANCE_API_KEY and BINANCE_API_SECRET")
        self.cfg = cfg
        self.fees = fees
        self.client = client or DerivativesTradingUsdsFutures(
            config_rest_api=ConfigurationRestAPI(
                api_key=key,
                api_secret=sec,
                base_path=REST_TEST if cfg.testnet else REST_PROD,
                timeout=cfg.rest_timeout_ms,
            ),
        )
        self.ws_url = WS_TEST if cfg.testnet else WS_PROD
        self.ws = None
        self.closed = ""
        self.wake: Callable[[], None] | None = None  # called after stream events
        self.filled: dict[str, float] = {}  # client id -> emitted filled qty
        self.child: dict[int, str] = {}  # triggered child order id -> stop client id
        self.orphans: dict[int, list[dict]] = {}  # child updates seen before ALGO_UPDATE
        self.armed: set[str] = set()  # symbols with a running order countdown
        self.session: aiohttp.ClientSession | None = None
        self.reader: asyncio.Task | None = None
        self.requery: set[str] = set()  # fills waiting for a price
        self.received = 0  # user stream messages, checked by the testnet check

    async def _rest(self, fn: Callable, *args, **kw) -> Any:
        resp = await asyncio.to_thread(fn, *args, **kw)
        return plain(resp.data())

    @property
    def api(self):
        return self.client.rest_api

    # ---- setup

    async def prepare(self, symbols: list[str], leverage: int):
        """One-way mode is required; set leverage per symbol."""
        mode = await self._rest(self.api.get_current_position_mode)
        if mode.get("dualSidePosition"):
            raise RuntimeError("account is in hedge mode, switch to one-way mode first")
        for s in symbols:
            await self._rest(self.api.change_initial_leverage, symbol=s, leverage=leverage)
            # a countdown left by a previous run would cancel adopted stops
            await self.auto_cancel(s, 0)

    async def commission(self, symbol: str) -> tuple[float, float]:
        d = await self._rest(self.api.user_commission_rate, symbol=symbol)
        return float(d["makerCommissionRate"]), float(d["takerCommissionRate"])

    # ---- Broker

    async def place(self, o: Order) -> Order:
        if o.type is OrderType.STOP_MARKET:
            kw = dict(
                algo_type="CONDITIONAL",
                symbol=o.symbol,
                side=o.side.value,
                type="STOP_MARKET",
                quantity=num(o.qty),
                trigger_price=num(o.stop_price),
                working_type="MARK_PRICE",
                client_algo_id=o.client_id,
            )
            if o.reduce_only:
                kw["reduce_only"] = "true"
            fn, conv = self.api.new_algo_order, order_from_algo
        else:
            kw = dict(
                symbol=o.symbol,
                side=o.side.value,
                type=o.type.value,
                quantity=num(o.qty),
                new_client_order_id=o.client_id,
                new_order_resp_type="RESULT",
            )
            if o.type is OrderType.LIMIT:
                kw.update(price=num(o.price), time_in_force=o.tif.value)
            if o.reduce_only:
                kw["reduce_only"] = "true"
            fn, conv = self.api.new_order, lambda d: order_from_rest(d, o.purpose)
        try:
            res = conv(await self._rest(fn, **kw))
        except BadRequestError as e:
            if e.status_code in UNKNOWN_CODES:
                raise
            reason = (
                "post_only" if e.status_code == POST_ONLY else f"{e.status_code} {e.error_message}"
            )
            return self._reject(o, reason)
        if o.type is OrderType.STOP_MARKET:
            res.qty = res.qty or o.qty  # algo response may omit quantity
        return self._apply(res)

    async def cancel(self, symbol: str, client_id: str) -> Order | None:
        stop = purpose_of(client_id) is Purpose.STOP
        accepted = False
        try:
            if stop:
                await self._rest(self.api.cancel_algo_order, client_algo_id=client_id)
                accepted = True
            else:
                d = await self._rest(
                    self.api.cancel_order, symbol=symbol, orig_client_order_id=client_id
                )
                return self._apply(order_from_rest(d, purpose_of(client_id)))
        except BadRequestError as e:
            if e.status_code in UNKNOWN_CODES:
                raise
            log.info("cancel %s: %s %s", client_id, e.status_code, e.error_message)
        # already final or algo cancel: read the final state
        o = await self.query(symbol, client_id)
        if accepted and o is not None and o.is_open:
            # algo query can lag right after a successful cancel (seen on testnet)
            o.status = OrderStatus.CANCELED
            self.emit(o)
        return o

    async def query(self, symbol: str, client_id: str) -> Order | None:
        p = purpose_of(client_id)
        try:
            if p is Purpose.STOP:
                d = await self._rest(self.api.query_algo_order, client_algo_id=client_id)
                return self._apply(order_from_algo(d))
            d = await self._rest(
                self.api.query_order, symbol=symbol, orig_client_order_id=client_id
            )
        except BadRequestError as e:
            if e.status_code in UNKNOWN_CODES:
                raise
            return None
        return self._apply(order_from_rest(d, p))

    async def open_orders(self, symbol: str) -> list[Order]:
        rows = await self._rest(self.api.current_all_open_orders, symbol=symbol)
        out = [order_from_rest(d, purpose_of(d["clientOrderId"])) for d in rows]
        algos = await self._rest(self.api.current_all_algo_open_orders, symbol=symbol)
        out += [order_from_algo(d) for d in algos]
        # orders from other tools are left alone
        return [o for o in out if o.purpose is not None]

    async def position_qty(self, symbol: str) -> float:
        rows = await self._rest(self.api.position_information_v3, symbol=symbol)
        return sum(float(r["positionAmt"]) for r in rows if r["symbol"] == symbol)

    async def entry_price(self, symbol: str) -> float:
        rows = await self._rest(self.api.position_information_v3, symbol=symbol)
        return next((float(r["entryPrice"]) for r in rows if float(r["positionAmt"])), 0.0)

    async def auto_cancel(self, symbol: str, seconds: int):
        """Exchange cancels open orders of the symbol if not refreshed in time; 0 disables."""
        await self._rest(
            self.api.auto_cancel_all_open_orders, symbol=symbol, countdown_time=seconds * 1000
        )

    async def arm(self, symbol: str, seconds: int):
        """Start or refresh the countdown. It also cancels stops (checked on testnet),
        so it may only run while the symbol has no position."""
        await self.auto_cancel(symbol, seconds)
        self.armed.add(symbol)

    async def before_protect(self, symbol: str):
        if symbol not in self.armed:
            return
        try:
            await self.auto_cancel(symbol, 0)
            self.armed.discard(symbol)
        except Exception:
            # stays armed, the live timer retries before the countdown runs out
            log.exception("%s countdown disarm failed", symbol)

    async def reconcile(self, orders: list[Order]):
        """Query tracked open orders after a stream gap and emit missed fills."""
        for o in orders:
            try:
                await self.query(o.symbol, o.client_id)
            except Exception:
                log.exception("reconcile %s failed", o.client_id)

    # ---- fills

    def _reject(self, o: Order, reason: str) -> Order:
        r = Order(**{k: getattr(o, k) for k in o.__slots__})
        r.status, r.reason = OrderStatus.REJECTED, reason
        self.emit(r)
        return r

    def _apply(
        self, o: Order, maker: bool | None = None, last_px: float = 0.0, fee: float | None = None
    ) -> Order:
        """Emit the order state plus a fill for any qty not emitted yet."""
        if maker is None:
            maker = o.type is OrderType.LIMIT and o.tif is TimeInForce.GTX
        prev = self.filled.get(o.client_id, 0.0)
        f = None
        if o.filled_qty > prev + EPS:
            qty = o.filled_qty - prev
            px = last_px or o.avg_price or o.stop_price or 0.0
            if px <= 0:
                # e.g. MARKET RESULT with avgPrice 0 on testnet: wait for the stream or a query
                self._later(o)
            else:
                if fee is None:
                    fee = qty * px * (self.fees.maker if maker else self.fees.taker)
                f = Fill(o.client_id, o.symbol, o.side, qty, px, fee, maker, o.time or _now())
                self.filled[o.client_id] = o.filled_qty
        self.emit(o, f)
        return o

    def _later(self, o: Order):
        if o.client_id in self.requery:
            return
        self.requery.add(o.client_id)

        async def run():
            try:
                for i in range(5):
                    await asyncio.sleep(1 + i)
                    if self.filled.get(o.client_id, 0.0) >= o.filled_qty - EPS:
                        return
                    await self.query(o.symbol, o.client_id)
                    if self.wake:
                        self.wake()
                log.error("%s fill price still unknown for %s", o.symbol, o.client_id)
            except Exception:
                log.exception("requery %s failed", o.client_id)
            finally:
                self.requery.discard(o.client_id)

        asyncio.get_running_loop().create_task(run())

    # ---- user data stream

    def on_user(self, d: dict):
        e = d.get("e")
        if e == "ORDER_TRADE_UPDATE":
            self._on_trade(d["o"])
        elif e == "ALGO_UPDATE":
            o = d["o"]
            if o.get("ai"):
                cid = o["caid"]
                oid = int(o["ai"])
                self.child[oid] = cid
                for x in self.orphans.pop(oid, []):
                    self._on_trade(x)
            if purpose_of(o.get("caid", "")) is Purpose.STOP:
                self._apply(algo_from_ws(o, int(d.get("E") or 0)))
        elif e == "listenKeyExpired":
            self.closed = "listen key expired"
        elif e == "ACCOUNT_UPDATE":
            log.debug("account update %s", d.get("a", {}).get("m"))
        if self.wake:
            self.wake()

    def _on_trade(self, o: dict):
        cid = o.get("c", "")
        p = purpose_of(cid)
        if p is None:
            oid = int(o["i"])
            if oid not in self.child:
                # maybe a stop child whose ALGO_UPDATE has not arrived yet
                self.orphans.setdefault(oid, []).append(o)
                return
            cid, p = self.child[oid], Purpose.STOP
        order = order_from_ws(o, cid, p)
        last_qty, last_px = _f(o.get("l")), _f(o.get("L"))
        prev = self.filled.get(cid, 0.0)
        exact = abs(order.filled_qty - prev - last_qty) < EPS
        fee = None
        if exact and o.get("N") == "USDT":
            fee = _f(o.get("n"))
        self._apply(order, maker=bool(o.get("m")), last_px=last_px if exact else 0.0, fee=fee)

    def user_url(self, listen_key: str) -> str:
        # private endpoint needs the event list, a bare listenKey subscription gets nothing
        base = self.ws_url.removesuffix("/stream").removesuffix("/ws")
        return f"{base}/private/ws?listenKey={listen_key}&events={'/'.join(USER_EVENTS)}"

    async def connect(self):
        lk = (await self._rest(self.api.start_user_data_stream))["listenKey"]
        self.session = aiohttp.ClientSession(trust_env=True)
        try:
            self.ws = await self.session.ws_connect(self.user_url(lk), heartbeat=30)
        except Exception:
            await self.session.close()
            raise
        self.closed = ""
        self.reader = asyncio.create_task(self._read(self.ws))
        log.info("user stream connected")

    async def _read(self, ws):
        async for msg in ws:
            if msg.type is aiohttp.WSMsgType.TEXT:
                self.received += 1
                self._on_message(json.loads(msg.data))
            elif msg.type is aiohttp.WSMsgType.ERROR:
                break
        if ws is self.ws and not self.closed:
            self.closed = "user stream closed"

    def _on_message(self, msg):
        try:
            # combined stream format wraps the event in "data"
            d = msg.get("data", msg) if isinstance(msg, dict) else msg
            if not isinstance(d, dict) or "e" not in d:
                log.info("user stream: %s", msg)
                return
            log.debug("user event %s", d)
            self.on_user(d)
        except Exception:
            log.exception("user stream message failed")

    async def keepalive(self):
        await self._rest(self.api.keepalive_user_data_stream)

    def stale(self) -> list[str]:
        return [self.closed] if self.closed else []

    async def close(self):
        ws, self.ws = self.ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                log.exception("close failed")
        if self.reader is not None:
            self.reader.cancel()
            self.reader = None
        if self.session is not None:
            await self.session.close()
            self.session = None


def _now() -> int:
    return int(time.time() * 1000)
