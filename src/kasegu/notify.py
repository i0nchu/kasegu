"""Optional event sinks. Each one is enabled from config; secrets come from env."""

import json
import logging
from pathlib import Path

import aiohttp
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from kasegu.config import AlertConfig
from kasegu.events import Event, EventBus

log = logging.getLogger(__name__)

ICON = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}


class TelegramSecrets(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", env_file=".env", extra="ignore")

    bot_token: SecretStr | None = None
    chat_id: str | None = None


class Http:
    """Shared session, created lazily inside the running loop."""

    def __init__(self):
        self.session: aiohttp.ClientSession | None = None

    async def post(self, url: str, **kw):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                trust_env=True, timeout=aiohttp.ClientTimeout(total=10)
            )
        async with self.session.post(url, **kw) as r:
            if r.status >= 400:
                raise RuntimeError(f"{url.split('?')[0]} -> {r.status} {await r.text()}")

    async def close(self):
        if self.session is not None:
            await self.session.close()
            self.session = None


def telegram_text(ev: Event, tag: str) -> str:
    head = f"{ICON.get(ev.level, '')} [{tag}] {ev.kind}" + (f" {ev.symbol}" if ev.symbol else "")
    return f"{head}\n{ev.message}"


def telegram_sink(http: Http, token: str, chat_id: str, tag: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    async def send(ev: Event):
        await http.post(url, json={"chat_id": chat_id, "text": telegram_text(ev, tag)})

    return send


def webhook_sink(http: Http, url: str, tag: str):
    # one JSON object per event, the contract for a future WebUI
    async def send(ev: Event):
        await http.post(url, json={"source": "kasegu", "tag": tag, **ev.to_dict()})

    return send


def file_sink(path: Path):
    # JSON lines, easy to tail or ingest
    path.parent.mkdir(parents=True, exist_ok=True)

    async def send(ev: Event):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev.to_dict(), default=str) + "\n")

    return send


def build_bus(cfg: AlertConfig, tag: str) -> tuple[EventBus, Http]:
    bus, http = EventBus(), Http()
    if cfg.events_path:
        bus.add(file_sink(cfg.events_path), "info")
    if cfg.telegram:
        sec = TelegramSecrets()
        if sec.bot_token and sec.chat_id:
            bus.add(
                telegram_sink(http, sec.bot_token.get_secret_value(), sec.chat_id, tag),
                cfg.min_level,
            )
        else:
            log.warning("alerts.telegram is on but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are unset")
    if cfg.webhook_url:
        bus.add(webhook_sink(http, cfg.webhook_url, tag), cfg.webhook_min_level)
    return bus, http
