from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect

logger = logging.getLogger("mf_agent")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required Zerodha configuration: {name}")
    return value


def create_client(access_token: str | None = None) -> KiteConnect:
    kite = KiteConnect(api_key=_required("ZERODHA_API_KEY"))

    if access_token:
        kite.set_access_token(access_token)

    return kite


def get_login_url() -> str:
    """Generate the Zerodha Kite Connect login URL."""
    kite = create_client()
    return kite.login_url()


def extract_request_token(redirect_url: str) -> str:
    """Extract request_token from Zerodha's redirect URL."""
    parsed = urlparse(redirect_url)

    token = parse_qs(parsed.query).get("request_token", [None])[0]

    if not token:
        raise ValueError(
            "No request_token found in Zerodha redirect URL."
        )

    return token


def generate_session(request_token: str) -> dict:
    """
    Exchange Zerodha request_token for an access token.
    """
    kite = create_client()

    return kite.generate_session(
        request_token,
        api_secret=_required("ZERODHA_API_SECRET"),
    )


def fetch_mf_holdings(access_token: str) -> list[dict]:
    """Fetch current mutual fund holdings from Zerodha."""
    kite = create_client(access_token)

    return kite.mf_holdings() or []


def normalise_holdings(items: list[dict]) -> list[dict]:
    """
    Convert Kite MF holdings into the existing mf_holdings.json format.
    """

    holdings = []

    for item in items:
        fund = str(item.get("fund") or "").strip()

        if not fund:
            continue

        quantity = float(item.get("quantity") or 0)
        average_price = float(item.get("average_price") or 0)
        last_price = float(item.get("last_price") or 0)

        invested_value = average_price * quantity
        current_value = last_price * quantity

        pnl = item.get("pnl")

        if pnl is None:
            pnl = current_value - invested_value

        pnl = float(pnl)

        pnl_pct = (
            pnl / invested_value * 100
            if invested_value
            else 0
        )

        holdings.append(
            {
                "fund": fund,
                "average_price": average_price,
                "quantity": quantity,
                "invested_value": round(invested_value, 2),
                "current_value": round(current_value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
                "folio": item.get("folio"),
                "tradingsymbol": item.get("tradingsymbol"),
                "last_price_date": item.get("last_price_date"),
                "isin": item.get("isin"),
            }
        )

    return holdings


def save_holdings(
    items: list[dict],
    path: str,
) -> None:

    holdings = normalise_holdings(items)

    payload = {
        "source": "zerodha_kite_connect",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "mf_holdings": holdings,
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Saved %d Zerodha MF holdings to %s",
        len(holdings),
        path,
    )


def refresh_holdings(
    request_token: str,
    path: str,
) -> dict:

    session = generate_session(request_token)

    access_token = session["access_token"]

    holdings = fetch_mf_holdings(access_token)

    save_holdings(
        holdings,
        path,
    )

    return {
        "user_id": session.get("user_id"),
        "user_name": session.get("user_name"),
        "holdings_count": len(holdings),
        "login_time": session.get("login_time"),
        "access_token": access_token,
    }