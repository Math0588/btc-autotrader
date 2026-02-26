"""
🤖 Polymarket BTC Autotrader v2.0 — Quant-Grade Paper Trading Bot
==================================================================
Runs 24/7, scans Polymarket markets, computes model probabilities
using Deribit IV, identifies edge, and executes paper trades.

v2.0 FIXES (Senior Quant Review):
  🔴 #1 — MTM from Polymarket CLOB mid (not model re-price)
  🔴 #2 — Active exit management (TP / SL / Time Decay)
  🔴 #3 — European-only probability (removed One-Touch bias)
  🔴 #4 — Kelly 25%, 8% max per trade, 60% max exposure
  🟠 #5 — Drawdown on total portfolio value (cash + exposure MTM)
  🟠 #6 — State reconciliation on startup (orphaned position fix)
  🟡 #7 — Market discovery via Gamma API tag search (not slug-only)

Usage:
    python autotrader.py                  # Run the bot
    python autotrader.py --once           # Single scan + report
    TELEGRAM_TOKEN=xxx TELEGRAM_CHAT=yyy python autotrader.py  # With alerts

Environment Variables:
    STARTING_CAPITAL   — Starting capital in USD (default: 100)
    SCAN_INTERVAL      — Seconds between scans (default: 300 = 5min)
    TELEGRAM_TOKEN     — Telegram bot token for notifications (optional)
    TELEGRAM_CHAT      — Telegram chat ID for notifications (optional)
    MIN_EDGE           — Minimum edge % to trade (default: 3.0)
    KELLY_FRAC         — Kelly fraction (default: 0.25)
    DRY_RUN            — If "false", attempt real trades (default: "true")
"""

import os
import sys
import json
import time
import math
import logging
import hashlib
import requests
import traceback
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from scipy.stats import norm

# ─── Setup Logging ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
        logging.FileHandler("autotrader.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("autotrader")

# ─── Configuration ────────────────────────────────────────────────────────────
# 🔴 FIX #4 — Conservative Kelly sizing to avoid ruin
CONFIG = {
    "starting_capital": float(os.getenv("STARTING_CAPITAL", "100")),
    "scan_interval": int(os.getenv("SCAN_INTERVAL", "300")),
    "min_edge_pct": float(os.getenv("MIN_EDGE", "3.0")),
    "kelly_fraction": float(os.getenv("KELLY_FRAC", "0.25")),   # 🔴 FIX #4: Quarter-Kelly (was 0.40)
    "max_position_pct": 0.08,        # 🔴 FIX #4: Max 8% per trade (was 25%)
    "max_exposure_pct": 0.60,        # 🔴 FIX #4: Max 60% exposed (was 100%)
    "max_positions": 5,              # 🔴 FIX #4: Max 5 simultaneous positions
    "polymarket_fee_pct": 2.0,
    "risk_free_rate": 0.045,
    "btc_drift_real": 0.10,
    "min_dte": 2,
    "max_dte": 45,
    "min_liquidity": 5000,
    "min_volume": 1000,
    "min_win_prob": 0.15,
    "max_drawdown_pct": 30,
    # 🔴 FIX #2 — Exit thresholds
    "take_profit_mult": 1.60,        # Exit if price >= 1.6x entry (+60%)
    "stop_loss_mult": 0.40,          # Exit if price <= 0.4x entry (-60%)
    "time_decay_hours": 6,           # Exit if DTE < 6h and position weak
    "time_decay_price_thresh": 0.35, # "weak" = current_mid < 0.35
    "telegram_token": os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat": os.getenv("TELEGRAM_CHAT", ""),
    "dry_run": os.getenv("DRY_RUN", "true").lower() != "false",
}

POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API = "https://clob.polymarket.com"
DERIBIT_API = "https://www.deribit.com/api/v2/public"

STATE_FILE = Path("autotrader_state.json")

# Month name mappings for slug generation (kept for backward compat)
_MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}


# ─── State Management ─────────────────────────────────────────────────────────

def load_state():
    """Load bot state from disk."""
    default = {
        "capital": CONFIG["starting_capital"],
        "initial_capital": CONFIG["starting_capital"],
        "positions": [],           # Open positions
        "closed_trades": [],       # Historical trades
        "total_pnl": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "peak_capital": CONFIG["starting_capital"],
        "max_drawdown": 0.0,
        "last_scan": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            # Merge with defaults for new fields
            for k, v in default.items():
                if k not in state:
                    state[k] = v
            return state
        except Exception as e:
            log.warning(f"Failed to load state: {e}, using defaults")
    return default


def save_state(state):
    """Persist state to disk."""
    state["last_saved"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


# ─── Telegram Notifications ──────────────────────────────────────────────────

def send_telegram(message):
    """Send a Telegram notification."""
    token = CONFIG["telegram_token"]
    chat = CONFIG["telegram_chat"]
    if not token or not chat:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={
            "chat_id": chat,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
    except Exception as e:
        log.warning(f"Telegram error: {e}")


# ─── Polymarket CLOB MTM ─────────────────────────────────────────────────────
# 🔴 FIX #1 — Fetch real mid price from Polymarket CLOB for mark-to-market

def get_polymarket_mid(token_id: str):
    """
    Fetch real mid price from Polymarket CLOB.
    This is the ACTUAL price you could exit at, not the model's opinion.
    """
    if not token_id:
        return None
    try:
        resp = requests.get(
            f"{POLYMARKET_CLOB_API}/midpoint",
            params={"token_id": token_id},
            timeout=5,
        )
        data = resp.json()
        mid = float(data.get("mid", 0))
        if 0 < mid <= 1:
            return mid
        return None
    except Exception as e:
        log.debug(f"CLOB midpoint fetch failed for {token_id[:16]}...: {e}")
        return None


def mark_to_market(state: dict, spot: float = None) -> float:
    """
    🔴 FIX #1 — Mark all positions to market using Polymarket CLOB mid.
    Returns total unrealized PnL.
    Falls back to cost-based valuation if CLOB is unavailable.
    """
    total_unrealized = 0.0
    mtm_count = 0

    for pos in state["positions"]:
        token_id = pos.get("token_id", "")
        mid = get_polymarket_mid(token_id) if token_id else None

        entry = pos.get("entry_price", pos.get("entry", 0))
        n = pos.get("n_contracts", pos.get("n", 1))

        if mid is not None:
            # Real MTM from Polymarket CLOB
            if pos["direction"] == "BUY_YES":
                unrealized = (mid - entry) * n
            else:  # BUY_NO
                unrealized = ((1.0 - mid) - entry) * n
            pos["current_mid"] = mid
            pos["unrealized_pnl"] = round(unrealized, 4)
            pos["mtm_source"] = "clob"
            mtm_count += 1
        else:
            # Fallback: use model re-price or cost as proxy
            pos["current_mid"] = entry
            pos["unrealized_pnl"] = 0.0
            pos["mtm_source"] = "cost_proxy"

        total_unrealized += pos.get("unrealized_pnl", 0.0)

    if mtm_count:
        log.info(f"📊 MTM: {mtm_count}/{len(state['positions'])} positions marked via CLOB")

    return total_unrealized


# ─── Deribit Data ─────────────────────────────────────────────────────────────

def fetch_btc_spot():
    """Get current BTC/USD price from Deribit."""
    try:
        resp = requests.get(
            f"{DERIBIT_API}/get_index_price",
            params={"index_name": "btc_usd"},
            timeout=10,
        )
        data = resp.json()
        return data["result"]["index_price"]
    except Exception as e:
        log.error(f"Failed to fetch BTC spot: {e}")
        return None


def fetch_deribit_iv_surface():
    """
    Fetch full IV surface from Deribit using a single API call.
    Returns list of {moneyness, dte, iv, strike, type, expiry}.
    """
    try:
        spot = fetch_btc_spot()
        if not spot:
            return []

        # Single API call gets all option summaries with mark_iv
        resp = requests.get(
            f"{DERIBIT_API}/get_book_summary_by_currency",
            params={"currency": "BTC", "kind": "option"},
            timeout=20,
        )
        summaries = resp.json().get("result", [])

        iv_points = []
        now = datetime.now(timezone.utc)

        for s in summaries:
            iv = s.get("mark_iv", 0)
            if not iv or iv <= 0:
                continue

            # Parse instrument name: BTC-28FEB26-64000-C
            parts = s.get("instrument_name", "").split("-")
            if len(parts) < 4:
                continue

            try:
                strike = float(parts[2])
                opt_type = parts[3]  # C or P
                exp_str = parts[1]
                exp_dt = datetime.strptime(exp_str, "%d%b%y").replace(tzinfo=timezone.utc)
                exp_dt = exp_dt.replace(hour=8)  # Deribit settles at 08:00 UTC

                dte = (exp_dt - now).total_seconds() / 86400
                if dte <= 0 or dte > 400:
                    continue

                iv_points.append({
                    "moneyness": strike / spot,
                    "dte": dte,
                    "iv": iv,
                    "strike": strike,
                    "type": opt_type,
                    "expiry": exp_dt.strftime("%Y-%m-%d"),
                })
            except Exception:
                continue

        log.info(f"Fetched {len(iv_points)} IV data points from Deribit (1 API call)")
        return iv_points

    except Exception as e:
        log.error(f"Failed to fetch Deribit IV surface: {e}")
        return []


# ─── Polymarket Market Discovery ──────────────────────────────────────────────

def _pm_get(url, params=None):
    """GET from Polymarket gamma API."""
    headers = {"Accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_btc_barrier_markets():
    """
    🟡 FIX #7 — Hybrid market discovery:
      1. Tag-based search via Gamma API (catches non-standard slugs)
      2. Slug-based search (backward compat for known naming patterns)
    """
    markets = []
    seen = set()
    now = datetime.now(timezone.utc)

    # ─── Method 1: Tag-based discovery (NEW) ──────────────────────────
    try:
        params = {
            "tag": "crypto",
            "active": True,
            "archived": False,
            "limit": 100,
        }
        events = _pm_get(f"{POLYMARKET_GAMMA_API}/events", params=params)
        if isinstance(events, list):
            for event in events:
                title = (event.get("title", "") or "").lower()
                if not any(kw in title for kw in ["bitcoin", "btc"]):
                    continue
                sub_markets = event.get("markets", [])
                for sm in sub_markets:
                    if sm.get("closed", False):
                        continue
                    parsed = _parse_market(sm)
                    if parsed and parsed["condition_id"] not in seen:
                        seen.add(parsed["condition_id"])
                        markets.append(parsed)
        log.info(f"Tag discovery: found {len(markets)} markets")
    except Exception as e:
        log.debug(f"Tag-based discovery error: {e}")

    # ─── Method 2: Slug-based discovery (original) ────────────────────
    for delta in range(-1, 21):
        dt = now + timedelta(days=delta)
        month_name = _MONTH_NAMES[dt.month]
        slug = f"bitcoin-above-on-{month_name}-{dt.day}"

        try:
            events = _pm_get(f"{POLYMARKET_GAMMA_API}/events", params={"slug": slug})
            if not events or not isinstance(events, list) or len(events) == 0:
                continue

            event = events[0]
            sub_markets = event.get("markets", [])

            for sm in sub_markets:
                if sm.get("closed", False):
                    continue
                parsed = _parse_market(sm)
                if parsed and parsed["condition_id"] not in seen:
                    seen.add(parsed["condition_id"])
                    markets.append(parsed)

        except requests.exceptions.HTTPError:
            continue
        except Exception as e:
            log.debug(f"Error fetching slug {slug}: {e}")

    # ─── Method 3: Annual "what price will bitcoin hit" markets ───────
    for annual_slug in ["what-price-will-bitcoin-hit-in-2026", "what-price-will-bitcoin-hit-in-2025"]:
        try:
            events = _pm_get(f"{POLYMARKET_GAMMA_API}/events", params={"slug": annual_slug})
            if events and isinstance(events, list) and len(events) > 0:
                event = events[0]
                for sm in event.get("markets", []):
                    if sm.get("closed", False):
                        continue
                    parsed = _parse_market(sm)
                    if parsed and parsed["condition_id"] not in seen:
                        seen.add(parsed["condition_id"])
                        markets.append(parsed)
        except Exception:
            pass

    markets.sort(key=lambda m: (m["expiry_dt"], m["barrier_price"]))
    log.info(f"Total: {len(markets)} BTC barrier markets on Polymarket")
    return markets


def _parse_market(data):
    """Parse a Polymarket sub-market into our standard format."""
    import re

    title = data.get("question", "") or data.get("title", "") or ""
    title_lower = title.lower()

    if not any(kw in title_lower for kw in ["bitcoin", "btc"]):
        return None
    if not any(kw in title_lower for kw in ["above", "below", "hit", "reach", "price"]):
        return None

    # Extract barrier price
    patterns = [
        r'\$([0-9]{1,3}(?:,[0-9]{3})+)',
        r'\$([0-9]+(?:\.[0-9]+)?)\s*[kK]',
        r'\$([0-9]+(?:,?[0-9]+)*)',
    ]
    barrier = None
    for pat in patterns:
        m = re.search(pat, title)
        if m:
            raw = m.group(1).replace(",", "")
            val = float(raw)
            if val < 1000:
                val *= 1000  # $100k → 100000
            if 10000 < val < 500000:
                barrier = val
                break
    if barrier is None:
        return None

    scenario = "below" if ("below" in title_lower or "under" in title_lower) else "above"

    # Parse expiry
    expiry_str = data.get("end_date_iso", "") or data.get("endDate", "")
    expiry_dt = None
    if expiry_str:
        try:
            expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    if expiry_dt is None:
        return None

    now = datetime.now(timezone.utc)
    dte = (expiry_dt - now).total_seconds() / 86400
    if dte <= 0:
        return None

    # Parse prices
    yes_price = None
    outcome_prices = data.get("outcomePrices", "")
    if outcome_prices:
        try:
            prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
            if len(prices) >= 2:
                yes_price = float(prices[0])
        except Exception:
            pass

    if yes_price is None:
        best_bid = data.get("bestBid")
        best_ask = data.get("bestAsk")
        if best_bid and best_ask:
            yes_price = (float(best_bid) + float(best_ask)) / 2

    if yes_price is None:
        return None
    no_price = 1.0 - yes_price

    volume = float(data.get("volume", 0) or data.get("volumeNum", 0) or 0)
    liquidity = float(data.get("liquidity", 0) or data.get("liquidityNum", 0) or 0)
    condition_id = data.get("conditionId", "") or data.get("condition_id", "")

    # Extract token IDs for CLOB MTM
    token_ids = []
    clob_token_ids = data.get("clobTokenIds", "")
    if clob_token_ids:
        try:
            if isinstance(clob_token_ids, str):
                token_ids = json.loads(clob_token_ids)
            else:
                token_ids = clob_token_ids
        except Exception:
            pass

    # Also check tokens array
    if not token_ids:
        tokens = data.get("tokens", [])
        for t in tokens:
            tid = t.get("token_id", "")
            if tid:
                token_ids.append(tid)

    return {
        "title": title,
        "condition_id": condition_id,
        "barrier_price": barrier,
        "scenario": scenario,
        "yes_price": round(yes_price, 4),
        "no_price": round(no_price, 4),
        "pm_prob": round(yes_price * 100, 2),
        "expiry_dt": expiry_dt,
        "dte_days": round(dte, 2),
        "volume": volume,
        "liquidity": liquidity,
        "token_ids": token_ids,  # 🔴 FIX #1: store for CLOB MTM
    }


# ─── Probability Models ──────────────────────────────────────────────────────
# 🔴 FIX #3 — European-only probability model (removed One-Touch)

def compute_european_prob(spot, barrier, dte_days, sigma, r=0.045, scenario="above"):
    """
    Standard Black-Scholes P(above/below barrier at expiry).
    
    🔴 FIX #3: Polymarket "BTC above $X on date Y" is a European digital.
    It pays if BTC is above the barrier AT EXPIRY, not if it ever touches it.
    One-Touch was systematically overestimating probabilities for OTM strikes.
    """
    if spot <= 0 or barrier <= 0 or dte_days <= 0 or sigma <= 0:
        return None
    T = dte_days / 365.0
    d2 = (math.log(spot / barrier) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    if scenario == "above":
        prob = norm.cdf(d2)
    else:
        prob = norm.cdf(-d2)

    # 🔴 FIX #3: Skew adjustment for BTC vol smile
    # BTC has negative skew: OTM puts are more expensive than OTM calls
    # This means downside probabilities are underestimated by flat-vol BS
    moneyness = barrier / spot
    if scenario == "above" and moneyness > 1.10:
        # Far OTM upside strike — model overestimates, apply negative adjustment
        skew_adj = -0.02
    elif scenario == "below" and moneyness < 0.90:
        # Far OTM downside strike — model underestimates, apply positive adjustment
        skew_adj = 0.02
    else:
        skew_adj = 0.0

    prob = max(0.0, min(1.0, prob + skew_adj))
    return prob


def interpolate_iv(iv_points, target_moneyness, target_dte):
    """
    Interpolate IV from Deribit surface data.
    Uses distance-weighted average of nearest options.
    """
    from scipy.interpolate import griddata

    if not iv_points or len(iv_points) < 3:
        return None

    call_points = [p for p in iv_points if p.get("type") == "C"]
    if len(call_points) < 5:
        call_points = iv_points

    m_arr = np.array([p["moneyness"] for p in call_points])
    d_arr = np.array([p["dte"] for p in call_points])
    iv_arr = np.array([p["iv"] for p in call_points])

    mask = (m_arr > 0.1) & (m_arr < 5.0) & (d_arr > 0)
    m_arr, d_arr, iv_arr = m_arr[mask], d_arr[mask], iv_arr[mask]

    if len(m_arr) < 3:
        return None

    # Try cubic → linear → weighted nearest
    for method in ["cubic", "linear", "nearest"]:
        try:
            iv = griddata((m_arr, d_arr), iv_arr,
                          (target_moneyness, target_dte), method=method)
            if not np.isnan(iv) and iv > 0:
                return float(iv)
        except Exception:
            continue

    # Distance-weighted fallback
    distances = np.sqrt(
        ((m_arr - target_moneyness) * 5) ** 2 +
        ((d_arr - target_dte) / 30) ** 2
    )
    nearest = np.argsort(distances)[:5]
    weights = 1.0 / (distances[nearest] + 0.001)
    return float(np.average(iv_arr[nearest], weights=weights))


# ─── Exit Strategy ────────────────────────────────────────────────────────────
# 🔴 FIX #2 — Active position management with TP/SL/Time Decay

def should_exit_early(pos: dict, current_mid: float, dte_hours: float) -> tuple:
    """
    Determine if a position should be exited before expiry.
    
    Exit rules:
      1. TAKE_PROFIT: current_mid >= entry * 1.6  (+60%)
      2. STOP_LOSS:   current_mid <= entry * 0.4   (-60%)
      3. TIME_DECAY:  DTE < 6h AND position is weak (mid < 0.35)
      4. EDGE_GONE:   Re-evaluated edge is negative (optional, future)
    
    Returns: (should_exit: bool, reason: str, exit_price: float)
    """
    entry = pos.get("entry_price", pos.get("entry", 0))
    direction = pos["direction"]

    # For BUY_NO, the relevant price is (1 - mid) since we bought NO token
    if direction == "BUY_NO":
        effective_price = 1.0 - current_mid
    else:
        effective_price = current_mid

    # Take Profit
    if effective_price >= entry * CONFIG["take_profit_mult"]:
        return True, "TAKE_PROFIT", effective_price

    # Stop Loss
    if effective_price <= entry * CONFIG["stop_loss_mult"]:
        return True, "STOP_LOSS", effective_price

    # Time Decay Exit: near expiry with weak position
    if dte_hours < CONFIG["time_decay_hours"] and effective_price < CONFIG["time_decay_price_thresh"]:
        return True, "TIME_DECAY_EXIT", effective_price

    return False, "", effective_price


def execute_early_exit(state: dict, pos: dict, exit_price: float, reason: str):
    """
    Execute an early exit by selling the position at current mid.
    """
    fee_rate = CONFIG["polymarket_fee_pct"] / 100
    n = pos.get("n_contracts", pos.get("n", 1))
    cost = pos.get("cost", 0)

    # Revenue from selling at current price (minus fees)
    revenue = exit_price * n * (1 - fee_rate)
    pnl = revenue - cost

    # Return revenue to capital
    state["capital"] += revenue

    if pnl >= 0:
        state["wins"] += 1
        result = "WIN"
        emoji = "🟢"
    else:
        state["losses"] += 1
        result = "LOSS"
        emoji = "🔴"

    state["total_pnl"] += pnl

    pos["status"] = "closed"
    pos["result"] = result
    pos["exit_reason"] = reason
    pos["exit_price"] = round(exit_price, 4)
    pos["pnl"] = round(pnl, 4)
    pos["settled_at"] = datetime.now(timezone.utc).isoformat()
    state["closed_trades"].append(pos)

    entry = pos.get("entry_price", pos.get("entry", 0))

    log.info(f"{emoji} EARLY EXIT [{reason}]: {pos.get('trade_desc', pos.get('desc', ''))} | "
             f"Entry: {entry:.3f} → Exit: {exit_price:.3f} | "
             f"PnL: ${pnl:+.2f} | Capital: ${state['capital']:.2f}")

    # Telegram alert for early exit
    pnl_pct = (pnl / cost * 100) if cost > 0 else 0
    msg = (f"{emoji} <b>EARLY EXIT [{reason}]</b>\n"
           f"{pos.get('trade_desc', pos.get('desc', ''))}\n"
           f"Entry: <code>{entry:.3f}</code> → Exit: <code>{exit_price:.3f}</code>\n"
           f"PnL: <code>${pnl:+.2f}</code> ({pnl_pct:+.1f}%)\n"
           f"Capital: <code>${state['capital']:.2f}</code>")
    send_telegram(msg)


# ─── Drawdown Computation ────────────────────────────────────────────────────
# 🟠 FIX #5 — Drawdown on total portfolio value (cash + MTM exposure)

def compute_portfolio_value(state: dict) -> float:
    """
    Compute total portfolio value = cash + sum of position MTM values.
    Positions with CLOB mid use real MTM; others fall back to cost.
    """
    exposure_value = 0.0
    for pos in state["positions"]:
        n = pos.get("n_contracts", pos.get("n", 1))
        if pos.get("mtm_source") == "clob" and "current_mid" in pos:
            # Use MTM value
            if pos["direction"] == "BUY_YES":
                exposure_value += pos["current_mid"] * n
            else:
                exposure_value += (1.0 - pos["current_mid"]) * n
        else:
            # Fallback to cost
            exposure_value += pos.get("cost", 0)
    return state["capital"] + exposure_value


def update_drawdown(state: dict):
    """
    🟠 FIX #5: Update peak capital and drawdown using total portfolio value,
    not just cash.
    """
    total_value = compute_portfolio_value(state)

    if total_value > state.get("peak_capital", CONFIG["starting_capital"]):
        state["peak_capital"] = total_value

    if state["peak_capital"] > 0:
        current_dd = (1 - total_value / state["peak_capital"]) * 100
        state["max_drawdown"] = max(state.get("max_drawdown", 0), current_dd)
        state["current_drawdown"] = round(current_dd, 2)
    else:
        state["current_drawdown"] = 0.0

    state["portfolio_value"] = round(total_value, 2)


# ─── State Reconciliation ────────────────────────────────────────────────────
# 🟠 FIX #6 — Reconcile orphaned positions on startup

def reconcile_on_startup(state: dict, spot: float):
    """
    At startup, settle all positions that expired during downtime.
    This prevents 'ghost' positions and ensures accurate capital/stats.
    """
    now = datetime.now(timezone.utc)
    reconciled = 0

    for pos in list(state["positions"]):
        expiry_str = pos.get("expiry_dt", "")
        if not expiry_str:
            continue
        try:
            expiry = datetime.fromisoformat(expiry_str)
        except Exception:
            continue

        if now >= expiry:
            log.warning(f"⚠️ Reconciling expired position from downtime: "
                        f"{pos.get('trade_desc', pos.get('desc', pos.get('title', '?')))}")
            settle_single_position(state, pos, spot, reason="RECONCILED_ON_STARTUP")
            reconciled += 1

    if reconciled:
        # Remove reconciled positions from open list
        state["positions"] = [p for p in state["positions"] if p.get("status") != "closed"]
        log.info(f"🔄 Reconciled {reconciled} expired position(s) from downtime")
        save_state(state)

        msg = (f"🔄 <b>Startup Reconciliation</b>\n"
               f"Settled {reconciled} position(s) expired during downtime.\n"
               f"Capital: <code>${state['capital']:.2f}</code>")
        send_telegram(msg)


def settle_single_position(state: dict, pos: dict, spot: float, reason: str = "EXPIRY"):
    """Settle a single position (used by both normal settlement and reconciliation)."""
    barrier = pos.get("barrier_price", pos.get("barrier", 0))
    direction = pos["direction"]
    cost = pos["cost"]
    n = pos.get("n_contracts", pos.get("n", 1))
    entry = pos.get("entry_price", pos.get("entry", 0))
    scenario = pos.get("scenario", "above")

    # Determine if the bet won
    if direction == "BUY_YES":
        won = spot >= barrier if scenario == "above" else spot < barrier
    else:  # BUY_NO
        won = spot < barrier if scenario == "above" else spot >= barrier

    if won:
        fee = CONFIG["polymarket_fee_pct"] / 100
        payout = n * (1.0 - entry) * (1 - fee)
        pnl = payout  # Cost was already deducted when opening
        state["capital"] += cost + pnl  # Return cost + profit
        state["wins"] += 1
        result = "WIN"
        emoji = "✅"
    else:
        pnl = -cost  # Lost the entire cost
        state["losses"] += 1
        result = "LOSS"
        emoji = "❌"

    state["total_pnl"] += pnl

    pos["status"] = "closed"
    pos["result"] = result
    pos["exit_reason"] = reason
    pos["pnl"] = round(pnl, 2)
    pos["settled_at"] = datetime.now(timezone.utc).isoformat()
    pos["spot_at_expiry"] = spot
    state["closed_trades"].append(pos)

    log.info(f"{emoji} SETTLED [{reason}]: {pos.get('trade_desc', pos.get('desc', ''))} → {result} | "
             f"PnL: ${pnl:+.2f} | Capital: ${state['capital']:.2f}")

    # Telegram alert
    msg = (f"{emoji} <b>{result}</b> [{reason}]: {pos.get('trade_desc', pos.get('desc', ''))}\n"
           f"PnL: <code>${pnl:+.2f}</code>\n"
           f"Capital: <code>${state['capital']:.2f}</code>")
    send_telegram(msg)


# ─── Core Strategy Engine ─────────────────────────────────────────────────────

def analyze_market(market, spot, iv_points, capital, current_exposure, n_open_positions):
    """
    Analyze a single market for edge. Returns opportunity dict or None.

    Quant-grade analysis:
      🔴 #3 — European probability only (no One-Touch)
      🔴 #4 — Conservative Kelly sizing with drawdown control
      Liquidity-adjusted edge
      Min win probability filter
    """
    barrier = market["barrier_price"]
    dte = market["dte_days"]
    scenario = market["scenario"]
    pm_prob = market["pm_prob"]  # In %

    # Filters
    if dte < CONFIG["min_dte"] or dte > CONFIG["max_dte"]:
        return None
    if market["liquidity"] < CONFIG["min_liquidity"]:
        return None
    if market["volume"] < CONFIG["min_volume"]:
        return None
    # 🔴 FIX #4: Enforce max positions
    if n_open_positions >= CONFIG["max_positions"]:
        return None

    # Get IV for this strike
    moneyness = barrier / spot
    iv = interpolate_iv(iv_points, moneyness, dte)
    if iv is None or iv <= 0:
        return None
    sigma = iv / 100.0

    # 🔴 FIX #3 — European probability ONLY
    # Polymarket "BTC above $X on date Y" pays at expiry, not on touch.
    model_prob = compute_european_prob(spot, barrier, dte, sigma,
                                       CONFIG["risk_free_rate"], scenario)
    if model_prob is None:
        return None

    model_prob_pct = model_prob * 100

    # ══════ EDGE CALCULATION ══════
    edge = model_prob_pct - pm_prob  # Positive = YES underpriced

    if abs(edge) < CONFIG["min_edge_pct"]:
        return None

    # Determine direction
    if edge > 0:
        direction = "BUY_YES"
        entry_price = market["yes_price"]
        win_prob = model_prob
        trade_desc = f"BTC AU-DESSUS ${barrier:,.0f}"
    else:
        direction = "BUY_NO"
        entry_price = market["no_price"]
        win_prob = 1 - model_prob
        trade_desc = f"BTC SOUS ${barrier:,.0f}"

    # Min win probability filter
    if win_prob < CONFIG["min_win_prob"]:
        return None

    # ══════ LIQUIDITY-ADJUSTED EDGE ══════
    liq = market["liquidity"]
    liq_factor = min(liq / 20000, 1.0)
    adjusted_edge = abs(edge) * liq_factor

    # ══════ 🔴 FIX #4 — KELLY SIZING (Conservative) ══════
    fee_rate = CONFIG["polymarket_fee_pct"] / 100
    payout = (1.0 - entry_price) * (1 - fee_rate)
    cost = entry_price

    if cost <= 0 or payout <= 0:
        return None

    b = payout / cost  # Odds ratio
    p = win_prob
    q = 1 - p
    kelly_full = (p * b - q) / b
    if kelly_full <= 0:
        return None

    kelly = kelly_full * CONFIG["kelly_fraction"]  # Quarter-Kelly

    # Drawdown control: reduce sizing during drawdowns
    drawdown_pct = (1 - capital / CONFIG["starting_capital"]) * 100
    if drawdown_pct > CONFIG["max_drawdown_pct"]:
        kelly *= 0.5  # Half size during big drawdowns
    elif drawdown_pct > CONFIG["max_drawdown_pct"] / 2:
        kelly *= 0.75  # 3/4 size during moderate drawdowns

    # 🔴 FIX #4 — Strict position sizing caps
    max_pos = capital * CONFIG["max_position_pct"]  # 8% of capital
    remaining_capacity = capital * CONFIG["max_exposure_pct"] - current_exposure  # 60% cap
    if remaining_capacity <= 0:
        return None

    position_usd = min(kelly * capital, max_pos, remaining_capacity)
    position_usd = max(position_usd, 1.0)  # Minimum $1

    n_contracts = int(position_usd / entry_price)
    if n_contracts < 1:
        return None

    actual_cost = n_contracts * entry_price
    expected_pnl = p * payout * n_contracts - q * cost * n_contracts
    profit_if_win_pct = (payout / cost) * 100

    # Get token_id for future CLOB MTM
    token_ids = market.get("token_ids", [])
    # YES token is typically first, NO token second
    if direction == "BUY_YES":
        token_id = token_ids[0] if len(token_ids) > 0 else ""
    else:
        token_id = token_ids[1] if len(token_ids) > 1 else (token_ids[0] if len(token_ids) > 0 else "")

    return {
        "title": market["title"],
        "condition_id": market["condition_id"],
        "barrier_price": barrier,
        "scenario": scenario,
        "direction": direction,
        "trade_desc": trade_desc,
        "dte_days": dte,
        "expiry_dt": market["expiry_dt"],
        "pm_prob": pm_prob,
        "model_prob": round(model_prob_pct, 2),
        "edge": round(abs(edge), 2),
        "adjusted_edge": round(adjusted_edge, 2),
        "entry_price": entry_price,
        "win_probability": round(win_prob * 100, 1),
        "profit_if_win_pct": round(profit_if_win_pct, 1),
        "kelly": round(kelly, 4),
        "n_contracts": n_contracts,
        "position_usd": round(actual_cost, 2),
        "expected_pnl": round(expected_pnl, 2),
        "liquidity": market["liquidity"],
        "iv_used": round(iv, 2),
        "token_id": token_id,          # 🔴 FIX #1: for CLOB MTM
        "token_ids": token_ids,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Trade Execution (Paper) ──────────────────────────────────────────────────

def execute_paper_trade(state, opportunity):
    """Execute a paper trade: deduct cost, add to positions."""
    cost = opportunity["position_usd"]

    if cost > state["capital"]:
        log.warning(f"Insufficient capital: ${state['capital']:.2f} < ${cost:.2f}")
        return False

    position = {
        "id": hashlib.md5(f"{opportunity['condition_id']}:{time.time()}".encode()).hexdigest()[:12],
        "condition_id": opportunity["condition_id"],
        "title": opportunity["title"],
        "trade_desc": opportunity["trade_desc"],
        "direction": opportunity["direction"],
        "scenario": opportunity["scenario"],
        "barrier_price": opportunity["barrier_price"],
        "entry_price": opportunity["entry_price"],
        "n_contracts": opportunity["n_contracts"],
        "cost": round(cost, 2),
        "win_probability": opportunity["win_probability"],
        "profit_if_win_pct": opportunity["profit_if_win_pct"],
        "expected_pnl": opportunity["expected_pnl"],
        "edge": opportunity["edge"],
        "model_prob": opportunity["model_prob"],
        "pm_prob": opportunity["pm_prob"],
        "token_id": opportunity.get("token_id", ""),    # 🔴 FIX #1
        "token_ids": opportunity.get("token_ids", []),
        "expiry_dt": opportunity["expiry_dt"].isoformat() if isinstance(opportunity["expiry_dt"], datetime) else opportunity["expiry_dt"],
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "status": "open",
    }

    state["capital"] -= cost
    state["positions"].append(position)
    state["total_trades"] += 1

    log.info(f"📈 TRADE: {opportunity['trade_desc']} | "
             f"${cost:.2f} | Win={opportunity['win_probability']:.0f}% | "
             f"Edge={opportunity['edge']:.1f}% | "
             f"Profit if win: +{opportunity['profit_if_win_pct']:.0f}%")

    return True


# ─── Position Management ──────────────────────────────────────────────────────

def manage_positions(state, spot):
    """
    🔴 FIX #1+2: Full position management cycle.
    1. Mark-to-market all positions via Polymarket CLOB
    2. Check for early exit signals (TP/SL/Time Decay)
    3. Settle expired positions
    """
    now = datetime.now(timezone.utc)
    still_open = []
    settled_count = 0
    exited_count = 0

    # First, MTM all positions
    mark_to_market(state, spot)

    for pos in state["positions"]:
        expiry_str = pos.get("expiry_dt", "")
        try:
            expiry = datetime.fromisoformat(expiry_str)
        except Exception:
            still_open.append(pos)
            continue

        # ─── Check for expiry settlement ──────────────────────────────
        if now >= expiry:
            settle_single_position(state, pos, spot, reason="EXPIRY")
            settled_count += 1
            continue

        # ─── 🔴 FIX #2: Check for early exit signals ─────────────────
        dte_hours = (expiry - now).total_seconds() / 3600
        current_mid = pos.get("current_mid", None)

        if current_mid is not None and current_mid > 0:
            should_exit, reason, effective_price = should_exit_early(pos, current_mid, dte_hours)
            if should_exit:
                execute_early_exit(state, pos, effective_price, reason)
                exited_count += 1
                continue

        still_open.append(pos)

    state["positions"] = still_open

    if settled_count or exited_count:
        log.info(f"Position mgmt: {settled_count} settled, {exited_count} early exits. "
                 f"Capital: ${state['capital']:.2f}")


# ─── Enhanced Telegram Reports ────────────────────────────────────────────────

def send_live_portfolio_report(state: dict, spot: float):
    """
    Send a comprehensive portfolio report to Telegram.
    Includes live PnL, position-level MTM, and risk metrics.
    """
    total_value = compute_portfolio_value(state)
    total_return_pct = (total_value - state["initial_capital"]) / state["initial_capital"] * 100
    n_positions = len(state["positions"])
    exposure = sum(p.get("cost", 0) for p in state["positions"])
    unrealized = sum(p.get("unrealized_pnl", 0) for p in state["positions"])
    win_rate = state["wins"] / max(state["total_trades"], 1) * 100
    current_dd = state.get("current_drawdown", 0)

    # Header
    if total_return_pct >= 0:
        trend = "📈"
    else:
        trend = "📉"

    msg = (
        f"{trend} <b>PORTFOLIO LIVE REPORT</b>\n"
        f"{'━' * 28}\n"
        f"💰 Total Value: <code>${total_value:.2f}</code>\n"
        f"💵 Cash: <code>${state['capital']:.2f}</code>\n"
        f"📊 Exposure: <code>${exposure:.2f}</code>\n"
        f"📈 Unrealized: <code>${unrealized:+.2f}</code>\n"
        f"💹 Total PnL: <code>${state['total_pnl']:+.2f}</code>\n"
        f"📊 Return: <code>{total_return_pct:+.1f}%</code>\n"
        f"{'━' * 28}\n"
        f"🎯 Win Rate: {win_rate:.0f}% ({state['wins']}W / {state['losses']}L)\n"
        f"📉 Max DD: {state['max_drawdown']:.1f}% | Current: {current_dd:.1f}%\n"
        f"📋 Open: {n_positions} | Total: {state['total_trades']}\n"
        f"₿ BTC: <code>${spot:,.0f}</code>\n"
    )

    # Position details
    if state["positions"]:
        msg += f"\n<b>Open Positions:</b>\n"
        for i, pos in enumerate(state["positions"][:5], 1):
            desc = pos.get("trade_desc", pos.get("desc", "?"))
            entry = pos.get("entry_price", pos.get("entry", 0))
            mid = pos.get("current_mid", entry)
            upnl = pos.get("unrealized_pnl", 0)
            mtm_src = pos.get("mtm_source", "?")
            cost = pos.get("cost", 0)
            pnl_pct = (upnl / cost * 100) if cost > 0 else 0

            emoji = "🟢" if upnl >= 0 else "🔴"
            msg += (f"  {emoji} {desc}\n"
                    f"     Entry: {entry:.3f} | Mid: {mid:.3f} [{mtm_src}]\n"
                    f"     PnL: ${upnl:+.2f} ({pnl_pct:+.1f}%)\n")

    # Objective tracker
    target = 10000
    progress = (total_value / target) * 100
    msg += (f"\n🎯 <b>Objective:</b> ${total_value:.0f} / ${target:,} "
            f"({progress:.1f}%)\n")

    send_telegram(msg)


def send_opportunity_report(opportunities: list):
    """Send a structured report of current trading opportunities."""
    if not opportunities:
        return

    msg = f"🔍 <b>Market Scanner</b> — {len(opportunities)} opportunities\n\n"

    for i, opp in enumerate(opportunities[:8], 1):
        direction_emoji = "🟢" if opp["direction"] == "BUY_YES" else "🔴"
        msg += (
            f"{i}. {direction_emoji} <b>{opp['trade_desc']}</b>\n"
            f"   Edge: {opp['edge']:.1f}% | Win: {opp['win_probability']:.0f}%\n"
            f"   PM: {opp['pm_prob']:.0f}% vs Model: {opp['model_prob']:.0f}%\n"
            f"   Size: ${opp['position_usd']:.2f} | Kelly: {opp['kelly']:.1%}\n"
            f"   IV: {opp['iv_used']:.1f}% | DTE: {opp['dte_days']:.0f}d\n\n"
        )

    send_telegram(msg)


# ─── Main Bot Loop ────────────────────────────────────────────────────────────

def run_scan(state):
    """Run one full scan cycle."""
    log.info("=" * 60)
    log.info(f"🔍 Starting scan... Capital: ${state['capital']:.2f} | "
             f"Open positions: {len(state['positions'])} | "
             f"Total PnL: ${state['total_pnl']:+.2f}")

    # 1. Fetch spot price
    spot = fetch_btc_spot()
    if not spot:
        log.error("Failed to fetch BTC price, skipping scan")
        return

    log.info(f"BTC Spot: ${spot:,.2f}")

    # 2. 🔴 FIX #1+2: Full position management (MTM + exits + settlement)
    manage_positions(state, spot)

    # 3. 🟠 FIX #5: Update drawdown on total portfolio value
    update_drawdown(state)

    # 4. Fetch IV surface
    iv_points = fetch_deribit_iv_surface()
    if not iv_points:
        log.warning("No IV data available, skipping market analysis")
        save_state(state)
        return

    # 5. 🟡 FIX #7: Fetch Polymarket markets (hybrid discovery)
    markets = fetch_btc_barrier_markets()
    if not markets:
        log.warning("No Polymarket markets found")
        save_state(state)
        return

    # 6. Analyze each market
    current_exposure = sum(p.get("cost", 0) for p in state["positions"])
    n_open = len(state["positions"])
    opportunities = []

    for market in markets:
        opp = analyze_market(market, spot, iv_points, state["capital"],
                             current_exposure, n_open)
        if opp:
            opportunities.append(opp)

    # 7. Sort by adjusted edge (best first)
    opportunities.sort(key=lambda o: o["adjusted_edge"], reverse=True)

    log.info(f"Found {len(opportunities)} tradeable opportunities")

    # 7.5 Send opportunity report to Telegram (once per scan if new opps)
    now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    last_opp_report = state.get("last_opp_report", "")

    if opportunities and now_date != last_opp_report:
        send_opportunity_report(opportunities)
        state["last_opp_report"] = now_date

    # 8. Execute top opportunities (avoid overconcentration)
    already_traded_barriers = {p.get("barrier_price", p.get("barrier", 0)) for p in state["positions"]}
    trades_this_scan = 0
    max_trades_per_scan = 2  # More conservative

    for opp in opportunities:
        if trades_this_scan >= max_trades_per_scan:
            break

        # 🔴 FIX #4: Check max positions
        if len(state["positions"]) >= CONFIG["max_positions"]:
            log.info("Max positions reached, no more trades")
            break

        # Don't double up on same barrier
        if opp["barrier_price"] in already_traded_barriers:
            continue

        # Check capacity
        if opp["position_usd"] > state["capital"]:
            continue

        # Execute paper trade
        success = execute_paper_trade(state, opp)
        if success:
            trades_this_scan += 1
            already_traded_barriers.add(opp["barrier_price"])
            current_exposure += opp["position_usd"]

            # Telegram alert
            msg = (f"📈 <b>NEW TRADE</b>: {opp['trade_desc']}\n"
                   f"Edge: {opp['edge']:.1f}% | Win: {opp['win_probability']:.0f}%\n"
                   f"Model: {opp['model_prob']:.1f}% vs PM: {opp['pm_prob']:.1f}%\n"
                   f"Cost: <code>${opp['position_usd']:.2f}</code> | "
                   f"Profit if win: +{opp['profit_if_win_pct']:.0f}%\n"
                   f"Kelly: {opp['kelly']:.1%} | IV: {opp['iv_used']:.1f}%")
            send_telegram(msg)

    # 9. 🟠 FIX #5: Update drawdown after trades
    update_drawdown(state)

    # 10. Report summary
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    state["scans_count"] = state.get("scans_count", 0) + 1

    portfolio_value = compute_portfolio_value(state)
    total_return = (portfolio_value - state["initial_capital"]) / state["initial_capital"] * 100
    win_rate = state["wins"] / max(state["total_trades"], 1) * 100
    unrealized = sum(p.get("unrealized_pnl", 0) for p in state["positions"])

    summary = (
        f"\n{'─'*50}\n"
        f"📊 PORTFOLIO SUMMARY (v2.0)\n"
        f"  Portfolio:   ${portfolio_value:.2f}\n"
        f"  Cash:        ${state['capital']:.2f}\n"
        f"  Exposure:    ${current_exposure:.2f}\n"
        f"  Unrealized:  ${unrealized:+.2f}\n"
        f"  Realized:    ${state['total_pnl']:+.2f}\n"
        f"  Return:      {total_return:+.1f}%\n"
        f"  Win Rate:    {win_rate:.0f}% ({state['wins']}W / {state['losses']}L)\n"
        f"  Max DD:      {state['max_drawdown']:.1f}%\n"
        f"  Current DD:  {state.get('current_drawdown', 0):.1f}%\n"
        f"  Open Pos:    {len(state['positions'])} / {CONFIG['max_positions']}\n"
        f"  Scans:       {state.get('scans_count', 0)}\n"
        f"{'─'*50}"
    )
    log.info(summary)

    # Send live portfolio report to Telegram every 6 scans (~30min at 5min interval)
    if state.get("scans_count", 0) % 6 == 0:
        send_live_portfolio_report(state, spot)

    save_state(state)


def main():
    """Main bot entry point."""
    one_shot = "--once" in sys.argv

    log.info("=" * 60)
    log.info("🤖 Polymarket BTC Autotrader v2.0 (Quant-Grade)")
    log.info(f"  Capital:       ${CONFIG['starting_capital']:.2f}")
    log.info(f"  Mode:          {'PAPER' if CONFIG['dry_run'] else '⚠️  LIVE'}")
    log.info(f"  Scan interval: {CONFIG['scan_interval']}s ({CONFIG['scan_interval']//60}min)")
    log.info(f"  Min edge:      {CONFIG['min_edge_pct']}%")
    log.info(f"  Kelly frac:    {CONFIG['kelly_fraction']} (Quarter-Kelly)")
    log.info(f"  Max pos size:  {CONFIG['max_position_pct']*100:.0f}% of capital")
    log.info(f"  Max exposure:  {CONFIG['max_exposure_pct']*100:.0f}%")
    log.info(f"  Max positions: {CONFIG['max_positions']}")
    log.info(f"  TP/SL:         +{(CONFIG['take_profit_mult']-1)*100:.0f}% / -{(1-CONFIG['stop_loss_mult'])*100:.0f}%")
    log.info(f"  Telegram:      {'✅' if CONFIG['telegram_token'] else '❌ (not set)'}")
    log.info(f"  FIXES:         MTM✅ Exits✅ European✅ Kelly✅ DD✅ Reconcile✅ Discovery✅")
    log.info("=" * 60)

    state = load_state()

    # 🟠 FIX #6: Reconcile expired positions from downtime
    spot = fetch_btc_spot()
    if spot:
        reconcile_on_startup(state, spot)

    if one_shot:
        run_scan(state)
        return

    # Startup notification
    portfolio_value = compute_portfolio_value(state)
    send_telegram(
        f"🤖 <b>Autotrader v2.0 Started!</b>\n"
        f"{'━' * 28}\n"
        f"💰 Portfolio: ${portfolio_value:.2f}\n"
        f"💵 Cash: ${state['capital']:.2f}\n"
        f"📋 Open: {len(state['positions'])} positions\n"
        f"Mode: {'PAPER' if CONFIG['dry_run'] else '⚠️ LIVE'}\n"
        f"\n<b>Quant Fixes Applied:</b>\n"
        f"🔴 MTM via CLOB ✅\n"
        f"🔴 TP/SL Active Exits ✅\n"
        f"🔴 European-only Model ✅\n"
        f"🔴 Quarter-Kelly Sizing ✅\n"
        f"🟠 Portfolio-level DD ✅\n"
        f"🟠 Startup Reconciliation ✅\n"
        f"🟡 Hybrid Market Discovery ✅"
    )

    # Main loop
    while True:
        try:
            run_scan(state)
        except Exception as e:
            log.error(f"Scan error: {e}\n{traceback.format_exc()}")
            send_telegram(f"⚠️ <b>Error:</b> {str(e)[:200]}")

        log.info(f"💤 Next scan in {CONFIG['scan_interval']}s...")
        time.sleep(CONFIG["scan_interval"])


if __name__ == "__main__":
    main()
