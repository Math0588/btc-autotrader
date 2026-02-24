"""
🤖 Polymarket BTC Autotrader — Autonomous Paper Trading Bot
==========================================================
Runs 24/7, scans Polymarket markets, computes model probabilities
using Deribit IV, identifies edge, and executes paper trades.

Designed to be deployed FREE on Render / Railway / Koyeb.

Usage:
    python autotrader.py                  # Run the bot
    python autotrader.py --once           # Single scan + report
    TELEGRAM_TOKEN=xxx TELEGRAM_CHAT=yyy python autotrader.py  # With alerts

Environment Variables:
    STARTING_CAPITAL   — Starting capital in USD (default: 100)
    SCAN_INTERVAL      — Seconds between scans (default: 900 = 15min)
    TELEGRAM_TOKEN     — Telegram bot token for notifications (optional)
    TELEGRAM_CHAT      — Telegram chat ID for notifications (optional)
    MIN_EDGE           — Minimum edge % to trade (default: 3.0)
    KELLY_FRAC         — Kelly fraction (default: 0.20)
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

CONFIG = {
    "starting_capital": float(os.getenv("STARTING_CAPITAL", "100")),
    "scan_interval": int(os.getenv("SCAN_INTERVAL", "900")),  # 15 min
    "min_edge_pct": float(os.getenv("MIN_EDGE", "2.0")),
    "kelly_fraction": float(os.getenv("KELLY_FRAC", "0.20")),  # Conservative 20% Kelly
    "max_position_pct": 0.15,       # Max 15% of capital per trade
    "max_exposure_pct": 0.80,       # Max 80% of capital deployed
    "polymarket_fee_pct": 2.0,
    "risk_free_rate": 0.045,
    "btc_drift_real": 0.10,
    "min_dte": 1,                   # Allow trades up to 24h before expiry (more trades)
    "max_dte": 45,                  # Extend timeframe
    "min_liquidity": 2000,          # Reduced liquidity requirement to catch newer markets
    "min_volume": 500,              # Reduced volume requirement
    "min_win_prob": 0.10,           # Lower win prob threshold to catch 10% "lotto tickets" with high edge
    "max_drawdown_pct": 30,         # Reduce sizing if drawdown exceeds 30%
    "telegram_token": os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat": os.getenv("TELEGRAM_CHAT", ""),
    "dry_run": os.getenv("DRY_RUN", "true").lower() != "false",
}

POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
DERIBIT_API = "https://www.deribit.com/api/v2/public"

STATE_FILE = Path("autotrader_state.json")

# Month name mappings for slug generation
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
                # Parse expiry from creation_timestamp
                creation_ts = s.get("creation_timestamp", 0)
                # Use the instrument name to get expiry date
                exp_str = parts[1]
                from datetime import datetime as dt_class
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
    Fetch all active BTC barrier markets from Polymarket.
    Uses date-based slug discovery: bitcoin-above-on-{month}-{day}
    """
    markets = []
    seen = set()
    now = datetime.now(timezone.utc)

    # Generate candidate slugs for current/upcoming events
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

    markets.sort(key=lambda m: (m["expiry_dt"], m["barrier_price"]))
    log.info(f"Found {len(markets)} BTC barrier markets on Polymarket")
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
    }


# ─── Probability Models ──────────────────────────────────────────────────────

def compute_one_touch_prob(spot, barrier, dte_days, sigma, r=0.045, scenario="above"):
    """
    One-Touch (First Passage Time) probability.
    More accurate than European for "Will BTC HIT $X by date?" bets.

    For barrier ABOVE spot:
        P(touch) = N(-d2) + (B/S)^(2μ/σ²) * N(-d1_adj)
    For barrier BELOW spot:
        P(touch) = N(d2_down) + (B/S)^(2μ/σ²) * N(d1_adj_down)
    """
    if spot <= 0 or barrier <= 0 or dte_days <= 0 or sigma <= 0:
        return None

    T = dte_days / 365.0
    sqrt_t = math.sqrt(T)

    # Drift under risk-neutral measure
    mu = r - 0.5 * sigma**2

    if scenario == "above":
        if barrier <= spot:
            # Already above barrier → high prob (European approx is fine)
            d2 = (math.log(spot / barrier) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt_t)
            return norm.cdf(d2)

        # Barrier above spot → First Passage Time
        log_ratio = math.log(barrier / spot)
        d_plus = (log_ratio - mu * T) / (sigma * sqrt_t)
        d_minus = (log_ratio + mu * T) / (sigma * sqrt_t)

        if sigma > 0:
            exponent = 2 * mu * log_ratio / (sigma**2)
            exponent = max(min(exponent, 50), -50)  # Clamp
            prob = norm.cdf(-d_plus) + math.exp(exponent) * norm.cdf(-d_minus)
        else:
            prob = 0.0

        return min(max(prob, 0), 1)

    else:  # below
        if barrier >= spot:
            d2 = (math.log(spot / barrier) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt_t)
            return norm.cdf(-d2)

        log_ratio = math.log(spot / barrier)
        d_plus = (log_ratio + mu * T) / (sigma * sqrt_t)
        d_minus = (log_ratio - mu * T) / (sigma * sqrt_t)

        if sigma > 0:
            exponent = -2 * mu * log_ratio / (sigma**2)
            exponent = max(min(exponent, 50), -50)
            prob = norm.cdf(-d_plus) + math.exp(exponent) * norm.cdf(-d_minus)
        else:
            prob = 0.0

        return min(max(prob, 0), 1)


def compute_european_prob(spot, barrier, dte_days, sigma, r=0.045, scenario="above"):
    """Standard Black-Scholes P(above/below barrier at expiry)."""
    T = dte_days / 365.0
    d2 = (math.log(spot / barrier) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    if scenario == "above":
        return norm.cdf(d2)
    else:
        return norm.cdf(-d2)


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


# ─── Core Strategy Engine ─────────────────────────────────────────────────────

def analyze_market(market, spot, iv_points, capital, current_exposure):
    """
    Analyze a single market for edge. Returns opportunity dict or None.

    Quant improvements over basic strategy:
    1. One-Touch probability (not European) for "will it HIT X" bets
    2. Liquidity-adjusted edge (discount edge for thin markets)
    3. Drawdown-aware Kelly sizing
    4. Min win probability filter
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

    # Get IV for this strike
    moneyness = barrier / spot
    iv = interpolate_iv(iv_points, moneyness, dte)
    if iv is None or iv <= 0:
        return None
    sigma = iv / 100.0

    # ══════ QUANT MODEL: One-Touch probability ══════
    # Polymarket bets are "will BTC be above $X on date Y?"
    # This is closer to European at-expiry probability
    # But for "hit" type bets, One-Touch is more accurate
    prob_european = compute_european_prob(spot, barrier, dte, sigma,
                                          CONFIG["risk_free_rate"], scenario)
    prob_one_touch = compute_one_touch_prob(spot, barrier, dte, sigma,
                                            CONFIG["risk_free_rate"], scenario)

    # Use blended probability (70% European + 30% One-Touch for "above" type)
    # Polymarket "above on date X" is European-style (at expiry, not first passage)
    model_prob = prob_european  # Primary model
    if prob_one_touch is not None:
        # Use One-Touch as sanity check / adjustment
        model_prob = 0.85 * prob_european + 0.15 * prob_one_touch

    model_prob_pct = model_prob * 100

    # ══════ EDGE CALCULATION ══════
    edge = model_prob_pct - pm_prob  # Positive = YES underpriced, Negative = NO underpriced

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
    liq_factor = min(liq / 20000, 1.0)  # Scale: $20k liq = full edge
    adjusted_edge = abs(edge) * liq_factor

    # ══════ KELLY SIZING with drawdown control ══════
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

    kelly = kelly_full * CONFIG["kelly_fraction"]

    # Drawdown control: reduce sizing during drawdowns
    drawdown_pct = (1 - capital / CONFIG["starting_capital"]) * 100
    if drawdown_pct > CONFIG["max_drawdown_pct"]:
        kelly *= 0.5  # Half size during big drawdowns
    elif drawdown_pct > CONFIG["max_drawdown_pct"] / 2:
        kelly *= 0.75  # 3/4 size during moderate drawdowns

    # Position sizing
    max_pos = capital * CONFIG["max_position_pct"]
    remaining_capacity = capital * CONFIG["max_exposure_pct"] - current_exposure
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
        "barrier_price": opportunity["barrier_price"],
        "entry_price": opportunity["entry_price"],
        "n_contracts": opportunity["n_contracts"],
        "cost": round(cost, 2),
        "win_probability": opportunity["win_probability"],
        "profit_if_win_pct": opportunity["profit_if_win_pct"],
        "expected_pnl": opportunity["expected_pnl"],
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


def settle_expired_positions(state, spot):
    """
    Check open positions and settle expired ones.
    A position settles when its expiry has passed.
    """
    now = datetime.now(timezone.utc)
    still_open = []
    settled_count = 0

    for pos in state["positions"]:
        expiry = datetime.fromisoformat(pos["expiry_dt"])
        if now < expiry:
            still_open.append(pos)
            continue

        # Position has expired — settle it
        barrier = pos["barrier_price"]
        direction = pos["direction"]
        cost = pos["cost"]
        n = pos["n_contracts"]

        # Determine if the bet won
        if direction == "BUY_YES":
            won = spot >= barrier if pos.get("scenario", "above") == "above" else spot < barrier
        else:  # BUY_NO
            won = spot < barrier if pos.get("scenario", "above") == "above" else spot >= barrier

        if won:
            fee = CONFIG["polymarket_fee_pct"] / 100
            payout = n * (1.0 - pos["entry_price"]) * (1 - fee)
            pnl = payout  # We already deducted cost when opening
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

        # Track peak capital and drawdown
        if state["capital"] > state["peak_capital"]:
            state["peak_capital"] = state["capital"]
        current_dd = (1 - state["capital"] / state["peak_capital"]) * 100
        state["max_drawdown"] = max(state["max_drawdown"], current_dd)

        pos["status"] = "closed"
        pos["result"] = result
        pos["pnl"] = round(pnl, 2)
        pos["settled_at"] = now.isoformat()
        pos["spot_at_expiry"] = spot
        state["closed_trades"].append(pos)
        settled_count += 1

        log.info(f"{emoji} SETTLED: {pos['trade_desc']} → {result} | "
                 f"PnL: ${pnl:+.2f} | Capital: ${state['capital']:.2f}")

        # Telegram alert
        msg = (f"{emoji} <b>{result}</b>: {pos['trade_desc']}\n"
               f"PnL: <code>${pnl:+.2f}</code>\n"
               f"Capital: <code>${state['capital']:.2f}</code>")
        send_telegram(msg)

    state["positions"] = still_open
    if settled_count:
        log.info(f"Settled {settled_count} position(s). Capital: ${state['capital']:.2f}")


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

    # 2. Settle expired positions
    settle_expired_positions(state, spot)

    # 3. Fetch IV surface
    iv_points = fetch_deribit_iv_surface()
    if not iv_points:
        log.warning("No IV data available, skipping market analysis")
        return

    # 4. Fetch Polymarket markets
    markets = fetch_btc_barrier_markets()
    if not markets:
        log.warning("No Polymarket markets found")
        return

    # 5. Analyze each market
    current_exposure = sum(p["cost"] for p in state["positions"])
    opportunities = []

    for market in markets:
        opp = analyze_market(market, spot, iv_points, state["capital"], current_exposure)
        if opp:
            opportunities.append(opp)

    # 6. Sort by edge (best first)
    opportunities.sort(key=lambda o: o["adjusted_edge"], reverse=True)

    log.info(f"Found {len(opportunities)} tradeable opportunities")

    # 7. Execute top opportunities (avoid overconcentration)
    already_traded_barriers = {p["barrier_price"] for p in state["positions"]}
    trades_this_scan = 0
    max_trades_per_scan = 3

    for opp in opportunities:
        if trades_this_scan >= max_trades_per_scan:
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
                   f"Cost: <code>${opp['position_usd']:.2f}</code> | "
                   f"Profit if win: +{opp['profit_if_win_pct']:.0f}%")
            send_telegram(msg)

    # 8. Report summary
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    win_rate = state["wins"] / max(state["total_trades"], 1) * 100
    total_return = (state["capital"] + current_exposure - state["initial_capital"]) / state["initial_capital"] * 100

    summary = (
        f"\n{'─'*50}\n"
        f"📊 PORTFOLIO SUMMARY\n"
        f"  Capital:     ${state['capital']:.2f}\n"
        f"  Exposure:    ${current_exposure:.2f}\n"
        f"  Total Value: ${state['capital'] + current_exposure:.2f}\n"
        f"  Total PnL:   ${state['total_pnl']:+.2f}\n"
        f"  Return:      {total_return:+.1f}%\n"
        f"  Win Rate:    {win_rate:.0f}% ({state['wins']}W / {state['losses']}L)\n"
        f"  Max DD:      {state['max_drawdown']:.1f}%\n"
        f"  Open Pos:    {len(state['positions'])}\n"
        f"{'─'*50}"
    )
    log.info(summary)

    save_state(state)


def main():
    """Main bot entry point."""
    one_shot = "--once" in sys.argv

    log.info("=" * 60)
    log.info("🤖 Polymarket BTC Autotrader v1.0")
    log.info(f"  Capital:       ${CONFIG['starting_capital']:.2f}")
    log.info(f"  Mode:          {'PAPER' if CONFIG['dry_run'] else '⚠️  LIVE'}")
    log.info(f"  Scan interval: {CONFIG['scan_interval']}s ({CONFIG['scan_interval']//60}min)")
    log.info(f"  Min edge:      {CONFIG['min_edge_pct']}%")
    log.info(f"  Kelly frac:    {CONFIG['kelly_fraction']}")
    log.info(f"  Telegram:      {'✅' if CONFIG['telegram_token'] else '❌ (not set)'}")
    log.info("=" * 60)

    state = load_state()

    if one_shot:
        run_scan(state)
        return

    # Startup notification
    send_telegram("🤖 <b>Autotrader Started!</b>\n"
                  f"Capital: ${state['capital']:.2f}\n"
                  f"Mode: {'PAPER' if CONFIG['dry_run'] else 'LIVE'}")

    # Main loop
    while True:
        try:
            run_scan(state)
        except Exception as e:
            log.error(f"Scan error: {e}\n{traceback.format_exc()}")
            send_telegram(f"⚠️ <b>Error:</b> {str(e)[:100]}")

        log.info(f"💤 Next scan in {CONFIG['scan_interval']}s...")
        time.sleep(CONFIG["scan_interval"])


if __name__ == "__main__":
    main()
