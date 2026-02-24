"""
Polymarket ↔ Options Arbitrage Strategy
========================================
Fetches Polymarket BTC barrier markets, compares implied probabilities
to our options-derived model, and identifies profitable trades.

Strategy Rules:
    1. Compute model probability P(BTC > K, T) from Deribit IV surface
    2. Compare to Polymarket YES price (= their implied probability)
    3. Edge = Model_Prob - Polymarket_Prob
    4. If Edge > threshold → BUY YES (mispriced cheap)
    5. If Edge < -threshold → BUY NO (mispriced expensive)
    6. Size with fractional Kelly Criterion
    7. Account for Polymarket fees (2% on winnings)
"""

import re
import time
import json
import math
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from scipy.stats import norm
from scipy.interpolate import griddata

# ─── Configuration ───────────────────────────────────────────────────────────

POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API = "https://clob.polymarket.com"

# Strategy parameters
STRATEGY_CONFIG = {
    "min_edge_pct": 3.0,           # Minimum edge to trade (in %)
    "kelly_fraction": 0.25,         # Fractional Kelly (25% = quarter Kelly)
    "max_position_usd": 500,        # Max per-market position
    "max_total_exposure": 5000,     # Max total portfolio exposure
    "polymarket_fee_pct": 2.0,     # Polymarket fee on winnings
    "risk_free_rate": 0.045,        # Annual risk-free rate
    "btc_drift_real": 0.10,         # Real-world BTC drift (annual)
    "min_days_to_expiry": 2,        # Don't trade markets expiring in < 2 days
    "max_days_to_expiry": 365,      # Don't trade markets expiring in > 1 year
    "min_liquidity": 1000,          # Minimum market liquidity ($)
    "refresh_interval": 60,         # Refresh interval (seconds)
}

# ─── Cache ───────────────────────────────────────────────────────────────────
_pm_cache = {}
PM_CACHE_TTL = 30


def _pm_get(url, params=None):
    """GET request to Polymarket API with caching."""
    cache_key = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
    now = time.time()
    if cache_key in _pm_cache and now - _pm_cache[cache_key]["ts"] < PM_CACHE_TTL:
        return _pm_cache[cache_key]["data"]

    headers = {"Accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    _pm_cache[cache_key] = {"data": data, "ts": now}
    return data


# ─── Polymarket Market Discovery ─────────────────────────────────────────────

# Month name mappings for slug generation
_MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}


def _generate_btc_event_slugs():
    """
    Generate event slugs for BTC weekly multi-strike markets.
    Polymarket uses slugs like: bitcoin-above-on-march-3
    These are weekly events, so we try dates within a ~4-week window.
    """
    now = datetime.now(timezone.utc)
    slugs = []

    # Try every day from today to +30 days forward (weekly events land ~weekly)
    for delta in range(-2, 31):
        dt = now + timedelta(days=delta)
        month_name = _MONTH_NAMES[dt.month]
        day = dt.day
        slug = f"bitcoin-above-on-{month_name}-{day}"
        slugs.append(slug)

    return slugs


def fetch_btc_barrier_markets():
    """
    Fetch all active Polymarket BTC barrier markets using the events API.
    
    Polymarket organizes BTC price bets as weekly events with slugs like:
        bitcoin-above-on-march-3
        bitcoin-above-on-february-28
    
    Each event contains multiple sub-markets at different strikes:
        "Will the price of Bitcoin be above $64,000 on March 3?"
        "Will the price of Bitcoin be above $68,000 on March 3?"
    
    Returns a list of parsed market objects.
    """
    markets = []
    seen_condition_ids = set()

    # Generate candidate slugs for current/upcoming weekly events
    candidate_slugs = _generate_btc_event_slugs()

    for slug in candidate_slugs:
        try:
            events = _pm_get(f"{POLYMARKET_GAMMA_API}/events", params={
                "slug": slug,
            })

            if not events or not isinstance(events, list) or len(events) == 0:
                continue

            event = events[0]
            event_title = event.get("title", "")
            sub_markets = event.get("markets", [])

            if not sub_markets:
                continue

            print(f"[Strategy] Found event: {event_title} ({len(sub_markets)} strikes)")

            for sub_market in sub_markets:
                # Skip closed markets
                if sub_market.get("closed", False):
                    continue
                if not sub_market.get("acceptingOrders", True):
                    continue

                parsed = _parse_btc_barrier_market(sub_market)
                if parsed and parsed["condition_id"] not in seen_condition_ids:
                    seen_condition_ids.add(parsed["condition_id"])
                    markets.append(parsed)

        except requests.exceptions.HTTPError:
            # 404 = no event for that date, expected
            continue
        except Exception as e:
            print(f"[Strategy] Error fetching event {slug}: {e}")
            continue

    # Also try fetching the "what price will bitcoin hit" annual markets
    for annual_slug in ["what-price-will-bitcoin-hit-in-2026", "what-price-will-bitcoin-hit-in-2025"]:
        try:
            events = _pm_get(f"{POLYMARKET_GAMMA_API}/events", params={"slug": annual_slug})
            if events and isinstance(events, list) and len(events) > 0:
                event = events[0]
                sub_markets = event.get("markets", [])
                print(f"[Strategy] Found annual event: {event.get('title', '')} ({len(sub_markets)} strikes)")
                for sub_market in sub_markets:
                    if sub_market.get("closed", False):
                        continue
                    parsed = _parse_btc_barrier_market(sub_market)
                    if parsed and parsed["condition_id"] not in seen_condition_ids:
                        seen_condition_ids.add(parsed["condition_id"])
                        markets.append(parsed)
        except Exception:
            pass

    # Sort by expiry then by barrier price
    markets.sort(key=lambda m: (
        m.get("expiry_dt", datetime.max.replace(tzinfo=timezone.utc)),
        m.get("barrier_price", 0),
    ))

    print(f"[Strategy] Total BTC barrier markets found: {len(markets)}")
    return markets


def _parse_btc_barrier_market(market_data):
    """
    Parse a Polymarket market and check if it's a BTC barrier bet.
    Returns parsed market dict or None.

    Matches patterns like:
        "Bitcoin above $100,000 on March 31?"
        "Will Bitcoin be above $75,000 on February 28?"
        "Bitcoin to hit $120,000 by April?"
    """
    title = market_data.get("question", "") or market_data.get("title", "") or ""
    description = market_data.get("description", "") or ""

    # Check if it's a Bitcoin barrier market
    title_lower = title.lower()
    if not any(kw in title_lower for kw in ["bitcoin", "btc"]):
        return None

    if not any(kw in title_lower for kw in ["above", "below", "hit", "reach", "price"]):
        return None

    # Extract barrier price from title
    barrier = _extract_barrier_price(title)
    if barrier is None:
        return None

    # Determine scenario
    scenario = "above"
    if "below" in title_lower or "under" in title_lower:
        scenario = "below"

    # Extract expiry date
    expiry_str = market_data.get("end_date_iso", "") or market_data.get("endDate", "")
    expiry_dt = None
    if expiry_str:
        try:
            # Handle various ISO formats
            expiry_str_clean = expiry_str.replace("Z", "+00:00")
            expiry_dt = datetime.fromisoformat(expiry_str_clean)
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass

    if expiry_dt is None:
        return None

    # Compute DTE
    now = datetime.now(timezone.utc)
    dte_days = (expiry_dt - now).total_seconds() / 86400.0
    if dte_days <= 0:
        return None

    # Get market prices
    yes_price = None
    no_price = None
    volume = 0
    liquidity = 0

    # Try to get prices from various fields
    outcomes = market_data.get("outcomes", [])
    outcome_prices = market_data.get("outcomePrices", "")
    tokens = market_data.get("tokens", [])
    clob_token_ids = market_data.get("clobTokenIds", "")

    # Parse outcome prices
    if outcome_prices:
        try:
            if isinstance(outcome_prices, str):
                prices = json.loads(outcome_prices)
            else:
                prices = outcome_prices
            if len(prices) >= 2:
                yes_price = float(prices[0])
                no_price = float(prices[1])
        except (json.JSONDecodeError, ValueError, IndexError):
            pass

    # Fallback: try tokens
    if yes_price is None and tokens:
        for token in tokens:
            outcome = token.get("outcome", "").lower()
            price = token.get("price", 0)
            if price and outcome in ("yes", "true", "1"):
                yes_price = float(price)
            elif price and outcome in ("no", "false", "0"):
                no_price = float(price)

    # Fallback: best bid/ask
    if yes_price is None:
        best_bid = market_data.get("bestBid")
        best_ask = market_data.get("bestAsk")
        if best_bid and best_ask:
            yes_price = (float(best_bid) + float(best_ask)) / 2

    if yes_price is None:
        return None

    if no_price is None:
        no_price = 1.0 - yes_price

    # Get volume and liquidity
    volume = float(market_data.get("volume", 0) or market_data.get("volumeNum", 0) or 0)
    liquidity = float(market_data.get("liquidity", 0) or market_data.get("liquidityNum", 0) or 0)

    # Token IDs for trading
    token_ids = []
    if clob_token_ids:
        try:
            if isinstance(clob_token_ids, str):
                token_ids = json.loads(clob_token_ids)
            else:
                token_ids = clob_token_ids
        except (json.JSONDecodeError, ValueError):
            pass

    condition_id = market_data.get("conditionId", "") or market_data.get("condition_id", "")

    return {
        "title": title,
        "condition_id": condition_id,
        "market_slug": market_data.get("slug", "") or market_data.get("market_slug", ""),
        "barrier_price": barrier,
        "scenario": scenario,
        "yes_price": round(yes_price, 4),
        "no_price": round(no_price, 4),
        "pm_implied_prob": round(yes_price * 100, 2),  # YES price = implied probability
        "expiry_str": expiry_str,
        "expiry_dt": expiry_dt,
        "dte_days": round(dte_days, 2),
        "volume": volume,
        "liquidity": liquidity,
        "token_ids": token_ids,
        "description": description[:200],
        "url": f"https://polymarket.com/event/{market_data.get('slug', '')}",
    }


def _extract_barrier_price(title):
    """Extract the USD barrier price from a market title."""
    # Match patterns like: $100,000  $100000  $100K  $100k
    patterns = [
        r'\$([0-9]{1,3}(?:,[0-9]{3})+)',          # $100,000
        r'\$([0-9]+(?:\.[0-9]+)?)\s*[kK]',        # $100K or $100k
        r'\$([0-9]+(?:,?[0-9]+)*)',                # $100000
    ]

    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            price_str = match.group(1).replace(",", "")
            try:
                price = float(price_str)
                # Handle K suffix
                if 'k' in title[match.end()-1:match.end()+1].lower():
                    price *= 1000
                # Sanity check: BTC price should be between $1k and $10M
                if 1000 <= price <= 10_000_000:
                    return price
            except ValueError:
                continue

    return None


# ─── Model Probability Computation ──────────────────────────────────────────

def compute_model_probability(spot, barrier_price, dte_days, iv_points, scenario="above"):
    """
    Compute the Black-Scholes probability using IV interpolated from
    the live options surface.

    Returns: dict with probability and metadata
    """
    if spot <= 0 or barrier_price <= 0 or dte_days <= 0:
        return None

    moneyness = barrier_price / spot
    iv, method, nearby = _interpolate_iv_from_points(iv_points, moneyness, dte_days)

    if iv is None or iv <= 0:
        return None

    # BS probability
    r = STRATEGY_CONFIG["risk_free_rate"]
    sigma = iv / 100.0
    T = dte_days / 365.0

    d1 = (math.log(spot / barrier_price) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    prob_above_rn = norm.cdf(d2)
    prob_below_rn = 1.0 - prob_above_rn

    # Real-world probability
    mu = STRATEGY_CONFIG["btc_drift_real"]
    d2_real = (math.log(spot / barrier_price) + (mu - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    prob_above_rw = norm.cdf(d2_real)
    prob_below_rw = 1.0 - prob_above_rw

    if scenario == "above":
        model_prob_rn = prob_above_rn
        model_prob_rw = prob_above_rw
    else:
        model_prob_rn = prob_below_rn
        model_prob_rw = prob_below_rw

    return {
        "model_prob_rn": round(model_prob_rn * 100, 2),
        "model_prob_rw": round(model_prob_rw * 100, 2),
        "iv_used": round(iv, 2),
        "iv_method": method,
        "sigma": round(sigma, 4),
        "d1": round(d1, 4),
        "d2": round(d2, 4),
        "T_years": round(T, 4),
        "moneyness": round(moneyness, 4),
        "nearby_options": nearby[:4],
    }


def _interpolate_iv_from_points(points, target_moneyness, target_dte):
    """Interpolate IV from raw options data points."""
    if not points or len(points) < 3:
        return None, "no_data", []

    call_points = [p for p in points if p.get("type") == "C"]
    if len(call_points) < 5:
        call_points = points

    moneyness_arr = np.array([p["moneyness"] for p in call_points])
    dte_arr = np.array([p["dte"] for p in call_points])
    iv_arr = np.array([p["iv"] for p in call_points])

    mask = (moneyness_arr > 0.1) & (moneyness_arr < 5.0) & (dte_arr > 0)
    moneyness_arr = moneyness_arr[mask]
    dte_arr = dte_arr[mask]
    iv_arr = iv_arr[mask]

    if len(moneyness_arr) < 3:
        return None, "insufficient_data", []

    # Find nearby
    distances = np.sqrt(
        ((moneyness_arr - target_moneyness) * 5) ** 2 +
        ((dte_arr - target_dte) / 30) ** 2
    )
    nearest_idx = np.argsort(distances)[:6]
    nearby = []
    for idx in nearest_idx:
        if int(idx) < len(call_points):
            cp = call_points[int(idx)]
            nearby.append({
                "strike": cp.get("strike", 0),
                "dte": round(cp.get("dte", 0), 1),
                "iv": round(cp.get("iv", 0), 2),
                "expiry": cp.get("expiry", ""),
                "moneyness": round(cp.get("moneyness", 0), 4),
                "distance": round(float(distances[idx]), 4),
            })

    # Try cubic
    try:
        iv = griddata((moneyness_arr, dte_arr), iv_arr,
                       (target_moneyness, target_dte), method="cubic")
        if not np.isnan(iv) and iv > 0:
            return float(iv), "cubic", nearby
    except Exception:
        pass

    # Try linear
    try:
        iv = griddata((moneyness_arr, dte_arr), iv_arr,
                       (target_moneyness, target_dte), method="linear")
        if not np.isnan(iv) and iv > 0:
            return float(iv), "linear", nearby
    except Exception:
        pass

    # Nearest
    try:
        iv = griddata((moneyness_arr, dte_arr), iv_arr,
                       (target_moneyness, target_dte), method="nearest")
        if not np.isnan(iv) and iv > 0:
            return float(iv), "nearest", nearby
    except Exception:
        pass

    # Weighted average
    if nearby:
        weights = [1 / (n["distance"] + 0.001) for n in nearby]
        total_w = sum(weights)
        weighted_iv = sum(n["iv"] * w for n, w in zip(nearby, weights)) / total_w
        return weighted_iv, "weighted", nearby

    return None, "failed", nearby


# ─── Strategy Engine ─────────────────────────────────────────────────────────

def analyze_opportunities(spot, iv_points):
    """
    Main strategy function: find all Polymarket BTC markets,
    compute model probabilities, calculate edges, and rank opportunities.
    """
    # Fetch Polymarket markets
    pm_markets = fetch_btc_barrier_markets()

    opportunities = []
    config = STRATEGY_CONFIG

    for market in pm_markets:
        barrier = market["barrier_price"]
        dte = market["dte_days"]
        scenario = market["scenario"]
        pm_prob = market["pm_implied_prob"]  # In %

        # Filter
        if dte < config["min_days_to_expiry"]:
            continue
        if dte > config["max_days_to_expiry"]:
            continue

        # Compute model probability
        model = compute_model_probability(spot, barrier, dte, iv_points, scenario)
        if model is None:
            continue

        model_prob_rn = model["model_prob_rn"]  # In %
        model_prob_rw = model["model_prob_rw"]  # In %

        # Use risk-neutral as primary (more conservative)
        # But also consider real-world for context
        edge_rn = model_prob_rn - pm_prob         # Positive = PM underpriced (BUY YES)
        edge_rw = model_prob_rw - pm_prob

        # Determine trade direction
        if edge_rn > config["min_edge_pct"]:
            direction = "BUY_YES"
            edge = edge_rn
            entry_price = market["yes_price"]
        elif edge_rn < -config["min_edge_pct"]:
            direction = "BUY_NO"
            edge = abs(edge_rn)
            entry_price = market["no_price"]
        else:
            direction = "NO_TRADE"
            edge = abs(edge_rn)
            entry_price = 0

        # Kelly Criterion sizing
        # For binary outcomes: f* = (bp - q) / b
        # where b = payout odds, p = true probability, q = 1 - p
        kelly_size = 0
        expected_pnl = 0
        fee_adjusted_edge = 0
        win_probability = 0
        profit_if_win_pct = 0
        loss_if_lose_pct = 0
        trade_description = ""
        trade_cost = 0
        trade_profit = 0

        if direction != "NO_TRADE":
            # Account for Polymarket fees
            fee_rate = config["polymarket_fee_pct"] / 100
            if direction == "BUY_YES":
                true_prob = model_prob_rn / 100
                payout = 1.0 - entry_price  # profit if win
                payout_after_fee = payout * (1 - fee_rate)
                cost = entry_price
                trade_description = f"BTC reste AU-DESSUS ${barrier:,.0f}"
            else:
                true_prob = (100 - model_prob_rn) / 100
                payout = 1.0 - market["no_price"]
                payout_after_fee = payout * (1 - fee_rate)
                cost = market["no_price"]
                trade_description = f"BTC reste SOUS ${barrier:,.0f}"

            # Win probability and return metrics
            win_probability = round(true_prob * 100, 1)
            if cost > 0:
                profit_if_win_pct = round((payout_after_fee / cost) * 100, 1)
                loss_if_lose_pct = 100.0  # Binary: you lose your stake
            trade_cost = cost
            trade_profit = payout_after_fee

            # Binary Kelly: f* = (p * b - q) / b
            # where b = net payout/cost ratio, p = true prob, q = 1-p
            if cost > 0 and payout_after_fee > 0:
                b = payout_after_fee / cost  # odds ratio
                p = true_prob
                q = 1.0 - p
                kelly_full = (p * b - q) / b if b > 0 else 0
                kelly_size = max(0, kelly_full * config["kelly_fraction"])

                # Fee-adjusted edge
                fee_adjusted_edge = edge - (config["polymarket_fee_pct"])
                if fee_adjusted_edge < 0:
                    direction = "NO_TRADE"
                    kelly_size = 0

                # Expected P&L per dollar risked
                expected_pnl = p * payout_after_fee - q * cost

        # Position size
        position_usd = min(
            kelly_size * config["max_total_exposure"],
            config["max_position_usd"]
        ) if direction != "NO_TRADE" else 0

        # Number of contracts
        n_contracts = int(position_usd / entry_price) if entry_price > 0 else 0

        # Annualized edge
        annualized_edge = edge * (365 / max(dte, 1)) if edge > 0 else 0

        # Grade: A = great, B = good, C = marginal
        grade = "C"
        if direction != "NO_TRADE":
            score = win_probability * profit_if_win_pct / 100
            if score > 40 and win_probability > 55:
                grade = "A"
            elif score > 20:
                grade = "B"

        opportunity = {
            # Market info
            "title": market["title"],
            "barrier_price": barrier,
            "scenario": scenario,
            "dte_days": dte,
            "expiry_str": market.get("expiry_str", ""),
            "url": market["url"],
            "slug": market.get("market_slug", ""),
            "condition_id": market.get("condition_id", ""),
            "volume": market["volume"],
            "liquidity": market["liquidity"],

            # Prices
            "pm_yes_price": market["yes_price"],
            "pm_no_price": market["no_price"],
            "pm_implied_prob": pm_prob,

            # Model
            "model_prob_rn": model_prob_rn,
            "model_prob_rw": model_prob_rw,
            "iv_used": model["iv_used"],
            "iv_method": model["iv_method"],
            "moneyness": model["moneyness"],
            "nearby_options": model["nearby_options"],

            # Strategy
            "direction": direction,
            "edge_rn": round(edge_rn, 2),
            "edge_rw": round(edge_rw, 2),
            "edge_abs": round(edge, 2),
            "fee_adjusted_edge": round(fee_adjusted_edge, 2),
            "annualized_edge": round(annualized_edge, 1),
            "kelly_fraction": round(kelly_size, 4),
            "position_usd": round(position_usd, 2),
            "n_contracts": n_contracts,
            "entry_price": round(entry_price, 4),
            "expected_pnl_per_dollar": round(expected_pnl, 4) if expected_pnl else 0,

            # Human-readable trade metrics
            "trade_description": trade_description,
            "win_probability": win_probability,
            "profit_if_win_pct": profit_if_win_pct,
            "loss_if_lose_pct": loss_if_lose_pct,
            "trade_cost": round(trade_cost, 4),
            "trade_profit": round(trade_profit, 4),
            "grade": grade,

            # Metadata
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        opportunities.append(opportunity)

    # Sort by absolute edge (best first)
    opportunities.sort(key=lambda x: x["edge_abs"], reverse=True)

    # Compute portfolio summary
    trades = [o for o in opportunities if o["direction"] != "NO_TRADE"]
    total_exposure = sum(t["position_usd"] for t in trades)
    avg_edge = np.mean([t["edge_abs"] for t in trades]) if trades else 0
    total_expected_pnl = sum(
        t["expected_pnl_per_dollar"] * t["position_usd"]
        for t in trades
    )

    # Weighted win rate across portfolio
    weighted_win_rate = 0
    if trades:
        total_pos = sum(t["position_usd"] for t in trades)
        if total_pos > 0:
            weighted_win_rate = sum(
                t["win_probability"] * t["position_usd"] for t in trades
            ) / total_pos

    # Best trade
    best_trade = max(trades, key=lambda t: t["expected_pnl_per_dollar"]) if trades else None
    grade_a_trades = [t for t in trades if t["grade"] == "A"]

    summary = {
        "total_markets_scanned": len(opportunities),
        "tradeable_opportunities": len(trades),
        "grade_a_count": len(grade_a_trades),
        "total_exposure_usd": round(total_exposure, 2),
        "avg_edge_pct": round(avg_edge, 2),
        "total_expected_pnl": round(total_expected_pnl, 2),
        "weighted_win_rate": round(weighted_win_rate, 1),
        "best_trade": best_trade["trade_description"] if best_trade else "—",
        "best_trade_return": round(best_trade["expected_pnl_per_dollar"] * 100, 1) if best_trade else 0,
        "spot_price": spot,
        "config": config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "summary": summary,
        "opportunities": opportunities,
        "trades": trades,
    }
