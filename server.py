"""
🤖 Polymarket Autotrader — Web Server + Bot Combined
====================================================
Flask web server with live PnL dashboard + autonomous trading bot.
Designed for FREE deployment on Render.com.

The web server keeps the Render service alive while the bot
scans markets in a background thread.
"""

import os
import sys
import json
import time
import math
import hashlib
import logging
import threading
import traceback
import io
import base64
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from scipy.stats import norm
from flask import Flask, jsonify, render_template_string

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
    ],
)
log = logging.getLogger("autotrader")

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "starting_capital": float(os.getenv("STARTING_CAPITAL", "100")),
    "scan_interval": int(os.getenv("SCAN_INTERVAL", "300")),
    "min_edge_pct": float(os.getenv("MIN_EDGE", "2.0")),      # 2% edge min to enter
    "exit_edge_pct": 0.5,                                     # RECORRELATION: Exit when edge drops below 0.5% (fair value reached)
    "kelly_fraction": 0.50,                                   # Half-Kelly (Optimum for rapid growth without guaranteed ruin)
    "max_position_pct": 0.50,                                 # Max 50% capital per trade (prevent 1 black swan wiping 100%)
    "max_exposure_pct": 0.95,                                 # Keep 5% cash minimum
    "min_profit_pct": 10.0,                                   # Minimum raw return
    "assumed_slippage": 0.015,                                # Assume 1.5 cents slip on polymarket (execution cost)
    "polymarket_fee_pct": 0.0,                                # Polymarket has no fees, only spread which we account for via slippage
    "risk_free_rate": 0.045,
    "min_dte": 0.5,                                           # Allow closer to expiry
    "max_dte": 60,
    "min_liquidity": 2000,
    "min_volume": 500,
    "min_win_prob": 0.25,                                     # Cut absolute lotteries, but allow 25% if heavily mispriced
    "stop_loss_prob": 0.10,                                   # THESIS INVALIDATION: If our model says < 10% chance, cut loss and run!
    "max_drawdown_pct": 50,
    "keepalive_interval": int(os.getenv("KEEPALIVE_INTERVAL", "600")),  # 10 min self-ping
    "heartbeat_interval": int(os.getenv("HEARTBEAT_INTERVAL", "7200")),  # 2h Telegram heartbeat
    "telegram_token": os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat": os.getenv("TELEGRAM_CHAT", ""),
}

POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
DERIBIT_API = "https://www.deribit.com/api/v2/public"
STATE_FILE = Path(os.getenv("STATE_FILE", "autotrader_state.json"))
MONTH_NAMES = {1:"january",2:"february",3:"march",4:"april",5:"may",6:"june",
               7:"july",8:"august",9:"september",10:"october",11:"november",12:"december"}

# ─── Global State ─────────────────────────────────────────────────────────────

BOT_STATE = None
PNL_HISTORY = []  # [(timestamp_iso, capital, total_value, pnl)]


def default_state():
    return {
        "capital": CONFIG["starting_capital"],
        "initial_capital": CONFIG["starting_capital"],
        "positions": [],
        "closed_trades": [],
        "total_pnl": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "early_tp": 0,  # Count of Take Profits
        "early_sl": 0,  # Count of Stop Losses
        "peak_capital": CONFIG["starting_capital"],
        "max_drawdown": 0.0,
        "last_scan": None,
        "scans_count": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_state():
    d = default_state()
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            for k, v in d.items():
                if k not in s:
                    s[k] = v
            return s
        except Exception:
            pass
    return d


def save_state(state):
    state["last_saved"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


# ─── Telegram ─────────────────────────────────────────────────────────────────

def tg_send(message):
    t, c = CONFIG["telegram_token"], CONFIG["telegram_chat"]
    if not t or not c:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{t}/sendMessage",
                      json={"chat_id": c, "text": message, "parse_mode": "HTML"},
                      timeout=10)
    except Exception:
        pass


def tg_send_chart():
    """Send PnL chart image to Telegram."""
    t, c = CONFIG["telegram_token"], CONFIG["telegram_chat"]
    if not t or not c or len(PNL_HISTORY) < 2:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        times = [datetime.fromisoformat(p[0]) for p in PNL_HISTORY]
        values = [p[2] for p in PNL_HISTORY]  # total_value
        pnls = [p[3] for p in PNL_HISTORY]    # pnl

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='#0f172a')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8')
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Portfolio Value
        ax1.plot(times, values, color='#10b981', linewidth=2)
        ax1.fill_between(times, CONFIG["starting_capital"], values,
                         where=[v >= CONFIG["starting_capital"] for v in values],
                         alpha=0.2, color='#10b981')
        ax1.fill_between(times, CONFIG["starting_capital"], values,
                         where=[v < CONFIG["starting_capital"] for v in values],
                         alpha=0.2, color='#ef4444')
        ax1.axhline(y=CONFIG["starting_capital"], color='#f59e0b', linestyle='--',
                    alpha=0.5, linewidth=1)
        ax1.set_ylabel('Portfolio Value ($)', color='#e2e8f0', fontsize=10)
        ax1.set_title('🤖 Polymarket Autotrader — Live PnL', color='#f1f5f9',
                       fontsize=13, fontweight='bold', pad=10)

        # Cumulative PnL
        ax2.plot(times, pnls, color='#3b82f6', linewidth=2)
        ax2.fill_between(times, 0, pnls, where=[p >= 0 for p in pnls], 
                         alpha=0.3, color='#10b981')
        ax2.fill_between(times, 0, pnls, where=[p < 0 for p in pnls], 
                         alpha=0.3, color='#ef4444')
        ax2.axhline(y=0, color='#64748b', linewidth=0.5)
        ax2.set_ylabel('Cumulative PnL ($)', color='#e2e8f0', fontsize=10)
        ax2.set_xlabel('Time', color='#e2e8f0', fontsize=10)

        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='#0f172a', edgecolor='none')
        buf.seek(0)
        plt.close()

        requests.post(
            f"https://api.telegram.org/bot{t}/sendPhoto",
            data={"chat_id": c, "caption": "📊 PnL Update"},
            files={"photo": ("pnl.png", buf, "image/png")},
            timeout=15,
        )
    except Exception as e:
        log.warning(f"Chart send error: {e}")


# ─── Deribit Data ─────────────────────────────────────────────────────────────

def fetch_spot():
    try:
        r = requests.get(f"{DERIBIT_API}/get_index_price",
                         params={"index_name": "btc_usd"}, timeout=10)
        return r.json()["result"]["index_price"]
    except Exception:
        return None


def fetch_iv_surface():
    try:
        spot = fetch_spot()
        if not spot:
            return [], None
        r = requests.get(f"{DERIBIT_API}/get_book_summary_by_currency",
                         params={"currency": "BTC", "kind": "option"}, timeout=20)
        summaries = r.json().get("result", [])
        pts = []
        now = datetime.now(timezone.utc)
        for s in summaries:
            iv = s.get("mark_iv", 0)
            if not iv or iv <= 0:
                continue
            parts = s.get("instrument_name", "").split("-")
            if len(parts) < 4:
                continue
            try:
                strike = float(parts[2])
                opt_type = parts[3]
                exp_dt = datetime.strptime(parts[1], "%d%b%y").replace(
                    tzinfo=timezone.utc, hour=8)
                dte = (exp_dt - now).total_seconds() / 86400
                if dte <= 0 or dte > 400:
                    continue
                pts.append({"moneyness": strike/spot, "dte": dte, "iv": iv,
                            "strike": strike, "type": opt_type})
            except Exception:
                continue
        log.info(f"IV surface: {len(pts)} points, BTC=${spot:,.0f}")
        return pts, spot
    except Exception as e:
        log.error(f"IV fetch error: {e}")
        return [], None


# ─── Polymarket Discovery ────────────────────────────────────────────────────

def pm_get(url, params=None):
    r = requests.get(url, params=params,
                     headers={"Accept": "application/json"}, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_markets():
    import re
    markets = []
    seen = set()
    now = datetime.now(timezone.utc)

    for delta in range(-1, 21):
        dt = now + timedelta(days=delta)
        slug = f"bitcoin-above-on-{MONTH_NAMES[dt.month]}-{dt.day}"
        try:
            events = pm_get(f"{POLYMARKET_GAMMA_API}/events", params={"slug": slug})
            if not events or not isinstance(events, list) or len(events) == 0:
                continue
            for sm in events[0].get("markets", []):
                if sm.get("closed", False):
                    continue
                p = parse_market(sm)
                if p and p["cid"] not in seen:
                    seen.add(p["cid"])
                    markets.append(p)
        except requests.exceptions.HTTPError:
            continue
        except Exception:
            continue

    markets.sort(key=lambda m: (m["expiry_dt"], m["barrier"]))
    log.info(f"Found {len(markets)} BTC barrier markets")
    return markets


def parse_market(data, token_type="BTC"):
    import re
    title = data.get("question", "") or data.get("title", "") or ""
    tl = title.lower()
    if not any(k in tl for k in ["bitcoin","btc"]):
        return None
    if not any(k in tl for k in ["above","below","hit","reach","price"]):
        return None

    barrier = None
    for pat in [r'\$([0-9]{1,3}(?:,[0-9]{3})+)', r'\$([0-9]+(?:\.[0-9]+)?)\s*[kK]',
                r'\$([0-9]+(?:,?[0-9]+)*)']:
        m = re.search(pat, title)
        if m:
            val = float(m.group(1).replace(",", ""))
            if val < 1000:
                val *= 1000
            if val > 100:
                barrier = val
                break
    if not barrier:
        return None

    scenario = "below" if ("below" in tl or "under" in tl) else "above"

    expiry_str = data.get("end_date_iso", "") or data.get("endDate", "")
    try:
        expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

    dte = (expiry_dt - datetime.now(timezone.utc)).total_seconds() / 86400
    if dte <= 0:
        return None

    bb, ba = data.get("bestBid"), data.get("bestAsk")
    if bb is None or ba is None:
        return None

    yes_bid = float(bb)
    yes_ask = float(ba)
    
    # If spread is absurd, skip
    if yes_ask - yes_bid > 0.15:
        return None

    # Taker entries
    entry_yes = yes_ask + CONFIG["assumed_slippage"]
    entry_no = (1 - yes_bid) + CONFIG["assumed_slippage"] # To buy NO, we pay 1 - yes_bid

    pm_prob = yes_ask * 100 # Implied probability if we want to buy YES

    return {
        "title": title, "cid": data.get("conditionId", ""),
        "barrier": barrier, "scenario": scenario, "token": token_type,
        "yes_ask": round(yes_ask, 4), "yes_bid": round(yes_bid, 4),
        "entry_yes": round(entry_yes, 4), "entry_no": round(entry_no, 4),
        "pm_prob": round(pm_prob, 2), "expiry_dt": expiry_dt,
        "dte": round(dte, 2),
        "vol": float(data.get("volume",0) or data.get("volumeNum",0) or 0),
        "liq": float(data.get("liquidity",0) or data.get("liquidityNum",0) or 0),
        "slug": data.get("slug", ""),
    }


# ─── Probability Models ──────────────────────────────────────────────────────


def one_touch_prob(spot, barrier, dte_days, sigma, r=0.045, scenario="above"):
    """
    One-Touch (First Passage Time) probability.
    Correct model for "Will BTC HIT $X by date?" barrier-style bets.

    For barrier ABOVE spot:
        P(touch) = N(-d2) + (B/S)^(2μ/σ²) * N(-d1_adj)
    For barrier BELOW spot:
        P(touch) = N(d2_down) + (B/S)^(2μ/σ²) * N(d1_adj_down)
    """
    T = max(dte_days / 365.0, 1e-6)
    mu = r - 0.5 * sigma ** 2
    sigma_sqrt_T = sigma * math.sqrt(T)

    if sigma_sqrt_T < 1e-10:
        if scenario == "above":
            return 1.0 if spot >= barrier else 0.0
        else:
            return 1.0 if spot <= barrier else 0.0

    log_ratio = math.log(barrier / spot)

    if scenario == "above":
        if spot >= barrier:
            return 1.0  # Already touched
        d1 = (log_ratio - mu * T) / sigma_sqrt_T
        d2 = (log_ratio + mu * T) / sigma_sqrt_T
        power = (2 * mu) / (sigma ** 2)
        try:
            ratio_term = (barrier / spot) ** power
        except (OverflowError, ZeroDivisionError):
            ratio_term = 0.0
        prob = norm.cdf(-d2) + ratio_term * norm.cdf(-d1)
    else:
        if spot <= barrier:
            return 1.0  # Already touched
        d1 = (-log_ratio + mu * T) / sigma_sqrt_T
        d2 = (-log_ratio - mu * T) / sigma_sqrt_T
        power = (2 * mu) / (sigma ** 2)
        try:
            ratio_term = (barrier / spot) ** power
        except (OverflowError, ZeroDivisionError):
            ratio_term = 0.0
        prob = norm.cdf(-d1) + ratio_term * norm.cdf(-d2)

    return max(0.0, min(1.0, prob))


def bs_prob(spot, barrier, dte, sigma, r=0.045, scenario="above"):
    """Standard Black-Scholes European probability (fallback)."""
    T = dte / 365.0
    if T < 1e-6 or sigma < 1e-6:
        return (1.0 if spot >= barrier else 0.0) if scenario == "above" else (1.0 if spot <= barrier else 0.0)
    d2 = (math.log(spot/barrier) + (r - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return norm.cdf(d2) if scenario == "above" else norm.cdf(-d2)


def _fit_svi_slice(moneyness_arr, iv_arr):
    """
    Fit SVI (Stochastic Volatility Inspired) parameterisation to a vol slice.
    SVI model: w(k) = a + b * (ρ*(k-m) + sqrt((k-m)² + σ²))
    where k = log(moneyness), w = total_variance = iv² * T
    Returns SVI params (a, b, rho, m, s) or None if fit fails.
    """
    from scipy.optimize import minimize

    if len(moneyness_arr) < 4:
        return None

    k = np.log(moneyness_arr)
    w = (iv_arr / 100.0) ** 2  # total variance proxy

    def svi(params, k_vals):
        a, b, rho, m_p, s = params
        return a + b * (rho * (k_vals - m_p) + np.sqrt((k_vals - m_p)**2 + s**2))

    def objective(params):
        pred = svi(params, k)
        return np.sum((pred - w) ** 2)

    # Initial guess
    a0 = np.mean(w)
    b0 = 0.1
    rho0 = -0.3
    m0 = np.mean(k)
    s0 = 0.1

    bounds = [(0, None), (0, None), (-0.99, 0.99), (None, None), (0.001, None)]

    try:
        res = minimize(objective, [a0, b0, rho0, m0, s0],
                       bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
        if res.success and res.fun / len(k) < 0.01:  # Good fit
            return res.x
    except Exception:
        pass
    return None


def interp_iv(pts, target_m, target_dte):
    """
    Interpolate IV from Deribit surface data using a multi-method approach:
    1. SVI parameterisation per expiry slice (industry standard)
    2. RBF (Radial Basis Function) interpolation as fallback
    3. Distance-weighted nearest-neighbor as last resort

    Returns IV in percentage (e.g. 55.0 for 55%).
    """
    from scipy.interpolate import RBFInterpolator

    if not pts or len(pts) < 3:
        return None

    # Prefer calls for above-spot, puts for below-spot
    if target_m >= 1.0:
        filtered = [p for p in pts if p.get("type") == "C"]
    else:
        filtered = [p for p in pts if p.get("type") == "P"]
    if len(filtered) < 5:
        filtered = pts  # Use all if not enough of one type

    m = np.array([p["moneyness"] for p in filtered])
    d = np.array([p["dte"] for p in filtered])
    iv = np.array([p["iv"] for p in filtered])

    # Filter outliers
    mask = (m > 0.3) & (m < 3.0) & (d > 0) & (iv > 5) & (iv < 300)
    m, d, iv = m[mask], d[mask], iv[mask]
    if len(m) < 3:
        return None

    # ── Method 1: SVI per-slice interpolation ──
    # Group by similar DTE (within 2 days)
    unique_dtes = sorted(set(d))
    slices = {}  # dte_bucket -> (moneyness_arr, iv_arr, actual_dte)
    for dte_val in unique_dtes:
        slice_mask = np.abs(d - dte_val) < 2.0
        if np.sum(slice_mask) >= 4:
            bucket_dte = round(dte_val)
            if bucket_dte not in slices or np.sum(slice_mask) > len(slices[bucket_dte][0]):
                slices[bucket_dte] = (m[slice_mask], iv[slice_mask], dte_val)

    if len(slices) >= 2:
        # Fit SVI to each slice, then interpolate across expiries
        svi_results = []  # (slice_dte, predicted_iv)
        for bucket_dte, (s_m, s_iv, actual_dte) in sorted(slices.items()):
            params = _fit_svi_slice(s_m, s_iv)
            if params is not None:
                a, b, rho, m_p, s = params
                k = math.log(target_m)
                w_pred = a + b * (rho * (k - m_p) + math.sqrt((k - m_p)**2 + s**2))
                if w_pred > 0:
                    iv_pred = math.sqrt(w_pred) * 100  # Convert back to IV %
                    if 5 < iv_pred < 300:
                        svi_results.append((actual_dte, iv_pred))

        if len(svi_results) >= 2:
            # Linear interpolation across expiry slices
            svi_dtes = np.array([r[0] for r in svi_results])
            svi_ivs = np.array([r[1] for r in svi_results])
            if target_dte <= svi_dtes[0]:
                return float(svi_ivs[0])  # Extrapolate flat
            if target_dte >= svi_dtes[-1]:
                return float(svi_ivs[-1])  # Extrapolate flat
            result = float(np.interp(target_dte, svi_dtes, svi_ivs))
            if 5 < result < 300:
                log.debug(f"IV via SVI: {result:.1f}% for m={target_m:.3f}, dte={target_dte:.1f}")
                return result
        elif len(svi_results) == 1:
            return float(svi_results[0][1])

    # ── Method 2: RBF interpolation ──
    try:
        # Normalize dimensions (moneyness and DTE have very different scales)
        m_norm = (m - np.mean(m)) / (np.std(m) + 1e-6)
        d_norm = (d - np.mean(d)) / (np.std(d) + 1e-6)
        coords = np.column_stack([m_norm, d_norm])

        target_m_norm = (target_m - np.mean(m)) / (np.std(m) + 1e-6)
        target_d_norm = (target_dte - np.mean(d)) / (np.std(d) + 1e-6)
        target_pt = np.array([[target_m_norm, target_d_norm]])

        rbf = RBFInterpolator(coords, iv, kernel='thin_plate_spline', smoothing=1.0)
        v = float(rbf(target_pt)[0])
        if 5 < v < 300:
            log.debug(f"IV via RBF: {v:.1f}% for m={target_m:.3f}, dte={target_dte:.1f}")
            return v
    except Exception:
        pass

    # ── Method 3: Distance-weighted nearest neighbors (last resort) ──
    # Weight moneyness distance more heavily (vol smile effect is strong)
    dists = np.sqrt(((m - target_m) * 8)**2 + ((d - target_dte) / 20)**2)
    nn = np.argsort(dists)[:7]
    w = 1.0 / (dists[nn] + 0.001)
    result = float(np.average(iv[nn], weights=w))
    if 5 < result < 300:
        log.debug(f"IV via KNN: {result:.1f}% for m={target_m:.3f}, dte={target_dte:.1f}")
        return result

    return None


# ─── Strategy Engine ─────────────────────────────────────────────────────────

def analyze(market, spot, iv_pts, capital, exposure):
    barrier = market["barrier"]
    dte = market["dte"]
    pm_prob = market["pm_prob"]

    if dte < CONFIG["min_dte"] or dte > CONFIG["max_dte"]:
        return None
    if market["liq"] < CONFIG["min_liquidity"]:
        return None

    moneyness = barrier / spot
    iv = interp_iv(iv_pts, moneyness, dte)
    if not iv or iv <= 0:
        return None
    sigma = iv / 100.0

    # European probability: Polymarket resolves "above X on date Y" (price AT expiry)
    # NOT "will it ever hit X" (which would be One-Touch)
    prob = bs_prob(spot, barrier, dte, sigma, CONFIG["risk_free_rate"], market["scenario"])
    model_pct = prob * 100
    
    # Evaluate edge based on what we would actually pay (Taker)
    entry_yes_price = market["entry_yes"]
    entry_no_price = market["entry_no"]
    
    edge_yes = prob - entry_yes_price
    edge_no = (1 - prob) - entry_no_price
    
    direction = None
    if edge_yes > CONFIG["min_edge_pct"] / 100 and prob >= CONFIG["min_win_prob"]:
        direction, entry, win_p, edge = "BUY_YES", entry_yes_price, prob, edge_yes
        desc = f"BTC AU-DESSUS ${barrier:,.0f}"
    elif edge_no > CONFIG["min_edge_pct"] / 100 and (1 - prob) >= CONFIG["min_win_prob"]:
        direction, entry, win_p, edge = "BUY_NO", entry_no_price, 1-prob, edge_no
        desc = f"BTC SOUS ${barrier:,.0f}"

    if not direction:
        return None

    fee = 0 # No explicit fee as we handle it via slippage now
    payout = (1.0 - entry)
    cost = entry
    if cost <= 0 or payout <= 0:
        return None

    profit_pct = (payout / cost) * 100
    if profit_pct < CONFIG["min_profit_pct"]:
        return None

    b = payout / cost
    kelly = max(0, (win_p*b - (1-win_p))/b) * CONFIG["kelly_fraction"]

    # Limit Kelly to strict parameters
    if kelly <= 0:
        return None

    # Drawdown control
    dd = (1 - capital / CONFIG["starting_capital"]) * 100
    if dd > CONFIG["max_drawdown_pct"]:
        kelly *= 0.4
    elif dd > CONFIG["max_drawdown_pct"]/2:
        kelly *= 0.7

    # Liquidity adjustment
    liq_factor = min(market["liq"] / 15000, 1.0)
    kelly *= liq_factor

    # Fixed account scaling logic
    max_pos = capital * CONFIG["max_position_pct"]
    remaining = capital * CONFIG["max_exposure_pct"] - exposure
    if remaining <= 0:
        return None

    pos = min(kelly * capital, max_pos, remaining)
    pos = max(pos, 10.0) # Simulate real money with minimum $10 bet
    n = int(pos / entry)
    if n < 1:
        n = 1 # Force at least 1 contract
    
    actual = n * entry
    e_pnl = win_p * payout * n - (1-win_p) * cost * n

    return {
        "title": market["title"], "cid": market["cid"],
        "barrier": barrier, "scenario": market["scenario"],
        "token": market["token"], "direction": direction,
        "desc": desc, "dte": dte, "expiry_dt": market["expiry_dt"],
        "pm_prob": pm_prob, "model_prob": round(model_pct, 1),
        "edge": round(abs(edge) * 100, 1), "edge_raw": abs(edge),  # edge_raw for dynamic exits
        "entry": entry,
        "win_prob": round(win_p*100, 1),
        "profit_pct": round((payout/cost)*100, 0),
        "n": n, "cost": round(actual, 2), "e_pnl": round(e_pnl, 2),
        "slug": market.get("slug", ""),
        "liq": market["liq"],
    }


# ─── Trade Execution & Settlement ────────────────────────────────────────────

def execute(state, opp):
    if opp["cost"] > state["capital"]:
        return False
    pos = {
        "id": hashlib.md5(f"{opp['cid']}:{time.time()}".encode()).hexdigest()[:10],
        "cid": opp["cid"], "title": opp["title"], "desc": opp["desc"],
        "direction": opp["direction"], "barrier": opp["barrier"],
        "token": opp["token"], "entry": opp["entry"], "n": opp["n"],
        "cost": opp["cost"], "win_prob": opp["win_prob"],
        "profit_pct": opp["profit_pct"], "e_pnl": opp["e_pnl"],
        "entry_edge": opp.get("edge_raw", opp["edge"] / 100),   # Store for dynamic TP/SL
        "entry_dte": opp["dte"],                                  # Store for dynamic TP/SL
        "slug": opp.get("slug", ""),
        "expiry_dt": opp["expiry_dt"].isoformat() if isinstance(opp["expiry_dt"], datetime) else opp["expiry_dt"],
        "opened_at": datetime.now(timezone.utc).isoformat(), "status": "open",
    }
    state["capital"] -= opp["cost"]
    state["positions"].append(pos)
    state["total_trades"] += 1
    log.info(f"TRADE: {opp['desc']} | ${opp['cost']:.2f} | Win={opp['win_prob']:.0f}% | "
             f"Edge={opp['edge']:.1f}% | +{opp['profit_pct']:.0f}% if win")
    
    pm_url = f"https://polymarket.com/event/{opp.get('slug', '')}" if opp.get('slug') else "Polymarket"
    
    tg_send(f"📈 <b>NEW TRADE</b>: <a href=\"{pm_url}\">{opp['desc']}</a>\n"
            f"Edge: {opp['edge']:.1f}% | Win: {opp['win_prob']:.0f}%\n"
            f"Cost: <code>${opp['cost']:.2f}</code>\n"
            f"Exits: Dynamic (edge-decay TP + thesis-invalidation SL)")
    return True


def settle(state, spot, iv_pts):
    now = datetime.now(timezone.utc)
    still_open = []
    
    for pos in state["positions"]:
        exp = datetime.fromisoformat(pos["expiry_dt"])
        dte = (exp - now).total_seconds() / 86400

        # --- 1. SETTLEMENT AT EXPIRY ---
        if dte <= 0:
            won = (spot >= pos["barrier"]) if pos["direction"] == "BUY_YES" else (spot < pos["barrier"])
            if won:
                fee = CONFIG["polymarket_fee_pct"] / 100
                payout = pos["n"] * (1.0 - pos["entry"]) * (1 - fee)
                pnl = payout
                state["capital"] += pos["cost"] + payout
                state["wins"] += 1
                emoji, result = "\u2705", "WIN(Exp)"
            else:
                pnl = -pos["cost"]
                state["losses"] += 1
                emoji, result = "\u274c", "LOSS(Exp)"
                
            state["total_pnl"] += pnl
            pos["pnl"] = round(pnl, 2)
            pos["status"], pos["result"] = "closed", result
            pos["settled_at"] = now.isoformat()
            state["closed_trades"].append(pos)
            
            log.info(f"{emoji} {result}: {pos['desc']} | PnL: ${pos['pnl']:+.2f}")
            tg_send(f"{emoji} <b>{result}</b>: {pos['desc']}\nPnL: <code>${pos['pnl']:+.2f}</code>")
            continue

        # --- 2. DYNAMIC EARLY EXITS ---
        # Get MTM fair value
        moneyness = pos["barrier"] / spot
        iv = interp_iv(iv_pts, moneyness, max(0.1, dte)) if iv_pts else None
        sigma = (iv / 100.0) if (iv and iv > 0) else 0.55

        # European probability (consistent with entry analysis)
        model_prob_above = bs_prob(spot, pos["barrier"], max(0.1, dte), sigma, CONFIG["risk_free_rate"], "above")
        
        # Fair price according to current models
        current_model_prob = model_prob_above if pos["direction"] == "BUY_YES" else (1 - model_prob_above)
        
        # Current edge vs entry
        current_edge = current_model_prob - pos["entry"]
        current_edge_pct = current_edge * 100
        
        # Simulated sell value
        sell_value = (pos["n"] * current_model_prob) * (1 - CONFIG["polymarket_fee_pct"] / 100)
        unrealized_pnl_pct = (sell_value / pos["cost"]) * 100 - 100
        pnl = sell_value - pos["cost"]

        # ─── Dynamic exit parameters ───
        entry_edge = pos.get("entry_edge", 0.03)  # Original edge at entry
        entry_dte = pos.get("entry_dte", 7)        # Original DTE at entry
        time_elapsed_ratio = max(0, 1 - dte / max(entry_dte, 0.1))  # 0 at entry → 1 at expiry

        # ━━━ DYNAMIC TAKE PROFIT ━━━
        # Logic: TP when the Kelly EV of continued holding becomes negative.
        # As edge decays toward 0, the risk of reversal outweighs the remaining upside.
        # The TP threshold TIGHTENS as we approach expiry (theta acceleration).
        #
        # tp_edge_threshold = base_threshold * (1 - time_decay_factor)
        # Base: we TP when edge drops below 40% of entry edge
        # Near expiry: we TP even more aggressively (below 80% of entry edge)
        tp_time_factor = 0.4 + 0.4 * time_elapsed_ratio  # 0.4 → 0.8 over lifetime
        tp_edge_threshold = entry_edge * tp_time_factor * 100  # In percentage
        
        # Additional condition: if we captured > 60% of max possible profit, lock it in
        max_possible_pnl_pct = (1.0 / pos["entry"] - 1) * 100  # Max profit if win at expiry
        profit_capture_ratio = unrealized_pnl_pct / max(max_possible_pnl_pct, 1) if max_possible_pnl_pct > 0 else 0
        
        tp_triggered = False
        tp_reason = ""
        
        # Don't take micro-profits unless edge is completely gone. Minimum 10% PnL.
        if current_edge_pct <= tp_edge_threshold and unrealized_pnl_pct > 10:
            tp_triggered = True
            tp_reason = f"Edge decay ({current_edge_pct:.1f}% < {tp_edge_threshold:.1f}% threshold)"
        elif profit_capture_ratio >= 0.6 and unrealized_pnl_pct > 5:
            # Captured 60%+ of max profit — diminishing returns to hold
            tp_triggered = True
            tp_reason = f"Profit capture {profit_capture_ratio*100:.0f}% of max"
        elif time_elapsed_ratio > 0.85 and unrealized_pnl_pct > 0:
            # Very close to expiry with ANY profit — gamma risk is too high
            tp_triggered = True
            tp_reason = f"Near-expiry profit lock (DTE={dte:.1f}d)"

        if tp_triggered:
            state["capital"] += sell_value
            state["total_pnl"] += pnl
            state.setdefault("early_tp", 0)
            state["early_tp"] += 1
            
            pos["pnl"] = round(pnl, 2)
            pos["status"], pos["result"] = "closed", "TP(Dynamic)"
            pos["settled_at"] = now.isoformat()
            state["closed_trades"].append(pos)
            
            log.info(f"\U0001f3af TP: {pos['desc']} | {tp_reason} | PnL: ${pos['pnl']:+.2f} ({unrealized_pnl_pct:+.1f}%)")
            tg_send(f"\U0001f3af <b>TAKE PROFIT</b>: {pos['desc']}\n"
                    f"{tp_reason}\n"
                    f"PnL: <code>${pos['pnl']:+.2f} ({unrealized_pnl_pct:+.1f}%)</code>")
            continue

        # ━━━ DYNAMIC STOP LOSS ━━━
        # Logic: SL threshold adapts based on:
        # 1. Entry edge (higher conviction = wider stop)
        # 2. Time elapsed (tightens as DTE decreases — theta burns us if wrong)
        # 3. Absolute floor: if model probability < 8%, thesis is dead regardless
        #
        # Base SL: model_prob < entry_edge * scaling_factor
        # The scaling allows the trade to breathe early, but cuts aggressively near expiry
        
        # SL probability floor: starts generous, tightens toward expiry
        # At entry: stop if prob < 15% (generous, let the trade breathe)
        # Near expiry: stop if prob < 30% (tight, theta is killing us)
        sl_base = 0.15
        sl_near_expiry = 0.35
        sl_prob_threshold = sl_base + (sl_near_expiry - sl_base) * time_elapsed_ratio
        
        # Adjust based on entry conviction: higher edge = slightly wider stop
        edge_adjustment = min(entry_edge * 0.5, 0.05)  # Max 5% wider
        sl_prob_threshold = max(0.08, sl_prob_threshold - edge_adjustment)  # Absolute floor 8%
        
        # Also check PnL-based stop: if we're losing > 50% of cost AND prob is bad
        pnl_based_sl = unrealized_pnl_pct < -50 and current_model_prob < 0.30
        
        sl_triggered = False
        sl_reason = ""
        
        if current_model_prob <= sl_prob_threshold:
            sl_triggered = True
            sl_reason = f"Prob {current_model_prob*100:.1f}% < {sl_prob_threshold*100:.0f}% dynamic threshold"
        elif pnl_based_sl:
            sl_triggered = True
            sl_reason = f"PnL {unrealized_pnl_pct:+.0f}% + low prob {current_model_prob*100:.1f}%"
        elif current_model_prob <= 0.08:
            # Absolute floor — thesis completely dead
            sl_triggered = True
            sl_reason = f"Thesis dead (prob={current_model_prob*100:.1f}%)"

        if sl_triggered:
            state["capital"] += sell_value
            state["total_pnl"] += pnl
            state.setdefault("early_sl", 0)
            state["early_sl"] += 1
            
            pos["pnl"] = round(pnl, 2)
            pos["status"], pos["result"] = "closed", "SL(Dynamic)"
            pos["settled_at"] = now.isoformat()
            state["closed_trades"].append(pos)

            log.info(f"\u2702\ufe0f SL: {pos['desc']} | {sl_reason} | PnL: ${pos['pnl']:+.2f} ({unrealized_pnl_pct:+.1f}%)")
            tg_send(f"\u2702\ufe0f <b>STOP LOSS</b>: {pos['desc']}\n"
                    f"{sl_reason}\n"
                    f"PnL: <code>${pos['pnl']:+.2f} ({unrealized_pnl_pct:+.1f}%)</code>")
            continue
            
        # Position remains open
        still_open.append(pos)
        
    # Drawdown tracking updates: use total portfolio value, not cash
    total_val = state["capital"] + sum(p["cost"] for p in still_open)
    if total_val > state.setdefault("peak_capital", CONFIG["starting_capital"]):
        state["peak_capital"] = total_val
    if state["peak_capital"] > 0:
        state["max_drawdown"] = max(state.setdefault("max_drawdown", 0), (1 - total_val / state["peak_capital"]) * 100)

    state["positions"] = still_open


# ─── Main Scan Loop ──────────────────────────────────────────────────────────

def run_scan(state):
    log.info(f"{'='*50}")
    log.info(f"SCAN #{state['scans_count']+1} | Capital: ${state['capital']:.2f} | "
             f"Positions: {len(state['positions'])} | PnL: ${state['total_pnl']:+.2f}")

    iv_pts, spot = fetch_iv_surface()
    if not spot:
        return

    # Settle Expiries + Execute TP/SL Recorrelation checks
    settle(state, spot, iv_pts)

    markets = fetch_markets()
    if not markets:
        return

    exposure = sum(p["cost"] for p in state["positions"])
    opps = []
    for m in markets:
        o = analyze(m, spot, iv_pts, state["capital"], exposure)
        if o:
            opps.append(o)

    opps.sort(key=lambda o: o["edge"] * o["win_prob"], reverse=True)  # Score = edge × win probability
    log.info(f"Opportunities: {len(opps)} tradeable")

    traded_barriers = {p["barrier"] for p in state["positions"]}
    trades_done = 0
    for opp in opps:
        if trades_done >= 3:
            break
        if opp["barrier"] in traded_barriers:
            continue
        if execute(state, opp):
            trades_done += 1
            traded_barriers.add(opp["barrier"])
            exposure += opp["cost"]

    state["scans_count"] += 1
    state["last_scan"] = datetime.now(timezone.utc).isoformat()

    total_val = state["capital"] + exposure
    ret = (total_val - state["initial_capital"]) / state["initial_capital"] * 100
    wr = state["wins"] / max(state["total_trades"], 1) * 100

    PNL_HISTORY.append((datetime.now(timezone.utc).isoformat(),
                        state["capital"], total_val, state["total_pnl"]))

    log.info(f"Portfolio: ${total_val:.2f} | Return: {ret:+.1f}% | "
             f"WR: {wr:.0f}% ({state['wins']}W/{state['losses']}L) | DD: {state['max_drawdown']:.1f}%")

    save_state(state)

    # Send chart every 12 scans (~1 hour)
    if state["scans_count"] % 12 == 0:
        tg_send_chart()


def bot_loop():
    global BOT_STATE
    BOT_STATE = load_state()
    log.info(f"🤖 Autotrader started | Capital: ${BOT_STATE['capital']:.2f}")
    tg_send(f"🤖 <b>Autotrader Started!</b>\nCapital: ${BOT_STATE['capital']:.2f}\n"
            f"Scan every {CONFIG['scan_interval']}s")

    while True:
        try:
            run_scan(BOT_STATE)
        except Exception as e:
            log.error(f"Scan error: {e}\n{traceback.format_exc()}")
        time.sleep(CONFIG["scan_interval"])


# ─── Flask Web App (keeps Render alive + dashboard) ──────────────────────────

app = Flask(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Polymarket Autotrader</title>
    <meta http-equiv="refresh" content="30">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 20px; }
        h1 { color: #10b981; font-size: 1.5rem; margin-bottom: 20px; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; text-align: center; }
        .card .label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .card .value { font-size: 1.4rem; font-weight: 700; margin-top: 6px; }
        .green { color: #10b981; }
        .red { color: #ef4444; }
        .yellow { color: #f59e0b; }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th { background: #1e293b; color: #94a3b8; padding: 10px; text-align: left; font-size: 0.75rem; text-transform: uppercase; }
        td { padding: 10px; border-bottom: 1px solid #1e293b; font-size: 0.85rem; }
        tr:hover { background: rgba(16,185,129,0.05); }
        .badge { padding: 2px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; }
        .badge-open { background: rgba(16,185,129,0.15); color: #10b981; }
        .badge-win { background: rgba(16,185,129,0.15); color: #10b981; }
        .badge-loss { background: rgba(239,68,68,0.15); color: #ef4444; }
        #chart { width: 100%; height: 250px; background: #1e293b; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; }
        .section { margin-top: 20px; }
        .section h2 { color: #f59e0b; font-size: 1rem; margin-bottom: 10px; }
        .footer { margin-top: 30px; text-align: center; color: #475569; font-size: 0.7rem; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>🤖 Polymarket Autotrader — Live Dashboard</h1>

    <div class="cards">
        <div class="card">
            <div class="label">Capital</div>
            <div class="value green" id="capital">$--</div>
        </div>
        <div class="card">
            <div class="label">Total PnL</div>
            <div class="value" id="pnl">$--</div>
        </div>
        <div class="card">
            <div class="label">Return</div>
            <div class="value" id="return">--%</div>
        </div>
        <div class="card">
            <div class="label">Win Rate</div>
            <div class="value" id="winrate">--%</div>
        </div>
        <div class="card">
            <div class="label">Trades</div>
            <div class="value" id="trades">--</div>
        </div>
        <div class="card">
            <div class="label">Open Positions</div>
            <div class="value yellow" id="open">--</div>
        </div>
        <div class="card">
            <div class="label">Max Drawdown</div>
            <div class="value red" id="dd">--%</div>
        </div>
        <div class="card">
            <div class="label">Scans</div>
            <div class="value" id="scans">--</div>
        </div>
    </div>

    <canvas id="chart"></canvas>

    <div class="section">
        <h2>📊 Open Positions</h2>
        <table>
            <thead><tr><th>Trade</th><th>Barrier</th><th>Cost</th><th>Win%</th><th>Profit%</th><th>DTE</th></tr></thead>
            <tbody id="positions"></tbody>
        </table>
    </div>

    <div class="section">
        <h2>📜 Recent Trades</h2>
        <table>
            <thead><tr><th>Result</th><th>Trade</th><th>PnL</th><th>Date</th></tr></thead>
            <tbody id="history"></tbody>
        </table>
    </div>

    <div class="footer">Auto-refresh: 30s | Paper Trading Mode | Polymarket ↔ Deribit</div>

    <script>
        let chart;
        async function update() {
            const r = await fetch('/api/status');
            const d = await r.json();
            const s = d.state;
            const exp = s.capital + d.exposure;
            const ret = ((exp - s.initial_capital) / s.initial_capital * 100);
            const wr = s.total_trades > 0 ? (s.wins / s.total_trades * 100) : 0;

            document.getElementById('capital').textContent = '$' + exp.toFixed(2);
            document.getElementById('capital').className = 'value ' + (exp >= s.initial_capital ? 'green' : 'red');
            document.getElementById('pnl').textContent = '$' + s.total_pnl.toFixed(2);
            document.getElementById('pnl').className = 'value ' + (s.total_pnl >= 0 ? 'green' : 'red');
            document.getElementById('return').textContent = (ret >= 0 ? '+' : '') + ret.toFixed(1) + '%';
            document.getElementById('return').className = 'value ' + (ret >= 0 ? 'green' : 'red');
            document.getElementById('winrate').textContent = wr.toFixed(0) + '%';
            document.getElementById('winrate').className = 'value ' + (wr >= 50 ? 'green' : 'yellow');
            document.getElementById('trades').textContent = s.wins + 'W / ' + s.losses + 'L';
            document.getElementById('open').textContent = s.positions.length;
            document.getElementById('dd').textContent = s.max_drawdown.toFixed(1) + '%';
            document.getElementById('scans').textContent = s.scans_count;

            // Positions
            let ph = '';
            s.positions.forEach(p => {
                const exp_dt = new Date(p.expiry_dt);
                const dte_h = Math.max(0, (exp_dt - new Date()) / 3600000).toFixed(0);
                ph += '<tr><td>' + p.desc + '</td><td>$' + p.barrier.toLocaleString() +
                    '</td><td>$' + p.cost.toFixed(2) + '</td><td>' + p.win_prob + '%</td><td>+' +
                    p.profit_pct + '%</td><td>' + dte_h + 'h</td></tr>';
            });
            document.getElementById('positions').innerHTML = ph || '<tr><td colspan="6" style="color:#475569">No open positions</td></tr>';

            // History
            let hh = '';
            s.closed_trades.slice(-10).reverse().forEach(t => {
                const cls = t.result === 'WIN' ? 'badge-win' : 'badge-loss';
                const dt = new Date(t.settled_at).toLocaleString();
                hh += '<tr><td><span class="badge ' + cls + '">' + (t.result === 'WIN' ? '✅' : '❌') + ' ' +
                    t.result + '</span></td><td>' + t.desc + '</td><td style="color:' +
                    (t.pnl >= 0 ? '#10b981' : '#ef4444') + '">$' + t.pnl.toFixed(2) +
                    '</td><td style="color:#64748b">' + dt + '</td></tr>';
            });
            document.getElementById('history').innerHTML = hh || '<tr><td colspan="4" style="color:#475569">No trades settled yet</td></tr>';

            // Chart
            const pnl = d.pnl_history || [];
            if (pnl.length > 1) {
                const labels = pnl.map(p => new Date(p[0]).toLocaleTimeString());
                const values = pnl.map(p => p[2]);
                if (chart) chart.destroy();
                chart = new Chart(document.getElementById('chart'), {
                    type: 'line',
                    data: { labels, datasets: [{
                        label: 'Portfolio Value ($)',
                        data: values,
                        borderColor: values[values.length-1] >= s.initial_capital ? '#10b981' : '#ef4444',
                        backgroundColor: (values[values.length-1] >= s.initial_capital ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)'),
                        fill: true, tension: 0.3, pointRadius: 0,
                    }]},
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { color: '#64748b', maxTicksLimit: 10 }, grid: { color: '#1e293b' } },
                            y: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' },
                                 suggestedMin: s.initial_capital * 0.9, suggestedMax: s.initial_capital * 1.1 }
                        }
                    }
                });
            }
        }
        update();
        setInterval(update, 30000);
    </script>
</body>
</html>"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    s = BOT_STATE or default_state()
    exposure = sum(p["cost"] for p in s.get("positions", []))
    return jsonify({
        "state": s,
        "exposure": exposure,
        "pnl_history": PNL_HISTORY[-200:],  # Last 200 data points
        "config": {k: v for k, v in CONFIG.items() if "token" not in k},
    })


@app.route("/health")
def health():
    return "OK", 200


# ─── Telegram Command Handler ────────────────────────────────────────────────

TG_LAST_UPDATE = 0


def mtm_position(pos, spot, iv_pts):
    """
    Mark-to-Market a position: calculate its current fair value
    based on the model's estimate, not the original entry price.
    """
    barrier = pos["barrier"]
    entry = pos["entry"]
    n = pos["n"]
    cost = pos["cost"]
    direction = pos["direction"]

    # Get time to expiry
    try:
        exp = datetime.fromisoformat(pos["expiry_dt"])
        dte = max(0.1, (exp - datetime.now(timezone.utc)).total_seconds() / 86400)
    except Exception:
        dte = 1

    # Get current model probability
    moneyness = barrier / spot
    iv = interp_iv(iv_pts, moneyness, dte) if iv_pts else None
    if not iv or iv <= 0:
        iv = 55.0  # Default IV
    sigma = iv / 100.0

    # European probability: Polymarket resolves at expiry, not barrier-touch
    model_prob_above = bs_prob(spot, barrier, dte, sigma, CONFIG["risk_free_rate"], "above")

    if direction == "BUY_YES":
        # We bought YES, current fair value = model probability
        fair_value_per_contract = model_prob_above
    else:
        # We bought NO, current fair value = 1 - model probability
        fair_value_per_contract = 1 - model_prob_above

    # Mark-to-market PnL
    mtm_value = fair_value_per_contract * n  # Total value of position at fair price
    unrealized_pnl = mtm_value - cost

    return {
        "fair_value": round(fair_value_per_contract, 4),
        "mtm_value": round(mtm_value, 4),
        "unrealized_pnl": round(unrealized_pnl, 4),
        "pnl_pct": round((unrealized_pnl / cost) * 100, 1) if cost > 0 else 0,
        "model_prob": round(model_prob_above * 100, 1),
        "dte_hours": round(dte * 24, 0),
    }


def handle_pnl_command():
    """Handle /pnl command: show live mark-to-market PnL."""
    state = BOT_STATE
    if not state:
        tg_send("Bot not initialized yet.")
        return

    spot = fetch_spot()
    if not spot:
        tg_send("Cannot fetch BTC price.")
        return

    iv_pts, _ = fetch_iv_surface()

    positions = state.get("positions", [])
    exposure = sum(p["cost"] for p in positions)
    total_mtm_pnl = 0

    msg = f"📊 <b>PnL LIVE — Mark-to-Market</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"BTC: <code>${spot:,.0f}</code>\n\n"

    if not positions:
        msg += "<i>Aucune position ouverte</i>\n"
    else:
        for pos in positions:
            mtm = mtm_position(pos, spot, iv_pts)
            total_mtm_pnl += mtm["unrealized_pnl"]
            emoji = "🟢" if mtm["unrealized_pnl"] >= 0 else "🔴"

            msg += f"{emoji} <b>{pos['desc']}</b>\n"
            msg += f"   Entry: {pos['entry']:.2f} → Fair: {mtm['fair_value']:.2f}\n"
            msg += f"   PnL: <code>${mtm['unrealized_pnl']:+.2f}</code> ({mtm['pnl_pct']:+.1f}%)\n"
            msg += f"   DTE: {mtm['dte_hours']:.0f}h\n\n"

    total_value = state["capital"] + exposure + total_mtm_pnl
    total_return = (total_value - state["initial_capital"]) / state["initial_capital"] * 100
    wr = state["wins"] / max(state["total_trades"], 1) * 100

    msg += f"━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 Capital libre: <code>${state['capital']:.2f}</code>\n"
    msg += f"📦 Exposition: <code>${exposure:.2f}</code>\n"
    msg += f"📈 Unrealized PnL: <code>${total_mtm_pnl:+.2f}</code>\n"
    msg += f"💼 <b>Valeur totale: <code>${total_value:.2f}</code></b>\n"
    msg += f"📊 Return: <code>{total_return:+.1f}%</code>\n"
    msg += f"🏆 Win Rate: {wr:.0f}% ({state['wins']}W/{state['losses']}L)\n"
    msg += f"📉 Max DD: {state['max_drawdown']:.1f}%\n"
    msg += f"🔄 Scans: {state.get('scans_count', 0)}"

    tg_send(msg)

    # Also send chart if we have data
    if len(PNL_HISTORY) >= 2:
        tg_send_chart()


def handle_status_command():
    """Handle /status command: brief status."""
    state = BOT_STATE
    if not state:
        tg_send("Bot not initialized yet.")
        return
    exposure = sum(p["cost"] for p in state.get("positions", []))
    total_value = state["capital"] + exposure
    ret = (total_value - state["initial_capital"]) / state["initial_capital"] * 100

    msg = (f"🤖 <b>Status</b>\n"
           f"Capital: ${total_value:.2f} ({ret:+.1f}%)\n"
           f"Positions: {len(state.get('positions', []))}\n"
           f"W/L: {state['wins']}/{state['losses']}\n"
           f"Scans: {state.get('scans_count', 0)}")
    tg_send(msg)


def telegram_poller():
    """Poll Telegram for commands like /pnl, /status, /positions."""
    global TG_LAST_UPDATE
    t = CONFIG["telegram_token"]
    if not t:
        return

    log.info("Telegram command handler started")

    while True:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{t}/getUpdates",
                params={"offset": TG_LAST_UPDATE + 1, "timeout": 5},
                timeout=10,
            )
            updates = resp.json().get("result", [])

            for upd in updates:
                TG_LAST_UPDATE = upd["update_id"]
                msg = upd.get("message", {})
                text = msg.get("text", "").strip().lower()
                chat_id = str(msg.get("chat", {}).get("id", ""))

                # Only respond to our configured chat
                if chat_id != CONFIG["telegram_chat"]:
                    continue

                if text == "/pnl":
                    handle_pnl_command()
                elif text == "/status":
                    handle_status_command()
                elif text == "/positions":
                    handle_pnl_command()  # Same as /pnl
                elif text == "/help" or text == "/start":
                    tg_send(
                        "🤖 <b>Polymarket Autotrader</b>\n\n"
                        "Commandes disponibles:\n"
                        "/pnl — PnL live mark-to-market\n"
                        "/status — Status rapide\n"
                        "/positions — Positions ouvertes\n"
                        "/help — Cette aide"
                    )

        except Exception as e:
            log.debug(f"Telegram poll error: {e}")

        time.sleep(3)


# ─── Keep-Alive (prevents Render free-tier inactivity shutdown) ──────────────

def keepalive_loop():
    """
    Self-ping the /health endpoint every 10 minutes to prevent Render from
    shutting down the service due to inactivity. Also sends a Telegram
    heartbeat every 2 hours so you know the bot is alive.
    """
    import urllib.request
    port = int(os.getenv("PORT", "5001"))
    url = f"http://localhost:{port}/health"
    # Also try the external URL if available
    render_url = os.getenv("RENDER_EXTERNAL_URL", "")
    heartbeat_counter = 0
    scans_between_heartbeats = CONFIG["heartbeat_interval"] // CONFIG["keepalive_interval"]

    # Wait for server to start
    time.sleep(30)
    log.info(f"🏓 Keep-alive started: self-ping every {CONFIG['keepalive_interval']}s")

    while True:
        try:
            # Self-ping local /health
            urllib.request.urlopen(url, timeout=10)
            log.debug("Keep-alive ping OK (local)")
        except Exception:
            pass

        # Also ping external URL if on Render (this is what Render monitors)
        if render_url:
            try:
                urllib.request.urlopen(f"{render_url}/health", timeout=10)
                log.debug("Keep-alive ping OK (external)")
            except Exception:
                pass

        # Telegram heartbeat every N pings (~2 hours)
        heartbeat_counter += 1
        if heartbeat_counter >= scans_between_heartbeats:
            heartbeat_counter = 0
            state = BOT_STATE
            if state:
                exposure = sum(p["cost"] for p in state.get("positions", []))
                total_val = state["capital"] + exposure
                ret = (total_val - state["initial_capital"]) / state["initial_capital"] * 100
                uptime_h = state.get("scans_count", 0) * CONFIG["scan_interval"] / 3600
                tg_send(
                    f"💓 <b>Heartbeat</b>\n"
                    f"Bot running — {uptime_h:.1f}h uptime\n"
                    f"💰 ${total_val:.2f} ({ret:+.1f}%)\n"
                    f"📊 {len(state.get('positions', []))} positions open\n"
                    f"🔄 {state.get('scans_count', 0)} scans completed"
                )

        time.sleep(CONFIG["keepalive_interval"])


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start bot in background thread
    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()

    # Start Telegram command handler
    tg_thread = threading.Thread(target=telegram_poller, daemon=True)
    tg_thread.start()

    # Start keep-alive self-pinger (prevents Render inactivity shutdown)
    keepalive_thread = threading.Thread(target=keepalive_loop, daemon=True)
    keepalive_thread.start()

    # Start web server
    port = int(os.getenv("PORT", "5001"))
    log.info(f"Dashboard: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

