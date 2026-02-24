"""
📊 Polymarket Strategy Backtest
================================
Backtests the BTC barrier market strategy using historical BTC prices.
Since we don't have historical Polymarket prices, we simulate realistic
barrier markets based on how Polymarket typically prices them.

Key insight: Polymarket systematically overprices "above" probabilities
(retail bullish bias). Our model exploits this by buying NO when the
model probability is lower than Polymarket's implied probability.
"""

import json
import math
import random
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from scipy.stats import norm

# ─── Configuration ────────────────────────────────────────────────────────────

BACKTEST_CONFIG = {
    "starting_capital": 100.0,
    "min_edge_pct": 2.5,
    "kelly_fraction": 0.15,
    "max_position_pct": 0.12,
    "max_exposure_pct": 0.70,
    "polymarket_fee_pct": 2.0,
    "risk_free_rate": 0.045,
    "min_win_prob": 0.20,
    "max_drawdown_reduce": 25,
    "max_trades_per_week": 5,
    "btc_annual_vol": 0.55,       # Typical BTC annualized vol
    "pm_bias_pct": 2.5,           # Conservative: PM overprices YES by ~2.5%
    "pm_noise_std": 3.0,          # Higher noise in PM pricing (stdev in %)
}


# ─── Fetch Historical BTC Prices ─────────────────────────────────────────────

def fetch_btc_history(days=365):
    """Fetch daily BTC prices from CoinGecko (free, no API key)."""
    print(f"Fetching {days} days of BTC price history...")
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        resp = requests.get(url, params={
            "vs_currency": "usd",
            "days": str(days),
            "interval": "daily",
        }, timeout=30)
        data = resp.json()
        prices = []
        for ts, price in data["prices"]:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            prices.append({"date": dt, "price": price})
        print(f"  Got {len(prices)} daily prices")
        print(f"  Range: ${prices[0]['price']:,.0f} to ${prices[-1]['price']:,.0f}")
        print(f"  Period: {prices[0]['date'].strftime('%Y-%m-%d')} to {prices[-1]['date'].strftime('%Y-%m-%d')}")
        return prices
    except Exception as e:
        print(f"  Error fetching prices: {e}")
        return []


# ─── Simulate Polymarket Markets ─────────────────────────────────────────────

def generate_barrier_markets(spot, date, vol):
    """
    Generate realistic barrier markets as Polymarket would have them.
    
    For each weekly expiry, create markets at various strikes around spot.
    Polymarket typically has: $50k, $52k, $54k, ..., $80k etc.
    """
    markets = []
    
    # Strike levels: every $2,000 around spot
    strike_step = 2000
    min_strike = max(10000, int((spot * 0.75) / strike_step) * strike_step)
    max_strike = int((spot * 1.35) / strike_step) * strike_step
    
    strikes = list(range(min_strike, max_strike + strike_step, strike_step))
    
    # Weekly expiry: 7 days out
    expiry = date + timedelta(days=7)
    dte = 7
    T = dte / 365.0
    sigma = vol
    r = BACKTEST_CONFIG["risk_free_rate"]
    
    for strike in strikes:
        # TRUE probability (Black-Scholes)
        d2 = (math.log(spot / strike) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        true_prob_above = norm.cdf(d2)
        
        # POLYMARKET price = true probability + bullish bias + noise
        # Retail traders on Polymarket tend to overvalue YES (bullish bias)
        bias = BACKTEST_CONFIG["pm_bias_pct"] / 100
        noise = random.gauss(0, BACKTEST_CONFIG["pm_noise_std"] / 100)
        
        pm_yes_price = min(0.99, max(0.01, true_prob_above + bias + noise))
        pm_no_price = 1.0 - pm_yes_price
        
        markets.append({
            "strike": strike,
            "scenario": "above",
            "true_prob": true_prob_above,
            "pm_yes": round(pm_yes_price, 4),
            "pm_no": round(pm_no_price, 4),
            "pm_prob": round(pm_yes_price * 100, 2),
            "dte": dte,
            "expiry": expiry,
            "date": date,
        })
    
    return markets


# ─── Model Probability ───────────────────────────────────────────────────────

def model_probability(spot, strike, dte, vol, scenario="above"):
    """Our model's estimate of P(above/below) at expiry."""
    T = dte / 365.0
    r = BACKTEST_CONFIG["risk_free_rate"]
    sigma = vol
    
    d2 = (math.log(spot / strike) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    
    if scenario == "above":
        return norm.cdf(d2)
    else:
        return norm.cdf(-d2)


# ─── Strategy Logic ──────────────────────────────────────────────────────────

def analyze_trade(market, spot, vol, capital, exposure):
    """Analyze a single market for edge."""
    strike = market["strike"]
    dte = market["dte"]
    pm_prob = market["pm_prob"]
    
    # Model probability
    model_prob = model_probability(spot, strike, dte, vol, "above") * 100
    
    edge = model_prob - pm_prob  # Positive = YES underpriced
    
    if abs(edge) < BACKTEST_CONFIG["min_edge_pct"]:
        return None
    
    if edge > 0:
        direction = "BUY_YES"
        entry = market["pm_yes"]
        win_prob = model_prob / 100
        desc = f"BTC > ${strike:,.0f}"
    else:
        direction = "BUY_NO"
        entry = market["pm_no"]
        win_prob = 1 - model_prob / 100
        desc = f"BTC < ${strike:,.0f}"
    
    if win_prob < BACKTEST_CONFIG["min_win_prob"]:
        return None
    
    # Kelly sizing
    fee = BACKTEST_CONFIG["polymarket_fee_pct"] / 100
    payout = (1.0 - entry) * (1 - fee)
    cost = entry
    
    if cost <= 0 or payout <= 0:
        return None
    
    b = payout / cost
    kelly = max(0, (win_prob * b - (1 - win_prob)) / b) * BACKTEST_CONFIG["kelly_fraction"]
    
    # Drawdown control
    dd = (1 - capital / BACKTEST_CONFIG["starting_capital"]) * 100
    if dd > BACKTEST_CONFIG["max_drawdown_reduce"]:
        kelly *= 0.4
    
    max_pos = capital * BACKTEST_CONFIG["max_position_pct"]
    remaining = capital * BACKTEST_CONFIG["max_exposure_pct"] - exposure
    if remaining <= 0:
        return None
    
    pos = min(kelly * capital, max_pos, remaining)
    n = int(pos / entry)
    if n < 1:
        return None
    
    actual_cost = n * entry
    
    return {
        "strike": strike, "direction": direction, "desc": desc,
        "entry": entry, "n": n, "cost": round(actual_cost, 4),
        "win_prob": round(win_prob * 100, 1),
        "profit_pct": round((payout / cost) * 100, 0),
        "edge": round(abs(edge), 1),
        "dte": dte, "expiry": market["expiry"],
        "true_prob": market["true_prob"],
    }


def settle_position(pos, spot_at_expiry):
    """Settle a position: did we win or lose?"""
    if pos["direction"] == "BUY_YES":
        won = spot_at_expiry >= pos["strike"]
    else:  # BUY_NO
        won = spot_at_expiry < pos["strike"]
    
    if won:
        fee = BACKTEST_CONFIG["polymarket_fee_pct"] / 100
        payout = pos["n"] * (1.0 - pos["entry"]) * (1 - fee)
        return payout, "WIN"
    else:
        return -pos["cost"], "LOSS"


# ─── Main Backtest ────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 60)
    print("  POLYMARKET BTC BARRIER STRATEGY — BACKTEST")
    print("=" * 60)
    
    # Fetch BTC prices
    prices = fetch_btc_history(365)
    if len(prices) < 30:
        print("Not enough price data!")
        return
    
    # Config
    capital = BACKTEST_CONFIG["starting_capital"]
    initial_capital = capital
    peak_capital = capital
    max_dd = 0.0
    
    positions = []  # Open positions
    trades_log = []
    wins = 0
    losses = 0
    total_pnl = 0.0
    equity_curve = []
    
    # Rolling volatility (30-day realized)
    def calc_vol(prices_slice):
        if len(prices_slice) < 10:
            return BACKTEST_CONFIG["btc_annual_vol"]
        returns = [math.log(prices_slice[i]["price"] / prices_slice[i-1]["price"]) 
                   for i in range(1, len(prices_slice))]
        daily_vol = np.std(returns)
        return daily_vol * math.sqrt(365)
    
    print(f"\nStarting capital: ${capital:.2f}")
    print(f"Period: {prices[0]['date'].strftime('%Y-%m-%d')} to {prices[-1]['date'].strftime('%Y-%m-%d')}")
    print(f"{'─'*60}")
    
    # Simulate week by week
    week_idx = 0
    i = 7  # Start after first week (need history for vol)
    
    while i < len(prices):
        current = prices[i]
        spot = current["price"]
        date = current["date"]
        
        # Settle expired positions
        still_open = []
        for pos in positions:
            if date >= pos["expiry"]:
                pnl, result = settle_position(pos, spot)
                if result == "WIN":
                    capital += pos["cost"] + pnl  # Return cost + profit
                    wins += 1
                    total_pnl += pnl
                else:
                    losses += 1
                    total_pnl += pnl  # pnl is negative
                
                trades_log.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "desc": pos["desc"],
                    "result": result,
                    "pnl": round(pnl, 4),
                    "capital_after": round(capital, 2),
                })
            else:
                still_open.append(pos)
        positions = still_open
        
        # Track equity
        exposure = sum(p["cost"] for p in positions)
        total_value = capital + exposure
        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "capital": round(capital, 2),
            "exposure": round(exposure, 2),
            "total_value": round(total_value, 2),
            "spot": round(spot, 0),
        })
        
        # Update drawdown
        if total_value > peak_capital:
            peak_capital = total_value
        dd = (1 - total_value / peak_capital) * 100
        max_dd = max(max_dd, dd)
        
        # Every 7 days: generate new markets and trade
        if i % 7 == 0:
            week_idx += 1
            
            # Calculate rolling vol
            vol_slice = prices[max(0, i-30):i]
            vol = calc_vol(vol_slice)
            vol = max(0.30, min(vol, 1.20))  # Clamp to reasonable range
            
            # Generate barrier markets
            markets = generate_barrier_markets(spot, date, vol)
            
            # Analyze all markets
            opps = []
            for m in markets:
                opp = analyze_trade(m, spot, vol, capital, exposure)
                if opp:
                    opps.append(opp)
            
            # Sort by edge * win probability
            opps.sort(key=lambda o: o["edge"] * o["win_prob"], reverse=True)
            
            # Execute top trades
            trades_this_week = 0
            traded_strikes = {p["strike"] for p in positions}
            
            for opp in opps:
                if trades_this_week >= BACKTEST_CONFIG["max_trades_per_week"]:
                    break
                if opp["strike"] in traded_strikes:
                    continue
                if opp["cost"] > capital:
                    continue
                
                capital -= opp["cost"]
                positions.append(opp)
                trades_this_week += 1
                traded_strikes.add(opp["strike"])
                exposure += opp["cost"]
        
        i += 1
    
    # Final settlement of remaining positions
    final_spot = prices[-1]["price"]
    for pos in positions:
        pnl, result = settle_position(pos, final_spot)
        if result == "WIN":
            capital += pos["cost"] + pnl
            wins += 1
            total_pnl += pnl
        else:
            losses += 1
            total_pnl += pnl
        trades_log.append({
            "date": prices[-1]["date"].strftime("%Y-%m-%d"),
            "desc": pos["desc"],
            "result": result,
            "pnl": round(pnl, 4),
            "capital_after": round(capital, 2),
        })
    
    # ─── Results ───────────────────────────────────────────────────────────
    total_trades = wins + losses
    win_rate = wins / max(total_trades, 1) * 100
    total_return = (capital - initial_capital) / initial_capital * 100
    
    # Annualized return
    n_days = (prices[-1]["date"] - prices[0]["date"]).days
    annual_return = total_return * (365 / max(n_days, 1))
    
    # Sharpe ratio (simplified)
    if len(equity_curve) > 10:
        values = [e["total_value"] for e in equity_curve]
        daily_returns = [(values[i] - values[i-1]) / values[i-1] 
                         for i in range(1, len(values)) if values[i-1] > 0]
        if daily_returns:
            sharpe = (np.mean(daily_returns) / max(np.std(daily_returns), 0.0001)) * math.sqrt(365)
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Period:          {n_days} days ({n_days//30} months)")
    print(f"  Starting Capital: ${initial_capital:.2f}")
    print(f"  Final Capital:    ${capital:.2f}")
    print(f"  Total Return:     {total_return:+.1f}%")
    print(f"  Annualized:       {annual_return:+.1f}%")
    print(f"  Sharpe Ratio:     {sharpe:.2f}")
    print(f"  Max Drawdown:     {max_dd:.1f}%")
    print(f"  Total Trades:     {total_trades}")
    print(f"  Win Rate:         {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Total PnL:        ${total_pnl:+.2f}")
    print(f"  Avg PnL/Trade:    ${total_pnl/max(total_trades,1):+.4f}")
    print(f"{'='*60}")
    
    # Show equity curve highlights
    if equity_curve:
        min_val = min(equity_curve, key=lambda e: e["total_value"])
        max_val = max(equity_curve, key=lambda e: e["total_value"])
        print(f"\n  Equity Curve:")
        print(f"    Lowest:  ${min_val['total_value']:.2f} on {min_val['date']}")
        print(f"    Highest: ${max_val['total_value']:.2f} on {max_val['date']}")
    
    # Show sample trades
    print(f"\n  Sample Trades (last 15):")
    for t in trades_log[-15:]:
        emoji = "+" if t["result"] == "WIN" else " "
        color = "WIN " if t["result"] == "WIN" else "LOSS"
        print(f"    {t['date']} | {color} | {t['desc']:>20s} | "
              f"PnL: ${t['pnl']:+.4f} | Capital: ${t['capital_after']:.2f}")
    
    # Save results
    results = {
        "config": BACKTEST_CONFIG,
        "summary": {
            "period_days": n_days,
            "starting_capital": initial_capital,
            "final_capital": round(capital, 2),
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annual_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 1),
            "total_pnl": round(total_pnl, 4),
        },
        "equity_curve": equity_curve,
        "trades": trades_log,
    }
    
    with open("backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to backtest_results.json")
    
    return results


if __name__ == "__main__":
    run_backtest()
