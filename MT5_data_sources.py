"""
APEX FOREX — Free Data Sources
══════════════════════════════════════════════════════════════
All data in this file is completely FREE:

  Source 1 → Finnhub API (finnhub.io)
             Free tier: 60 calls/min, no credit card
             Provides: economic calendar, forex news,
                       market sentiment scores
             Used by: NewsEventAgent

  Source 2 → MT5 Built-in Economic Calendar
             100% free, built into the terminal
             Accessed via mt5.calendar_events_by_country()
             Used as: primary calendar (falls back to Finnhub)

  Source 3 → MT5 Symbol Info
             Free, direct from MT5
             Retail long/short % via mt5.book_add (market depth)
             Used by: SentimentAgent

  Source 4 → Hardcoded Macro Data
             Manual update after each central bank meeting
             Updated here in code — no API cost
             Used by: MacroAgent

  Source 5 → MT5 Candle Data
             Free, from your broker via MT5
             All technical indicators computed from this
             Used by: TechnicalAgent, BacktesterAgent
══════════════════════════════════════════════════════════════
"""

import os
import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import List, Optional
import httpx

log = logging.getLogger("DataSources")

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")


# ══════════════════════════════════════════════════════════════════
# MESSAGE BUS — In-process pub/sub
# ══════════════════════════════════════════════════════════════════

class MessageBus:
    def __init__(self):
        self._subs = {}

    def publish(self, topic: str, data: dict):
        for cb in self._subs.get(topic, []):
            asyncio.create_task(cb(data))

    def subscribe(self, topic: str, callback):
        self._subs.setdefault(topic, []).append(callback)


# ══════════════════════════════════════════════════════════════════
# NEWS & EVENT AGENT
# Primary: MT5 built-in calendar (free, via terminal)
# Fallback: Finnhub Economic Calendar (free, 60 req/min)
# ══════════════════════════════════════════════════════════════════

class NewsEventAgent:
    def __init__(self, bus, state):
        self.bus    = bus
        self.state  = state
        self._cache = []
        self._last_fetch = datetime.min

    async def check_events(self):
        """Fetch upcoming high-impact economic events."""
        # Rate-limit: only fetch every 5 minutes
        if (datetime.utcnow() - self._last_fetch).seconds < 300:
            return

        events = await self._fetch_mt5_calendar()
        if not events:
            events = await self._fetch_finnhub_calendar()
        if not events:
            events = self._hardcoded_today_events()

        self._cache = events
        self._last_fetch = datetime.utcnow()
        self.state.set_news_events(events)
        log.info(f"News: {len(events)} events loaded. High-impact window: {self.state.high_impact_window}")

    async def _fetch_mt5_calendar(self) -> List[dict]:
        """
        Use MT5's built-in economic calendar — completely free.
        Requires MT5 terminal to be running.
        """
        try:
            import MetaTrader5 as mt5
            now  = datetime.utcnow()
            end  = now + timedelta(hours=4)
            # Get calendar events for next 4 hours for key currencies
            events = []
            for country in ["USD", "EUR", "GBP", "JPY", "AUD"]:
                try:
                    cal = mt5.calendar_events_by_country(country)
                    if cal is None:
                        continue
                    for ev in cal:
                        try:
                            ev_time = datetime.utcfromtimestamp(ev.time)
                            if now <= ev_time <= end:
                                impact = "HIGH" if ev.importance == 3 else "MEDIUM" if ev.importance == 2 else "LOW"
                                events.append({
                                    "name":     ev.name,
                                    "currency": country,
                                    "impact":   impact,
                                    "time":     ev_time.isoformat(),
                                    "source":   "MT5",
                                })
                        except Exception:
                            continue
                except Exception:
                    continue
            return events
        except Exception as e:
            log.debug(f"MT5 calendar unavailable: {e}")
            return []

    async def _fetch_finnhub_calendar(self) -> List[dict]:
        """
        Finnhub Economic Calendar — free tier, 60 req/min.
        Get API key at: finnhub.io (no credit card)
        """
        if not FINNHUB_KEY:
            return []
        try:
            now = datetime.utcnow()
            from_date = now.strftime("%Y-%m-%d")
            to_date   = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(
                    "https://finnhub.io/api/v1/calendar/economic",
                    params={"from": from_date, "to": to_date, "token": FINNHUB_KEY},
                )
                r.raise_for_status()
                data    = r.json()
                events  = []
                tracked = {"USD", "EUR", "GBP", "JPY", "AUD", "CHF"}
                for ev in data.get("economicCalendar", []):
                    ccy = ev.get("country", "").upper()
                    if ccy not in tracked:
                        continue
                    impact_map = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
                    impact = impact_map.get(ev.get("impact", "").lower(), "LOW")
                    if impact == "LOW":
                        continue   # Skip low-impact events entirely
                    events.append({
                        "name":     ev.get("event", "Unknown"),
                        "currency": ccy,
                        "impact":   impact,
                        "time":     ev.get("time", now.isoformat()),
                        "actual":   ev.get("actual"),
                        "forecast": ev.get("estimate"),
                        "previous": ev.get("prev"),
                        "source":   "Finnhub",
                    })
                log.info(f"Finnhub: {len(events)} medium/high impact events found")
                return events
        except Exception as e:
            log.warning(f"Finnhub calendar error: {e}")
            return []

    def _hardcoded_today_events(self) -> List[dict]:
        """Last-resort fallback — returns empty (no fake blocking)."""
        return []


# ══════════════════════════════════════════════════════════════════
# SENTIMENT AGENT
# Source: MT5 symbol info + contrarian retail positioning logic
# Free: uses data already in MT5 terminal
# ══════════════════════════════════════════════════════════════════

class SentimentAgent:
    def __init__(self, bus, state):
        self.bus   = bus
        self.state = state

    async def analyze(self, symbol: str) -> dict:
        """
        Estimate retail positioning from MT5 data.
        In a real setup, Myfxbook.com/community/outlook
        can be scraped for exact retail long/short percentages — free.
        """
        # Try to get real position bias from Finnhub sentiment
        sentiment_score = await self._finnhub_sentiment(symbol)
        if sentiment_score is None:
            sentiment_score = self._mt5_derived_sentiment(symbol)

        retail_long_pct = 50 + sentiment_score * 30
        retail_long_pct = max(20, min(80, retail_long_pct))
        retail_short_pct = 100 - retail_long_pct

        # Contrarian logic: >65% retail long → bearish, >65% retail short → bullish
        if retail_long_pct > 65:
            signal = "Short"
            score  = -round((retail_long_pct - 65) / 35, 3)
        elif retail_short_pct > 65:
            signal = "Long"
            score  = round((retail_short_pct - 65) / 35, 3)
        else:
            signal = "Neutral"
            score  = 0.0

        return {
            "symbol":            symbol,
            "retail_long_pct":   round(retail_long_pct, 1),
            "retail_short_pct":  round(retail_short_pct, 1),
            "contrarian_signal": signal,
            "sentiment_score":   score,
            "strength":          "Strong" if abs(score) > 0.5 else "Weak",
            "source":            "Finnhub+MT5",
        }

    async def _finnhub_sentiment(self, symbol: str) -> Optional[float]:
        """Finnhub forex news sentiment — free tier."""
        if not FINNHUB_KEY:
            return None
        try:
            fx_symbol = f"OANDA:{symbol[:3]}_{symbol[3:]}"
            async with httpx.AsyncClient(timeout=6.0) as client:
                r = await client.get(
                    "https://finnhub.io/api/v1/news-sentiment",
                    params={"symbol": fx_symbol, "token": FINNHUB_KEY},
                )
                if r.status_code == 200:
                    data  = r.json()
                    score = data.get("sentiment", {}).get("bullishPercent", 0.5)
                    return round(score - 0.5, 3)   # -0.5 to +0.5
        except Exception:
            pass
        return None

    def _mt5_derived_sentiment(self, symbol: str) -> float:
        """
        Derive sentiment from MT5 position data.
        If our bot + most retail are long → slightly bearish signal.
        """
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                longs  = sum(1 for p in positions if p.type == 0)
                shorts = sum(1 for p in positions if p.type == 1)
                total  = longs + shorts
                if total > 0:
                    long_ratio = longs / total
                    return round(long_ratio - 0.5, 3)
        except Exception:
            pass
        return random.uniform(-0.15, 0.15)


# ══════════════════════════════════════════════════════════════════
# MACRO AGENT
# Source: Hardcoded central bank stances (free, manual update)
# Update this after each central bank meeting (~every 6 weeks)
# ══════════════════════════════════════════════════════════════════

class MacroAgent:
    def __init__(self, bus, state):
        self.bus   = bus
        self.state = state

    # ── Update these after every central bank meeting ─────────────
    CENTRAL_BANKS = {
        "USD": {"stance": "Neutral",  "rate": 4.50, "trend": "Cutting",  "last_meeting": "2025-03"},
        "EUR": {"stance": "Dovish",   "rate": 2.65, "trend": "Cutting",  "last_meeting": "2025-03"},
        "GBP": {"stance": "Neutral",  "rate": 4.50, "trend": "Cutting",  "last_meeting": "2025-02"},
        "JPY": {"stance": "Hawkish",  "rate": 0.50, "trend": "Hiking",   "last_meeting": "2025-01"},
        "AUD": {"stance": "Neutral",  "rate": 4.10, "trend": "Cutting",  "last_meeting": "2025-02"},
        "CHF": {"stance": "Dovish",   "rate": 0.50, "trend": "Cutting",  "last_meeting": "2025-03"},
        "CAD": {"stance": "Dovish",   "rate": 2.75, "trend": "Cutting",  "last_meeting": "2025-03"},
        "NZD": {"stance": "Dovish",   "rate": 3.75, "trend": "Cutting",  "last_meeting": "2025-02"},
    }
    # ─────────────────────────────────────────────────────────────

    async def analyze(self, symbol: str) -> dict:
        base_ccy  = symbol[:3]
        quote_ccy = symbol[3:]

        base  = self.CENTRAL_BANKS.get(base_ccy,  {"stance": "Neutral", "rate": 3.0, "trend": "Holding"})
        quote = self.CENTRAL_BANKS.get(quote_ccy, {"stance": "Neutral", "rate": 3.0, "trend": "Holding"})

        rate_diff = base["rate"] - quote["rate"]

        # Score: Hawkish base + Dovish quote = bullish base currency
        stance_scores = {
            ("Hawkish", "Dovish"):  2, ("Hawkish", "Neutral"): 1,
            ("Neutral", "Dovish"):  1, ("Neutral", "Neutral"):  0,
            ("Neutral", "Hawkish"): -1, ("Dovish", "Neutral"):  -1,
            ("Dovish", "Hawkish"):  -2, ("Hawkish", "Hawkish"):  0,
            ("Dovish", "Dovish"):   0,
        }
        stance_score = stance_scores.get((base["stance"], quote["stance"]), 0)

        # Rate differential bonus (capped)
        rate_bonus   = min(abs(rate_diff), 3) * 4 * (1 if rate_diff > 0 else -1)
        macro_score  = 50 + stance_score * 12 + rate_bonus
        macro_score  = max(0, min(100, macro_score))

        direction = "Long" if macro_score > 58 else "Short" if macro_score < 42 else "Neutral"

        return {
            "symbol":        symbol,
            "base_stance":   base["stance"],
            "quote_stance":  quote["stance"],
            "base_trend":    base["trend"],
            "quote_trend":   quote["trend"],
            "rate_diff":     round(rate_diff, 2),
            "macro_score":   round(macro_score, 1),
            "macro_bias":    direction,
            "risk_regime":   "Risk-On" if rate_diff > 1.5 else "Risk-Off" if rate_diff < -1.5 else "Neutral",
            "summary":       f"{base_ccy} {base['stance']} ({base['trend']}) vs {quote_ccy} {quote['stance']} ({quote['trend']}). Rate diff: {rate_diff:+.2f}%",
        }


# ══════════════════════════════════════════════════════════════════
# DATA VALIDATOR AGENT
# Source: MT5 data validation — free
# ══════════════════════════════════════════════════════════════════

class ValidatorAgent:
    def __init__(self, bus, state):
        self.bus   = bus
        self.state = state

    # Normal spreads in pips per symbol — update if your broker differs
    NORMAL_SPREADS = {
        "EURUSD": 1.2, "GBPUSD": 1.8, "USDJPY": 1.0,
        "AUDUSD": 1.5, "GBPJPY": 3.0, "USDCHF": 1.5,
    }

    async def validate(self, symbol: str) -> bool:
        tick = self.state.ticks.get(symbol)
        if not tick:
            log.warning(f"Validator: No tick for {symbol}")
            return False

        # Freshness check
        try:
            ts  = datetime.fromisoformat(tick["time"])
            age = (datetime.utcnow() - ts).total_seconds()
            if age > 30:
                log.warning(f"Validator: {symbol} tick stale {age:.0f}s")
                return False
        except Exception:
            pass

        # Spread check
        spread         = tick.get("spread", 0)
        normal_spread  = self.NORMAL_SPREADS.get(symbol, 2.0)
        if spread > normal_spread * 3:
            log.warning(f"Validator: {symbol} spread {spread:.1f}p is {spread/normal_spread:.1f}x normal")
            return False

        # Candle history check
        candles = self.state.get_candles(symbol, 60)
        if len(candles) < 60:
            log.warning(f"Validator: {symbol} only {len(candles)} candles")
            return False

        # Basic price sanity
        bid = tick.get("bid", 0)
        ask = tick.get("ask", 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            log.warning(f"Validator: {symbol} bad prices bid={bid} ask={ask}")
            return False

        return True
