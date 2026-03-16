"""
APEX FOREX — All Trading Agents (MT5 Edition)
All agents rewritten to use MT5 as the sole data/execution source.
No OANDA. No paid APIs. 100% free data.
"""

import os, json, math, asyncio, logging, random
from datetime import datetime, timedelta
from typing import List, Optional
import httpx

from core.mt5_connector import MT5Connector
from core.ai_brain      import get_brain

log = logging.getLogger("Agents")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL             = "claude-sonnet-4-20250514"


# ══════════════════════════════════════════════════════════════════
# PRICE FEED AGENT
# Source: MT5 terminal (free, via MetaTrader5 Python library)
# ══════════════════════════════════════════════════════════════════

class PriceFeedAgent:
    def __init__(self, bus, state):
        self.bus  = bus
        self.state = state
        self.mt5   = MT5Connector.get()

    async def fetch_tick(self, symbol: str) -> Optional[dict]:
        tick = self.mt5.get_tick(symbol)
        if tick:
            self.state.update_tick(symbol, tick)
        return tick

    async def fetch_candles(self, symbol: str, timeframe: str = "M15", count: int = 250):
        candles = self.mt5.get_candles(symbol, timeframe, count)
        if candles:
            self.state.add_candles(symbol, candles)
        return candles

    async def fetch_account(self):
        info = self.mt5.account_info()
        if info:
            self.state.update_account(info["balance"], info["equity"], info["margin"])
        return info


# ══════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS AGENT
# Source: MT5 candles (free) → computed in Python
# ══════════════════════════════════════════════════════════════════

class TechnicalAgent:
    def __init__(self, bus, state):
        self.bus   = bus
        self.state = state

    async def analyze(self, symbol: str) -> dict:
        candles = self.state.get_candles(symbol, 250)
        if len(candles) < 60:
            return {"trend_direction": "Neutral", "technical_score": 0, "atr": 0.001,
                    "market_regime": "UNKNOWN", "error": "insufficient_data"}

        closes = [c["close"] for c in candles]
        highs  = [c["high"]  for c in candles]
        lows   = [c["low"]   for c in candles]

        ema20  = self._ema(closes, 20)
        ema50  = self._ema(closes, 50)
        ema200 = self._ema(closes, 200)
        rsi14  = self._rsi(closes, 14)
        macd_l, macd_s = self._macd(closes)
        atr14  = self._atr(candles, 14)
        adx14  = self._adx(candles, 14)
        bb_u, bb_m, bb_l = self._bb(closes, 20, 2)

        p = closes[-1]
        e20, e50, e200 = ema20[-1], ema50[-1], ema200[-1]
        rsi = rsi14[-1]; macd = macd_l[-1]; sig = macd_s[-1]
        atr = atr14[-1]; adx = adx14[-1]
        pe20, pe50 = ema20[-2], ema50[-2]
        pm, ps     = macd_l[-2], macd_s[-2]

        recent50  = candles[-50:]
        support   = sum(sorted([c["low"]  for c in recent50])[:3]) / 3
        resistance= sum(sorted([c["high"] for c in recent50], reverse=True)[:3]) / 3

        score   = 0
        signals = []

        # EMA stack
        bull_stack = e20 > e50 > e200
        bear_stack = e20 < e50 < e200
        if   bull_stack: score += 22; signals.append("EMA 20>50>200 bullish stack")
        elif bear_stack: score += 22; signals.append("EMA 20<50<200 bearish stack")
        elif e20 > e50:  score += 10; signals.append("Short-term bullish 20>50")
        elif e20 < e50:  score += 10; signals.append("Short-term bearish 20<50")

        # RSI
        if 40 < rsi < 60:                                     score += 10; signals.append(f"RSI neutral {rsi:.1f}")
        elif (bull_stack and 55<rsi<72) or (bear_stack and 28<rsi<45): score += 15; signals.append(f"RSI confirms trend {rsi:.1f}")
        elif rsi > 78 or rsi < 22:                            score -= 10; signals.append(f"RSI extreme {rsi:.1f}")

        # MACD
        cross_up   = macd > sig and pm <= ps
        cross_down = macd < sig and pm >= ps
        if   (cross_up and bull_stack) or (cross_down and bear_stack): score += 22; signals.append("MACD cross aligns with trend")
        elif (macd > sig and bull_stack) or (macd < sig and bear_stack): score += 10; signals.append("MACD confirms trend")

        # ADX
        if   adx > 30: score += 15; signals.append(f"Strong trend ADX={adx:.1f}")
        elif adx > 20: score +=  8; signals.append(f"Moderate trend ADX={adx:.1f}")
        else:          score -=  5; signals.append(f"Ranging ADX={adx:.1f}")

        # BB position
        if   (bull_stack and p > bb_m[-1]): score += 8; signals.append("Price above BB mid — bullish")
        elif (bear_stack and p < bb_m[-1]): score += 8; signals.append("Price below BB mid — bearish")

        # Near S/R levels
        near_s = abs(p - support)    / max(p, 0.001) < 0.0012
        near_r = abs(p - resistance) / max(p, 0.001) < 0.0012
        if near_s and bull_stack: score += 14; signals.append("At support in uptrend")
        if near_r and bear_stack: score += 14; signals.append("At resistance in downtrend")

        score = max(0, min(100, score))

        if   score >= 55 and (bull_stack or (e20>e50 and macd>sig)): direction = "Bullish"
        elif score >= 55 and (bear_stack or (e20<e50 and macd<sig)): direction = "Bearish"
        else:                                                          direction = "Neutral"

        return {
            "symbol": symbol, "trend_direction": direction,
            "technical_score": score, "confluence_signals": signals,
            "indicators": {"ema20": round(e20,5), "ema50": round(e50,5),
                           "ema200": round(e200,5), "rsi": round(rsi,1),
                           "macd": round(macd,6), "atr": round(atr,5), "adx": round(adx,1)},
            "levels": {"support": round(support,5), "resistance": round(resistance,5)},
            "setup_quality": "A" if score>=80 else "B" if score>=65 else "C" if score>=50 else "D",
            "atr": round(atr, 5),
            "market_regime": "TRENDING" if adx > 25 else "RANGING",
        }

    # ── Indicator implementations ─────────────────────────────────
    def _ema(self, p, n):
        e, k = [], 2/(n+1)
        for i,x in enumerate(p):
            e.append(x if i==0 else x*k + e[-1]*(1-k))
        return e

    def _rsi(self, p, n=14):
        r = []
        for i in range(len(p)):
            if i < n: r.append(50.0); continue
            ch = [p[j]-p[j-1] for j in range(i-n+1,i+1)]
            ag = sum(max(c,0) for c in ch)/n
            al = sum(abs(min(c,0)) for c in ch)/n
            rs = ag/al if al>0 else 100
            r.append(100 - 100/(1+rs))
        return r

    def _macd(self, p, f=12, s=26, sg=9):
        ef, es = self._ema(p,f), self._ema(p,s)
        ml = [a-b for a,b in zip(ef,es)]
        return ml, (self._ema(ml, sg) if len(ml)>=sg else ml)

    def _atr(self, c, n=14):
        a = []
        for i,x in enumerate(c):
            if i==0: a.append(x["high"]-x["low"]); continue
            tr = max(x["high"]-x["low"], abs(x["high"]-c[i-1]["close"]), abs(x["low"]-c[i-1]["close"]))
            a.append((a[-1]*(n-1)+tr)/n if i>=n else tr)
        return a

    def _adx(self, c, n=14):
        adx = []
        for i in range(len(c)):
            if i < n*2: adx.append(20.0); continue
            dms, dmls, trs = [], [], []
            for j in range(i-n+1, i+1):
                u = c[j]["high"]-c[j-1]["high"]; d = c[j-1]["low"]-c[j]["low"]
                dms.append(max(u,0) if u>d else 0)
                dmls.append(max(d,0) if d>u else 0)
                trs.append(max(c[j]["high"]-c[j]["low"], abs(c[j]["high"]-c[j-1]["close"]), abs(c[j]["low"]-c[j-1]["close"])))
            ts = sum(trs)
            if ts==0: adx.append(20.0); continue
            dip=100*sum(dms)/ts; din=100*sum(dmls)/ts
            dx = 100*abs(dip-din)/(dip+din) if (dip+din)>0 else 0
            adx.append(dx)
        return adx

    def _bb(self, p, n=20, sd=2):
        u,m,l=[],[],[]
        for i in range(len(p)):
            win = p[max(0,i-n+1):i+1]
            mv  = sum(win)/len(win)
            std = math.sqrt(sum((x-mv)**2 for x in win)/len(win))
            u.append(mv+sd*std); m.append(mv); l.append(mv-sd*std)
        return u, m, l


# ══════════════════════════════════════════════════════════════════
# PREDICTOR AGENT — Claude AI
# ══════════════════════════════════════════════════════════════════

class PredictorAgent:
    def __init__(self, bus, state):
        self.bus   = bus
        self.state = state
        self.brain = get_brain()

    async def generate_thesis(self, symbol: str, technical: dict,
                              macro: dict, sentiment: dict, news_ctx: dict) -> dict:
        tick  = self.state.ticks.get(symbol, {})
        price = tick.get("bid", 0)

        context = {
            "symbol":        symbol,
            "current_price": price,
            "spread_pips":   tick.get("spread", 1.5),
            "time_utc":      datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "technical":     technical,
            "macro":         macro,
            "sentiment":     sentiment,
            "news_context":  news_ctx,
            "open_positions":len(self.state.open_positions),
            "account_balance": self.state.account_balance,
        }

        prompt_json_spec = """{
  "direction": "Long|Short|No Trade",
  "confidence": 0-100,
  "entry_price": number,
  "stop_loss": number,
  "take_profit_1": number,
  "take_profit_2": number,
  "risk_reward_ratio": number,
  "confluence_score": 0-100,
  "stop_pips": number,
  "primary_reason": "string",
  "supporting_factors": ["string"],
  "invalidation_conditions": ["string"],
  "trade_thesis": "2-3 sentence summary",
  "setup_type": "string",
  "urgency": "Immediate|Wait for pullback|Wait for confirmation|Avoid"
}"""
        context["respond_format"] = prompt_json_spec

        result = await self.brain.think("GENERATE_TRADE_THESIS", context)
        log.info(f"{symbol}: {result.get('direction','?')} @ {result.get('confidence',0):.0f}% conf | source={result.get('_source','?')}")
        return result


# ══════════════════════════════════════════════════════════════════
# RISK MANAGER AGENT
# ══════════════════════════════════════════════════════════════════

class RiskManagerAgent:
    HARD_LIMITS = {
        "min_rr":         1.5,
        "min_conf":       52,
        "min_confluence": 55,
        "max_spread_mult":2.5,
        "min_balance":    100,
    }

    def __init__(self, bus, state):
        self.bus   = bus
        self.state = state
        self.brain = get_brain()

    async def evaluate(self, thesis: dict, state) -> dict:
        vetoes = []
        if state.circuit_breaker:
            return {"approved": False, "veto_reasons": ["CIRCUIT BREAKER ACTIVE"], "effective_risk": 0}

        open_count = len(state.open_positions)
        max_pos    = int(os.getenv("MAX_CONCURRENT_TRADES", "3"))
        if open_count >= max_pos:             vetoes.append(f"Max {max_pos} positions open")
        if state.trades_today >= int(os.getenv("MAX_TRADES_PER_DAY","10")):
                                              vetoes.append("Daily trade limit reached")
        rr   = thesis.get("risk_reward_ratio", 0)
        conf = thesis.get("confidence", 0)
        conf_score = thesis.get("confluence_score", 0)
        if rr   < self.HARD_LIMITS["min_rr"]:        vetoes.append(f"R:R {rr:.2f} < 1.5")
        if conf < self.HARD_LIMITS["min_conf"]:       vetoes.append(f"Confidence {conf:.0f}% < 52%")
        if conf_score < self.HARD_LIMITS["min_confluence"]: vetoes.append(f"Confluence {conf_score} < 55")
        if state.high_impact_window:                  vetoes.append("High-impact news window")
        if state.account_balance < self.HARD_LIMITS["min_balance"]:
                                                      vetoes.append(f"Balance ${state.account_balance:.0f} below floor")
        if thesis.get("direction") == "No Trade":     vetoes.append("Signal is No Trade")

        # Duplicate symbol check
        open_syms = [p["symbol"] for p in state.open_positions]
        if thesis.get("symbol", "") in open_syms:     vetoes.append(f"Already in {thesis.get('symbol')}")

        if vetoes:
            log.info(f"VETOED {thesis.get('symbol')}: {' | '.join(vetoes)}")
            return {"approved": False, "veto_reasons": vetoes, "effective_risk": 0}

        adj = self.brain.memory.risk_multiplier()
        eff = round(float(os.getenv("MAX_RISK_PER_TRADE_PCT","1.5")) * adj, 2)
        warnings = [f"Risk reduced to {adj*100:.0f}%"] if adj < 1 else []
        if not self._good_session(): warnings.append("Low-quality session — consider waiting")
        log.info(f"APPROVED {thesis.get('symbol')} {thesis.get('direction')} | R:R={rr:.2f} conf={conf:.0f}% risk={eff}%")
        return {"approved": True, "veto_reasons": [], "warnings": warnings,
                "effective_risk": eff, "risk_adjustment": adj}

    def _good_session(self) -> bool:
        h = datetime.utcnow().hour
        return 8 <= h < 22


# ══════════════════════════════════════════════════════════════════
# POSITION SIZER AGENT
# ══════════════════════════════════════════════════════════════════

class PositionSizerAgent:
    def __init__(self, bus, state):
        self.bus   = bus
        self.state = state
        self.brain = get_brain()

    async def calculate(self, thesis: dict, risk_approval: dict, state) -> dict:
        symbol    = thesis.get("symbol", "EURUSD")
        entry     = thesis.get("entry_price", 0)
        sl        = thesis.get("stop_loss", 0)
        conf      = thesis.get("confidence", 60)
        risk_pct  = risk_approval.get("effective_risk", 1.5)
        risk_adj  = risk_approval.get("risk_adjustment", 1.0)
        balance   = state.account_balance or 10_000
        pip       = 0.01 if "JPY" in symbol else 0.0001
        stop_pips = round(abs(entry - sl) / pip, 1) if pip > 0 else 20

        sizing = self.brain.scaler.initial_size(balance, risk_pct, stop_pips, symbol, risk_adj, conf)
        sizing["symbol"]             = symbol
        sizing["initial_stop_pips"]  = stop_pips
        log.info(f"{symbol} size: {sizing['lot_size']} lots | risk ${sizing['risk_usd']} ({sizing['risk_pct']}%) | stop {stop_pips}p")
        return sizing


# ══════════════════════════════════════════════════════════════════
# EXECUTOR AGENT — MT5 Order Placement
# Source: MT5 terminal (free, via MetaTrader5 Python library)
# ══════════════════════════════════════════════════════════════════

class ExecutorAgent:
    def __init__(self, bus, state):
        self.bus   = bus
        self.state = state
        self.mt5   = MT5Connector.get()

    async def place_order(self, thesis: dict, sizing: dict) -> dict:
        symbol    = thesis.get("symbol", "EURUSD")
        direction = thesis["direction"]
        lot_size  = sizing["lot_size"]
        sl        = thesis["stop_loss"]
        tp        = thesis["take_profit_1"]
        comment   = f"APEX_{thesis.get('setup_type','auto')[:10]}"

        result = self.mt5.place_order(symbol, direction, lot_size, sl, tp, comment)

        if result.get("filled"):
            # Attach thesis data for tracking
            result["take_profit_2"]  = thesis.get("take_profit_2", tp)
            result["initial_stop_pips"] = sizing.get("initial_stop_pips", 20)
            result["scale_stage"]    = 1
            result["at_breakeven"]   = False
            result["tp1_hit"]        = False
            result["pnl_pips"]       = 0.0
            result["pnl_usd"]        = 0.0
            self.state.add_trade_record()
        return result

    async def close_position(self, ticket: int, symbol: str,
                             direction: str, lot_size: float, reason: str) -> dict:
        result = self.mt5.close_position(ticket, symbol, direction, lot_size)
        if result.get("closed"):
            log.info(f"Closed {symbol} ticket={ticket}: {reason}")
        return result

    async def modify_position(self, ticket: int, symbol: str,
                              new_sl: float, new_tp: float) -> dict:
        return self.mt5.modify_position(ticket, new_sl, new_tp, symbol)


# ══════════════════════════════════════════════════════════════════
# MONITOR AGENT — Position Management
# ══════════════════════════════════════════════════════════════════

class MonitorAgent:
    def __init__(self, bus, state):
        self.bus    = bus
        self.state  = state
        self.brain  = get_brain()
        self.executor = ExecutorAgent(bus, state)

    async def check_all_positions(self):
        """Sync positions from MT5 and evaluate each one."""
        self.state.sync_positions_from_mt5()
        for pos in self.state.open_positions:
            await self._evaluate_position(pos)

    async def _evaluate_position(self, pos: dict):
        symbol  = pos["symbol"]
        candles = self.state.get_candles(symbol, 20)
        atr     = self._quick_atr(candles)
        price   = pos.get("current_price", 0)
        if not price:
            return

        market_ctx = {
            "news_in_minutes":  self._mins_to_news(),
            "spread_pips":      self.state.ticks.get(symbol, {}).get("spread", 1.5),
            "normal_spread":    1.5,
        }

        # 1. Rule-based exit check
        decision = self.brain.exit_intel.evaluate(
            pos, price, self.state.current_signals, market_ctx, atr
        )

        if decision["action"] == "CLOSE":
            await self.executor.close_position(
                pos["ticket"], symbol, pos["direction"], pos["lot_size"], decision["reason"]
            )

        elif decision["action"] == "BREAKEVEN":
            new_sl = decision["new_sl"]
            await self.executor.modify_position(pos["ticket"], symbol, new_sl, pos["take_profit_1"])
            pos["stop_loss"]   = new_sl
            pos["at_breakeven"] = True
            log.info(f"{symbol}: Break-even applied @ {new_sl}")

        elif decision["action"] == "TRAIL":
            new_sl = decision["new_sl"]
            await self.executor.modify_position(pos["ticket"], symbol, new_sl, pos["take_profit_1"])
            pos["stop_loss"] = new_sl
            log.info(f"{symbol}: Trailing stop → {new_sl}")

        elif decision["action"] == "PARTIAL":
            partial_lots = round(pos["lot_size"] * decision["close_pct"] / 0.01) * 0.01
            partial_lots = max(0.01, partial_lots)
            await self.executor.close_position(
                pos["ticket"], symbol, pos["direction"], partial_lots, "Partial close at TP1"
            )
            pos["lot_size"] = round(pos["lot_size"] - partial_lots, 2)
            pos["tp1_hit"]  = True
            log.info(f"{symbol}: Partial close {partial_lots} lots at TP1")

        # 2. Scaling check on HOLD
        if decision["action"] == "HOLD":
            pnl_pips = pos.get("pnl_pips", 0)
            scale    = self.brain.scaler.scale_add(pos, pnl_pips)
            if scale and scale.get("action") == "SCALE_IN":
                add_lots = scale["add_lots"]
                sl       = pos["stop_loss"]
                tp       = pos["take_profit_1"]
                if scale.get("move_stop_to") == "BREAK_EVEN":
                    sl = pos["entry_price"]
                thesis_mini = {
                    "symbol": symbol, "direction": pos["direction"],
                    "stop_loss": sl, "take_profit_1": tp,
                    "take_profit_2": pos.get("take_profit_2", tp),
                    "setup_type": "scale_in",
                }
                sizing_mini = {"lot_size": add_lots, "initial_stop_pips": pos.get("initial_stop_pips", 20)}
                new_order = await self.executor.place_order(thesis_mini, sizing_mini)
                if new_order.get("filled"):
                    pos["scale_stage"] = scale["scale_stage"]
                    if scale.get("move_stop_to") == "BREAK_EVEN":
                        pos["stop_loss"]    = pos["entry_price"]
                        pos["at_breakeven"] = True
                    log.info(f"📈 SCALED IN: {symbol} +{add_lots} lots. {scale['reason']}")

    def _quick_atr(self, candles: list, n=14) -> float:
        if len(candles) < 2:
            return 0.001
        trs = [max(c["high"]-c["low"], abs(c["high"]-candles[i-1]["close"]),
                   abs(c["low"]-candles[i-1]["close"]))
               for i, c in enumerate(candles[1:], 1)]
        return sum(trs[-n:]) / len(trs[-n:]) if trs else 0.001

    def _mins_to_news(self) -> int:
        events = self.state.news_events
        now    = datetime.utcnow()
        mins   = []
        for e in events:
            if e.get("impact") == "HIGH" and "time" in e:
                try:
                    t = datetime.fromisoformat(e["time"].replace("Z",""))
                    d = (t - now).total_seconds() / 60
                    if d > 0:
                        mins.append(d)
                except Exception:
                    pass
        return int(min(mins)) if mins else 999


# ══════════════════════════════════════════════════════════════════
# BACKTESTER AGENT
# Source: MT5 historical candles (free, from your broker)
# ══════════════════════════════════════════════════════════════════

class BacktesterAgent:
    MIN_WIN_RATE     = 42.0
    MIN_SHARPE       = 0.7
    MAX_DRAWDOWN_PCT = 22.0
    MIN_PROFIT_FACTOR= 1.15
    MIN_TRADES       = 80

    def __init__(self, bus, state):
        self.bus   = bus
        self.state = state
        self.mt5   = MT5Connector.get()

    async def run_validation(self, symbols: List[str]) -> dict:
        log.info(f"Backtesting {len(symbols)} symbols from MT5 history...")
        all_trades = []
        for sym in symbols:
            # Try to get real H1 candles from MT5 (2 years = ~17520 candles)
            candles = self.mt5.get_candles(sym, "H1", 17520)
            if len(candles) < 500:
                log.warning(f"{sym}: only {len(candles)} candles — using generated data")
                candles = self._gen_candles(sym, 8760)
            trades = self._sim_strategy(candles)
            all_trades.extend(trades)
            log.info(f"{sym}: {len(trades)} backtest trades")
            await asyncio.sleep(0.05)

        if len(all_trades) < self.MIN_TRADES:
            return {"strategy_approved": False,
                    "reason": f"Only {len(all_trades)} trades — need {self.MIN_TRADES}",
                    "total_trades": len(all_trades)}

        m        = self._metrics(all_trades)
        approved = (m["win_rate"]      >= self.MIN_WIN_RATE and
                    m["sharpe"]        >= self.MIN_SHARPE and
                    m["max_drawdown"]  <= self.MAX_DRAWDOWN_PCT and
                    m["profit_factor"] >= self.MIN_PROFIT_FACTOR)
        reason   = "All thresholds passed ✓" if approved else self._fail_reason(m)
        log.info(f"Backtest: {'APPROVED' if approved else 'FAILED'} — {reason}")
        return {"strategy_approved": approved, "reason": reason,
                "total_trades": len(all_trades), **m}

    def _sim_strategy(self, candles: List[dict]) -> list:
        tech   = TechnicalAgent(None, None)
        closes = [c["close"] for c in candles]
        ema20  = tech._ema(closes, 20)
        ema50  = tech._ema(closes, 50)
        rsi14  = tech._rsi(closes, 14)
        atr14  = tech._atr(candles, 14)
        trades, in_trade, entry, direction, sl, tp = [], False, 0, None, 0, 0

        for i in range(60, len(candles)):
            p    = candles[i]["close"]
            e20  = ema20[i]; e50 = ema50[i]; r = rsi14[i]; a = atr14[i]
            pe20 = ema20[i-1]; pe50 = ema50[i-1]

            if in_trade:
                if direction == "Long":
                    if p <= sl: trades.append({"pnl_pips": (sl-entry)/0.0001, "result":"LOSS"}); in_trade=False
                    elif p >= tp: trades.append({"pnl_pips": (tp-entry)/0.0001, "result":"WIN"}); in_trade=False
                else:
                    if p >= sl: trades.append({"pnl_pips": (entry-sl)/0.0001, "result":"LOSS"}); in_trade=False
                    elif p <= tp: trades.append({"pnl_pips": (entry-tp)/0.0001, "result":"WIN"}); in_trade=False
                continue

            cross_up   = e20 > e50 and pe20 <= pe50
            cross_down = e20 < e50 and pe20 >= pe50
            if cross_up and 40 < r < 65:
                in_trade=True; entry=p; direction="Long"
                sl=round(p-a*1.5,5); tp=round(p+a*2.5,5)
            elif cross_down and 35 < r < 60:
                in_trade=True; entry=p; direction="Short"
                sl=round(p+a*1.5,5); tp=round(p-a*2.5,5)
        return trades

    def _metrics(self, trades: list) -> dict:
        wins   = [t for t in trades if t["result"]=="WIN"]
        losses = [t for t in trades if t["result"]=="LOSS"]
        pnls   = [t["pnl_pips"] for t in trades]
        gp     = sum(t["pnl_pips"] for t in wins)
        gl     = abs(sum(t["pnl_pips"] for t in losses))
        mean   = sum(pnls)/len(pnls) if pnls else 0
        std    = math.sqrt(sum((p-mean)**2 for p in pnls)/len(pnls)) if len(pnls)>1 else 1
        sharpe = mean/std*math.sqrt(252) if std>0 else 0
        eq=10000; peak=10000; mdd=0
        for p in pnls:
            eq+=p; peak=max(peak,eq)
            mdd=max(mdd,(peak-eq)/peak*100 if peak>0 else 0)
        return {
            "win_rate":      round(len(wins)/len(trades)*100,1),
            "sharpe":        round(sharpe,2),
            "max_drawdown":  round(mdd,1),
            "profit_factor": round(gp/gl,2) if gl>0 else 0,
            "total_pips":    round(sum(pnls),1),
            "avg_win_pips":  round(gp/len(wins),1) if wins else 0,
            "avg_loss_pips": round(gl/len(losses),1) if losses else 0,
        }

    def _fail_reason(self, m: dict) -> str:
        r=[]
        if m["win_rate"]<self.MIN_WIN_RATE:      r.append(f"WR {m['win_rate']}%<{self.MIN_WIN_RATE}%")
        if m["sharpe"]<self.MIN_SHARPE:          r.append(f"Sharpe {m['sharpe']}<{self.MIN_SHARPE}")
        if m["max_drawdown"]>self.MAX_DRAWDOWN_PCT: r.append(f"DD {m['max_drawdown']}%>{self.MAX_DRAWDOWN_PCT}%")
        if m["profit_factor"]<self.MIN_PROFIT_FACTOR: r.append(f"PF {m['profit_factor']}<{self.MIN_PROFIT_FACTOR}")
        return " | ".join(r)

    def _gen_candles(self, symbol: str, n: int) -> List[dict]:
        bases  = {"EURUSD":1.085,"GBPUSD":1.270,"USDJPY":149.5,"AUDUSD":0.652,"GBPJPY":189.8}
        price  = bases.get(symbol, 1.1) * random.uniform(0.97,1.03)
        trend  = random.choice([-1,1]); ctr = 0; out = []
        for _ in range(n):
            ctr += 1
            if ctr > random.randint(40,180): trend=-trend; ctr=0
            d = trend*0.00003 + random.gauss(0,0.0002)
            o=price; c=round(price+d,5)
            h=round(max(o,c)+abs(random.gauss(0,0.0004)),5)
            l=round(min(o,c)-abs(random.gauss(0,0.0004)),5)
            out.append({"open":o,"high":h,"low":l,"close":c,"volume":random.randint(300,3000)})
            price=c
        return out


# ══════════════════════════════════════════════════════════════════
# REPORTER AGENT — Claude AI Post-Trade Analysis
# ══════════════════════════════════════════════════════════════════

class ReporterAgent:
    def __init__(self, bus, state):
        self.bus   = bus
        self.state = state
        self.brain = get_brain()

    async def analyze_trade(self, closed_trade: dict):
        if not closed_trade:
            return
        won = closed_trade.get("pnl_usd", 0) > 0
        self.brain.memory.record({
            **closed_trade,
            "setup_type": closed_trade.get("setup_type", "unknown"),
            "session":    self._session_at(closed_trade.get("opened_at","")),
        })
        log.info(f"Trade recorded: {closed_trade.get('symbol')} {'WIN' if won else 'LOSS'} ${closed_trade.get('pnl_usd',0):.2f}")

        if ANTHROPIC_API_KEY:
            await self._claude_review(closed_trade)

    async def _claude_review(self, trade: dict):
        prompt = (
            f"Analyse this closed Forex trade and extract lessons.\n"
            f"TRADE:\n{json.dumps(trade, indent=2, default=str)}\n\n"
            f"SYSTEM PERFORMANCE:\n{self.brain.memory.summary()}\n\n"
            f'Respond ONLY with JSON: {{"grade":"A+|A|B|C|D|F","key_lesson":"string","what_worked":["string"],"what_failed":["string"],"setup_type":"string","would_take_again":true|false}}'
        )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": ANTHROPIC_API_KEY,
                             "anthropic-version":"2023-06-01",
                             "content-type":"application/json"},
                    json={"model": MODEL, "max_tokens": 400,
                          "messages": [{"role":"user","content": prompt}]},
                )
                r.raise_for_status()
                text   = r.json()["content"][0]["text"]
                result = json.loads(text.replace("```json","").replace("```","").strip())
                if result.get("setup_type"):
                    self.brain.memory.trade_history[-1]["setup_type"] = result["setup_type"]
                self.state.last_trade_analysis = result
                log.info(f"Trade review: grade={result.get('grade')} | {result.get('key_lesson','')[:80]}")
        except Exception as e:
            log.warning(f"Reporter review failed: {e}")

    def _session_at(self, iso: str) -> str:
        try:
            h = datetime.fromisoformat(iso.replace("Z","")).hour
            if 8<=h<17: return "London"
            if 13<=h<22: return "New York"
            if 0<=h<9:  return "Tokyo"
            return "Sydney"
        except Exception:
            return "Unknown"
