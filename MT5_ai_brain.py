"""
APEX FOREX — AI Brain (MT5 Edition)
All intelligence logic unchanged from original design.
The Brain doesn't care whether data comes from OANDA or MT5 —
it only works on processed signals passed by the agents.
Full trading knowledge, memory, scaler, exit intelligence.
"""

import os, json, math, asyncio, logging
from datetime import datetime
from typing import List, Dict, Optional
import httpx

log    = logging.getLogger("AIBrain")
MODEL  = "claude-sonnet-4-20250514"
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

TRADING_KNOWLEDGE = """
You are the AI brain of APEX FOREX — an autonomous Forex trading system.

═══ CORE RULES (never break these) ═══
1. Capital preservation FIRST. Profit second.
2. Never risk more than 1.5% of account on any single trade.
3. Never enter without defined SL and TP.
4. Never trade ±15 minutes around HIGH impact news.
5. Always trade WITH the higher timeframe trend.
6. Minimum R:R = 1.5:1. Target 2:1. Ideal 3:1.
7. Cut losses fast. Let winners run with trailing stops.
8. Scale IN on winners. NEVER add to losers.
9. After 2 consecutive losses: reduce size 50%.
10. Max 3 open positions simultaneously.

═══ SESSION TIMING ═══
Best: London open (08:00–10:00 GMT) + NY open (13:00–15:00 GMT)
Best overlap: London/NY (13:00–17:00 GMT) — highest volume
Avoid: Asian session for EUR/GBP pairs (low volume, choppy)
Avoid: Friday after 20:00 GMT (gap risk over weekend)

═══ POSITION SIZING — MICRO LOT PYRAMID ═══
Start: 0.01 lots (micro) = $0.10/pip
Stage 2 add: when profit ≥ 75% of stop distance → add 0.01 lots, move SL to break-even
Stage 3 add: when profit ≥ 1.5× stop distance → add 0.01 lots, trail aggressively
Worst case after Stage 2: break-even. Best case: 3× normal profit.

═══ ATR-BASED STOPS (always) ═══
Normal:    SL = Entry ± ATR × 1.5
Volatile:  SL = Entry ± ATR × 2.0
Tight:     SL = Entry ± ATR × 1.0
Place stop BEYOND nearest structure (swing high/low).

═══ ENTRY SIGNALS — HIGH CONFIDENCE (use these) ═══
1. EMA Crossover + RSI: 20 EMA crosses 50 EMA, RSI 40-60, ADX > 20
2. S/R Break + Retest: Break key level, wait for retest, enter on rejection
3. London Breakout: Mark 06:00-07:45 range, trade 08:00 breakout
4. Trend Pullback: ADX > 25, price retraces to 38-62% Fibonacci, RSI returning

═══ CONFLUENCE SCORE ═══
8-10 signals → STRONG  → full size 0.01 lots, extended TP (3:1)
6-7  signals → GOOD    → full size 0.01 lots, normal TP (2:1)
4-5  signals → FAIR    → half size, conservative TP (1.5:1)
0-3  signals → SKIP    → do not trade

═══ EXIT RULES ═══
Close IMMEDIATELY: news in <10min, spread 3× normal, strong opposing signal, Friday 20:00
Move to break-even: when profit = stop distance
Partial close (40%): when TP1 hit
Tighten trail: when profit ≥ 2× stop distance
Hold: during normal retracements (20-40% pullback is healthy)
"""


class TradeMemory:
    def __init__(self):
        self.trade_history:       List[dict] = []
        self.pattern_performance: Dict[str, dict] = {}
        self.pair_performance:    Dict[str, dict] = {}
        self.session_performance: Dict[str, dict] = {}
        self.consecutive_losses = 0
        self.consecutive_wins   = 0

    def record(self, trade: dict):
        self.trade_history.append({**trade, "ts": datetime.utcnow().isoformat()})
        won     = trade.get("pnl_usd", 0) > 0
        setup   = trade.get("setup_type", "unknown")
        pair    = trade.get("symbol", "unknown")
        session = trade.get("session", "unknown")

        if won:
            self.consecutive_wins += 1; self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1; self.consecutive_wins = 0

        for key, bucket in [(setup, self.pattern_performance),
                             (pair,  self.pair_performance),
                             (session, self.session_performance)]:
            if key not in bucket:
                bucket[key] = {"wins": 0, "losses": 0, "total_pnl": 0}
            bucket[key]["wins" if won else "losses"] += 1
            bucket[key]["total_pnl"] += trade.get("pnl_usd", 0)

    def risk_multiplier(self) -> float:
        if self.consecutive_losses >= 3: return 0.25
        if self.consecutive_losses == 2: return 0.50
        if self.consecutive_losses == 1: return 0.75
        if self.consecutive_wins   >= 7: return 0.80
        return 1.0

    def summary(self) -> str:
        recent = self.trade_history[-20:]
        if not recent:
            return "No trade history yet."
        wins = sum(1 for t in recent if t.get("pnl_usd", 0) > 0)
        pnl  = sum(t.get("pnl_usd", 0) for t in recent)
        streak_type = "win" if self.consecutive_wins > 0 else "loss"
        streak_n    = max(self.consecutive_wins, self.consecutive_losses)
        worst = []
        for pair, s in self.pair_performance.items():
            total = s["wins"] + s["losses"]
            if total >= 5 and s["total_pnl"] < -30:
                worst.append(f"{pair}(${s['total_pnl']:.0f})")
        return (
            f"Last {len(recent)} trades: {wins/len(recent)*100:.0f}% win rate, ${pnl:.2f} P&L. "
            f"Current {streak_n}-{streak_type} streak. Risk multiplier: {self.risk_multiplier():.0%}. "
            f"Underperforming: {', '.join(worst) if worst else 'none'}."
        )


class PositionScaler:
    PIP_VALUES = {
        "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0,
        "USDCHF": 10.0, "USDJPY": 6.7,  "GBPJPY": 6.7,
    }

    def initial_size(self, balance: float, risk_pct: float, stop_pips: float,
                     symbol: str, risk_adj: float = 1.0, confidence: float = 70) -> dict:
        pip_val  = self.PIP_VALUES.get(symbol, 10.0)
        eff_risk = (balance * risk_pct / 100) * risk_adj
        conf_mod = 0.7 if confidence < 60 else (1.0 if confidence < 75 else 1.15)
        stop_pips = max(stop_pips, 5)  # minimum 5 pip stop
        raw_lots  = (eff_risk * conf_mod) / (stop_pips * pip_val)
        lot_size  = max(0.01, min(0.50, round(raw_lots / 0.01) * 0.01))
        return {
            "lot_size":    lot_size,
            "units":       int(lot_size * 100_000),
            "risk_usd":    round(lot_size * stop_pips * pip_val, 2),
            "risk_pct":    round(lot_size * stop_pips * pip_val / balance * 100, 2),
            "stop_pips":   stop_pips,
            "scale_stage": 1,
        }

    def scale_add(self, position: dict, pnl_pips: float) -> Optional[dict]:
        stage       = position.get("scale_stage", 1)
        init_stop   = position.get("initial_stop_pips", 20)
        lot         = position.get("lot_size", 0.01)
        if pnl_pips <= 0:
            return None
        if stage == 1 and pnl_pips >= init_stop * 0.75:
            return {"action": "SCALE_IN", "add_lots": max(0.01, round(lot*0.5/0.01)*0.01),
                    "scale_stage": 2, "move_stop_to": "BREAK_EVEN",
                    "reason": f"Up {pnl_pips:.1f}p — adding 50% position, stop to break-even"}
        if stage == 2 and pnl_pips >= init_stop * 1.5:
            return {"action": "SCALE_IN", "add_lots": 0.01,
                    "scale_stage": 3, "move_stop_to": "TRAILING",
                    "reason": f"Up {pnl_pips:.1f}p — final micro add, trailing aggressively"}
        return None


class ExitIntelligence:
    def evaluate(self, position: dict, current_price: float,
                 signals: dict, market: dict, atr: float) -> dict:
        direction   = position["direction"]
        entry       = position["entry_price"]
        sl          = position["stop_loss"]
        tp1         = position["take_profit_1"]
        pnl_pips    = position.get("pnl_pips", 0)
        symbol      = position["symbol"]
        pip         = 0.01 if "JPY" in symbol else 0.0001
        init_stop_p = abs(entry - sl) / pip if pip > 0 else 20

        # ── Immediate close ────────────────────────────────────────
        if market.get("news_in_minutes", 999) < 10:
            return {"action": "CLOSE", "reason": "High-impact news in <10 min"}
        if market.get("spread_pips", 0) > market.get("normal_spread", 2) * 3:
            return {"action": "CLOSE", "reason": "Spread 3× normal — liquidity issue"}
        sig = signals.get(symbol, {})
        if ((direction == "Long"  and sig.get("direction") == "Short" and sig.get("confidence", 0) > 80) or
            (direction == "Short" and sig.get("direction") == "Long"  and sig.get("confidence", 0) > 80)):
            return {"action": "CLOSE", "reason": "Strong opposing signal"}
        now = datetime.utcnow()
        if now.weekday() == 4 and now.hour >= 20:
            return {"action": "CLOSE", "reason": "Friday close — avoid weekend gap"}

        # ── Break-even ─────────────────────────────────────────────
        if pnl_pips >= init_stop_p and not position.get("at_breakeven"):
            return {"action": "BREAKEVEN", "new_sl": entry,
                    "reason": f"Profit={pnl_pips:.1f}p equals risk — free trade"}

        # ── Partial close at TP1 ───────────────────────────────────
        if not position.get("tp1_hit"):
            tp_dist  = abs(tp1 - entry)
            cur_dist = abs(current_price - entry)
            if tp_dist > 0 and cur_dist / tp_dist >= 1.0:
                return {"action": "PARTIAL", "close_pct": 0.40, "tp1_hit": True,
                        "reason": "TP1 reached — closing 40%, letting 60% run"}

        # ── Trailing stop ──────────────────────────────────────────
        if pnl_pips >= init_stop_p * 2:
            trail = atr * 0.8
            if direction == "Long":
                new_sl = round(current_price - trail, 5)
                if new_sl > sl:
                    return {"action": "TRAIL", "new_sl": new_sl,
                            "reason": f"Strong profit {pnl_pips:.1f}p — trailing stop up"}
            else:
                new_sl = round(current_price + trail, 5)
                if new_sl < sl:
                    return {"action": "TRAIL", "new_sl": new_sl,
                            "reason": f"Strong profit {pnl_pips:.1f}p — trailing stop down"}

        return {"action": "HOLD", "reason": f"Trade healthy at {pnl_pips:.1f}p"}


class AIBrain:
    def __init__(self):
        self.memory    = TradeMemory()
        self.scaler    = PositionScaler()
        self.exit_intel = ExitIntelligence()

    async def think(self, task: str, context: dict) -> dict:
        """Main reasoning — Claude AI with rule-based fallback."""
        if not API_KEY:
            return self._rules(task, context)

        system = f"{TRADING_KNOWLEDGE}\n\nPERFORMANCE: {self.memory.summary()}\n\nRespond ONLY with valid JSON."
        prompt = f"TASK: {task}\n\nCONTEXT:\n{json.dumps(context, indent=2, default=str)}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json={"model": MODEL, "max_tokens": 1200,
                          "system": system, "messages": [{"role": "user", "content": prompt}]},
                )
                r.raise_for_status()
                text  = r.json()["content"][0]["text"]
                clean = text.replace("```json","").replace("```","").strip()
                result = json.loads(clean)
                result["_source"] = "claude_ai"
                return result
        except Exception as e:
            log.warning(f"Claude fallback ({e})")
            result = self._rules(task, context)
            result["_source"] = "rule_based"
            return result

    def _rules(self, task: str, ctx: dict) -> dict:
        if task == "GENERATE_TRADE_THESIS": return self._thesis(ctx)
        if task == "EVALUATE_EXIT":         return self.exit_intel.evaluate(
            ctx["position"], ctx["current_price"], ctx["signals"], ctx["market"], ctx["atr"])
        if task == "EVALUATE_RISK":         return self._risk(ctx)
        return {"decision": "PASS", "reason": "Unknown task"}

    def _thesis(self, ctx: dict) -> dict:
        tech   = ctx.get("technical", {})
        score  = tech.get("technical_score", 0)
        dir_   = tech.get("trend_direction", "Neutral")
        atr    = tech.get("atr", 0.001)
        price  = ctx.get("current_price", 1.0)
        symbol = ctx.get("symbol", "EURUSD")
        if score < 60 or dir_ == "Neutral" or ctx.get("news_context", {}).get("high_impact_window"):
            return {"direction": "No Trade", "confidence": 0, "confluence_score": score,
                    "trade_thesis": "No Trade — insufficient confluence or news risk"}
        is_long = dir_ == "Bullish"
        sl_d, tp_d = atr * 1.5, atr * 2.5
        pip = 0.01 if "JPY" in symbol else 0.0001
        return {
            "direction":   "Long" if is_long else "Short",
            "confidence":  min(score, 72),
            "entry_price": price,
            "stop_loss":   round(price - sl_d if is_long else price + sl_d, 5),
            "take_profit_1": round(price + tp_d if is_long else price - tp_d, 5),
            "take_profit_2": round(price + tp_d*1.6 if is_long else price - tp_d*1.6, 5),
            "risk_reward_ratio": round(tp_d / sl_d, 2),
            "confluence_score": score,
            "stop_pips":   round(sl_d / pip, 1),
            "trade_thesis": f"Rule-based: {dir_} EMA alignment, score {score}",
            "setup_type":   "ema_trend_fallback",
            "urgency":      "Wait for confirmation",
        }

    def _risk(self, ctx: dict) -> dict:
        thesis = ctx.get("thesis", {})
        vetoes = []
        if ctx.get("open_count", 0) >= int(os.getenv("MAX_CONCURRENT_TRADES", "3")):
            vetoes.append("Max concurrent positions reached")
        if thesis.get("risk_reward_ratio", 0) < 1.5:
            vetoes.append(f"R:R {thesis.get('risk_reward_ratio', 0):.2f} < 1.5")
        if thesis.get("confidence", 0) < 52:
            vetoes.append(f"Confidence {thesis.get('confidence', 0)}% < 52%")
        if thesis.get("confluence_score", 0) < 55:
            vetoes.append(f"Confluence {thesis.get('confluence_score', 0)} < 55")
        if ctx.get("high_impact_news"):
            vetoes.append("High-impact news window")
        if thesis.get("direction") == "No Trade":
            vetoes.append("AI returned No Trade")
        adj = self.memory.risk_multiplier()
        return {
            "approved":        len(vetoes) == 0,
            "veto_reasons":    vetoes,
            "risk_adjustment": adj,
            "effective_risk":  round(float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.5")) * adj, 2),
            "warnings":        [f"Risk at {adj*100:.0f}% due to loss streak"] if adj < 1 else [],
        }


_brain: Optional[AIBrain] = None
def get_brain() -> AIBrain:
    global _brain
    if _brain is None:
        _brain = AIBrain()
    return _brain
