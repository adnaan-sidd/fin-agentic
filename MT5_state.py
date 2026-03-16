"""
APEX FOREX — System State (MT5 Edition)
Single source of truth. All agents read and write here.
Thread-safe. Updated in real-time by the monitor loop.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from core.mt5_connector import MT5Connector


class SystemState:
    def __init__(self):
        # System
        self.system_status          = "STARTING"
        self.started_at             = datetime.utcnow()
        self.circuit_breaker        = False

        # Agent heartbeats
        self.agent_statuses: Dict[str, str] = {
            "price_feed": "IDLE", "news_event": "IDLE", "sentiment": "IDLE",
            "validator":  "IDLE", "technical":  "IDLE", "macro":     "IDLE",
            "backtester": "IDLE", "predictor":  "IDLE", "risk":      "IDLE",
            "sizer":      "IDLE", "executor":   "IDLE", "monitor":   "IDLE",
            "reporter":   "IDLE",
        }

        # Market data (symbol → data)
        self.ticks:   Dict[str, dict] = {}
        self.candles: Dict[str, List[dict]] = {}

        # News
        self.news_events:       List[dict] = []
        self.high_impact_window: bool      = False

        # Signals
        self.current_signals: Dict[str, dict] = {}

        # Positions (synced from MT5 every 5s)
        self.open_positions:   List[dict] = []
        self.closed_positions: List[dict] = []

        # Account (synced from MT5 every 5s)
        self.account_balance:   float = 0.0
        self.account_equity:    float = 0.0
        self.account_margin:    float = 0.0
        self.starting_balance:  float = 0.0

        # Performance
        self.equity_curve: List[dict] = []
        self.daily_pnl:    float      = 0.0
        self.trades_today: int        = 0

        # Backtest results
        self.backtest_results: dict = {}
        self.last_trade_analysis: Optional[dict] = None

    # ── Agent status ──────────────────────────────────────────────

    def set_agent_status(self, agent: str, status: str):
        self.agent_statuses[agent] = status

    def set_system_status(self, status: str):
        self.system_status = status

    # ── Market data ───────────────────────────────────────────────

    def update_tick(self, symbol: str, tick: dict):
        self.ticks[symbol] = tick

    def add_candles(self, symbol: str, candles: List[dict]):
        """Replace candle list for symbol (MT5 returns full history)."""
        self.candles[symbol] = candles[-500:]

    def get_candles(self, symbol: str, count: int = 200) -> List[dict]:
        return self.candles.get(symbol, [])[-count:]

    # ── News ──────────────────────────────────────────────────────

    def set_news_events(self, events: List[dict]):
        self.news_events = events
        now = datetime.utcnow()
        self.high_impact_window = any(
            e.get("impact") == "HIGH" and "time" in e and
            abs((datetime.fromisoformat(e["time"].replace("Z","")) - now).total_seconds()) < 900
            for e in events
        )

    def get_news_context(self) -> dict:
        return {
            "events":              self.news_events[-5:],
            "high_impact_window":  self.high_impact_window,
        }

    # ── Signals ───────────────────────────────────────────────────

    def update_signal(self, symbol: str, thesis: dict):
        self.current_signals[symbol] = {**thesis, "ts": datetime.utcnow().isoformat()}

    # ── Positions (synced from MT5) ───────────────────────────────

    def sync_positions_from_mt5(self):
        """Pull live positions directly from MT5 terminal."""
        connector   = MT5Connector.get()
        mt5_positions = connector.get_open_positions()

        # Detect closed positions
        prev_tickets = {p["ticket"] for p in self.open_positions if "ticket" in p}
        curr_tickets = {p["ticket"] for p in mt5_positions}
        newly_closed = prev_tickets - curr_tickets

        for ticket in newly_closed:
            pos = next((p for p in self.open_positions if p.get("ticket") == ticket), None)
            if pos:
                pos["closed_at"] = datetime.utcnow().isoformat()
                self.closed_positions.append(pos)
                self.daily_pnl += pos.get("pnl_usd", 0)

        self.open_positions = mt5_positions

        # Update circuit breaker
        if self.account_balance > 0:
            dd_pct = abs(self.daily_pnl) / self.account_balance * 100
            if self.daily_pnl < 0 and dd_pct >= float(
                __import__("os").getenv("MAX_DAILY_DRAWDOWN_PCT", "5.0")
            ):
                self.circuit_breaker = True

    def add_trade_record(self):
        self.trades_today += 1
        if self.trades_today >= int(__import__("os").getenv("MAX_TRADES_PER_DAY", "10")):
            self.circuit_breaker = True

    # ── Account ───────────────────────────────────────────────────

    def update_account(self, balance: float, equity: float, margin: float):
        if self.starting_balance == 0 and balance > 0:
            self.starting_balance = balance
        self.account_balance = balance
        self.account_equity  = equity
        self.account_margin  = margin
        self.equity_curve.append({
            "ts":      datetime.utcnow().isoformat(),
            "equity":  equity,
            "balance": balance,
        })
        if len(self.equity_curve) > 1440:
            self.equity_curve = self.equity_curve[-1440:]

    def update_backtest(self, results: dict):
        self.backtest_results = results

    # ── Performance ───────────────────────────────────────────────

    def get_performance(self) -> dict:
        all_closed = self.closed_positions
        if not all_closed:
            return {
                "total_trades": 0, "win_rate": 0, "total_pnl": 0,
                "daily_pnl": self.daily_pnl, "profit_factor": 0,
                "max_drawdown": 0, "avg_win": 0, "avg_loss": 0,
                "open_trades": len(self.open_positions),
            }
        wins   = [t for t in all_closed if t.get("pnl_usd", 0) > 0]
        losses = [t for t in all_closed if t.get("pnl_usd", 0) <= 0]
        gp     = sum(t["pnl_usd"] for t in wins)
        gl     = abs(sum(t["pnl_usd"] for t in losses))

        eq  = [e["equity"] for e in self.equity_curve]
        mdd = 0
        if eq:
            peak = eq[0]
            for e in eq:
                peak = max(peak, e)
                dd   = (peak - e) / peak * 100 if peak > 0 else 0
                mdd  = max(mdd, dd)

        return {
            "total_trades":  len(all_closed),
            "open_trades":   len(self.open_positions),
            "win_rate":      len(wins) / len(all_closed) * 100,
            "total_pnl":     sum(t.get("pnl_usd", 0) for t in all_closed),
            "daily_pnl":     self.daily_pnl,
            "profit_factor": gp / gl if gl > 0 else 0,
            "max_drawdown":  round(mdd, 2),
            "avg_win":       gp / len(wins) if wins else 0,
            "avg_loss":      gl / len(losses) if losses else 0,
        }

    # ── Dashboard payload ─────────────────────────────────────────

    def get_dashboard_payload(self) -> dict:
        return {
            "ts":              datetime.utcnow().isoformat(),
            "system_status":   self.system_status,
            "circuit_breaker": self.circuit_breaker,
            "agent_statuses":  self.agent_statuses.copy(),
            "account": {
                "balance":           self.account_balance,
                "equity":            self.account_equity,
                "margin":            self.account_margin,
                "starting_balance":  self.starting_balance,
            },
            "open_positions":   self.open_positions.copy(),
            "closed_positions": self.closed_positions[-20:],
            "equity_curve":     self.equity_curve[-120:],
            "current_signals":  self.current_signals.copy(),
            "performance":      self.get_performance(),
            "backtest":         self.backtest_results,
            "high_impact_news": self.high_impact_window,
            "ticks":            self.ticks.copy(),
        }
