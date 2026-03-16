"""
APEX FOREX — Master Orchestrator (MT5 Edition)
══════════════════════════════════════════════════════════════
Entry point. Run this file to start the entire system.

    python orchestrator.py

What happens on startup:
  1. Connects to MT5 terminal (PU Prime or any MT5 broker)
  2. Runs backtest validation on 2yr MT5 candle history
  3. If backtest passes → starts autonomous trading
  4. Four async loops run forever:
       • Data loop      (every 5s)  — fetches ticks + candles
       • Analysis loop  (every 60s) — runs all 7 analysis agents
       • Monitor loop   (every 5s)  — manages open positions
       • Broadcast loop (every 1s)  — streams state to dashboard

Prerequisites:
  • MT5 terminal open and logged into PU Prime
  • Algo Trading enabled in MT5 (Tools → Options → Expert Advisors)
  • .env file configured (copy .env.example → .env, fill in values)
  • pip install -r requirements.txt
══════════════════════════════════════════════════════════════
"""

import asyncio
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from core.mt5_connector import MT5Connector
from core.state         import SystemState
from core.data_sources  import (MessageBus, NewsEventAgent,
                                SentimentAgent, MacroAgent, ValidatorAgent)
from core.ai_brain      import get_brain
from core.websocket_server import WebSocketServer
from agents.all_agents  import (PriceFeedAgent, TechnicalAgent,
                                PredictorAgent, RiskManagerAgent,
                                PositionSizerAgent, ExecutorAgent,
                                MonitorAgent, BacktesterAgent, ReporterAgent)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)-18s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Orchestrator")

# ── Configuration from .env ───────────────────────────────────────
SYMBOLS          = os.getenv("TRADING_PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,GBPJPY").split(",")
DATA_INTERVAL    = 5     # seconds between tick fetches
ANALYSIS_INTERVAL= 60    # seconds between full analysis cycles
MONITOR_INTERVAL = 5     # seconds between position checks
BACKTEST_ON_START= True  # validate strategy before trading


class Orchestrator:
    def __init__(self):
        self.bus      = MessageBus()
        self.state    = SystemState()
        self.ws       = WebSocketServer(self.state)
        self.running  = False
        self.mt5      = MT5Connector.get()

        # Instantiate all agents
        self.price_feed = PriceFeedAgent(self.bus, self.state)
        self.news       = NewsEventAgent(self.bus, self.state)
        self.sentiment  = SentimentAgent(self.bus, self.state)
        self.validator  = ValidatorAgent(self.bus, self.state)
        self.technical  = TechnicalAgent(self.bus, self.state)
        self.macro      = MacroAgent(self.bus, self.state)
        self.backtester = BacktesterAgent(self.bus, self.state)
        self.predictor  = PredictorAgent(self.bus, self.state)
        self.risk       = RiskManagerAgent(self.bus, self.state)
        self.sizer      = PositionSizerAgent(self.bus, self.state)
        self.executor   = ExecutorAgent(self.bus, self.state)
        self.monitor    = MonitorAgent(self.bus, self.state)
        self.reporter   = ReporterAgent(self.bus, self.state)

    # ── Startup ───────────────────────────────────────────────────

    async def startup(self) -> bool:
        log.info("=" * 60)
        log.info("  APEX FOREX  —  MT5 Autonomous Trading System")
        log.info("=" * 60)
        self.state.set_system_status("STARTING")

        # 1. Connect to MT5
        log.info("Connecting to MT5 terminal...")
        ok = self.mt5.connect()
        if not ok:
            log.warning("MT5 not connected — running in MOCK mode (no real trades)")

        # 2. Fetch account info
        acct = await self.price_feed.fetch_account()
        if acct:
            log.info(f"Account: ${acct.get('balance',0):.2f} balance | {acct.get('leverage',0)}x leverage | {acct.get('currency','USD')}")

        # 3. Warm up candle history for all symbols
        log.info(f"Loading candle history for {len(SYMBOLS)} pairs...")
        for sym in SYMBOLS:
            self.state.set_agent_status("price_feed", "RUNNING")
            await self.price_feed.fetch_candles(sym, "M15", 500)
            await self.price_feed.fetch_candles(sym, "H1",  200)
            await asyncio.sleep(0.2)
        self.state.set_agent_status("price_feed", "IDLE")
        log.info("Candle history loaded ✓")

        # 4. Backtest validation
        if BACKTEST_ON_START:
            log.info("Running backtest validation on MT5 historical data...")
            self.state.set_agent_status("backtester", "RUNNING")
            results = await self.backtester.run_validation(SYMBOLS)
            self.state.set_agent_status("backtester", "IDLE")
            self.state.update_backtest(results)

            if not results["strategy_approved"]:
                log.error(f"Backtest FAILED: {results['reason']}")
                log.error("System will NOT trade until strategy passes backtest.")
                self.state.set_system_status("HALTED — BACKTEST FAILED")
                return False

            log.info(f"Backtest PASSED ✓  WR={results.get('win_rate')}%  "
                     f"Sharpe={results.get('sharpe')}  DD={results.get('max_drawdown')}%  "
                     f"PF={results.get('profit_factor')}")

        self.state.set_system_status("LIVE")
        log.info("System LIVE — autonomous trading active")
        return True

    # ── Data Loop (every 5s) ──────────────────────────────────────

    async def data_loop(self):
        while self.running:
            try:
                self.state.set_agent_status("price_feed", "RUNNING")
                for sym in SYMBOLS:
                    tick = await self.price_feed.fetch_tick(sym)
                    if not tick:
                        log.warning(f"No tick for {sym}")

                # Refresh account balance
                await self.price_feed.fetch_account()
                self.state.set_agent_status("price_feed", "IDLE")

                # Refresh candles every 60 data cycles (~5 min)
                if not hasattr(self, "_candle_counter"):
                    self._candle_counter = 0
                self._candle_counter += 1
                if self._candle_counter % 60 == 0:
                    for sym in SYMBOLS:
                        await self.price_feed.fetch_candles(sym, "M15", 500)

                # Check news events
                self.state.set_agent_status("news_event", "RUNNING")
                await self.news.check_events()
                self.state.set_agent_status("news_event", "IDLE")

            except Exception as e:
                log.error(f"Data loop error: {e}")

            await asyncio.sleep(DATA_INTERVAL)

    # ── Analysis Loop (every 60s) ─────────────────────────────────

    async def analysis_loop(self):
        while self.running:
            try:
                for sym in SYMBOLS:
                    if self.state.circuit_breaker:
                        log.warning("Circuit breaker active — skipping analysis")
                        break

                    # Validate data first
                    self.state.set_agent_status("validator", "RUNNING")
                    valid = await self.validator.validate(sym)
                    self.state.set_agent_status("validator", "IDLE")
                    if not valid:
                        continue

                    # Technical analysis
                    self.state.set_agent_status("technical", "RUNNING")
                    tech = await self.technical.analyze(sym)
                    self.state.set_agent_status("technical", "IDLE")

                    # Macro context
                    self.state.set_agent_status("macro", "RUNNING")
                    macro = await self.macro.analyze(sym)
                    self.state.set_agent_status("macro", "IDLE")

                    # Sentiment
                    self.state.set_agent_status("sentiment", "RUNNING")
                    sent = await self.sentiment.analyze(sym)
                    self.state.set_agent_status("sentiment", "IDLE")

                    # Claude AI prediction
                    self.state.set_agent_status("predictor", "RUNNING")
                    thesis = await self.predictor.generate_thesis(
                        sym, tech, macro, sent, self.state.get_news_context()
                    )
                    thesis["symbol"] = sym
                    self.state.update_signal(sym, thesis)
                    self.state.set_agent_status("predictor", "IDLE")

                    if thesis.get("direction") == "No Trade":
                        log.info(f"{sym}: No Trade — {thesis.get('trade_thesis','')[:60]}")
                        continue

                    # Risk check
                    self.state.set_agent_status("risk", "RUNNING")
                    risk = await self.risk.evaluate(thesis, self.state)
                    self.state.set_agent_status("risk", "IDLE")
                    if not risk["approved"]:
                        continue

                    # Position sizing
                    self.state.set_agent_status("sizer", "RUNNING")
                    sizing = await self.sizer.calculate(thesis, risk, self.state)
                    self.state.set_agent_status("sizer", "IDLE")

                    # Execute trade
                    self.state.set_agent_status("executor", "RUNNING")
                    order = await self.executor.place_order(thesis, sizing)
                    self.state.set_agent_status("executor", "IDLE")

                    if order.get("filled"):
                        log.info(f"✅ TRADE OPENED: {sym} {thesis['direction']} "
                                 f"{sizing['lot_size']} lots @ {order['fill_price']}  "
                                 f"SL={thesis['stop_loss']}  TP={thesis['take_profit_1']}")

                    await asyncio.sleep(1)   # brief pause between pairs

            except Exception as e:
                log.error(f"Analysis loop error: {e}", exc_info=True)

            await asyncio.sleep(ANALYSIS_INTERVAL)

    # ── Monitor Loop (every 5s) ───────────────────────────────────

    async def monitor_loop(self):
        while self.running:
            try:
                self.state.set_agent_status("monitor", "RUNNING")
                prev_count = len(self.state.open_positions)
                await self.monitor.check_all_positions()

                # Detect newly closed trades → generate report
                new_count = len(self.state.open_positions)
                if new_count < prev_count:
                    newly_closed = self.state.closed_positions[-(prev_count - new_count):]
                    for trade in newly_closed:
                        self.state.set_agent_status("reporter", "RUNNING")
                        await self.reporter.analyze_trade(trade)
                        self.state.set_agent_status("reporter", "IDLE")

                self.state.set_agent_status("monitor", "IDLE")

            except Exception as e:
                log.error(f"Monitor loop error: {e}")

            await asyncio.sleep(MONITOR_INTERVAL)

    # ── Broadcast Loop (every 1s) ─────────────────────────────────

    async def broadcast_loop(self):
        while self.running:
            try:
                await self.ws.broadcast(self.state.get_dashboard_payload())
            except Exception as e:
                log.error(f"Broadcast error: {e}")
            await asyncio.sleep(1)

    # ── Main ──────────────────────────────────────────────────────

    async def run(self):
        if not await self.startup():
            return

        self.running = True
        await self.ws.start()
        log.info(f"WebSocket dashboard: ws://localhost:{os.getenv('WEBSOCKET_PORT','8765')}")
        log.info(f"Trading pairs: {', '.join(SYMBOLS)}")
        log.info("Press Ctrl+C to stop\n")

        try:
            await asyncio.gather(
                self.data_loop(),
                self.analysis_loop(),
                self.monitor_loop(),
                self.broadcast_loop(),
            )
        except KeyboardInterrupt:
            log.info("Shutdown requested...")
        finally:
            self.running = False
            self.mt5.disconnect()
            log.info("APEX FOREX stopped.")


if __name__ == "__main__":
    asyncio.run(Orchestrator().run())
