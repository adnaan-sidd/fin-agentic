"""
APEX FOREX — MT5 Connector
══════════════════════════════════════════════════════════════
Single module that wraps the official MetaQuotes MetaTrader5
Python library. Every agent uses this — no direct MT5 calls
anywhere else in the codebase.

Replaces ALL OANDA functionality:
  ✓ Real-time tick prices   (was: OANDA /v3/pricing)
  ✓ OHLCV candle history    (was: OANDA /v3/instruments/candles)
  ✓ Account balance/equity  (was: OANDA /v3/accounts/summary)
  ✓ Place market orders     (was: OANDA /v3/orders)
  ✓ Close / modify trades   (was: OANDA /v3/trades)
  ✓ Fetch open positions    (was: OANDA /v3/positions)

Setup:
  1. Install PU Prime MT5 terminal from puprime.com
  2. Log in to your demo or live account
  3. Enable Algo Trading: Tools → Options → Expert Advisors
     → tick "Allow automated trading"
  4. Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env
  5. Run this system — it connects automatically

Windows ONLY note:
  The official MetaTrader5 package is Windows-only.
  On Linux VPS: pip install mt5linux and change the import below.
══════════════════════════════════════════════════════════════
"""

import os
import time
import logging
import asyncio
from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd

log = logging.getLogger("MT5Connector")

# ── Import MT5 library ────────────────────────────────────────────
# Windows (default): pip install MetaTrader5
# Linux VPS:         pip install mt5linux  →  change to:
#                    from mt5linux import MetaTrader5; mt5 = MetaTrader5()
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    log.warning("MetaTrader5 package not installed — running in MOCK mode")

# MT5 timeframe map
TF_MAP = {
    "M1":  1,   "M5":  5,   "M15": 15,  "M30": 30,
    "H1":  16385, "H4": 16388, "D1": 16408,
}

MAGIC_NUMBER = 20250001   # Identifies our bot's orders in MT5

class MT5Connector:
    """
    Singleton connector to the MT5 terminal.
    Call MT5Connector.get() to get the shared instance.
    """
    _instance = None

    def __init__(self):
        self.connected  = False
        self.login      = int(os.getenv("MT5_LOGIN", "0"))
        self.password   = os.getenv("MT5_PASSWORD", "")
        self.server     = os.getenv("MT5_SERVER", "")
        self.path       = os.getenv("MT5_PATH", "")  # optional explicit path
        self.mode       = os.getenv("MT5_MODE", "demo")
        self._mock_prices: Dict[str, float] = {
            "EURUSD": 1.08500, "GBPUSD": 1.27000, "USDJPY": 149.500,
            "AUDUSD": 0.65200, "GBPJPY": 189.800, "USDCHF": 0.90500,
        }

    @classmethod
    def get(cls) -> "MT5Connector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Connection ────────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect and authenticate with MT5 terminal."""
        if not MT5_AVAILABLE:
            log.warning("MT5 not available — using mock data (install MetaTrader5 on Windows)")
            self.connected = False
            return False

        # Initialize terminal
        kwargs = {}
        if self.path:
            kwargs["path"] = self.path

        if not mt5.initialize(**kwargs):
            log.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        # Login
        if self.login and self.password and self.server:
            ok = mt5.login(self.login, password=self.password, server=self.server)
            if not ok:
                log.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            log.info(f"MT5 connected: account {self.login} @ {self.server} [{self.mode.upper()}]")
        else:
            log.info("MT5 connected to already-open terminal (no login credentials provided)")

        self.connected = True
        info = self.account_info()
        if info:
            log.info(f"Account: balance=${info['balance']:.2f}  equity=${info['equity']:.2f}  leverage=1:{info['leverage']}")
        return True

    def disconnect(self):
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            log.info("MT5 disconnected")

    def ensure_connected(self) -> bool:
        """Auto-reconnect if terminal dropped."""
        if not MT5_AVAILABLE:
            return False
        if not self.connected or not mt5.terminal_info():
            log.warning("MT5 connection lost — reconnecting...")
            return self.connect()
        return True

    # ── Price Data ────────────────────────────────────────────────

    def get_tick(self, symbol: str) -> Optional[dict]:
        """Get latest bid/ask for a symbol."""
        if not self.ensure_connected():
            return self._mock_tick(symbol)
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self._enable_symbol(symbol)
                tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return self._mock_tick(symbol)

            spread_pips = self._calc_spread_pips(symbol, tick.ask - tick.bid)
            return {
                "symbol": symbol,
                "bid":    round(tick.bid, 5),
                "ask":    round(tick.ask, 5),
                "spread": round(spread_pips, 1),
                "time":   datetime.utcfromtimestamp(tick.time).isoformat(),
            }
        except Exception as e:
            log.error(f"get_tick {symbol}: {e}")
            return self._mock_tick(symbol)

    def get_candles(self, symbol: str, timeframe: str = "M15", count: int = 250) -> List[dict]:
        """Fetch OHLCV candles as list of dicts."""
        if not self.ensure_connected():
            return self._mock_candles(symbol, count)
        try:
            tf = self._tf(timeframe)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None or len(rates) == 0:
                self._enable_symbol(symbol)
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None:
                return self._mock_candles(symbol, count)

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            return [
                {
                    "ts":     str(row["time"]),
                    "open":   row["open"],
                    "high":   row["high"],
                    "low":    row["low"],
                    "close":  row["close"],
                    "volume": int(row.get("tick_volume", 0)),
                }
                for _, row in df.iterrows()
            ]
        except Exception as e:
            log.error(f"get_candles {symbol}/{timeframe}: {e}")
            return self._mock_candles(symbol, count)

    # ── Account ───────────────────────────────────────────────────

    def account_info(self) -> Optional[dict]:
        """Fetch live account balance, equity, margin."""
        if not self.ensure_connected():
            return {"balance": 10000.0, "equity": 10000.0, "margin": 0.0,
                    "free_margin": 10000.0, "leverage": 500, "currency": "USD"}
        try:
            info = mt5.account_info()
            if info is None:
                return None
            return {
                "balance":     info.balance,
                "equity":      info.equity,
                "margin":      info.margin,
                "free_margin": info.margin_free,
                "leverage":    info.leverage,
                "currency":    info.currency,
                "profit":      info.profit,
            }
        except Exception as e:
            log.error(f"account_info: {e}")
            return None

    # ── Trading ───────────────────────────────────────────────────

    def place_order(self, symbol: str, direction: str, lot_size: float,
                    sl: float, tp: float, comment: str = "APEX") -> dict:
        """
        Place a market order with stop-loss and take-profit.
        direction: "Long" or "Short"
        Returns dict with filled=True/False and order details.
        """
        if not self.ensure_connected():
            return self._mock_place_order(symbol, direction, lot_size, sl, tp)

        try:
            tick     = mt5.symbol_info_tick(symbol)
            info     = mt5.symbol_info(symbol)
            if tick is None or info is None:
                return {"filled": False, "error": f"Symbol {symbol} not available"}

            # Normalize SL/TP to broker's digit precision
            digits   = info.digits
            sl       = round(sl, digits)
            tp       = round(tp, digits)
            lot_size = self._normalize_lot(symbol, lot_size)

            if direction == "Long":
                order_type = mt5.ORDER_TYPE_BUY
                price      = tick.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price      = tick.bid

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       lot_size,
                "type":         order_type,
                "price":        price,
                "sl":           sl,
                "tp":           tp,
                "deviation":    10,          # max slippage in points
                "magic":        MAGIC_NUMBER,
                "comment":      comment,
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                return {"filled": False, "error": str(mt5.last_error())}

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"ORDER FILLED: {direction} {lot_size} {symbol} @ {result.price}  SL={sl}  TP={tp}")
                return {
                    "filled":     True,
                    "ticket":     result.order,
                    "symbol":     symbol,
                    "direction":  direction,
                    "lot_size":   lot_size,
                    "fill_price": result.price,
                    "stop_loss":  sl,
                    "take_profit_1": tp,
                    "slippage":   abs(result.price - price),
                    "ts":         datetime.utcnow().isoformat(),
                }
            else:
                log.error(f"Order failed retcode={result.retcode}: {result.comment}")
                return {"filled": False, "error": result.comment, "retcode": result.retcode}

        except Exception as e:
            log.error(f"place_order {symbol}: {e}")
            return {"filled": False, "error": str(e)}

    def close_position(self, ticket: int, symbol: str, direction: str, lot_size: float) -> dict:
        """Close a specific open position by ticket number."""
        if not self.ensure_connected():
            return {"closed": True, "ticket": ticket}
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"closed": False, "error": "No tick data"}

            close_type  = mt5.ORDER_TYPE_SELL if direction == "Long" else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if direction == "Long" else tick.ask

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       lot_size,
                "type":         close_type,
                "position":     ticket,
                "price":        close_price,
                "deviation":    10,
                "magic":        MAGIC_NUMBER,
                "comment":      "APEX close",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"POSITION CLOSED: ticket={ticket} {direction} {symbol} @ {result.price}")
                return {"closed": True, "ticket": ticket, "close_price": result.price}
            else:
                err = result.comment if result else str(mt5.last_error())
                log.error(f"Close failed ticket={ticket}: {err}")
                return {"closed": False, "error": err}
        except Exception as e:
            log.error(f"close_position ticket={ticket}: {e}")
            return {"closed": False, "error": str(e)}

    def modify_position(self, ticket: int, new_sl: float, new_tp: float, symbol: str) -> dict:
        """Modify SL/TP on an existing position (for trailing stops, break-even)."""
        if not self.ensure_connected():
            return {"modified": True}
        try:
            info   = mt5.symbol_info(symbol)
            digits = info.digits if info else 5
            request = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "symbol":   symbol,
                "position": ticket,
                "sl":       round(new_sl, digits),
                "tp":       round(new_tp, digits),
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return {"modified": True, "ticket": ticket, "sl": new_sl, "tp": new_tp}
            else:
                err = result.comment if result else str(mt5.last_error())
                return {"modified": False, "error": err}
        except Exception as e:
            return {"modified": False, "error": str(e)}

    def get_open_positions(self) -> List[dict]:
        """Get all positions opened by our bot (identified by magic number)."""
        if not self.ensure_connected():
            return []
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            result = []
            for p in positions:
                if p.magic != MAGIC_NUMBER:
                    continue
                result.append({
                    "ticket":    p.ticket,
                    "symbol":    p.symbol,
                    "direction": "Long" if p.type == 0 else "Short",
                    "lot_size":  p.volume,
                    "entry_price": p.price_open,
                    "current_price": p.price_current,
                    "stop_loss": p.sl,
                    "take_profit_1": p.tp,
                    "pnl_usd":   p.profit,
                    "swap":      p.swap,
                    "opened_at": datetime.utcfromtimestamp(p.time).isoformat(),
                })
            return result
        except Exception as e:
            log.error(f"get_open_positions: {e}")
            return []

    # ── Helpers ───────────────────────────────────────────────────

    def _tf(self, tf_str: str) -> int:
        if not MT5_AVAILABLE:
            return TF_MAP.get(tf_str, 15)
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,   "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,   "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return tf_map.get(tf_str, mt5.TIMEFRAME_M15)

    def _enable_symbol(self, symbol: str):
        """Add symbol to Market Watch if not visible."""
        if MT5_AVAILABLE:
            mt5.symbol_select(symbol, True)

    def _normalize_lot(self, symbol: str, lot: float) -> float:
        """Round lot size to broker's minimum step."""
        if not MT5_AVAILABLE:
            return round(lot / 0.01) * 0.01
        info = mt5.symbol_info(symbol)
        if info is None:
            return max(0.01, round(lot / 0.01) * 0.01)
        step = info.volume_step
        min_vol = info.volume_min
        normalized = round(lot / step) * step
        return max(min_vol, round(normalized, 2))

    def _calc_spread_pips(self, symbol: str, spread_price: float) -> float:
        if "JPY" in symbol:
            return spread_price * 100
        return spread_price * 10000

    def _pip_size(self, symbol: str) -> float:
        return 0.01 if "JPY" in symbol else 0.0001

    # ── Mock data (when MT5 not available) ───────────────────────

    def _mock_tick(self, symbol: str) -> dict:
        import random
        base = self._mock_prices.get(symbol, 1.1000)
        noise = random.uniform(-0.0003, 0.0003)
        spread = 0.00012 if "JPY" not in symbol else 0.012
        bid = round(base + noise, 5)
        ask = round(bid + spread, 5)
        self._mock_prices[symbol] = bid
        return {"symbol": symbol, "bid": bid, "ask": ask,
                "spread": round(spread * 10000, 1), "time": datetime.utcnow().isoformat()}

    def _mock_candles(self, symbol: str, count: int) -> List[dict]:
        import random, math
        base  = self._mock_prices.get(symbol, 1.1000)
        candles, price = [], base
        trend = random.choice([-1, 1])
        for i in range(count):
            if i % 80 == 0:
                trend = -trend
            drift = trend * 0.00003 + random.gauss(0, 0.00018)
            o = price
            c = round(price + drift, 5)
            h = round(max(o, c) + abs(random.gauss(0, 0.00035)), 5)
            l = round(min(o, c) - abs(random.gauss(0, 0.00035)), 5)
            candles.append({"ts": f"T-{count-i}", "open": o, "high": h,
                            "low": l, "close": c, "volume": random.randint(200, 2500)})
            price = c
        return candles

    def _mock_place_order(self, symbol, direction, lot_size, sl, tp) -> dict:
        import random, uuid
        tick   = self._mock_tick(symbol)
        price  = tick["ask"] if direction == "Long" else tick["bid"]
        slip   = random.uniform(0, 0.3) * self._pip_size(symbol)
        filled = price + slip if direction == "Long" else price - slip
        return {
            "filled": True, "ticket": random.randint(100000, 999999),
            "symbol": symbol, "direction": direction,
            "lot_size": lot_size, "fill_price": round(filled, 5),
            "stop_loss": sl, "take_profit_1": tp,
            "slippage": round(slip / self._pip_size(symbol), 1),
            "ts": datetime.utcnow().isoformat(),
        }
