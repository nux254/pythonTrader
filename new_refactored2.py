import json
import logging
import math
from time import sleep
from datetime import datetime, timedelta, timezone
import pandas as pd
import pandas_ta as ta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from api.dwx_client import dwx_client

# Configure logging to log to file and console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('strategy_refactored.log'), logging.StreamHandler()]
)

class ScalpingStrategy:
    def __init__(self, mt4_dir_path, symbol, timeframe, risk_percentage=0.01,
                 min_volatility=0.2, use_limit_orders=True):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_percentage = risk_percentage  # % of account balance risked per trade
        self.min_volatility = min_volatility    # Minimum ATR threshold to allow a trade
        self.use_limit_orders = use_limit_orders

        # Strategy parameters (to be optimized periodically)
        self.slatrcoef = 1.2   # ATR multiplier for stop-loss calculation.
        self.tpsl_ratio = 1.5  # Ratio of take profit relative to stop loss.
        self.max_spread = 0.45 # Maximum bid-ask spread allowed.

        # Data storage and tick variables.
        self.df = pd.DataFrame()
        self.current_bid = None
        self.current_ask = None

        # Initialize DWX client.
        self.dwx = dwx_client(
            self,
            mt4_dir_path,
            sleep_delay=0.005,
            max_retry_command_seconds=10,
            verbose=True
        )
        # Assign callbacks.
        self.on_historic_data = self.on_historic_data
        self.on_message = self.on_message
        self.on_tick = self.on_tick

        # Initialize connection and load history.
        self._initialize_connection()

    def _initialize_connection(self):
        """Establish connection to MT4 and subscribe to data feeds."""
        try:
            self.dwx.start()
            sleep(1)  # Allow time for connection.
            self.logger.info("Account info: %s", self.dwx.account_info)
            # Subscribe to symbol and bar data.
            self.dwx.subscribe_symbols([self.symbol])
            self.dwx.subscribe_symbols_bar_data([[self.symbol, self.timeframe]])
            # Load historical data (30 days by default).
            self._load_historical_data()
        except Exception as e:
            self.logger.error("Failed to initialize connection: %s", e)
            raise

    def _load_historical_data(self, days=30):
        """Fetch historical data from MT4 for initialization."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        try:
            self.dwx.get_historic_data(
                self.symbol,
                self.timeframe,
                start.timestamp(),
                end.timestamp()
            )
        except Exception as e:
            self.logger.error("Error loading historical data: %s", e)

    def on_bar_data(self, symbol, time_frame, time_val, open_price, high, low, close_price, tick_volume):
        """Process an incoming bar (OHLCV) and update DataFrame & indicators."""
        if symbol != self.symbol or time_frame != self.timeframe:
            return
        try:
            # Convert time using Unix timestamp or custom format.
            try:
                timestamp = pd.to_datetime(time_val, unit='s')
            except Exception:
                timestamp = pd.to_datetime(time_val, format='%Y.%m.%d %H:%M', errors='coerce')
                if pd.isnull(timestamp):
                    raise ValueError(f"Time conversion failed for value: {time_val}")
            new_row = pd.DataFrame([{
                'Time': timestamp,
                'Open': float(open_price),
                'High': float(high),
                'Low': float(low),
                'Close': float(close_price),
                'Volume': int(tick_volume)
            }])
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            self._calculate_indicators()
        except Exception as e:
            self.logger.error("Error processing bar data: %s", e)

    def _calculate_indicators(self):
        """Calculate ATR, EMA, RSI, Bollinger Bands, MACD and Volume Average."""
        if len(self.df) < 50:
            return  # Ensure sufficient data.
        self.df = self.df.reset_index(drop=True)
        self.df["ATR"] = ta.atr(self.df.High, self.df.Low, self.df.Close, length=7)
        self.df["EMA_fast"] = ta.ema(self.df.Close, length=30)
        self.df["EMA_slow"] = ta.ema(self.df.Close, length=50)
        self.df["RSI"] = ta.rsi(self.df.Close, length=10)
        bbands = ta.bbands(self.df.Close, length=15, std=1.5)
        for col in bbands.columns:
            self.df[col] = bbands[col]
        # Additional indicator: MACD (for additional filter)
        macd = ta.macd(self.df.Close)
        for col in macd.columns:
            self.df[col] = macd[col]
        # Additional filter: Calculate average volume over the last 20 bars.
        self.df["Vol_Avg"] = self.df.Volume.rolling(window=20).mean()

    def get_signal(self):
        """Combine indicators to generate a trading signal.
           Returns 2 for buy, 1 for sell, 0 for no action.
        """
        if len(self.df) < 10:
            return 0
        current_idx = len(self.df) - 1
        ema_signal = self._ema_signal(current_idx, backcandles=7)
        bbl = self.df['BBL_15_1.5'].iloc[current_idx]
        bbu = self.df['BBU_15_1.5'].iloc[current_idx]
        rsi = self.df['RSI'].iloc[current_idx]
        open_price = self.df['Open'].iloc[current_idx]
        atr = self.df['ATR'].iloc[current_idx]
        macd_hist = self.df['MACDh_12_26_9'].iloc[current_idx]  # Using MACD histogram.
        volume = self.df['Volume'].iloc[current_idx]
        vol_avg = self.df['Vol_Avg'].iloc[current_idx]

        # Filter: Only trade if volatility (ATR) is above threshold.
        if atr < self.min_volatility:
            self.logger.info("ATR %.2f below minimum %.2f; skipping trade", atr, self.min_volatility)
            return 0

        # Filter: Basic MACD filter – require histogram to agree with the trend.
        if ema_signal == 2 and macd_hist < 0:
            self.logger.info("MACD histogram negative while bullish signal; skipping trade")
            return 0
        if ema_signal == 1 and macd_hist > 0:
            self.logger.info("MACD histogram positive while bearish signal; skipping trade")
            return 0

        # Filter: Volume spike confirmation – trade only if current volume is at least 20% higher than average.
        if vol_avg > 0 and (volume / vol_avg) < 1.2:
            self.logger.info("Volume not high enough (%.2f vs. avg %.2f); skipping trade", volume, vol_avg)
            return 0

        # Entry Conditions: using EMA, Bollinger Bands, RSI.
        if ema_signal == 2 and open_price <= bbl and rsi < 60:
            return 2  # Buy signal.
        if ema_signal == 1 and open_price >= bbu and rsi > 40:
            return 1  # Sell signal.
        return 0

    def _ema_signal(self, current_idx, backcandles=7):
        """Determine bullish or bearish trend from EMA crossover over a lookback period."""
        start_idx = max(0, current_idx - backcandles)
        ema_fast = self.df['EMA_fast'].iloc[start_idx:current_idx+1]
        ema_slow = self.df['EMA_slow'].iloc[start_idx:current_idx+1]
        if all(ema_fast > ema_slow):
            return 2  # Bullish.
        if all(ema_fast < ema_slow):
            return 1  # Bearish.
        return 0

    def compute_position_size(self, stop_loss_distance):
        """Calculate dynamic position size based on account balance, risk percentage, and stop loss distance."""
        try:
            account_balance = float(self.dwx.account_info.get("balance", 10000))
        except Exception:
            account_balance = 10000
        risk_amount = account_balance * self.risk_percentage
        pos_size = risk_amount / stop_loss_distance
        self.logger.info("Position size calculated: %.4f units (Risk: %.2f, SL distance: %.2f)", pos_size, risk_amount, stop_loss_distance)
        return pos_size

    def run_strategy(self):
        """Main method to run the trading strategy, incorporating all filters and risk checks."""
        try:
            if not self.dwx.ACTIVE:
                self.logger.warning("DWX client inactive; attempting reconnection...")
                self._initialize_connection()
                return

            signal = self.get_signal()
            if self.current_bid is None or self.current_ask is None:
                self.logger.warning("No tick data available yet.")
                return

            spread = self.current_ask - self.current_bid
            if spread > self.max_spread:
                self.logger.warning("Spread too wide: %.5f", spread)
                return

            if signal == 0:
                self.logger.debug("No valid trade signal generated.")
                return

            # Calculate stop-loss and take-profit distances.
            atr = self.df['ATR'].iloc[-1]
            stop_loss_distance = atr * self.slatrcoef
            take_profit_distance = stop_loss_distance * self.tpsl_ratio

            # Dynamic position sizing.
            pos_size = self.compute_position_size(stop_loss_distance)
            pos_size = max(pos_size, 0.01)  # Ensure a minimum trade size.

            # Risk-reward check: only trade if potential reward is at least twice the risk.
            if take_profit_distance < 2 * stop_loss_distance:
                self.logger.warning("Risk-reward ratio insufficient: SL=%.2f, TP=%.2f", stop_loss_distance, take_profit_distance)
                return

            # Placeholder for trailing stops:
            # (In a production system, you would monitor open trades and adjust TP dynamically.)
            trailing_stop_enabled = True  # Dummy flag; can be expanded.
            if trailing_stop_enabled:
                self.logger.debug("Trailing stop enabled (placeholder).")

            # Execute trade if no open orders.
            if not self.dwx.open_orders:
                if signal == 2:  # Buy signal.
                    stop_loss_price = self.current_bid - stop_loss_distance
                    take_profit_price = self.current_bid + take_profit_distance
                    order_side = 'buy'
                    ref_price = self.current_bid
                elif signal == 1:  # Sell signal.
                    stop_loss_price = self.current_ask + stop_loss_distance
                    take_profit_price = self.current_ask - take_profit_distance
                    order_side = 'sell'
                    ref_price = self.current_ask
                else:
                    return

                self._execute_trade(order_side, pos_size, stop_loss_price, take_profit_price, ref_price)
        except Exception as e:
            self.logger.error("Error in run_strategy: %s", e, exc_info=True)

    def _execute_trade(self, direction, pos_size, stop_loss, take_profit, ref_price):
        """Execute trade using either limit or market orders with a small buffer to reduce slippage."""
        try:
            order_type = 'limit' if self.use_limit_orders else 'market'
            order_price = ref_price
            if self.use_limit_orders:
                buffer = 0.05  # Buffer to increase fill probability.
                order_price = ref_price + buffer if direction == 'buy' else ref_price - buffer

            self.dwx.open_order(
                symbol=self.symbol,
                order_type=direction,
                take_profit=take_profit,
                stop_loss=stop_loss,
                lots=pos_size,
                order_price=order_price,      # Price to be used if limit order.
                order_execution_type=order_type
            )
            self.logger.info("Executed %s order at %.2f (SL: %.2f, TP: %.2f, Size: %.4f)",
                             direction, order_price, stop_loss, take_profit, pos_size)
        except Exception as e:
            self.logger.error("Failed to execute %s order: %s", direction, e)

    def schedule_jobs(self):
        """Schedule the strategy and optimization jobs using a background scheduler."""
        self.scheduler = BackgroundScheduler()
        try:
            # Schedule the trading strategy every 5 minutes during market hours (06:00–21:00 UTC, Mon-Fri).
            self.scheduler.add_job(
                self.run_strategy,
                trigger=CronTrigger(minute='*/5', hour='6-21', day_of_week='mon-fri', timezone='UTC'),
                max_instances=1,
                misfire_grace_time=300
            )
            # Schedule parameter optimization every Sunday at 23:00 UTC.
            self.scheduler.add_job(
                self.optimize_parameters,
                trigger=CronTrigger(day_of_week='sun', hour='23', timezone='UTC'),
                misfire_grace_time=3600
            )
            self.scheduler.start()
            self.logger.info("Background scheduler started successfully.")
        except Exception as e:
            self.logger.error("Scheduler failed: %s", e, exc_info=True)

    def optimize_parameters(self):
        """Optimize strategy parameters using historical data (with basic grid search)."""
        from backtesting import Backtest, Strategy

        class OptimizationStrategy(Strategy):
            def init(self):
                self.atr = self.I(ta.atr, self.data.High, self.data.Low, self.data.Close, length=7)
            def next(self):
                pass  # Placeholder for further optimization logic.

        try:
            bt = Backtest(self.df, OptimizationStrategy, cash=250, margin=1/30)
            stats = bt.optimize(
                sl_coef=[i/10 for i in range(10, 26)],  # Test 1.0 to 2.5
                tp_ratio=[i/10 for i in range(10, 26)],
                maximize='Return [%]',
                max_tries=100
            )
            self.slatrcoef = stats['sl_coef']
            self.tpsl_ratio = stats['tp_ratio']
            self.logger.info("Optimized parameters: SL Coef=%.2f, TP Ratio=%.2f", self.slatrcoef, self.tpsl_ratio)
        except Exception as e:
            self.logger.error("Parameter optimization failed: %s", e)

    def on_tick(self, symbol, bid, ask):
        """Update current bid and ask prices on receiving tick data."""
        if symbol == self.symbol:
            self.current_bid = bid
            self.current_ask = ask

    def on_message(self, message):
        """Log messages from DWX with proper error categorization."""
        try:
            msg_type = message.get("type", "")
            if msg_type == "ERROR":
                self.logger.error("DWX ERROR: %s", message)
            elif msg_type == "INFO":
                self.logger.info("DWX INFO: %s", message)
            else:
                self.logger.debug("DWX Message: %s", message)
        except Exception as e:
            self.logger.error("Error in on_message: %s", e)

    def on_historic_data(self, symbol, time_frame, data):
        """Process each bar from historical data (supporting dict or string formats)."""
        self.logger.info("Received historic data for %s %s", symbol, time_frame)
        for bar in data:
            try:
                if isinstance(bar, dict):
                    t = bar.get('time')
                    op = bar.get('open')
                    hi = bar.get('high')
                    lo = bar.get('low')
                    cl = bar.get('close')
                    vol = bar.get('volume')
                elif isinstance(bar, str):
                    parts = bar.split(',')
                    if len(parts) == 6:
                        t, op, hi, lo, cl, vol = parts
                    elif len(parts) == 1:
                        t = parts[0]
                        if len(self.df) > 0:
                            last_close = self.df['Close'].iloc[-1]
                            op = hi = lo = cl = last_close
                        else:
                            self.logger.warning("No previous data to fill missing bar: %s", bar)
                            continue
                        vol = 0
                    else:
                        self.logger.warning("Skipping historic bar due to unexpected format: %s", bar)
                        continue
                else:
                    self.logger.error("Unknown historic bar type: %s", type(bar))
                    continue

                self.on_bar_data(symbol, time_frame, t, op, hi, lo, cl, vol)
            except Exception as e:
                self.logger.error("Error processing historic bar: %s", e)

    def shutdown(self):
        """Cleanly shutdown the DWX client and scheduler."""
        try:
            self.dwx.stop()
        except Exception as e:
            self.logger.error("Error stopping DWX client: %s", e)
        try:
            if hasattr(self, 'scheduler'):
                self.scheduler.shutdown(wait=False)
        except Exception as e:
            self.logger.error("Error shutting down scheduler: %s", e)
        self.logger.info("Strategy shutdown complete")

    def wait_for_next_bar(self):
        """Wait precisely until the start of the next minute bar."""
        now = datetime.now()
        next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        pause = math.ceil((next_min - now).total_seconds())
        self.logger.info("Sleeping for %d seconds until next bar", pause)
        sleep(pause)

if __name__ == "__main__":
    strategy = ScalpingStrategy(
        mt4_dir_path='C:/Users/User/AppData/Roaming/MetaQuotes/Terminal/B7238334EF4B2B20A39D097B2877DE6E/MQL4/Files/',
        symbol='XAUUSD',
        timeframe='M5',
        risk_percentage=0.01,   # Risk 1% of account balance per trade.
        min_volatility=0.2,     # Only trade if ATR >= 0.2.
        use_limit_orders=True   # Attempt to use limit orders to reduce slippage.
    )

    try:
        strategy.schedule_jobs()
        # Main loop: Check connection health and wait.
        while True:
            if not strategy.dwx.ACTIVE:
                strategy.logger.warning("DWX client inactive; attempting reconnection...")
                strategy._initialize_connection()
            sleep(1)
    except KeyboardInterrupt:
        strategy.shutdown()
