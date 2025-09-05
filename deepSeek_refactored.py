import json
import logging
from time import sleep
from datetime import datetime, timedelta, timezone
import pandas as pd
import pandas_ta as ta
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from api.dwx_client import dwx_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('strategy.log'), logging.StreamHandler()]
)

class ScalpingStrategy:
    def __init__(self, mt4_dir_path, symbol, timeframe, lots, risk_percentage=0.01):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbol = symbol
        self.timeframe = timeframe
        self.lots = lots
        self.risk_percentage = risk_percentage

        # Initialize DWX client
        self.dwx = dwx_client(
            self,
            mt4_dir_path,
            sleep_delay=0.005,
            max_retry_command_seconds=10,
            verbose=True
        )
        
        # Explicitly assign callback methods to the instance attributes.
        self.on_historic_data = self.on_historic_data
        self.on_message = self.on_message
        self.on_tick = self.on_tick

        # Data storage
        self.df = pd.DataFrame()
        self.current_bid = None
        self.current_ask = None
        
        # Strategy parameters with validation
        self.slatrcoef = 1.2  # type: float
        self.tpsl_ratio = 1.5  # type: float
        # self.max_spread = 5e-2  # type: float
        self.max_spread = 0.45
        
        # Initialize connection
        self._initialize_connection()

    def _initialize_connection(self):
        """Establish connection to MT4 and subscribe to data feeds."""
        try:
            self.dwx.start()
            sleep(1)  # Allow connection to establish
            self.logger.info("Account info: %s", self.dwx.account_info)
            
            # Subscribe to market data
            self.dwx.subscribe_symbols([self.symbol])
            self.dwx.subscribe_symbols_bar_data([[self.symbol, self.timeframe]])
            
            # Load historical data
            self._load_historical_data()
            
        except Exception as e:
            self.logger.error("Failed to initialize connection: %s", e)
            raise

    def _load_historical_data(self, days=30):
        """Load historical data for initialization."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        self.dwx.get_historic_data(
            self.symbol,
            self.timeframe,
            start.timestamp(),
            end.timestamp()
        )

    def on_bar_data(self, symbol, time_frame, time, open_price, high, low, close_price, tick_volume):
        """Handle incoming bar data."""
        if symbol != self.symbol or time_frame != self.timeframe:
            return

        try:
            # Try to convert the time input.
            try:
                timestamp = pd.to_datetime(time, unit='s')
            except Exception:
                # If conversion using Unix timestamp fails, try custom format:
                timestamp = pd.to_datetime(time, format='%Y.%m.%d %H:%M', errors='coerce')
                if pd.isnull(timestamp):
                    raise ValueError(f"Time conversion failed for value: {time}")
            
            # Append new data to DataFrame (reset index to ensure uniqueness)
            new_row = pd.DataFrame([{
                'Time': timestamp,
                'Open': float(open_price),
                'High': float(high),
                'Low': float(low),
                'Close': float(close_price),
                'Volume': int(tick_volume)
            }])
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            
            # Calculate indicators
            self._calculate_indicators()
            
        except Exception as e:
            self.logger.error("Error processing bar data: %s", e)

    def _calculate_indicators(self):
        """Calculate technical indicators."""
        if len(self.df) < 50:
            return  # Ensure enough data for calculations
        
        # Reset index to ensure it is unique
        self.df = self.df.reset_index(drop=True)

        self.df["ATR"] = ta.atr(self.df.High, self.df.Low, self.df.Close, length=7)
        self.df["EMA_fast"] = ta.ema(self.df.Close, length=30)
        self.df["EMA_slow"] = ta.ema(self.df.Close, length=50)
        self.df["RSI"] = ta.rsi(self.df.Close, length=10)
        
        # Compute Bollinger Bands
        bbands = ta.bbands(self.df.Close, length=15, std=1.5)
        # Instead of concatenating, assign (or overwrite) the Bollinger Bands columns:
        for col in bbands.columns:
            self.df[col] = bbands[col]

    def get_signal(self):
        """Generate trading signal based on current market conditions."""
        if len(self.df) < 10:
            return 0  # Not enough data

        current_idx = len(self.df) - 1
        ema_signal = self._ema_signal(current_idx, backcandles=7)
        
        bbl = self.df['BBL_15_1.5'].iloc[current_idx]
        bbu = self.df['BBU_15_1.5'].iloc[current_idx]
        rsi = self.df['RSI'].iloc[current_idx]
        open_price = self.df['Open'].iloc[current_idx]

        if ema_signal == 2 and open_price <= bbl and rsi < 60:
            return 2  # Buy signal
        if ema_signal == 1 and open_price >= bbu and rsi > 40:
            return 1  # Sell signal
        return 0

    def _ema_signal(self, current_idx, backcandles=7):
        """Determine EMA crossover signal."""
        start_idx = max(0, current_idx - backcandles)
        ema_fast = self.df['EMA_fast'].iloc[start_idx:current_idx+1]
        ema_slow = self.df['EMA_slow'].iloc[start_idx:current_idx+1]

        if all(ema_fast > ema_slow):
            return 2
        if all(ema_fast < ema_slow):
            return 1
        return 0

    def run_strategy(self):
        """Main strategy execution method."""
        try:
            signal = self.get_signal()
            if self.current_bid is None or self.current_ask is None:
                self.logger.warning("No tick data available yet.")
                return

            spread = self.current_ask - self.current_bid
            if spread > self.max_spread:
                self.logger.warning("Spread too wide: %.5f", spread)
                return

            if not self.dwx.open_orders:
                if signal == 1:  # Sell signal
                    self._execute_trade('sell')
                elif signal == 2:  # Buy signal
                    self._execute_trade('buy')

        except Exception as e:
            self.logger.error("Strategy execution error: %s", e)

    def _execute_trade(self, direction):
        """Execute trade based on signal direction."""
        if len(self.df) == 0 or 'ATR' not in self.df.columns:
            self.logger.error("Not enough data to compute trade parameters.")
            return

        atr = self.df['ATR'].iloc[-1]
        sl = atr * self.slatrcoef
        tp = sl * self.tpsl_ratio

        if direction == 'buy':
            stop_loss = self.current_bid - sl
            take_profit = self.current_bid + tp
            order_type = 'buy'
        else:
            stop_loss = self.current_ask + sl
            take_profit = self.current_ask - tp
            order_type = 'sell'

        try:
            self.dwx.open_order(
                symbol=self.symbol,
                order_type=order_type,
                take_profit=take_profit,
                stop_loss=stop_loss,
                lots=self.lots
            )
            self.logger.info("Opened %s order at price reference %.2f",
                             order_type, self.current_bid if direction == 'buy' else self.current_ask)
        except Exception as e:
            self.logger.error("Failed to execute %s order: %s", order_type, e)
    

    
    def schedule_jobs(self):
        """Enhanced scheduler with job persistence"""
        scheduler = BlockingScheduler()
        
        # Trading job
        scheduler.add_job(
            self.run_strategy,
            trigger=CronTrigger(
                minute='*/5',
                hour='6-21',
                day_of_week='mon-fri',
                timezone='UTC'
            ),
            max_instances=1,
            misfire_grace_time=300
        )
        
        # Optimization job
        scheduler.add_job(
            self.optimize_parameters,
            trigger=CronTrigger(
                day_of_week='sun',
                hour=23,
                timezone='UTC'
            ),
            misfire_grace_time=3600
        )
        
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown(wait=False)
            self.logger.info("Scheduler stopped")
        except Exception as e:
            self.logger.error("Scheduler failed: %s", e, exc_info=True)

    def optimize_parameters(self):
        """Optimize strategy parameters using historical data."""
        from backtesting import Backtest, Strategy
        
        class OptimizationStrategy(Strategy):
            def init(self):
                self.atr = self.I(ta.atr, self.data.High, self.data.Low, self.data.Close, length=7)
            def next(self):
                pass  # Custom optimization logic here
        
        try:
            bt = Backtest(self.df, OptimizationStrategy, cash=250, margin=1/30)
            stats = bt.optimize(
                sl_coef=[i/10 for i in range(10, 26)],
                tp_ratio=[i/10 for i in range(10, 26)],
                maximize='Return [%]',
                max_tries=100
            )
            
            self.slatrcoef = stats['sl_coef']
            self.tpsl_ratio = stats['tp_ratio']
            self.logger.info("Updated parameters: SL=%.2f, TP Ratio=%.2f", self.slatrcoef, self.tpsl_ratio)
            
        except Exception as e:
            self.logger.error("Parameter optimization failed: %s", e)

    def on_tick(self, symbol, bid, ask):
        """Handle incoming tick data."""
        if symbol == self.symbol:
            self.current_bid = bid
            self.current_ask = ask

    def on_message(self, message):
        """Handle incoming messages from DWX."""
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
        """Handle incoming historic data.
        
        For each bar:
         - If it's a dictionary, process it directly.
         - If it's a string:
             * If it contains 6 comma-separated values, use them.
             * If it contains only 1 value (just a timestamp), then fill missing values:
                 - Use the previous close for Open, High, Low, and Close (if available).
                 - Use volume = 0.
         - Otherwise, log a warning and skip.
        """
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
                        # Only time is provided.
                        t = parts[0]
                        if len(self.df) > 0:
                            last_close = self.df['Close'].iloc[-1]
                            op = hi = lo = cl = last_close
                        else:
                            self.logger.warning("No previous data to fill missing values for bar: %s", bar)
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
        """Clean shutdown procedure."""
        self.dwx.stop()
        self.logger.info("Strategy shutdown complete")

if __name__ == "__main__":
    strategy = ScalpingStrategy(
        mt4_dir_path='C:/Users/User/AppData/Roaming/MetaQuotes/Terminal/B7238334EF4B2B20A39D097B2877DE6E/MQL4/Files/',
        symbol='XAUUSD',
        timeframe='M5',
        lots=0.01
    )
    
    try:
        strategy.schedule_jobs()
        while strategy.dwx.ACTIVE:
            sleep(1)
    except KeyboardInterrupt:
        strategy.shutdown()
