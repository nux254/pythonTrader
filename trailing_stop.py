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

# For machine learning
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Configure logging: log both to file and console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('strategy_refactored.log'), logging.StreamHandler()]
)


class ScalpingStrategy:
    def __init__(self, mt4_dir_path, symbol, timeframe, risk_percentage=0.01,
                 min_volatility=0.2, use_limit_orders=True, min_adx=25):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbol = symbol
        self.timeframe = timeframe  # Use 'M1' or 'M15' for tick-level scalping.
        self.risk_percentage = risk_percentage  # Fraction of account balance risked per trade.
        self.min_volatility = min_volatility    # Minimum ATR required to trade.
        self.use_limit_orders = use_limit_orders
        self.min_adx = min_adx  # Market regime filter: require ADX >= this threshold.
        
        # Strategy parameters – subject to walk‑forward optimization.
        self.slatrcoef = 1.2   # Multiplier for ATR to compute stop-loss distance.
        self.tpsl_ratio = 2.0  # Risk-reward ratio: TP must be at least 2x SL.
        self.max_spread = 0.45
        
        # Trailing stop settings.
        self.trailing_stop_multiplier = 0.5  # Fraction of ATR used to trail the stop.
        self.active_trade = None  # Stores active trade details for trailing stops.
        
        # Data storage.
        self.df = pd.DataFrame()
        self.current_bid = None
        self.current_ask = None
        
        # Machine Learning model for adaptive signals.
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_trained = False
        
        # Initialize DWX client.
        self.dwx = dwx_client(
            self,
            mt4_dir_path,
            sleep_delay=0.005,
            max_retry_command_seconds=10,
            verbose=True
        )
        
        # Set callback methods.
        self.on_historic_data = self.on_historic_data
        self.on_message = self.on_message
        self.on_tick = self.on_tick
        
        # Establish connection.
        self._initialize_connection()

    def _initialize_connection(self):
        """Establish connection to MT4 and subscribe to data feeds (with retry)."""
        try:
            self.dwx.start()
            sleep(1)  # Allow connection to establish.
            self.logger.info("Account info: %s", self.dwx.account_info)
            self.dwx.subscribe_symbols([self.symbol])
            self.dwx.subscribe_symbols_bar_data([[self.symbol, self.timeframe]])
            self._load_historical_data()
        except Exception as e:
            self.logger.error("Failed to initialize connection: %s", e)
            raise

    def _load_historical_data(self, days=30):
        """Load historical data for initialization."""
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
        """Handle incoming bar data."""
        if symbol != self.symbol or time_frame != self.timeframe:
            return
        try:
            # Convert the time input.
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
            # Train ML model once enough data is collected.
            if len(self.df) >= 100 and not self.ml_trained:
                self.train_ml_model()
        except Exception as e:
            self.logger.error("Error processing bar data: %s", e)

    def _calculate_indicators(self):
        """Calculate technical indicators on the latest data."""
        if len(self.df) < 50:
            return
        self.df = self.df.reset_index(drop=True)
        # Volatility indicator.
        self.df["ATR"] = ta.atr(self.df.High, self.df.Low, self.df.Close, length=7)
        # Exponential Moving Averages for trend detection.
        self.df["EMA_fast"] = ta.ema(self.df.Close, length=30)
        self.df["EMA_slow"] = ta.ema(self.df.Close, length=50)
        # RSI with traditional thresholds.
        self.df["RSI"] = ta.rsi(self.df.Close, length=14)
        # Bollinger Bands.
        bbands = ta.bbands(self.df.Close, length=15, std=1.5)
        for col in bbands.columns:
            self.df[col] = bbands[col]
        # MACD for momentum and potential divergence.
        macd = ta.macd(self.df.Close, fast=12, slow=26)
        self.df["MACD"] = macd["MACD_12_26_9"]
        self.df["MACD_signal"] = macd["MACDs_12_26_9"]
        # ADX for market regime detection.
        adx = ta.adx(self.df.High, self.df.Low, self.df.Close)
        self.df["ADX"] = adx["ADX_14"]
        # Rolling average volume for volume spike detection.
        self.df["Volume_Avg"] = self.df.Volume.rolling(window=20).mean()

    def train_ml_model(self):
        """Train ML model on historical indicator data to predict price movement."""
        try:
            if len(self.df) < 100:
                return
            features = self.df[['ATR', 'EMA_fast', 'EMA_slow', 'RSI', 'MACD', 'ADX', 'Volume']].dropna()
            # Use a binary label: 1 if next close is higher than current close, else 0.
            labels = (self.df['Close'].diff().shift(-1) > 0).astype(int).dropna()
            min_len = min(len(features), len(labels))
            features = features.iloc[:min_len]
            labels = labels.iloc[:min_len]
            self.ml_model.fit(features, labels)
            self.ml_trained = True
            self.logger.info("ML model trained on historical data.")
        except Exception as e:
            self.logger.error("ML training error: %s", e)

    def get_ml_signal(self):
        """Return a signal based on the ML model: 2 for buy, 1 for sell, 0 for neutral."""
        try:
            if not self.ml_trained or len(self.df) < 1:
                return 0
            latest = self.df[['ATR', 'EMA_fast', 'EMA_slow', 'RSI', 'MACD', 'ADX', 'Volume']].iloc[-1:]
            prediction = self.ml_model.predict(latest)[0]
            # Map the prediction: if 1, assume upward move (buy); if 0, assume downward (sell).
            return 2 if prediction == 1 else 1
        except Exception as e:
            self.logger.error("Error in ML signal: %s", e)
            return 0

    def _ema_signal(self, current_idx, backcandles=7):
        """Determine a baseline EMA crossover signal over a lookback period."""
        start_idx = max(0, current_idx - backcandles)
        ema_fast = self.df['EMA_fast'].iloc[start_idx:current_idx + 1]
        ema_slow = self.df['EMA_slow'].iloc[start_idx:current_idx + 1]
        if all(ema_fast > ema_slow):
            return 2  # Bullish.
        if all(ema_fast < ema_slow):
            return 1  # Bearish.
        return 0

    def get_alternative_signal(self):
        """Placeholder for alternative data integration (e.g. news sentiment)."""
        return self._ema_signal(len(self.df) - 1, backcandles=7)

    def get_signal(self):
        """Generate trading signal by combining technical indicators, ML output, and additional filters."""
        if len(self.df) < 10:
            return 0  # Not enough data.
        current_idx = len(self.df) - 1

        # Get baseline EMA and ML signals.
        ema_signal = self._ema_signal(current_idx, backcandles=7)
        ml_signal = self.get_ml_signal()
        alt_signal = self.get_alternative_signal()

        # Market regime check using ADX.
        current_adx = self.df['ADX'].iloc[current_idx]
        if current_adx < self.min_adx:
            self.logger.info("ADX %.2f below threshold %.2f (choppy market); no trade.", current_adx, self.min_adx)
            return 0

        # Volume spike filter: require current volume to be at least 2x the rolling average.
        current_volume = self.df['Volume'].iloc[current_idx]
        avg_volume = self.df['Volume_Avg'].iloc[current_idx]
        if avg_volume > 0 and current_volume < 2 * avg_volume:
            self.logger.info("Volume spike not detected (current: %d, avg: %.2f); no trade.", current_volume, avg_volume)
            return 0

        # Simple MACD divergence check: For a buy, require MACD is rising; for a sell, falling.
        current_macd = self.df['MACD'].iloc[current_idx]
        prev_macd = self.df['MACD'].iloc[current_idx - 1] if current_idx > 0 else current_macd
        if ml_signal == 2 and current_macd <= prev_macd:
            self.logger.info("MACD divergence not confirmed for buy (current: %.2f, previous: %.2f).", current_macd, prev_macd)
            return 0
        if ml_signal == 1 and current_macd >= prev_macd:
            self.logger.info("MACD divergence not confirmed for sell (current: %.2f, previous: %.2f).", current_macd, prev_macd)
            return 0

        # Use ML signal if trained; otherwise, fallback to EMA signal.
        signal = ml_signal if self.ml_trained else ema_signal
        # Require agreement with the alternative signal.
        if signal != alt_signal:
            self.logger.info("Signal mismatch: primary %d vs alternative %d; no trade.", signal, alt_signal)
            return 0

        return signal

    def compute_position_size(self, stop_loss_distance):
        """Calculate dynamic position size based on account balance and stop-loss distance."""
        try:
            account_balance = float(self.dwx.account_info.get("balance", 10000))
        except Exception:
            account_balance = 10000
        risk_amount = account_balance * self.risk_percentage
        pos_size = risk_amount / stop_loss_distance
        self.logger.info("Calculated position size: %.4f units (balance: %.2f, risk: %.2f, SL distance: %.2f)",
                         pos_size, account_balance, risk_amount, stop_loss_distance)
        return max(pos_size, 0.01)

    def run_strategy(self):
        """Main execution routine with connection checks and risk filters."""
        try:
            if not self.dwx.ACTIVE:
                self.logger.warning("DWX client inactive. Attempting reconnection...")
                self._initialize_connection()
                return

            signal = self.get_signal()
            if self.current_bid is None or self.current_ask is None:
                self.logger.warning("Tick data not available yet.")
                return

            spread = self.current_ask - self.current_bid
            if spread > self.max_spread:
                self.logger.warning("Spread too wide: %.5f", spread)
                return

            atr = self.df['ATR'].iloc[-1]
            if atr < self.min_volatility:
                self.logger.info("ATR %.2f below minimum %.2f; no trade.", atr, self.min_volatility)
                return

            # Calculate stop-loss and take-profit distances.
            sl_distance = atr * self.slatrcoef
            tp_distance = sl_distance * self.tpsl_ratio

            # Risk-reward check: Ensure TP is at least twice the risk.
            if tp_distance < 2 * sl_distance:
                self.logger.info("Risk-reward ratio insufficient (TP: %.2f, SL: %.2f); no trade.", tp_distance, sl_distance)
                return

            # Determine dynamic position sizing.
            pos_size = self.compute_position_size(sl_distance)
            pos_size = max(pos_size, 0.01)

            # Execute order if there are no open orders.
            if not self.dwx.open_orders:
                if signal == 2:  # Buy signal.
                    stop_loss_price = self.current_bid - sl_distance
                    take_profit_price = self.current_bid + tp_distance
                    order_side = 'buy'
                    ref_price = self.current_bid
                elif signal == 1:  # Sell signal.
                    stop_loss_price = self.current_ask + sl_distance
                    take_profit_price = self.current_ask - tp_distance
                    order_side = 'sell'
                    ref_price = self.current_ask
                else:
                    return

                self._execute_trade(order_side, pos_size, stop_loss_price, take_profit_price, ref_price)
        except Exception as e:
            self.logger.error("Error in run_strategy: %s", e, exc_info=True)

    def _execute_trade(self, direction, pos_size, stop_loss, take_profit, ref_price):
        """Execute trade using limit or market orders and record the active trade for trailing stops."""
        try:
            order_type = 'limit' if self.use_limit_orders else 'market'
            order_price = ref_price
            if self.use_limit_orders:
                buffer = 0.05  # Small buffer to improve fill probability.
                order_price = ref_price + buffer if direction == 'buy' else ref_price - buffer

            self.dwx.open_order(
                symbol=self.symbol,
                order_type=direction,
                take_profit=take_profit,
                stop_loss=stop_loss,
                lots=pos_size,
                order_price=order_price,
                order_execution_type=order_type
            )
            self.logger.info("Opened %s order at price %.2f (SL: %.2f, TP: %.2f, size: %.4f)",
                             direction, order_price, stop_loss, take_profit, pos_size)
            # Record active trade for trailing stop management.
            self.active_trade = {
                'direction': direction,
                'entry_price': ref_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        except Exception as e:
            self.logger.error("Failed to execute %s order: %s", direction, e)

    def update_trailing_stop(self, tick_price):
        """Adjust the stop loss to lock in profits as the market moves in our favor."""
        if not self.active_trade:
            return
        try:
            direction = self.active_trade['direction']
            current_sl = self.active_trade['stop_loss']
            atr = self.df['ATR'].iloc[-1]
            trail_amount = atr * self.trailing_stop_multiplier
            new_sl = current_sl

            if direction == 'buy':
                # For a buy, update SL if the bid price minus trail_amount is higher.
                if tick_price - trail_amount > current_sl:
                    new_sl = tick_price - trail_amount
            elif direction == 'sell':
                # For a sell, update SL if the ask price plus trail_amount is lower.
                if tick_price + trail_amount < current_sl:
                    new_sl = tick_price + trail_amount

            if new_sl != current_sl:
                self.logger.info("Updating trailing stop from %.2f to %.2f", current_sl, new_sl)
                # Placeholder: Update the order via DWX API if available.
                self.dwx.modify_order(self.symbol, new_stop_loss=new_sl)
                self.active_trade['stop_loss'] = new_sl
        except Exception as e:
            self.logger.error("Error updating trailing stop: %s", e)

    def on_tick(self, symbol, bid, ask):
        """Handle incoming tick data, update bid/ask, and manage trailing stops."""
        if symbol == self.symbol:
            self.current_bid = bid
            self.current_ask = ask
            # Use the bid price for buy orders and ask price for sell orders.
            tick_price = bid if (self.active_trade and self.active_trade['direction'] == 'buy') else ask
            self.update_trailing_stop(tick_price)

    def on_message(self, message):
        """Handle messages from DWX with proper error categorization."""
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
        """Process historical data received from DWX."""
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

    def optimize_parameters(self):
        """Optimize strategy parameters using historical data with walk-forward validation (placeholder)."""
        from backtesting import Backtest, Strategy

        class OptimizationStrategy(Strategy):
            def init(self):
                self.atr = self.I(ta.atr, self.data.High, self.data.Low, self.data.Close, length=7)

            def next(self):
                pass  # Extend optimization logic as needed.

        try:
            # Placeholder: Implement walk-forward optimization by splitting data into training and out-of-sample sets.
            bt = Backtest(self.df, OptimizationStrategy, cash=250, margin=1/30)
            stats = bt.optimize(
                sl_coef=[i/10 for i in range(10, 26)],  # 1.0 to 2.5
                tp_ratio=[i/10 for i in range(10, 26)],
                maximize='Return [%]',
                max_tries=100
            )
            self.slatrcoef = stats['sl_coef']
            self.tpsl_ratio = stats['tp_ratio']
            self.logger.info("Optimized parameters updated: SL Coef=%.2f, TP Ratio=%.2f",
                             self.slatrcoef, self.tpsl_ratio)
        except Exception as e:
            self.logger.error("Parameter optimization failed: %s", e)

    def wait_for_next_bar(self):
        """Sleep until the start of the next minute bar."""
        now = datetime.now()
        next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        pause = math.ceil((next_min - now).total_seconds())
        self.logger.info("Sleeping for %d seconds until next bar", pause)
        sleep(pause)

    def schedule_jobs(self):
        """Schedule trading and optimization jobs using a background scheduler."""
        self.scheduler = BackgroundScheduler()
        try:
            # Trading job: runs every 5 minutes between 06:00 and 21:00 UTC, Monday–Friday.
            self.scheduler.add_job(
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
            # Optimization job: runs every Sunday at 23:00 UTC.
            self.scheduler.add_job(
                self.optimize_parameters,
                trigger=CronTrigger(
                    day_of_week='sun',
                    hour='23',
                    timezone='UTC'
                ),
                misfire_grace_time=3600
            )
            self.scheduler.start()
            self.logger.info("Scheduler started successfully.")
        except Exception as e:
            self.logger.error("Scheduler failed: %s", e, exc_info=True)

    def shutdown(self):
        """Cleanly shut down the DWX client and scheduler."""
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


if __name__ == "__main__":
    # Instantiate the strategy with improved risk management and scalability.
    strategy = ScalpingStrategy(
        mt4_dir_path='C:/Users/User/AppData/Roaming/MetaQuotes/Terminal/B7238334EF4B2B20A39D097B2877DE6E/MQL4/Files/',
        symbol='XAUUSD',
        timeframe='M1',  # Use M1 or M15 for faster, tick-level scalping.
        risk_percentage=0.01,  # Risk 1% of account balance per trade.
        min_volatility=0.2,    # Only trade if ATR >= 0.2 (subject to optimization).
        use_limit_orders=True,  # Use limit orders to reduce slippage.
        min_adx=25             # Require a minimum ADX for trending market conditions.
    )

    try:
        strategy.schedule_jobs()
        # Main loop: Monitor connection health and process events.
        while True:
            if not strategy.dwx.ACTIVE:
                strategy.logger.warning("DWX client inactive; attempting reconnection...")
                strategy._initialize_connection()
            sleep(1)
    except KeyboardInterrupt:
        strategy.shutdown()
