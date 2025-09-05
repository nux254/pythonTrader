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
    def __init__(self, mt4_dir_path, symbols, timeframe, risk_percentage=0.01,
                 min_volatility=0.2, use_limit_orders=True, min_adx=25):
        """
        symbols: either a single symbol (string) or a list of symbols to trade.
        timeframe: e.g. 'M1' or 'M15' for tick-level scalping.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        # Ensure symbols is a list.
        if isinstance(symbols, str):
            self.symbols = [symbols]
        else:
            self.symbols = symbols

        self.timeframe = timeframe
        self.risk_percentage = risk_percentage   # Fraction of account balance risked per trade.
        self.min_volatility = min_volatility       # Minimum ATR required to trade.
        self.use_limit_orders = use_limit_orders
        self.min_adx = min_adx                     # ADX threshold for market regime.

        # Strategy parameters (subject to walk‑forward optimization).
        self.slatrcoef = 1.2   # Multiplier for ATR to compute stop-loss distance.
        self.tpsl_ratio = 2.0  # TP must be at least 2x the SL.
        self.max_spread = 0.45

        # Trailing stop settings.
        self.trailing_stop_multiplier = 0.5  # Fraction of ATR used to trail the stop.
        # For each symbol, we record the active trade (if any).
        self.active_trades = {sym: None for sym in self.symbols}

        # Data storage: one DataFrame per symbol.
        self.data = {sym: pd.DataFrame() for sym in self.symbols}
        self.current_bid = {sym: None for sym in self.symbols}
        self.current_ask = {sym: None for sym in self.symbols}

        # Machine Learning model for adaptive signals.
        # For simplicity, a single ML model is used for all assets.
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
        """Establish connection to MT4 and subscribe to data feeds for each symbol."""
        try:
            self.dwx.start()
            sleep(1)  # Allow connection to establish.
            self.logger.info("Account info: %s", self.dwx.account_info)
            # Subscribe to each symbol.
            self.dwx.subscribe_symbols(self.symbols)
            # Subscribe to bar data for each symbol.
            for sym in self.symbols:
                self.dwx.subscribe_symbols_bar_data([[sym, self.timeframe]])
            self._load_historical_data()
        except Exception as e:
            self.logger.error("Failed to initialize connection: %s", e)
            raise

    def _load_historical_data(self, days=30):
        """Load historical data for each symbol."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        for sym in self.symbols:
            try:
                self.dwx.get_historic_data(
                    sym,
                    self.timeframe,
                    start.timestamp(),
                    end.timestamp()
                )
            except Exception as e:
                self.logger.error("Error loading historical data for %s: %s", sym, e)

    def on_bar_data(self, symbol, time_frame, time_val, open_price, high, low, close_price, tick_volume):
        """Handle incoming bar data for the given symbol."""
        if symbol not in self.symbols or time_frame != self.timeframe:
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
            # Append to the DataFrame for this symbol.
            self.data[symbol] = pd.concat([self.data[symbol], new_row], ignore_index=True)
            self._calculate_indicators(symbol)
            # Train ML model once enough data is collected (shared for all assets).
            if len(self.data[symbol]) >= 100 and not self.ml_trained:
                self.train_ml_model()
        except Exception as e:
            self.logger.error("Error processing bar data for %s: %s", symbol, e)

    def _calculate_indicators(self, symbol):
        """Calculate technical indicators on the latest data for the given symbol."""
        df = self.data[symbol]
        if len(df) < 50:
            return
        df = df.reset_index(drop=True)
        # ATR for volatility.
        df["ATR"] = ta.atr(df.High, df.Low, df.Close, length=7)
        # EMAs for trend detection.
        df["EMA_fast"] = ta.ema(df.Close, length=30)
        df["EMA_slow"] = ta.ema(df.Close, length=50)
        # RSI with traditional length (14) and thresholds.
        df["RSI"] = ta.rsi(df.Close, length=14)
        # Bollinger Bands.
        bbands = ta.bbands(df.Close, length=15, std=1.5)
        for col in bbands.columns:
            df[col] = bbands[col]
        # MACD for momentum/divergence.
        macd = ta.macd(df.Close, fast=12, slow=26)
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        # ADX for market regime detection.
        adx = ta.adx(df.High, df.Low, df.Close)
        df["ADX"] = adx["ADX_14"]
        # Rolling average volume for volume spike detection.
        df["Volume_Avg"] = df.Volume.rolling(window=20).mean()
        self.data[symbol] = df

    def train_ml_model(self):
        """Train an ML model on historical indicator data (aggregated across symbols)."""
        try:
            # For simplicity, we use data from the first symbol with enough data.
            for sym in self.symbols:
                if len(self.data[sym]) >= 100:
                    df = self.data[sym]
                    break
            else:
                return

            features = df[['ATR', 'EMA_fast', 'EMA_slow', 'RSI', 'MACD', 'ADX', 'Volume']].dropna()
            # Label: 1 if next close > current close, else 0.
            labels = (df['Close'].diff().shift(-1) > 0).astype(int).dropna()
            min_len = min(len(features), len(labels))
            features = features.iloc[:min_len]
            labels = labels.iloc[:min_len]
            self.ml_model.fit(features, labels)
            self.ml_trained = True
            self.logger.info("ML model trained on historical data.")
        except Exception as e:
            self.logger.error("ML training error: %s", e)

    def get_ml_signal(self, symbol):
        """Return a signal based on the ML model for the given symbol: 2 for buy, 1 for sell, 0 for neutral."""
        try:
            if not self.ml_trained or len(self.data[symbol]) < 1:
                return 0
            latest = self.data[symbol][['ATR', 'EMA_fast', 'EMA_slow', 'RSI', 'MACD', 'ADX', 'Volume']].iloc[-1:]
            prediction = self.ml_model.predict(latest)[0]
            return 2 if prediction == 1 else 1
        except Exception as e:
            self.logger.error("Error in ML signal for %s: %s", symbol, e)
            return 0

    def _ema_signal(self, symbol, current_idx, backcandles=7):
        """Determine a baseline EMA crossover signal for the given symbol."""
        df = self.data[symbol]
        start_idx = max(0, current_idx - backcandles)
        ema_fast = df['EMA_fast'].iloc[start_idx:current_idx + 1]
        ema_slow = df['EMA_slow'].iloc[start_idx:current_idx + 1]
        if all(ema_fast > ema_slow):
            return 2  # Bullish.
        if all(ema_fast < ema_slow):
            return 1  # Bearish.
        return 0

    def get_alternative_signal(self, symbol):
        """Placeholder for alternative data (e.g. news sentiment) for the given symbol."""
        return self._ema_signal(symbol, len(self.data[symbol]) - 1, backcandles=7)

    def get_signal(self, symbol):
        """Combine technical indicators, ML output, and additional filters to generate a signal for the symbol."""
        df = self.data[symbol]
        if len(df) < 10:
            return 0  # Not enough data.
        current_idx = len(df) - 1

        # Baseline EMA and ML signals.
        ema_signal = self._ema_signal(symbol, current_idx, backcandles=7)
        ml_signal = self.get_ml_signal(symbol)
        alt_signal = self.get_alternative_signal(symbol)

        # Market regime check using ADX.
        current_adx = df['ADX'].iloc[current_idx]
        if current_adx < self.min_adx:
            self.logger.info("For %s: ADX %.2f below threshold %.2f (choppy market); no trade.", symbol, current_adx, self.min_adx)
            return 0

        # Volume spike filter.
        current_volume = df['Volume'].iloc[current_idx]
        avg_volume = df['Volume_Avg'].iloc[current_idx]
        if avg_volume > 0 and current_volume < 2 * avg_volume:
            self.logger.info("For %s: Volume spike not detected (current: %d, avg: %.2f); no trade.", symbol, current_volume, avg_volume)
            return 0

        # Simple MACD divergence check.
        current_macd = df['MACD'].iloc[current_idx]
        prev_macd = df['MACD'].iloc[current_idx - 1] if current_idx > 0 else current_macd
        if ml_signal == 2 and current_macd <= prev_macd:
            self.logger.info("For %s: MACD divergence not confirmed for buy (current: %.2f, previous: %.2f).", symbol, current_macd, prev_macd)
            return 0
        if ml_signal == 1 and current_macd >= prev_macd:
            self.logger.info("For %s: MACD divergence not confirmed for sell (current: %.2f, previous: %.2f).", symbol, current_macd, prev_macd)
            return 0

        # Use ML signal if trained; otherwise fallback to EMA.
        signal = ml_signal if self.ml_trained else ema_signal
        if signal != alt_signal:
            self.logger.info("For %s: Signal mismatch: primary %d vs alternative %d; no trade.", symbol, signal, alt_signal)
            return 0

        return signal

    def compute_position_size(self, stop_loss_distance):
        """Calculate dynamic position size based on account balance and stop-loss distance (global across assets)."""
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
        """Main execution routine that cycles through each symbol, applying filters and executing trades."""
        try:
            if not self.dwx.ACTIVE:
                self.logger.warning("DWX client inactive. Attempting reconnection...")
                self._initialize_connection()
                return

            for sym in self.symbols:
                signal = self.get_signal(sym)
                if self.current_bid[sym] is None or self.current_ask[sym] is None:
                    self.logger.warning("For %s: Tick data not available yet.", sym)
                    continue

                spread = self.current_ask[sym] - self.current_bid[sym]
                if spread > self.max_spread:
                    self.logger.warning("For %s: Spread too wide: %.5f", sym, spread)
                    continue

                df = self.data[sym]
                atr = df['ATR'].iloc[-1]
                if atr < self.min_volatility:
                    self.logger.info("For %s: ATR %.2f below minimum %.2f; no trade.", sym, atr, self.min_volatility)
                    continue

                # Calculate stop-loss and take-profit distances.
                sl_distance = atr * self.slatrcoef
                tp_distance = sl_distance * self.tpsl_ratio
                if tp_distance < 2 * sl_distance:
                    self.logger.info("For %s: Risk-reward ratio insufficient (TP: %.2f, SL: %.2f); no trade.", sym, tp_distance, sl_distance)
                    continue

                pos_size = self.compute_position_size(sl_distance)
                pos_size = max(pos_size, 0.01)

                if not self.dwx.open_orders:
                    if signal == 2:  # Buy signal.
                        stop_loss_price = self.current_bid[sym] - sl_distance
                        take_profit_price = self.current_bid[sym] + tp_distance
                        order_side = 'buy'
                        ref_price = self.current_bid[sym]
                    elif signal == 1:  # Sell signal.
                        stop_loss_price = self.current_ask[sym] + sl_distance
                        take_profit_price = self.current_ask[sym] - tp_distance
                        order_side = 'sell'
                        ref_price = self.current_ask[sym]
                    else:
                        continue

                    self._execute_trade(sym, order_side, pos_size, stop_loss_price, take_profit_price, ref_price)
        except Exception as e:
            self.logger.error("Error in run_strategy: %s", e, exc_info=True)

    def _execute_trade(self, symbol, direction, pos_size, stop_loss, take_profit, ref_price):
        """Execute an order for a given symbol and record the active trade for trailing stops."""
        try:
            order_type = 'limit' if self.use_limit_orders else 'market'
            order_price = ref_price
            if self.use_limit_orders:
                buffer = 0.05  # Adjust as needed.
                order_price = ref_price + buffer if direction == 'buy' else ref_price - buffer

            self.dwx.open_order(
                symbol=symbol,
                order_type=direction,
                take_profit=take_profit,
                stop_loss=stop_loss,
                lots=pos_size,
                order_price=order_price,
                order_execution_type=order_type
            )
            self.logger.info("For %s: Opened %s order at price %.2f (SL: %.2f, TP: %.2f, size: %.4f)",
                             symbol, direction, order_price, stop_loss, take_profit, pos_size)
            # Record the active trade for trailing stop updates.
            self.active_trades[symbol] = {
                'direction': direction,
                'entry_price': ref_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        except Exception as e:
            self.logger.error("Failed to execute %s order for %s: %s", direction, symbol, e)

    def update_trailing_stop(self, symbol, tick_price):
        """Update the stop loss for an active trade for the given symbol to lock in profits."""
        trade = self.active_trades.get(symbol)
        if not trade:
            return
        try:
            direction = trade['direction']
            current_sl = trade['stop_loss']
            atr = self.data[symbol]['ATR'].iloc[-1]
            trail_amount = atr * self.trailing_stop_multiplier
            new_sl = current_sl

            if direction == 'buy' and (tick_price - trail_amount > current_sl):
                new_sl = tick_price - trail_amount
            elif direction == 'sell' and (tick_price + trail_amount < current_sl):
                new_sl = tick_price + trail_amount

            if new_sl != current_sl:
                self.logger.info("For %s: Updating trailing stop from %.2f to %.2f", symbol, current_sl, new_sl)
                # Placeholder: Update order via DWX API if available.
                self.dwx.modify_order(symbol, new_stop_loss=new_sl)
                trade['stop_loss'] = new_sl
        except Exception as e:
            self.logger.error("Error updating trailing stop for %s: %s", symbol, e)

    def on_tick(self, symbol, bid, ask):
        """Handle tick data: update bid/ask for the symbol and adjust trailing stops if necessary."""
        if symbol not in self.symbols:
            return
        self.current_bid[symbol] = bid
        self.current_ask[symbol] = ask
        # Use bid price for buys, ask for sells.
        if self.active_trades.get(symbol) and self.active_trades[symbol]['direction'] == 'buy':
            tick_price = bid
        else:
            tick_price = ask
        self.update_trailing_stop(symbol, tick_price)

    def on_message(self, message):
        """Process messages from DWX with appropriate logging."""
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
        """Process historical data from DWX for a given symbol."""
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
                        if len(self.data[symbol]) > 0:
                            last_close = self.data[symbol]['Close'].iloc[-1]
                            op = hi = lo = cl = last_close
                        else:
                            self.logger.warning("For %s: No previous data to fill missing bar: %s", symbol, bar)
                            continue
                        vol = 0
                    else:
                        self.logger.warning("For %s: Skipping historic bar due to unexpected format: %s", symbol, bar)
                        continue
                else:
                    self.logger.error("For %s: Unknown historic bar type: %s", symbol, type(bar))
                    continue

                self.on_bar_data(symbol, time_frame, t, op, hi, lo, cl, vol)
            except Exception as e:
                self.logger.error("Error processing historic bar for %s: %s", symbol, e)

    def optimize_parameters(self):
        """Placeholder: Optimize parameters using walk-forward validation across each symbol."""
        from backtesting import Backtest, Strategy

        class OptimizationStrategy(Strategy):
            def init(self):
                self.atr = self.I(ta.atr, self.data.High, self.data.Low, self.data.Close, length=7)

            def next(self):
                pass  # Extend with further optimization logic.

        try:
            for sym in self.symbols:
                df = self.data[sym]
                if len(df) < 50:
                    continue
                bt = Backtest(df, OptimizationStrategy, cash=250, margin=1/30)
                stats = bt.optimize(
                    sl_coef=[i/10 for i in range(10, 26)],
                    tp_ratio=[i/10 for i in range(10, 26)],
                    maximize='Return [%]',
                    max_tries=100
                )
                self.slatrcoef = stats['sl_coef']
                self.tpsl_ratio = stats['tp_ratio']
                self.logger.info("For %s: Optimized parameters updated: SL Coef=%.2f, TP Ratio=%.2f", sym, self.slatrcoef, self.tpsl_ratio)
        except Exception as e:
            self.logger.error("Parameter optimization failed: %s", e)

    def wait_for_next_bar(self):
        """Wait until the start of the next minute bar."""
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
            # Optimization job: run every Sunday at 23:00 UTC.
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
    # Instantiate the strategy for multiple assets.
    strategy = ScalpingStrategy(
        mt4_dir_path='C:/Users/User/AppData/Roaming/MetaQuotes/Terminal/B7238334EF4B2B20A39D097B2877DE6E/MQL4/Files/',
        symbols=['XAUUSD', 'EURUSD', 'GBPUSD'],  # List of assets/pairs.
        timeframe='M1',  # Use M1 or M15 for tick-level scalping.
        risk_percentage=0.01,  # Risk 1% of account balance per trade.
        min_volatility=0.2,
        use_limit_orders=True,
        min_adx=25
    )

    try:
        strategy.schedule_jobs()
        # Main loop: Monitor connection health.
        while True:
            if not strategy.dwx.ACTIVE:
                strategy.logger.warning("DWX client inactive; attempting reconnection...")
                strategy._initialize_connection()
            sleep(1)
    except KeyboardInterrupt:
        strategy.shutdown()
