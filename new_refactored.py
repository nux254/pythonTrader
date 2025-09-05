import json
import logging
import math
from time import sleep, time
from datetime import datetime, timedelta, timezone
import pandas as pd
import pandas_ta as ta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from api.dwx_client import dwx_client

# Configure logging: log both to file and console.
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
        self.risk_percentage = risk_percentage  # fraction of account balance risked per trade
        self.min_volatility = min_volatility    # minimum ATR required to trade
        self.use_limit_orders = use_limit_orders  # flag to use limit orders if possible

        # Strategy parameters: these will be optimized periodically.
        self.slatrcoef = 1.2   # Multiplier for ATR to compute stop loss distance.
        self.tpsl_ratio = 1.5  # Ratio to derive take profit distance from stop loss distance.
        self.max_spread = 0.45 # Maximum acceptable spread for trade execution.

        # Data storage
        self.df = pd.DataFrame()
        self.current_bid = None
        self.current_ask = None

        # Initialize DWX client with robust connection settings.
        self.dwx = dwx_client(
            self,
            mt4_dir_path,
            sleep_delay=0.005,
            max_retry_command_seconds=10,
            verbose=True
        )

        # Assign callback methods.
        self.on_historic_data = self.on_historic_data
        self.on_message = self.on_message
        self.on_tick = self.on_tick

        # Initialize connection.
        self._initialize_connection()

    def _initialize_connection(self):
        """Establish connection to MT4 and subscribe to data feeds, with retry."""
        try:
            self.dwx.start()
            sleep(1)  # Allow connection to establish.
            self.logger.info("Account info: %s", self.dwx.account_info)

            # Subscribe to market data.
            self.dwx.subscribe_symbols([self.symbol])
            self.dwx.subscribe_symbols_bar_data([[self.symbol, self.timeframe]])

            # Load historical data.
            self._load_historical_data()

        except Exception as e:
            self.logger.error("Failed to initialize connection: %s", e)
            # Optionally, implement reconnection logic here.
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
            # Try to convert the time input.
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
        """Calculate technical indicators on latest data; require minimum bars."""
        if len(self.df) < 50:
            return
        # Ensure unique index.
        self.df = self.df.reset_index(drop=True)
        # Calculate technical indicators.
        self.df["ATR"] = ta.atr(self.df.High, self.df.Low, self.df.Close, length=7)
        self.df["EMA_fast"] = ta.ema(self.df.Close, length=30)
        self.df["EMA_slow"] = ta.ema(self.df.Close, length=50)
        self.df["RSI"] = ta.rsi(self.df.Close, length=10)
        bbands = ta.bbands(self.df.Close, length=15, std=1.5)
        for col in bbands.columns:
            self.df[col] = bbands[col]

    def get_signal(self):
        """Generate trading signal based on current market conditions and alternative data."""
        if len(self.df) < 10:
            return 0  # Not enough data.
        current_idx = len(self.df) - 1
        ema_signal = self._ema_signal(current_idx, backcandles=7)
        bbl = self.df['BBL_15_1.5'].iloc[current_idx]
        bbu = self.df['BBU_15_1.5'].iloc[current_idx]
        rsi = self.df['RSI'].iloc[current_idx]
        open_price = self.df['Open'].iloc[current_idx]
        atr = self.df['ATR'].iloc[current_idx]

        # Check volatility filter: Only trade if ATR exceeds a minimum threshold.
        if atr < self.min_volatility:
            self.logger.info("ATR %.2f below min volatility %.2f; no trade", atr, self.min_volatility)
            return 0

        # Incorporate a dummy alternative data signal (e.g. news sentiment)
        alt_signal = self.get_alternative_signal()
        # Combine signals: For instance, require that alternative signal agrees.
        if alt_signal != ema_signal:
            return 0

        if ema_signal == 2 and open_price <= bbl and rsi < 60:
            return 2  # Buy signal.
        if ema_signal == 1 and open_price >= bbu and rsi > 40:
            return 1  # Sell signal.
        return 0

    def _ema_signal(self, current_idx, backcandles=7):
        """Determine EMA crossover signal over a lookback period."""
        start_idx = max(0, current_idx - backcandles)
        ema_fast = self.df['EMA_fast'].iloc[start_idx:current_idx+1]
        ema_slow = self.df['EMA_slow'].iloc[start_idx:current_idx+1]
        if all(ema_fast > ema_slow):
            return 2  # Bullish trend.
        if all(ema_fast < ema_slow):
            return 1  # Bearish trend.
        return 0

    def get_alternative_signal(self):
        """Placeholder for alternative data integration (e.g., news sentiment). 
        Return 2 for bullish, 1 for bearish, 0 for neutral.
        Here we simply return the EMA signal to simulate agreement."""
        # In production, you could integrate an API for news sentiment, etc.
        return self._ema_signal(len(self.df) - 1, backcandles=7)

    def compute_position_size(self, stop_loss_distance):
        """Calculate dynamic position size based on account balance, risk percentage, and stop loss."""
        try:
            account_balance = float(self.dwx.account_info.get("balance", 10000))  # Default to 10k if not available.
        except Exception:
            account_balance = 10000
        risk_amount = account_balance * self.risk_percentage
        # Position size (in lots or units) is risk amount divided by stop loss distance.
        pos_size = risk_amount / stop_loss_distance
        self.logger.info("Calculated position size: %.4f units (balance %.2f, risk %.2f, stop loss %.2f)", pos_size, account_balance, risk_amount, stop_loss_distance)
        return pos_size

    def run_strategy(self):
        """Main strategy execution method with enhanced error handling and reconnection logic."""
        try:
            # Check connection; if inactive, try to reconnect.
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

            # Check for volatility filter already applied in get_signal.
            if signal == 0:
                self.logger.debug("No valid signal generated.")
                return

            # Calculate stop loss and take profit.
            atr = self.df['ATR'].iloc[-1]
            sl_distance = atr * self.slatrcoef
            tp_distance = sl_distance * self.tpsl_ratio

            # Dynamic position sizing based on risk.
            pos_size = self.compute_position_size(sl_distance)
            # For simplicity, ensure pos_size is at least 0.01.
            pos_size = max(pos_size, 0.01)

            # Execute order only if no open orders.
            if not self.dwx.open_orders:
                if signal == 2:  # Buy signal.
                    # For buy, use current_bid as reference price.
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
        """Execute trade using either limit or market orders."""
        try:
            order_type = 'limit' if self.use_limit_orders else 'market'
            order_price = ref_price
            if self.use_limit_orders:
                # For limit orders, adjust price by a small buffer to improve fill probability.
                buffer = 0.05  # Example buffer; adjust as needed.
                if direction == 'buy':
                    order_price = ref_price + buffer
                else:
                    order_price = ref_price - buffer

            self.dwx.open_order(
                symbol=self.symbol,
                order_type=direction,
                take_profit=take_profit,
                stop_loss=stop_loss,
                lots=pos_size,
                order_price=order_price,   # New parameter for limit orders.
                order_execution_type=order_type
            )
            self.logger.info("Opened %s order at price %.2f (stop loss: %.2f, take profit: %.2f, size: %.4f)",
                             direction, order_price, stop_loss, take_profit, pos_size)
        except Exception as e:
            self.logger.error("Failed to execute %s order: %s", direction, e)

    def schedule_jobs(self):
        """Schedule trading and optimization jobs using a background scheduler."""
        self.scheduler = BackgroundScheduler()
        try:
            # Trading job: runs every 5 minutes between 06:00 and 21:00 UTC, Mon-Fri.
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

    def optimize_parameters(self):
        """Optimize strategy parameters using historical data and a rolling window approach."""
        from backtesting import Backtest, Strategy

        class OptimizationStrategy(Strategy):
            def init(self):
                self.atr = self.I(ta.atr, self.data.High, self.data.Low, self.data.Close, length=7)
            def next(self):
                pass  # Add more optimization logic if desired.

        try:
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

    def on_tick(self, symbol, bid, ask):
        """Handle incoming tick data: update current bid/ask."""
        if symbol == self.symbol:
            self.current_bid = bid
            self.current_ask = ask

    def on_message(self, message):
        """Handle incoming messages from DWX with error categorization."""
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
        """Process incoming historical data from DWX."""
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
        """Perform a clean shutdown: stop DWX client and scheduler."""
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
        """Wait until the next minute bar starts (more precise than sleep(60))."""
        now = datetime.now()
        next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        pause = math.ceil((next_min - now).total_seconds())
        self.logger.info("Sleeping for %d seconds until next bar", pause)
        sleep(pause)


if __name__ == "__main__":
    # Instantiate the strategy with improved risk management.
    strategy = ScalpingStrategy(
        mt4_dir_path='C:/Users/User/AppData/Roaming/MetaQuotes/Terminal/B7238334EF4B2B20A39D097B2877DE6E/MQL4/Files/',
        symbol='XAUUSD',
        timeframe='M5',
        risk_percentage=0.01,  # Risk 1% of account balance per trade.
        min_volatility=0.2,    # Only trade if ATR >= 0.2 (this threshold can be optimized).
        use_limit_orders=True  # Attempt to use limit orders to reduce slippage.
    )

    try:
        strategy.schedule_jobs()
        # Main loop: in addition to scheduler jobs, monitor connection health.
        while True:
            if not strategy.dwx.ACTIVE:
                strategy.logger.warning("DWX client inactive; attempting reconnection...")
                strategy._initialize_connection()
            sleep(1)
    except KeyboardInterrupt:
        strategy.shutdown()

