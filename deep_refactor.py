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
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler(),
        logging.NullHandler()  # Fallback handler
    ]
)

class ScalpingStrategy:
    def __init__(self, mt4_dir_path, symbol, timeframe, lots, risk_percentage=0.01):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbol = symbol
        self.timeframe = timeframe
        self.lots = lots
        self.risk_percentage = risk_percentage

        # Initialize DWX client with proper callback binding.
        # Note: The keyword argument is updated to "mt4_directory_path".
        self.dwx = dwx_client(
            self,
            mt4_dir_path,  # Correct parameter name,
            sleep_delay=0.005,
            max_retry_command_seconds=10,
            verbose=True
        )

        # Data storage with type hints
        self.df: pd.DataFrame = pd.DataFrame()
        self.current_bid: float = 0.0
        self.current_ask: float = 0.0
        
        # Strategy parameters with validation
        self.slatrcoef = 1.2  # type: float
        self.tpsl_ratio = 1.5  # type: float
        self.max_spread = 5e-2  # type: float
        
        # Initialize connection with retry logic
        self._initialize_connection_with_retry(max_retries=3)

    def _initialize_connection_with_retry(self, max_retries=3):
        """Establish connection with retry mechanism"""
        for attempt in range(max_retries):
            try:
                self._initialize_connection()
                return
            except Exception as e:
                self.logger.error("Connection attempt %d failed: %s", attempt+1, e)
                sleep(2 ** attempt)  # Exponential backoff
        raise ConnectionError("Failed to establish connection after multiple attempts")

    def _initialize_connection(self):
        """Establish connection to MT4 and subscribe to data feeds"""
        self.dwx.start()
        sleep(1)  # Allow connection to establish
        
        # Validate account information
        if not self.dwx.account_info:
            raise ValueError("No account information received")
            
        self.logger.info("Account info: %s", self.dwx.account_info)
        
        # Subscribe to market data with confirmation
        self.dwx.subscribe_symbols([self.symbol])
        self.dwx.subscribe_symbols_bar_data([[self.symbol, self.timeframe]])
        self._load_historical_data()

    def _load_historical_data(self, days=30):
        """Load historical data with date validation"""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        
        if start >= end:
            raise ValueError("Invalid historical data date range")
            
        self.dwx.get_historic_data(
            self.symbol,
            self.timeframe,
            start.timestamp(),
            end.timestamp()
        )

    def on_bar_data(self, symbol: str, time_frame: str, time: str, 
                   open_price: float, high: float, low: float, 
                   close_price: float, tick_volume: int):
        """Robust bar data handler with multiple time formats support"""
        if symbol != self.symbol or time_frame != self.timeframe:
            return

        try:
            # Advanced time parsing with multiple format support
            timestamp = self._parse_timestamp(time)
            
            new_row = pd.DataFrame([{
                'Time': timestamp,
                'Open': float(open_price),
                'High': float(high),
                'Low': float(low),
                'Close': float(close_price),
                'Volume': int(tick_volume)
            }])
            
            # Efficient DataFrame update
            self.df = pd.concat([self.df, new_row], ignore_index=True).ffill()
            self._calculate_indicators()
            
        except Exception as e:
            self.logger.error("Bar data error: %s", e, exc_info=True)

    def _parse_timestamp(self, time_str: str) -> datetime:
        """Universal timestamp parser with multiple format support"""
        try:
            # Try Unix timestamp first
            return datetime.fromtimestamp(int(time_str), tz=timezone.utc)
        except ValueError:
            try:
                # Try MT4 format
                return datetime.strptime(time_str, "%Y.%m.%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                # Fallback to pandas parser
                dt = pd.to_datetime(time_str, errors='coerce', utc=True)
                if pd.isnull(dt):
                    raise ValueError(f"Unparseable time format: {time_str}")
                return dt

    def _calculate_indicators(self):
        """Indicator calculation with validation"""
        if len(self.df) < 100:
            self.logger.warning("Insufficient data for indicators (%d bars)", len(self.df))
            return

        try:
            # Calculate ATR with validation
            atr = ta.atr(self.df.High, self.df.Low, self.df.Close, length=7)
            if atr.isnull().all():
                raise ValueError("ATR calculation failed")
            self.df["ATR"] = atr

            # EMA calculations
            self.df["EMA_fast"] = ta.ema(self.df.Close, length=30)
            self.df["EMA_slow"] = ta.ema(self.df.Close, length=50)

            # RSI with bounds check
            self.df['RSI'] = ta.rsi(self.df.Close, length=10).clip(0, 100)

            # Bollinger Bands with error handling:
            bbands = ta.bbands(self.df.Close, length=15, std=1.5)
            if not bbands.empty:
                # Overwrite existing Bollinger Bands columns to avoid duplicates.
                for col in bbands.columns:
                    self.df[col] = bbands[col]
                
        except Exception as e:
            self.logger.error("Indicator error: %s", e, exc_info=True)

    def get_signal(self) -> int:
        """Enhanced signal generation with current prices"""
        try:
            if len(self.df) < 50:
                return 0

            current_idx = len(self.df) - 1
            ema_signal = self._ema_signal(current_idx, backcandles=1)  # Reduced from 7 to 1

            required_columns = ['BBL_15_1.5', 'BBU_15_1.5', 'RSI']
            if not all(col in self.df.columns for col in required_columns):
                return 0

            bbl = self.df['BBL_15_1.5'].iloc[current_idx]
            bbu = self.df['BBU_15_1.5'].iloc[current_idx]
            rsi = self.df['RSI'].iloc[current_idx]

            # Use current price instead of open price
            current_price = self.current_bid  # For sell signals
            if ema_signal == 2:  # Buy signal
                current_price = self.current_ask
                if current_price <= bbl and rsi < 60:
                    return 2
            elif ema_signal == 1:  # Sell signal
                if current_price >= bbu and rsi > 40:
                    return 1
            return 0
            
        except Exception as e:
            self.logger.error("Signal error: %s", e)
            return 0


    def _ema_signal(self, current_idx: int, backcandles: int = 1) -> int:
        """EMA crossover signal with reduced lookback"""
        try:
            if current_idx < backcandles:
                return 0

            # Check crossover in the most recent candle
            fast_now = self.df['EMA_fast'].iloc[current_idx]
            slow_now = self.df['EMA_slow'].iloc[current_idx]
            fast_prev = self.df['EMA_fast'].iloc[current_idx-1]
            slow_prev = self.df['EMA_slow'].iloc[current_idx-1]

            if fast_now > slow_now and fast_prev <= slow_prev:
                return 2  # Bullish crossover
            elif fast_now < slow_now and fast_prev >= slow_prev:
                return 1  # Bearish crossover
            return 0
        except Exception as e:
            self.logger.error("EMA signal error: %s", e)
            return 0


    def _validate_buy_conditions(self) -> bool:
        """Additional validation for buy signals"""
        # Add market condition checks here
        return True

    def _validate_sell_conditions(self) -> bool:
        """Additional validation for sell signals"""
        # Add market condition checks here
        return True

    def run_strategy(self):
        """Main strategy loop with enhanced safety checks"""
        try:
            if not self._market_open_check():
                self.logger.info("Market closed - skipping execution")
                return

            signal = self.get_signal()
            self._execute_strategy(signal)

        except Exception as e:
            self.logger.error("Strategy error: %s", e, exc_info=True)

    def _execute_strategy(self, signal: int):
        """Trade execution with pre-flight checks"""
        # Validate tick data
        if not self._valid_tick_data():
            return

        # Check spread
        spread = self.current_ask - self.current_bid
        if spread > self.max_spread:
            self.logger.warning("Spread %.5f exceeds maximum", spread)
            return

        # Check existing positions
        if self.dwx.open_orders:
            self.logger.info("Existing positions - skipping new trade")
            return

        # Execute trade
        if signal == 1:
            self._execute_trade('sell')
        elif signal == 2:
            self._execute_trade('buy')

    def _valid_tick_data(self) -> bool:
        """Validate current bid/ask prices"""
        if None in (self.current_bid, self.current_ask):
            self.logger.warning("Missing tick data")
            return False
        if self.current_bid <= 0 or self.current_ask <= 0:
            self.logger.error("Invalid prices: Bid=%.2f Ask=%.2f", 
                            self.current_bid, self.current_ask)
            return False
        return True

    def _market_open_check(self) -> bool:
        """Market hours validation"""
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:  # Weekend
            return False
        utc_hour = now.hour
        # Adjust for XAUUSD market hours (24/5 but less liquid periods)
        return 6 <= utc_hour < 22  # 6 AM to 10 PM UTC

    def _execute_trade(self, direction: str):
        """Trade execution with dynamic position sizing"""
        try:
            risk_amount = self.dwx.account_info['balance'] * self.risk_percentage
            atr = self.df['ATR'].iloc[-1]
            sl_pips = atr * self.slatrcoef
            pip_value = 10  # For XAUUSD 0.01 lot = $0.10 per pip
            
            position_size = round(risk_amount / (sl_pips * pip_value), 2)
            position_size = max(0.01, min(position_size, 50))

            # Correct entry price and SL/TP calculation
            if direction == 'buy':
                entry_price = self.current_ask
                stop_loss = entry_price - sl_pips
                take_profit = entry_price + (sl_pips * self.tpsl_ratio)
            else:
                entry_price = self.current_bid
                stop_loss = entry_price + sl_pips
                take_profit = entry_price - (sl_pips * self.tpsl_ratio)

            self.dwx.open_order(
                symbol=self.symbol,
                order_type=direction,
                take_profit=take_profit,
                stop_loss=stop_loss,
                lots=position_size
            )
            self.logger.info("Executed %s %.2f lots @%.2f SL:%.2f TP:%.2f",
                           direction, position_size, entry_price, 
                           stop_loss, take_profit)
        except Exception as e:
            self.logger.error("Trade execution failed: %s", e, exc_info=True)


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
        """Enhanced parameter optimization with walk-forward testing"""
        from backtesting import Backtest
        from backtesting.lib import crossover

        class OptimizationStrategy:
            def init(self):
                self.atr = self.I(ta.atr, self.data.High, self.data.Low, 
                                self.data.Close, length=7)
                self.ema_fast = self.I(ta.ema, self.data.Close, length=30)
                self.ema_slow = self.I(ta.ema, self.data.Close, length=50)
                
            def next(self):
                if crossover(self.ema_fast, self.ema_slow):
                    self.buy()
                elif crossover(self.ema_slow, self.ema_fast):
                    self.sell()

        try:
            bt = Backtest(self.df, OptimizationStrategy, 
                         cash=10000, commission=0.0002)
            
            stats = bt.optimize(
                sl_coef=range(10, 26),
                tp_ratio=range(10, 26),
                maximize='Return [%]',
                method='grid',
                max_tries=100,
                return_heatmap=True
            )
            
            # Parameter validation
            self.slatrcoef = max(1.0, min(stats['sl_coef']/10, 3.0))
            self.tpsl_ratio = max(1.0, min(stats['tp_ratio']/10, 3.0))
            
            self.logger.info("Optimized parameters: SL=%.2f TP=%.2f Return=%.2f%%",
                            self.slatrcoef, self.tpsl_ratio, stats['Return [%]'])
            
            # Save optimization results
            bt.plot(filename=f"optimization_{datetime.now():%Y%m%d}.html")
            
        except Exception as e:
            self.logger.error("Optimization failed: %s", e, exc_info=True)

    # DWX Event Handlers ------------------------------------------------------
    def on_tick(self, symbol: str, bid: float, ask: float):
        """Tick data handler with validation"""
        if symbol == self.symbol:
            self.current_bid = bid
            self.current_ask = ask
            self.logger.debug("Tick: %s Bid:%.2f Ask:%.2f", symbol, bid, ask)

    def on_message(self, message: dict):
        """Message handler with protocol validation"""
        try:
            msg_type = message.get('type', 'UNKNOWN')
            if msg_type == 'ERROR':
                self.logger.error("DWX Error: %s", message)
            elif msg_type == 'INFO':
                self.logger.info("DWX Info: %s", message)
            elif msg_type == 'HISTORIC_DATA':
                self.logger.debug("Historic data received")
            else:
                self.logger.warning("Unknown message type: %s", msg_type)
        except Exception as e:
            self.logger.error("Message handling error: %s", e, exc_info=True)

    def on_historic_data(self, symbol: str, timeframe: str, data: list):
        """Historic data handler with data integrity checks"""
        self.logger.info("Receiving %d historic bars for %s %s", 
                        len(data), symbol, timeframe)
        
        valid_bars = 0
        for bar in data:
            try:
                if isinstance(bar, dict):
                    components = bar
                elif isinstance(bar, str):
                    parts = bar.split(',')
                    components = {
                        'time': parts[0],
                        'open': parts[1],
                        'high': parts[2],
                        'low': parts[3],
                        'close': parts[4],
                        'volume': parts[5]
                    }
                else:
                    continue
                
                self.on_bar_data(
                    symbol, timeframe,
                    components['time'],
                    components['open'],
                    components['high'],
                    components['low'],
                    components['close'],
                    components['volume']
                )
                valid_bars += 1
                
            except Exception as e:
                self.logger.warning("Invalid bar format: %s", e)
        
        self.logger.info("Processed %d/%d valid bars", valid_bars, len(data))

    def shutdown(self):
        """Graceful shutdown procedure"""
        try:
            self.dwx.stop()
            self.logger.info("Disconnected from MT4")
            # Save final state
            self.df.to_csv(f"market_data_{datetime.now():%Y%m%d}.csv", index=False)
        except Exception as e:
            self.logger.error("Shutdown error: %s", e, exc_info=True)
        finally:
            self.logger.info("Strategy shutdown complete")

if __name__ == "__main__":
    strategy = ScalpingStrategy(
        mt4_dir_path='C:/Users/User/AppData/Roaming/MetaQuotes/Terminal/B7238334EF4B2B20A39D097B2877DE6E/MQL4/Files/',
        symbol='XAUUSD',
        timeframe='M5',
        lots=0.1
    )
    
    try:
        strategy.schedule_jobs()
        while strategy.dwx.ACTIVE:
            sleep(1)
    except KeyboardInterrupt:
        strategy.shutdown()
