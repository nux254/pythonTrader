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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('enhanced_strategy.log'), logging.StreamHandler()]
)

class EnhancedScalpingStrategy:
    def __init__(self, mt4_dir_path, symbol, timeframe, risk_percentage=0.01,
                 min_volatility=0.2, pip_size=0.1, contract_size=100,
                 trailing_atr_coef=1.5, adx_threshold=25, volume_spike_multiplier=1.5):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_percentage = risk_percentage
        self.min_volatility = min_volatility
        self.pip_size = pip_size
        self.contract_size = contract_size
        self.pip_value = pip_size * contract_size
        self.trailing_atr_coef = trailing_atr_coef
        self.adx_threshold = adx_threshold
        self.volume_spike_multiplier = volume_spike_multiplier

        # Strategy parameters
        self.slatrcoef = 2.0
        self.tpsl_ratio = 2.0
        self.max_spread = 0.5

        # Data storage
        self.df = pd.DataFrame()
        self.current_bid = None
        self.current_ask = None

        # Initialize DWX client
        self.dwx = dwx_client(
            self,
            mt4_dir_path,
            sleep_delay=0.005,
            max_retry_command_seconds=10,
            verbose=True
        )

        self._initialize_connection()

    def _initialize_connection(self):
        try:
            self.dwx.start()
            sleep(1)
            self.dwx.subscribe_symbols([self.symbol])
            self.dwx.subscribe_symbols_bar_data([[self.symbol, self.timeframe]])
            self._load_historical_data()
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            raise

    def _load_historical_data(self, days=7):
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
            self.logger.error(f"History load error: {e}")

    def on_bar_data(self, symbol, time_frame, time_val, open_price, high, low, close_price, tick_volume):
        if symbol != self.symbol or time_frame != self.timeframe:
            return

        try:
            timestamp = pd.to_datetime(time_val, unit='s')
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
            self.logger.error(f"Bar data error: {e}")

    def _calculate_indicators(self):
        if len(self.df) < 100:
            return

        self.df["ATR"] = ta.atr(self.df.High, self.df.Low, self.df.Close, length=14)
        self.df["EMA_fast"] = ta.ema(self.df.Close, length=21)
        self.df["EMA_slow"] = ta.ema(self.df.Close, length=50)
        self.df["RSI"] = ta.rsi(self.df.Close, length=14)
        
        # Bollinger Bands with 2 std dev
        bbands = ta.bbands(self.df.Close, length=20, std=2)
        self.df = pd.concat([self.df, bbands], axis=1)
        
        # Volume and Momentum Indicators
        self.df["Volume_MA"] = ta.sma(self.df.Volume, length=20)
        macd = ta.macd(self.df.Close, fast=12, slow=26, signal=9)
        self.df = pd.concat([self.df, macd], axis=1)
        
        # Trend Strength
        adx_data = ta.adx(self.df.High, self.df.Low, self.df.Close, length=14)
        self.df["ADX"] = adx_data["ADX_14"]

    def get_signal(self):
        if len(self.df) < 50:
            return 0

        current = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        # Trend Filter
        if current["ADX"] < self.adx_threshold:
            self.logger.debug("ADX below threshold - no clear trend")
            return 0

        # Volume Filter
        if current["Volume"] < self.volume_spike_multiplier * current["Volume_MA"]:
            self.logger.debug("Insufficient volume spike")
            return 0

        # MACD Crossover
        macd_bullish = current["MACD_12_26_9"] > current["MACDs_12_26_9"]
        macd_bearish = current["MACD_12_26_9"] < current["MACDs_12_26_9"]

        # EMA Crossover
        ema_bullish = current["EMA_fast"] > current["EMA_slow"]
        ema_bearish = current["EMA_fast"] < current["EMA_slow"]

        # Bollinger Bands & RSI
        bb_low = current["BBL_20_2.0"]
        bb_high = current["BBU_20_2.0"]
        rsi_oversold = current["RSI"] < 30
        rsi_overbought = current["RSI"] > 70

        if ema_bullish and macd_bullish and current["Close"] < bb_low and rsi_oversold:
            return 2  # Buy signal

        if ema_bearish and macd_bearish and current["Close"] > bb_high and rsi_overbought:
            return 1  # Sell signal

        return 0

    def compute_position_size(self, stop_loss_pips):
        try:
            balance = float(self.dwx.account_info.get("balance", 10000))
        except:
            balance = 10000
            
        risk_amount = balance * self.risk_percentage
        risk_per_lot = stop_loss_pips * self.pip_value
        return max(risk_amount / risk_per_lot, 0.01)

    def run_strategy(self):
        try:
            if not self.dwx.ACTIVE:
                self._initialize_connection()
                return

            self._manage_trailing_stops()
            signal = self.get_signal()
            
            if signal == 0:
                return

            spread = self.current_ask - self.current_bid
            if spread > self.max_spread:
                return

            atr = self.df["ATR"].iloc[-1]
            sl_distance = atr * self.slatrcoef
            tp_distance = sl_distance * self.tpsl_ratio
            
            if (tp_distance / sl_distance) < 2.0:
                self.logger.warning("Risk-reward ratio below 2:1")
                return

            stop_loss_pips = sl_distance / self.pip_size
            position_size = self.compute_position_size(stop_loss_pips)

            if signal == 2:
                price = self.current_bid
                sl = price - sl_distance
                tp = price + tp_distance
            else:
                price = self.current_ask
                sl = price + sl_distance
                tp = price - tp_distance

            self._execute_trade(signal, position_size, price, sl, tp)
        except Exception as e:
            self.logger.error(f"Strategy error: {e}")

    def _execute_trade(self, direction, size, price, sl, tp):
        try:
            order_type = 'buy' if direction == 2 else 'sell'
            self.dwx.open_order(
                symbol=self.symbol,
                order_type=order_type,
                lots=size,
                stop_loss=sl,
                take_profit=tp,
                order_price=price,
                order_execution_type='limit'
            )
            self.logger.info(f"Opened {order_type} order at {price}")
        except Exception as e:
            self.logger.error(f"Order failed: {e}")

    def _manage_trailing_stops(self):
        if not self.dwx.open_orders:
            return

        current_atr = self.df["ATR"].iloc[-1]
        trail_distance = current_atr * self.trailing_atr_coef

        for order in self.dwx.open_orders:
            if order['symbol'] != self.symbol:
                continue

            current_price = self.current_bid if order['type'] == 'buy' else self.current_ask
            new_sl = None

            if order['type'] == 'buy':
                new_sl = current_price - trail_distance
                if new_sl > order['stop_loss']:
                    self.dwx.modify_order(order['ticket'], stop_loss=new_sl)
            else:
                new_sl = current_price + trail_distance
                if new_sl < order['stop_loss']:
                    self.dwx.modify_order(order['ticket'], stop_loss=new_sl)

    def schedule_jobs(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.run_strategy,
            trigger=CronTrigger(
                minute=f'*/{self.timeframe[1:]}',
                hour='6-21',
                day_of_week='mon-fri',
                timezone='UTC'
            )
        )
        self.scheduler.add_job(
            self.optimize_parameters,
            trigger=CronTrigger(
                day_of_week='sun',
                hour='23',
                timezone='UTC'
            )
        )
        self.scheduler.start()

    def optimize_parameters(self):
        from backtesting import Backtest, Strategy

        class WalkForwardStrategy(Strategy):
            def init(self):
                self.atr = self.I(ta.atr, self.data.High, self.data.Low, self.data.Close, 14)
                self.adx = self.I(ta.adx, self.data.High, self.data.Low, self.data.Close, 14)

            def next(self):
                pass

        best_params = None
        best_sharpe = -float('inf')

        for fold in range(3):
            train_size = int(len(self.df) * 0.6)
            test_size = int(len(self.df) * 0.2)
            train_start = fold * test_size
            train_end = train_start + train_size
            test_end = train_end + test_size

            train_df = self.df.iloc[train_start:train_end]
            test_df = self.df.iloc[train_end:test_end]

            bt = Backtest(train_df, WalkForwardStrategy, cash=10000, margin=1/30)
            stats = bt.optimize(
                slatrcoef=[x/10 for x in range(15, 31)],
                tpsl_ratio=[2.0],
                maximize='Sharpe Ratio',
                max_tries=50
            )

            validate = Backtest(test_df, WalkForwardStrategy, cash=10000, margin=1/30)
            test_stats = validate.run(**stats._params)
            
            if test_stats['Sharpe Ratio'] > best_sharpe:
                best_sharpe = test_stats['Sharpe Ratio']
                best_params = stats._params

        if best_params:
            self.slatrcoef = best_params['slatrcoef']
            self.tpsl_ratio = best_params['tpsl_ratio']
            self.logger.info(f"Optimized params: SL={self.slatrcoef:.1f} TP={self.tpsl_ratio:.1f}")

    def on_tick(self, symbol, bid, ask):
        if symbol == self.symbol:
            self.current_bid = bid
            self.current_ask = ask

    def shutdown(self):
        try:
            self.dwx.stop()
            self.scheduler.shutdown()
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    strategy = EnhancedScalpingStrategy(
        mt4_dir_path='YOUR_MT4_PATH',
        symbol='XAUUSD',
        timeframe='M15',
        risk_percentage=0.02,
        pip_size=0.1,
        contract_size=100
    )
    
    try:
        strategy.schedule_jobs()
        while True:
            sleep(1)
    except KeyboardInterrupt:
        strategy.shutdown()