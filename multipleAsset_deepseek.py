import logging
import threading
from queue import Queue
from datetime import datetime, timedelta, timezone
import pandas as pd
import pandas_ta as ta
from apscheduler.schedulers.background import BackgroundScheduler
from api.dwx_client import dwx_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('multi_asset_strategy.log'), logging.StreamHandler()]
)

class AssetManager:
    """Manages trading logic and state for a single asset"""
    def __init__(self, symbol, config, dwx_connection):
        self.symbol = symbol
        self.config = config
        self.dwx = dwx_connection
        self.df = pd.DataFrame()
        self.current_bid = None
        self.current_ask = None
        self.lock = threading.Lock()
        
        # Initialize indicators
        self._calculate_indicators()

    def update_data(self, bid, ask, bar_data=None):
        """Thread-safe data update"""
        with self.lock:
            self.current_bid = bid
            self.current_ask = ask
            if bar_data:
                self._process_bar_data(bar_data)

    def _process_bar_data(self, bar_data):
        """Add new bar and calculate indicators"""
        new_row = pd.DataFrame([{
            'Time': pd.to_datetime(bar_data['time'], unit='s'),
            'Open': float(bar_data['open']),
            'High': float(bar_data['high']),
            'Low': float(bar_data['low']),
            'Close': float(bar_data['close']),
            'Volume': int(bar_data['volume'])
        }])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self._calculate_indicators()

    def _calculate_indicators(self):
        """Calculate technical indicators"""
        if len(self.df) < 100:
            return

        # Core indicators
        self.df["ATR"] = ta.atr(self.df.High, self.df.Low, self.df.Close, length=14)
        self.df["EMA_fast"] = ta.ema(self.df.Close, length=21)
        self.df["EMA_slow"] = ta.ema(self.df.Close, length=50)
        
        # Momentum indicators
        macd = ta.macd(self.df.Close, fast=12, slow=26, signal=9)
        self.df = pd.concat([self.df, macd], axis=1)
        
        # Volatility indicators
        bbands = ta.bbands(self.df.Close, length=20, std=2)
        self.df = pd.concat([self.df, bbands], axis=1)

    def generate_signal(self):
        """Generate trading signal for this asset"""
        if len(self.df) < 50:
            return 0

        current = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        # Trend filter
        if current["EMA_fast"] < current["EMA_slow"]:
            return 0

        # Momentum confirmation
        if current["MACD_12_26_9"] < current["MACDs_12_26_9"]:
            return 0

        # Bollinger Bands logic
        if current["Close"] < current["BBL_20_2.0"]:
            return 2  # Buy signal
        elif current["Close"] > current["BBU_20_2.0"]:
            return 1  # Sell signal
        
        return 0

    def calculate_position_size(self, account_balance):
        """Calculate position size based on volatility and risk"""
        atr = self.df["ATR"].iloc[-1]
        risk_amount = account_balance * self.config['risk_per_trade']
        return min(risk_amount / (atr * self.config['sl_multiplier']), 
                  self.config['max_position_size'])

class MultiAssetStrategy:
    """Main strategy class managing multiple assets"""
    def __init__(self, mt4_path, assets_config):
        self.assets = {}
        self.data_queue = Queue()
        self.scheduler = BackgroundScheduler()
        self.dwx = dwx_client(self, mt4_path, sleep_delay=0.001)
        
        # Initialize asset managers
        for config in assets_config:
            self.assets[config['symbol']] = AssetManager(
                config['symbol'], 
                config,
                self.dwx
            )

        # Start connection and data feeds
        self._initialize_mt4_connection()
        self._start_data_processor()

    def _initialize_mt4_connection(self):
        """Connect to MT4 and subscribe to symbols"""
        try:
            self.dwx.start()
            sleep(1)
            symbols = [asset.symbol for asset in self.assets.values()]
            self.dwx.subscribe_symbols(symbols)
            for config in self.assets.values():
                self.dwx.subscribe_symbols_bar_data([[config.symbol, config.timeframe]])
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            raise

    def _start_data_processor(self):
        """Start background thread for data processing"""
        def process_data():
            while True:
                item = self.data_queue.get()
                if item['type'] == 'tick':
                    self._process_tick_data(item)
                elif item['type'] == 'bar':
                    self._process_bar_data(item)
                self.data_queue.task_done()

        processor_thread = threading.Thread(
            target=process_data, 
            daemon=True
        )
        processor_thread.start()

    def _process_tick_data(self, tick_data):
        """Handle tick updates"""
        asset = self.assets.get(tick_data['symbol'])
        if asset:
            asset.update_data(
                tick_data['bid'], 
                tick_data['ask']
            )

    def _process_bar_data(self, bar_data):
        """Handle bar updates"""
        asset = self.assets.get(bar_data['symbol'])
        if asset:
            asset.update_data(
                asset.current_bid,
                asset.current_ask,
                bar_data
            )

    def on_tick(self, symbol, bid, ask):
        """MT4 tick callback"""
        self.data_queue.put({
            'type': 'tick',
            'symbol': symbol,
            'bid': bid,
            'ask': ask
        })

    def on_bar_data(self, symbol, timeframe, time_val, open_price, high, low, close_price, volume):
        """MT4 bar callback"""
        self.data_queue.put({
            'type': 'bar',
            'symbol': symbol,
            'timeframe': timeframe,
            'time': time_val,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })

    def run_strategy_cycle(self):
        """Main strategy execution cycle"""
        try:
            account_balance = float(self.dwx.account_info.get("balance", 10000))
            total_risk = 0
            
            for symbol, asset in self.assets.items():
                signal = asset.generate_signal()
                if signal == 0:
                    continue

                # Calculate position size
                position_size = asset.calculate_position_size(account_balance)
                spread = asset.current_ask - asset.current_bid
                
                # Check spread limits
                if spread > asset.config['max_spread']:
                    continue

                # Check total exposure
                risk_amount = position_size * asset.config['sl_multiplier'] * asset.df["ATR"].iloc[-1]
                if (total_risk + risk_amount) > asset.config['max_portfolio_risk']:
                    continue

                # Execute trade
                self._execute_trade(asset, signal, position_size)
                total_risk += risk_amount

        except Exception as e:
            logging.error(f"Strategy cycle error: {e}")

    def _execute_trade(self, asset, signal, size):
        """Execute trade for specific asset"""
        try:
            sl_distance = asset.df["ATR"].iloc[-1] * asset.config['sl_multiplier']
            tp_distance = sl_distance * asset.config['tp_ratio']

            if signal == 2:  # Buy
                price = asset.current_ask
                sl = price - sl_distance
                tp = price + tp_distance
            else:  # Sell
                price = asset.current_bid
                sl = price + sl_distance
                tp = price - tp_distance

            self.dwx.open_order(
                symbol=asset.symbol,
                order_type='buy' if signal == 2 else 'sell',
                lots=size,
                stop_loss=sl,
                take_profit=tp,
                order_price=price,
                order_execution_type='limit'
            )
            logging.info(f"Executed {asset.symbol} {'BUY' if signal == 2 else 'SELL'} @ {price}")

        except Exception as e:
            logging.error(f"Trade execution failed for {asset.symbol}: {e}")

    def schedule_jobs(self):
        """Schedule trading and maintenance jobs"""
        self.scheduler.add_job(
            self.run_strategy_cycle,
            'interval',
            minutes=5,
            next_run_time=datetime.now() + timedelta(seconds=30)
        )
        self.scheduler.start()

    def shutdown(self):
        """Clean shutdown procedure"""
        self.dwx.stop()
        self.scheduler.shutdown()
        logging.info("Strategy shutdown complete")

# Configuration Example
ASSETS_CONFIG = [
    {
        'symbol': 'EURUSD',
        'timeframe': 'M15',
        'risk_per_trade': 0.01,
        'sl_multiplier': 1.5,
        'tp_ratio': 2.0,
        'max_spread': 0.0002,
        'max_position_size': 10.0,
        'max_portfolio_risk': 0.05
    },
    {
        'symbol': 'XAUUSD',
        'timeframe': 'M30',
        'risk_per_trade': 0.015,
        'sl_multiplier': 2.0,
        'tp_ratio': 1.8,
        'max_spread': 0.5,
        'max_position_size': 5.0,
        'max_portfolio_risk': 0.07
    }
]

if __name__ == "__main__":
    strategy = MultiAssetStrategy(
        mt4_path='your_mt4_path_here',
        assets_config=ASSETS_CONFIG
    )
    
    try:
        strategy.schedule_jobs()
        while True:
            sleep(1)
    except KeyboardInterrupt:
        strategy.shutdown()