import logging
import threading
from time import sleep
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from api.dwx_client import dwx_client
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('multi_asset_strategy.log'), logging.StreamHandler()]
)

class Asset:
    """Manages all asset-specific trading logic and data."""
    def __init__(self, symbol: str, config: dict):
        self.symbol = symbol
        self.config = config
        self.data = pd.DataFrame()
        self.current_bid: Optional[float] = None
        self.current_ask: Optional[float] = None
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_trained = False
        self.active_trade: Optional[dict] = None
        self._lock = threading.Lock()
        self.config.setdefault('min_trading_bars', 200)
        self.config.setdefault('training_interval', 'daily')

    def update_data(self, bid: float, ask: float, bar: Optional[dict] = None):
        with self._lock:
            self.current_bid = bid
            self.current_ask = ask
            if bar:
                self._process_bar(bar)

    def _process_bar(self, bar: dict):
        new_row = pd.DataFrame([{
            'Time': pd.to_datetime(bar['time']),
            'Open': float(bar['open']),
            'High': float(bar['high']),
            'Low': float(bar['low']),
            'Close': float(bar['close']),
            'Volume': int(bar['volume'])
        }])
        with self._lock:
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            self.data = self.data.drop_duplicates(subset='Time', keep='last').reset_index(drop=True)
        self._calculate_indicators()

    def _calculate_indicators(self):
        if len(self.data) < self.config['min_trading_bars']:
            logging.warning(f"{self.symbol}: Insufficient data for indicators")
            return

        df = self.data.copy()
        df["ATR"] = ta.atr(df.High, df.Low, df.Close, length=14)
        df["EMA_fast"] = ta.ema(df.Close, length=21)
        df["EMA_slow"] = ta.ema(df.Close, length=50)
        df["RSI"] = ta.rsi(df.Close, length=14)
        df["ADX"] = ta.adx(df.High, df.Low, df.Close)["ADX_14"]
        df["Volume_MA"] = df.Volume.rolling(20).mean()
        
        macd = ta.macd(df.Close, fast=12, slow=26)
        for col in macd.columns:
            df[col] = macd[col]
        
        bbands = ta.bbands(df.Close, length=20, std=2)
        for col in bbands.columns:
            df[col] = bbands[col]
        
        self.data = df.copy()

    def train_model(self):
        if len(self.data) < 200:
            return

        features = self.data[['ATR', 'EMA_fast', 'EMA_slow', 'RSI', 'MACD_12_26_9', 'ADX', 'Volume']].dropna()
        labels = (self.data['Close'].diff().shift(-1) > 0).astype(int).dropna()
        min_len = min(len(features), len(labels))
        if min_len == 0:
            return
            
        self.ml_model.fit(features.iloc[:min_len], labels.iloc[:min_len])
        self.ml_trained = True
        logging.info(f"{self.symbol}: Model trained ({min_len} samples)")
        logging.debug(f"{self.symbol}: Feature importances {self.ml_model.feature_importances_}")

    def get_ml_signal(self) -> int:
        try:
            if not self.ml_trained or len(self.data) < 1:
                return 0
            latest = self.data[['ATR', 'EMA_fast', 'EMA_slow', 'RSI', 'Volume', 'MACD_12_26_9', 'ADX']].iloc[-1:]
            return 2 if self.ml_model.predict(latest)[0] == 1 else 1
        except Exception as e:
            logging.error(f"{self.symbol}: ML signal error: {e}")
            return 0

    def generate_signal(self) -> int:
        if len(self.data) < 50:
            return 0

        current = self.data.iloc[-1]
        if current["ADX"] < self.config.get('min_adx', 25):
            logging.debug(f"{self.symbol}: Low ADX")
            return 0

        ema_bullish = current["EMA_fast"] > current["EMA_slow"]
        macd_bullish = current["MACD_12_26_9"] > current["MACDs_12_26_9"]
        bb_low = current.get("BBL_20_2.0", 0)
        bb_high = current.get("BBU_20_2.0", 0)

        tech_signal = 0
        if ema_bullish and macd_bullish and current["Close"] < bb_low:
            tech_signal = 2
        elif not ema_bullish and not macd_bullish and current["Close"] > bb_high:
            tech_signal = 1

        if self.ml_trained and self.get_ml_signal() != tech_signal:
            logging.info(f"{self.symbol}: ML/Technical mismatch")
            return 0

        return tech_signal

class MultiAssetStrategy:
    """Main trading strategy controller."""
    def __init__(self, mt4_path: str, assets_config: List[dict]):
        self.logger = logging.getLogger(__name__)
        self.assets: Dict[str, Asset] = {
            cfg['symbol']: Asset(cfg['symbol'], cfg) for cfg in assets_config
        }
        self.dwx = dwx_client(self, mt4_path, sleep_delay=0.005)
        self.scheduler = BackgroundScheduler()
        self._initialize_connection()

    def _initialize_connection(self):
        try:
            self.dwx.start()
            sleep(5)
            if not self.dwx.START:
                raise ConnectionError("Connection failed: DWX client did not start")
                
            symbols = list(self.assets.keys())
            self.dwx.subscribe_symbols(symbols)
            for symbol in symbols:
                timeframe = self.assets[symbol].config.get('timeframe')
                self.dwx.subscribe_symbols_bar_data([[symbol, timeframe]])
            self._load_historical_data()
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise

    def _load_historical_data(self, days: int = 30):
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(days, 90))
        for symbol, asset in self.assets.items():
            try:
                self.dwx.get_historic_data(
                    symbol,
                    asset.config.get('timeframe'),
                    start.timestamp(),
                    end.timestamp()
                )
            except Exception as e:
                self.logger.error(f"History load failed {symbol}: {e}")

    def on_bar_data(self, symbol: str, timeframe: str, time_val, open_price, high, low, close_price, volume):
        """Handle incoming bar data from DWX."""
        asset = self.assets.get(symbol)
        if asset and timeframe == asset.config.get('timeframe'):
            bid = asset.current_bid if asset.current_bid is not None else float(close_price)
            ask = asset.current_ask if asset.current_ask is not None else float(close_price)
            asset.update_data(bid, ask, {
                'time': time_val,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })

    def on_historic_data(self, symbol: str, timeframe: str, data: List):
        """Handle historic data, ensuring no duplicate bars."""
        for bar in data:
            if isinstance(bar, dict):
                self.on_bar_data(symbol, timeframe, bar.get('time'), bar.get('open'),
                                 bar.get('high'), bar.get('low'), bar.get('close'), bar.get('volume'))
            elif isinstance(bar, str):
                parts = bar.split(',')
                if len(parts) == 6:
                    self.on_bar_data(symbol, timeframe, parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
                elif len(parts) == 1:
                    asset = self.assets.get(symbol)
                    if asset is not None and not asset.data.empty:
                        last_close = asset.data['Close'].iloc[-1]
                        # Pass volume=0 to avoid influencing volume-based indicators with synthetic data
                        self.on_bar_data(symbol, timeframe, parts[0], last_close, last_close, last_close, last_close, 0)
                    else:
                        self.logger.debug(f"{symbol}: No previous data to fill missing bar: {bar}")
                else:
                    self.logger.warning(f"{symbol}: Historic bar string format unexpected: {bar}")
            else:
                self.logger.warning(f"{symbol}: Unknown historic bar type: {type(bar)}")

    def on_message(self, message: dict):
        """Handle messages from DWX."""
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

    def on_tick(self, symbol: str, bid: float, ask: float):
        """Update tick data for a given asset."""
        asset = self.assets.get(symbol)
        if asset:
            asset.update_data(bid, ask)

    def run_strategy_cycle(self):
        try:
            self._run_strategy_logic()
        except Exception as e:
            self.logger.error(f"Strategy cycle failed: {e}", exc_info=True)

    def _run_strategy_logic(self):
        account_balance = self._get_validated_balance()
        if not account_balance:
            return

        for asset in self.assets.values():
            self._process_asset(asset, account_balance)

    def _get_validated_balance(self) -> Optional[float]:
        balance_val = self.dwx.account_info.get("balance")
        if balance_val is None:
            self.logger.error("Missing balance data")
            return None
        try:
            return float(balance_val)
        except ValueError:
            self.logger.error("Invalid balance format")
            return None

    def _process_asset(self, asset: Asset, balance: float):
        if not self._validate_asset(asset):
            return

        signal = asset.generate_signal()
        if not signal:
            return

        trade_params = self._calculate_trade_params(asset, balance, signal)
        if trade_params:
            self._execute_trade(asset, signal, *trade_params)

    def _validate_asset(self, asset: Asset) -> bool:
        if None in (asset.current_bid, asset.current_ask):
            self.logger.debug(f"{asset.symbol}: Missing prices")
            return False
        if "ATR" not in asset.data.columns or asset.data["ATR"].empty:
            self.logger.debug(f"{asset.symbol}: Missing ATR")
            return False
        return True

    def _calculate_trade_params(self, asset: Asset, balance: float, signal: int) -> Optional[Tuple[float, float, float, float]]:
        atr = asset.data["ATR"].iloc[-1]
        sl_distance = atr * asset.config.get('sl_multiplier', 1.5)
        
        if not self._validate_spread(asset):
            return None

        position_size = max(
            (balance * asset.config.get('risk_per_trade', 0.01)) / sl_distance,
            0.01
        )
        price, sl, tp = self._get_trade_levels(asset, signal, sl_distance)
        return (position_size, price, sl, tp)

    def _get_trade_levels(self, asset: Asset, signal: int, sl_distance: float) -> Tuple[float, float, float]:
        if signal == 2:
            price = asset.current_bid
            sl = price - sl_distance
            tp = price + (sl_distance * asset.config.get('tp_ratio', 2.0))
        else:
            price = asset.current_ask
            sl = price + sl_distance
            tp = price - (sl_distance * asset.config.get('tp_ratio', 2.0))
        return price, sl, tp

    def _validate_spread(self, asset: Asset) -> bool:
        spread = asset.current_ask - asset.current_bid
        max_spread = asset.config.get('max_spread', 0.5)
        if spread > max_spread:
            self.logger.warning(f"{asset.symbol}: Spread {spread:.5f} > {max_spread}")
            return False
        return True

    def _execute_trade(self, asset: Asset, signal: int, size: float, price: float, sl: float, tp: float):
        try:
            order_type = 'buy' if signal == 2 else 'sell'
            self.dwx.open_order(
                symbol=asset.symbol,
                order_type=order_type,
                lots=size,
                stop_loss=sl,
                take_profit=tp,
                order_price=price,
                order_execution_type='limit'
            )
            self.logger.info(f"{asset.symbol}: {order_type.upper()} @ {price}")
            asset.active_trade = {
                'direction': order_type,
                'entry_price': price,
                'stop_loss': sl,
                'take_profit': tp
            }
        except Exception as e:
            self.logger.error(f"{asset.symbol}: Trade failed: {e}")


    def _manage_trailing_stops(self):
        """Update trailing stops for all open orders across assets."""
        if not self.dwx.open_orders:
            return

        for symbol, asset in self.assets.items():
            if len(asset.data) < 1:
                continue
            current_atr = asset.data["ATR"].iloc[-1]
            trail_distance = current_atr * asset.config.get('trailing_atr_coef', 1.5)
            for order in self.dwx.open_orders:
                if order['symbol'] != symbol:
                    continue
                current_price = asset.current_bid if order['type'] == 'buy' else asset.current_ask
                new_sl = None
                if order['type'] == 'buy':
                    new_sl = current_price - trail_distance
                    if new_sl > order['stop_loss']:
                        self.dwx.modify_order(order['ticket'], stop_loss=new_sl)
                else:
                    new_sl = current_price + trail_distance
                    if new_sl < order['stop_loss']:
                        self.dwx.modify_order(order['ticket'], stop_loss=new_sl)

    def optimize_parameters(self):
        """Walk-forward optimization to tune strategy parameters (placeholder)."""
        from backtesting import Backtest, Strategy

        class WalkForwardStrategy(Strategy):
            def init(self):
                self.atr = self.I(ta.atr, self.data.High, self.data.Low, self.data.Close, 14)
                self.adx = self.I(ta.adx, self.data.High, self.data.Low, self.data.Close, 14)
            
            def next(self):
                if self.data.ADX_14[-1] > 25:
                    if self.data.EMA_fast_21[-1] > self.data.EMA_slow_50[-1]:
                        self.buy(sl=self.data.Close[-1] - self.data.ATR_14[-1]*1.5)
                    else:
                        self.sell(sl=self.data.Close[-1] + self.data.ATR_14[-1]*1.5)

        best_params = None
        best_sharpe = -float('inf')

        asset0 = list(self.assets.values())[0]
        df_opt = asset0.data
        if len(df_opt) < 50:
            self.logger.info("Not enough data for optimization.")
            return

        for fold in range(3):
            train_size = int(len(df_opt) * 0.6)
            test_size = int(len(df_opt) * 0.2)
            train_start = fold * test_size
            train_end = train_start + train_size
            test_end = train_end + test_size

            train_df = df_opt.iloc[train_start:train_end]
            test_df = df_opt.iloc[train_end:test_end]

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
            for asset in self.assets.values():
                asset.config['sl_multiplier'] = best_params['slatrcoef']
                asset.config['tp_ratio'] = best_params['tpsl_ratio']
            self.logger.info(f"Optimized params: SL={best_params['slatrcoef']:.1f} TP={best_params['tpsl_ratio']:.1f}")

    def on_tick(self, symbol: str, bid: float, ask: float):
        """Update tick data for a given asset."""
        asset = self.assets.get(symbol)
        if asset:
            asset.update_data(bid, ask)

    def schedule_jobs(self):
        """Schedule periodic strategy and optimization jobs."""
        # Use an intermediate variable for the minute interval.
        timeframe_interval = list(self.assets.values())[0].config.get("timeframe")[1:]
        self.scheduler.add_job(
            self.run_strategy_cycle,
            trigger=CronTrigger(
                minute=f'*/{timeframe_interval}',
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

    def shutdown(self):
        """Cleanly shut down the DWX client and scheduler."""
        self.dwx.stop()
        self.scheduler.shutdown()
        self.logger.info("Strategy shutdown complete")

if __name__ == "__main__":
    ASSETS_CONFIG = [
        {
            'symbol': 'XAUUSD',
            'timeframe': 'M1',
            'risk_per_trade': 0.01,
            'sl_multiplier': 1.5,
            'tp_ratio': 2.0,
            'min_adx': 25,
            'max_spread': 0.5,
            'trailing_atr_coef': 1.5
        },
        {
            'symbol': 'EURUSD',
            'timeframe': 'M1',
            'risk_per_trade': 0.015,
            'sl_multiplier': 1.8,
            'tp_ratio': 2.2,
            'min_adx': 20,
            'max_spread': 0.0002,
            'trailing_atr_coef': 1.5
        },
        {
            'symbol': 'XAGUSD',
            'timeframe': 'M1',
            'risk_per_trade': 0.01,
            'sl_multiplier': 1.5,
            'tp_ratio': 2.0,
            'min_adx': 25,
            'max_spread': 0.5,
            'trailing_atr_coef': 1.5
        },
        {
            'symbol': 'USDJPY',
            'timeframe': 'M1',
            'risk_per_trade': 0.015,
            'sl_multiplier': 1.8,
            'tp_ratio': 2.2,
            'min_adx': 20,
            'max_spread': 0.02,
            'trailing_atr_coef': 1.5
        },
        {
            'symbol': 'GBPUSD',
            'timeframe': 'M1',
            'risk_per_trade': 0.01,
            'sl_multiplier': 1.5,
            'tp_ratio': 2.0,
            'min_adx': 25,
            'max_spread': 0.5,
            'trailing_atr_coef': 1.5
        },
        {
            'symbol': 'AUDUSD',
            'timeframe': 'M1',
            'risk_per_trade': 0.015,
            'sl_multiplier': 1.8,
            'tp_ratio': 2.2,
            'min_adx': 20,
            'max_spread': 0.02,
            'trailing_atr_coef': 1.5
        },
        {
            'symbol': 'USDCAD',
            'timeframe': 'M1',
            'risk_per_trade': 0.01,
            'sl_multiplier': 1.5,
            'tp_ratio': 2.0,
            'min_adx': 25,
            'max_spread': 0.5,
            'trailing_atr_coef': 1.5
        },
        {
            'symbol': 'USDCHF',
            'timeframe': 'M1',
            'risk_per_trade': 0.015,
            'sl_multiplier': 1.8,
            'tp_ratio': 2.2,
            'min_adx': 20,
            'max_spread': 0.02,
            'trailing_atr_coef': 1.5
        }
    ]

    strategy = MultiAssetStrategy(
        mt4_path='C:/Users/User/AppData/Roaming/MetaQuotes/Terminal/B7238334EF4B2B20A39D097B2877DE6E/MQL4/Files/',
        assets_config=ASSETS_CONFIG
    )

    try:
        strategy.schedule_jobs()
        while True:
            sleep(1)
    except KeyboardInterrupt:
        strategy.shutdown()
