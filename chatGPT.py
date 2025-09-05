import json
from time import sleep
from datetime import datetime, timedelta, timezone
from random import random

import pandas as pd
import pandas_ta as ta
from apscheduler.schedulers.blocking import BlockingScheduler

# Import your DWX client and backtesting libraries
from api.dwx_client import dwx_client
from backtesting import Strategy, Backtest


# Configuration
MT4_DIR_PATH = r'C:/Users/User/AppData/Roaming/MetaQuotes/Terminal/B7238334EF4B2B20A39D097B2877DE6E/MQL4/Files/'
SYMBOL = 'XAUUSD'
TIMEFRAME = 'M5'  # 5-minute timeframe for scalping
LOTS = 0.1  # Trade size


class ScalpingStrategy:
    """
    A scalping strategy that subscribes to market data, builds a bar DataFrame, calculates technical
    indicators and signals, and then uses a backtester to optimize parameters. It also executes trades
    via the DWX client.
    """

    def __init__(self, mt4_directory_path, sleep_delay=0.005, max_retry_command_seconds=10, verbose=True):
        # Instance variables for trading and fitting coefficients
        self.slatrcoef = 0.0
        self.TPSLRatio_coef = 0.0

        # Create an empty DataFrame to hold bar data.
        self.df = pd.DataFrame(columns=["Time", "Open", "High", "Low", "Close", "Volume"])

        # DWX client initialization
        self.dwx = dwx_client(
            self, mt4_directory_path, sleep_delay, max_retry_command_seconds, verbose=verbose
        )
        self.open_test_trades = False  # Disable test trades
        self.last_open_time = datetime.now(timezone.utc)
        self.last_modification_time = datetime.now(timezone.utc)

        sleep(1)
        self.dwx.start()

        # Print account information from DWX client.
        print("Account info:", self.dwx.account_info)

        # Subscribe to tick and bar data.
        self.dwx.subscribe_symbols([SYMBOL])
        self.dwx.subscribe_symbols_bar_data([[SYMBOL, TIMEFRAME]])

        # Request historic data (last 30 days)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        self.dwx.get_historic_data(SYMBOL, TIMEFRAME, start.timestamp(), end.timestamp())

    def on_bar_data(self, symbol, time_frame, time, open_price, high, low, close_price, tick_volume):
        """
        Callback method for receiving new bar data.
        Appends a new row to the internal DataFrame and recalculates technical indicators.
        """
        new_row = {
            'Time': time,
            'Open': float(open_price),
            'High': float(high),
            'Low': float(low),
            'Close': float(close_price),
            'Volume': tick_volume
        }
        # Append the new row using pd.concat for efficiency.
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

        # Calculate technical indicators over the entire DataFrame.
        # (For better performance you might only update the last few rows.)
        self.df["ATR"] = ta.atr(self.df["High"], self.df["Low"], self.df["Close"], length=7)
        self.df["EMA_fast"] = ta.ema(self.df["Close"], length=30)
        self.df["EMA_slow"] = ta.ema(self.df["Close"], length=50)
        self.df['RSI'] = ta.rsi(self.df["Close"], length=10)
        bbands = ta.bbands(self.df["Close"], length=15, std=1.5)
        self.df = self.df.join(bbands)
        # Calculate the total signal for each row (using a lookback of 7 candles)
        self.df['TotalSignal'] = self.df.index.to_series().apply(lambda idx: self.total_signal(idx, backcandles=7))

    def ema_signal(self, current_index: int, backcandles: int) -> int:
        """
        Returns:
            1 if the EMA_fast is below EMA_slow for the last `backcandles` candles,
            2 if the EMA_fast is above EMA_slow for the last `backcandles` candles,
            0 otherwise.
        """
        start = max(0, current_index - backcandles)
        relevant = self.df.iloc[start:current_index]
        if relevant.empty:
            return 0

        if (relevant["EMA_fast"] < relevant["EMA_slow"]).all():
            return 1
        elif (relevant["EMA_fast"] > relevant["EMA_slow"]).all():
            return 2
        else:
            return 0

    def total_signal(self, current_index: int, backcandles: int) -> int:
        """
        Combines EMA and other conditions to produce a total signal.
        Returns:
            2 for a buy signal, 1 for a sell signal, and 0 for no signal.
        """
        ema_sig = self.ema_signal(current_index, backcandles)
        # Ensure the index exists in the DataFrame.
        try:
            candle_open = self.df.at[current_index, "Open"]
            bbl = self.df.at[current_index, "BBL_15_1.5"]
            bbu = self.df.at[current_index, "BBU_15_1.5"]
            rsi = self.df.at[current_index, "RSI"]
        except KeyError:
            return 0

        # For a BUY signal:
        if ema_sig == 2 and candle_open <= bbl and rsi < 60:
            return 2
        # For a SELL signal:
        if ema_sig == 1 and candle_open >= bbu and rsi > 40:
            return 1
        return 0

    def get_recent_data(self, n: int = 70) -> pd.DataFrame:
        """
        Returns the last n rows of the DataFrame.
        """
        return self.df.tail(n).reset_index(drop=True)

    def fitting_job(self):
        """
        Uses backtesting to optimize strategy parameters.
        Writes the best parameters and performance stats to a file.
        """
        # Use the most recent data for backtesting.
        dfstream = self.get_recent_data(n=7000)

        def SIGNAL():
            # Return the TotalSignal series from the DataFrame.
            return dfstream["TotalSignal"]

        class MyStrat(Strategy):
            mysize = 3000
            slcoef = 1.1
            TPSLRatio = 1.5

            def init(self):
                self.signal1 = self.I(SIGNAL)

            def next(self):
                # Calculate ATR-based stop-loss.
                slatr = self.slcoef * self.data.ATR[-1]
                if self.signal1 == 2 and len(self.trades) == 0:
                    sl1 = self.data.Close[-1] - slatr
                    tp1 = self.data.Close[-1] + slatr * self.TPSLRatio
                    self.buy(sl=sl1, tp=tp1, size=self.mysize)
                elif self.signal1 == 1 and len(self.trades) == 0:
                    sl1 = self.data.Close[-1] + slatr
                    tp1 = self.data.Close[-1] - slatr * self.TPSLRatio
                    self.sell(sl=sl1, tp=tp1, size=self.mysize)

        # Optimize strategy parameters.
        bt = Backtest(dfstream, MyStrat, cash=250, margin=1/30)
        stats, heatmap = bt.optimize(
            slcoef=[i/10 for i in range(10, 26)],
            TPSLRatio=[i/10 for i in range(10, 26)],
            maximize='Return [%]', max_tries=300,
            random_state=0,
            return_heatmap=True
        )

        # Save optimized parameters to instance variables.
        self.slatrcoef = stats["_strategy"].slcoef
        self.TPSLRatio_coef = stats["_strategy"].TPSLRatio
        print("Optimized Parameters:", self.slatrcoef, self.TPSLRatio_coef)

        with open("fitting_data_file.txt", "a") as file:
            file.write(f"{self.slatrcoef}, {self.TPSLRatio_coef}, expected return, {stats['Return [%]']}\n")

    def trading_job(self, symbol, bid, ask, order_type, price_func, sl_func, risk_percentage):
        """
        Called periodically to evaluate trading signals and execute trades.
        """
        dfstream = self.get_recent_data(n=70)
        # Use the most recent candle for signal generation.
        signal = self.total_signal(current_index=len(dfstream) - 1, backcandles=7)

        open_order_count = len(self.dwx.open_orders)

        # If it's Monday before 07:05, run the fitting job.
        now = datetime.now()
        if now.weekday() == 0 and now.hour < 7 and now.minute < 5:
            self.fitting_job()

        # Compute stop loss distance and TP/SL levels.
        last_atr = dfstream["ATR"].iloc[-1]
        slatr = self.slatrcoef * last_atr
        TPSLRatio = self.TPSLRatio_coef
        max_spread = 16e-5

        # Assume current candle prices are based on bid/ask.
        spread = ask - bid
        if spread >= max_spread:
            return  # Do not trade if spread is too high

        SLBuy = bid - slatr - spread
        SLSell = ask + slatr + spread
        TPBuy = ask + slatr * TPSLRatio + spread
        TPSell = bid - slatr * TPSLRatio - spread

        # Check if there are no open orders before placing a new trade.
        if open_order_count == 0:
            if signal == 1:
                print("Sell Signal Found...")
                self.dwx.open_order(
                    symbol=symbol,
                    order_type=order_type,
                    take_profit=tp_func(TPSell),
                    stop_loss=sl_func(SLSell),
                    lots=LOTS
                )
                print(f"Opened SELL trade for {SYMBOL} with SL={SLSell} and TP={TPSell}")
                with open("trading_data_file.txt", "a") as file:
                    file.write(f"{self.slatrcoef}, {self.TPSLRatio_coef}\n")
            elif signal == 2:
                print("Buy Signal Found...")
                self.dwx.open_order(
                    symbol=symbol,
                    order_type=order_type,
                    take_profit=tp_func(TPBuy),
                    stop_loss=sl_func(SLBuy),
                    lots=LOTS
                )
                print(f"Opened BUY trade for {SYMBOL} with SL={SLBuy} and TP={TPBuy}")
                with open("trading_data_file.txt", "a") as file:
                    file.write(f"{self.slatrcoef}, {self.TPSLRatio_coef}\n")

    def scheduler(self):
        """
        Schedules the trading job to run on weekdays during specified hours.
        """
        scheduler = BlockingScheduler(timezone='Asia/Beirut')
        scheduler.add_job(
            self.trading_job,
            trigger='cron',
            day_of_week='mon-fri',
            hour='07-18',
            minute='1,6,11,16,21,26,31,36,41,46,51,56',
            misfire_grace_time=15,
            args=[SYMBOL, 0.0, 0.0, "MARKET",  # Replace bid/ask with real-time data as needed.
                  lambda x: x, lambda x: x, 0]
        )
        scheduler.start()

    def on_historic_trades(self):
        """Callback to print the number of historic trades."""
        print(f"Historic trades: {len(self.dwx.historic_trades)}")

    def on_message(self, message):
        """Callback for handling messages from DWX."""
        msg_type = message.get("type", "")
        if msg_type == "ERROR":
            print(f"{msg_type} | {message.get('error_type')} | {message.get('description')}")
        elif msg_type == "INFO":
            print(f"{msg_type} | {message.get('message')}")

    def on_order_event(self):
        """Callback for order events (added/removed orders)."""
        print(f"Order event: {len(self.dwx.open_orders)} open orders.")


if __name__ == '__main__':
    strategy = ScalpingStrategy(MT4_DIR_PATH)

    # Option 1: Run the scheduler in a separate thread (or process) if needed.
    # strategy.scheduler()

    # Option 2: Run a simple loop to keep the client active.
    try:
        while strategy.dwx.ACTIVE:
            sleep(1)
    except KeyboardInterrupt:
        print("Shutting down strategy.")
