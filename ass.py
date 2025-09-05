import json
from time import sleep
from datetime import datetime, timedelta, timezone
from random import random
import pandas as pd
import pandas_ta as ta
from apscheduler.schedulers.blocking import BlockingScheduler
from api.dwx_client import dwx_client


# Configuration
MT4_DIR_PATH = 'C:/Users/User/AppData/Roaming/MetaQuotes/Terminal/B7238334EF4B2B20A39D097B2877DE6E/MQL4/Files/'
SYMBOL = 'XAUUSD'
TIMEFRAME = 'M5'  # 5-minute timeframe for scalping
LOTS = 0.1  # Trade size
slatrcoef = 0
TPSLRatio_coef = 0



class ScalpingStrategy():
    def __init__(self, MT4_directory_path, sleep_delay=0.005, max_retry_command_seconds=10, verbose=True):
        self.dwx = dwx_client(self, MT4_directory_path, sleep_delay, max_retry_command_seconds, verbose=verbose)
        self.df = pd.DataFrame()  # DataFrame to store bar data
        self.open_test_trades = False  # Disable test trades
        self.last_open_time = datetime.now(timezone.utc)
        self.last_modification_time = datetime.now(timezone.utc)
        
        
        sleep(1)

        self.dwx.start()
        
        # account information is stored in self.dwx.account_info.
        print("Account info:", self.dwx.account_info)
        
        self.dwx.subscribe_symbols([SYMBOL])  # Subscribe to tick data
        self.dwx.subscribe_symbols_bar_data([[SYMBOL, TIMEFRAME]])  # Subscribe to bar data
        
        # request historic data:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)  # last 30 days
        self.dwx.get_historic_data(SYMBOL, TIMEFRAME, start.timestamp(), end.timestamp())
        
        
    
    
    def on_bar_data(self, symbol, time_frame, time, open_price, high, low, close_price, tick_volume):
        # Update DataFrame with new bar data
        new_row = {
            'Time': time,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close_price,
            'Volume': tick_volume
        }
        self.df = self.df.append(new_row, ignore_index=True)

        # Calculate indicators
        self.df['Open'] = self.df['Open'].astype(float)
        self.df['Close'] = self.df['Close'].astype(float)
        self.df['High'] = self.df['High'].astype(float)
        self.df['Low'] = self.df['Low'].astype(float)

        self.df["ATR"] = ta.atr(self.df.High, self.df.Low, self.df.Close, length=7)
        self.df["EMA_fast"]=ta.ema(self.df.Close, length=30)
        self.df["EMA_slow"]=ta.ema(self.df.Close, length=50)
        self.df['RSI']=ta.rsi(self.df.Close, length=10)
        my_bbands = ta.bbands(self.df.Close, length=15, std=1.5)
        self.df=self.df.join(my_bbands) 
        self.df['TotalSignal'] = self.df.apply(lambda row: self.total_signal(self.df, row.name, 7), axis=1)
        
        
    def ema_signal(self, current_candle, backcandles):
        df_slice = self.df.reset_index().copy()
        # Get the range of candles to consider
        start = max(0, current_candle - backcandles)
        end = current_candle
        relevant_rows = df_slice.iloc[start:end]

        # Check if all EMA_fast values are below EMA_slow values
        if all(relevant_rows["EMA_fast"] < relevant_rows["EMA_slow"]):
            return 1
        elif all(relevant_rows["EMA_fast"] > relevant_rows["EMA_slow"]):
            return 2
        else:
            return 0
    
    def total_signal(self, current_candle, backcandles):
        ema_signal_result = self.ema_signal(self, current_candle, backcandles)
        candle_open_price = self.df.Open[current_candle]
        bbl = self.df['BBL_15_1.5'][current_candle]
        bbu = self.df['BBU_15_1.5'][current_candle]

        if (ema_signal_result==2 and candle_open_price<=bbl and self.df.RSI[current_candle]<60): #and df.RSI[current_candle]<60
                return 2
        if (ema_signal_result==1 and candle_open_price>=bbu and self.df.RSI[current_candle]>40): #and df.RSI[current_candle]>40
                return 1
        return 0

    def fitting_job(self):
        global slatrcoef
        global TPSLRatio_coef
        
        dfstream = self.on_bar_data(7000)
        
        def SIGNAL():
            return dfstream.TotalSignal
        
        from backtesting import Strategy
        from backtesting import Backtest

        class MyStrat(Strategy):
            mysize = 3000
            slcoef = 1.1
            TPSLRatio = 1.5
            
            def init(self):
                super().init()
                self.signal1 = self.I(SIGNAL)

            def next(self):
                super().next()
                slatr = self.slcoef*self.data.ATR[-1]
                TPSLRatio = self.TPSLRatio
            
                if self.signal1==2 and len(self.trades)==0:
                    sl1 = self.data.Close[-1] - slatr
                    tp1 = self.data.Close[-1] + slatr*TPSLRatio
                    self.buy(sl=sl1, tp=tp1, size=self.mysize)
                
                elif self.signal1==1 and len(self.trades)==0:         
                    sl1 = self.data.Close[-1] + slatr
                    tp1 = self.data.Close[-1] - slatr*TPSLRatio
                    self.sell(sl=sl1, tp=tp1, size=self.mysize)

        bt = Backtest(dfstream, MyStrat, cash=250, margin=1/30)
        stats, heatmap = bt.optimize(slcoef=[i/10 for i in range(10, 26)],
                            TPSLRatio=[i/10 for i in range(10, 26)],
                            maximize='Return [%]', max_tries=300,
                            random_state=0,
                            return_heatmap=True)
        #print(stats)

        slatrcoef = stats["_strategy"].slcoef
        TPSLRatio_coef = stats["_strategy"].TPSLRatio
        print(slatrcoef, TPSLRatio_coef)

        with open("fitting_data_file.txt", "a") as file:
            file.write(f"{slatrcoef}, {TPSLRatio_coef}, expected return, {stats['Return [%]']}\n")
            
            
    def trading_job(self, symbol, bid, ask, order_type, price, sl, tp, risk_percentage):

        dfstream = self.on_bar_data(70)
        signal = self.total_signal(dfstream, len(dfstream)-1, 7) # current candle looking for open price entry
        
         # Count the number of open orders
        open_order_count = len(self.dwx.open_orders)
        
        global slatrcoef
        global TPSLRatio_coef    
        
        from datetime import datetime
        now = datetime.now()
        if now.weekday() == 0 and now.hour < 7 and now.minute < 5:  # Monday before 07:05
            self.fitting_job()
            print(slatrcoef, TPSLRatio_coef)

        slatr = slatrcoef*dfstream.ATR.iloc[-1]
        TPSLRatio = TPSLRatio_coef
        max_spread = 16e-5
        
        candle = self.on_bar_data(1)[-1]
        candle_open_bid = bid
        candle_open_ask = ask
        spread = candle_open_ask-candle_open_bid

        SLBuy = candle_open_bid-slatr-spread
        SLSell = candle_open_ask+slatr+spread

        TPBuy = candle_open_ask+slatr*TPSLRatio+spread
        TPSell = candle_open_bid-slatr*TPSLRatio-spread
        
        stop_loss = sl
        take_profit = tp
        
        #Sell
        if signal == 1 and open_order_count() == 0 and spread<max_spread:
            print("Sell Signal Found...")
            self.dwx.open_order(symbol=symbol, order_type=order_type, take_profit=tp(price=TPSell), stop_loss=sl(price= SLSell), lots=LOTS)
            print(f"Opened {order_type} trade for {SYMBOL} with SL={stop_loss} and TP={take_profit}")
            with open("trading_data_file.txt", "a") as file:
                file.write(f"{slatrcoef}, {TPSLRatio_coef}\n")

        #Buy
        elif signal == 2 and open_order_count() == 0 and spread<max_spread:
            print("Buy Signal Found...")
            self.dwx.open_order(symbol=symbol, order_type=order_type, take_profit=tp(price=TPBuy), stop_loss=sl(price= SLBuy), lots=LOTS)
            print(f"Opened {order_type} trade for {SYMBOL} with SL={stop_loss} and TP={take_profit}")
            with open("trading_data_file.txt", "a") as file:
                file.write(f"{slatrcoef}, {TPSLRatio_coef}\n")
                
                
    def scheduler(self):
        scheduler = BlockingScheduler()
        scheduler.add_job(self.trading_job, 'cron', day_of_week='mon-fri', hour='07-18', minute='1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56', timezone='Asia/Beirut', misfire_grace_time=15)
        scheduler.start()
                
                
    def on_historic_trades(self):
        print(f"historic_trades: {len(self.dwx.historic_trades)}")

    def on_message(self, message):
        if message["type"] == "ERROR":
            print(
                message["type"], "|", message["error_type"], "|", message["description"]
            )
        elif message["type"] == "INFO":
            print(message["type"], "|", message["message"])

    # triggers when an order is added or removed, not when only modified.
    def on_order_event(self):
        print(f"on_order_event. open_orders: {len(self.dwx.open_orders)} open orders")
        
        
strategy = ScalpingStrategy(MT4_DIR_PATH)
while strategy.dwx.ACTIVE:
    sleep(1)
