#!/usr/bin/env python
# coding: utf-8

# ### Initialise

# In[ ]:


import pandas_ta as ta
import MetaTrader5 as mt5
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from json import JSONDecodeError
import random 
from scipy.stats import norm,t,gamma
from scipy.stats import skewnorm
import scipy.stats as stats
from datetime import datetime, timedelta
#### Telegram Configuration

import requests
import math
import inspect
import json

# === 1. Initialize MetaTrader 5 ===
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed. Check terminal connection.")


# ### Importing Functions

# In[ ]:


def weibull_plot(data):
    global x, mu, loc, sigma, skew_pdf, ninety_percentile
    if type(data) != np.ndarray:
        data = np.array(data)
        print(data)
    eps = 1e-6
    #time_to_tp_long = time_to_tp_long + eps
    x = np.linspace(data.min(), data.max(), 200)
    mu, loc, sigma  = weibull_min.fit(data)
    skew_pdf = weibull_min.pdf(x, mu, loc, sigma)
    ninety_percentile = weibull_min.ppf(0.9, mu, loc, sigma)


# In[ ]:


def gen_trade_id(*size):
    #n = size.get("size")
    if len(size) == 0:
        trade_id = np.random.randint(1000000,9999999)
        trade_id = str("#") + str(trade_id)
    else:
        n = size[0]
        trade_id = np.random.randint(1000000,9999999, n)
        if n > 1:
            idens = []
            for iden in trade_id:
                #print(iden)
                idens.append(str("#") + str(iden))
            trade_id = np.array(idens)
    return trade_id


# In[ ]:


def normalised_difference(x,y):
    try:
        
        return round(math.log(x/y),7)

    except Exception as e:
        send_telegram_message("MTTF failed to execute due to error in normalised_difference() module!")
        return 0


# In[ ]:


def risk_reward(a,b,c): # a = sl, b= entry level, c= tp
     
    try:
        return round(abs((c-b)/(b-a)),2)
        
    except ZeroDivisionError as e:
        
        send_telegram_message(f"{e}.")
        
    except Exception as e:
        
        send_telegram_message(f"{e}.")
    
    else:
        return False
        


# In[ ]:


def choose_instrument(x):
    global symbol, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    
    TELEGRAM_BOT_TOKEN = x.get("telegram token")
    TELEGRAM_CHAT_ID = x.get("telegram chat id")
    symbol= x.get("id")

    
    return symbol


# In[ ]:


def log_trade(r):

    file = f"{symbol} MTTF Magnitude Vector Backtest Log.csv"
    
    if os.path.exists(file):
        
        if type(r) == pd.core.frame.DataFrame:
    
            log_row = {
                "Trade id" : gen_trade_id(),
                "Datetime" : r["timestamp"].iloc[last_vector_trough] if macro_slope_m5 > 0 else r["timestamp"].iloc[last_vector_peak],
                "Instrument" : symbol,
                "Session" : check_session(str(r["timestamp"].iloc[last_vector_peak])) if macro_slope_m5 > 0 else check_session(str(r["timestamp"].iloc[last_vector_peak])) ,
                "Trade Mode" : "MTTF BAU",
                "Type of Trade":(
                    "Long" if macro_slope_m5 > 0 else
                    "Short" if macro_slope_m5 < 0 else
                    "N/A"
                ),
                "M5 Trend": (
                    "Uptrend" if macro_slope_m5 > 0 else
                    "Downtrend" if macro_slope_m5 < 0 else
                    "N/A"
                ),
                "macro_slope_m5": macro_slope_m5,
                "RSI at peak/trough" : (
                    round((r["rsi"].iloc[last_vector_peak]),2) if macro_slope_m5 < 0 else
                    round((r["rsi"].iloc[last_vector_trough]),2) if macro_slope_m5 > 0 else
                    "N/A"
                ),
                "M5 200 EMA" : (
                    "Above" if r["close"].iloc[-1] > r["moving average"].iloc[-1] else
                    "Below" if r["close"].iloc[-1] < r["moving average"].iloc[-1] else
                    "N/A"
                ),
                "Peak/Trough Z-Score" : (
                    round((r["z_score"].iloc[last_vector_peak]),2) if macro_slope_m5 < 0 else
                    round((r["z_score"].iloc[last_vector_trough]),2) if macro_slope_m5 > 0 else
                    "N/A"
                ),
                          
                "m5 log returns" : (m5_log_returns if m5_log_returns !=None else "N/A") ,
                "m5 peak returns" : (m5_peak_returns if m5_peak_returns !=None else "N/A") ,
                "m5 trough returns" : (m5_trough_returns if m5_trough_returns !=None else "N/A") ,

                "win rr" : rr,
                "loss rr" : loss_rr,
                
                "Number of Peaks" : len(valid_peaks_m5),
                "Number of Troughs" : len(valid_troughs_m5),
    
                "m5 ema distance" : (m5_ema_difference if m5_ema_difference != None else "N/A"),
                "m5 peak distance" : m5_peak_difference  if m5_peak_difference is not None else "N/A",
                "m5 trough distance" : m5_trough_difference if m5_trough_difference is not None else "N/A",  
                "Trade Size" : trade_size if trade_size is not None else 0,
                "std" : (std / r["close"].iloc[-1]) if std is not None else "N/A",
                "atr_std" : (sdv / r["close"].iloc[-1]) if sdv is not None else "N/A",
                "Trade Result" : (1 if min([tp_idx, sl_idx]) == tp_idx else 0)
            }
            log = pd.DataFrame([log_row])
            data = pd.read_csv(file)
            combined = pd.concat([data,log])
            combined.to_csv(file, mode = "w", index = False)
            print(f"Length of CSV = {len(combined)}")
            print(f"Trade {log_row.get("Trade id")} Logged!")

    else:
        
        if type(r) == pd.core.frame.DataFrame:
    
            log_row = {
                "Trade id" : gen_trade_id(),
                "Datetime" : r["timestamp"].iloc[last_vector_trough] if macro_slope_m5 > 0 else r["timestamp"].iloc[last_vector_peak],
                "Instrument" : symbol,
                "Session" : check_session(str(r["timestamp"].iloc[last_vector_peak])) if macro_slope_m5 > 0 else check_session(str(r["timestamp"].iloc[last_vector_peak])),
                "Trade Mode" : "MTTF BAU",
                "Type of Trade":(
                    "Long" if macro_slope_m5 > 0 else
                    "Short" if macro_slope_m5 < 0 else
                    "N/A"
                ),
                "M5 Trend": (
                    "Uptrend" if macro_slope_m5 > 0 else
                    "Downtrend" if macro_slope_m5 < 0 else
                    "N/A"
                ),
                "macro_slope_m5": macro_slope_m5,

                "RSI at peak/trough" : (
                    round((r["rsi"].iloc[last_vector_peak]),2) if macro_slope_m5 < 0 else
                    round((r["rsi"].iloc[last_vector_trough]),2) if macro_slope_m5 > 0 else
                    "N/A"
                ),
                "M5 200 EMA" : (
                    "Above" if r["close"].iloc[-1] > r["moving average"].iloc[-1] else
                    "Below" if r["close"].iloc[-1] < r["moving average"].iloc[-1] else
                    "N/A"
                ),
                "Peak/Trough Z-Score" : (
                    round((r["z_score"].iloc[last_vector_peak]),2) if macro_slope_m5 < 0 else
                    round((r["z_score"].iloc[last_vector_trough]),2) if macro_slope_m5 > 0 else
                    "N/A"
                ),
                          
                "m5 log returns" : (m5_log_returns if m5_log_returns !=None else "N/A") ,
                "m5 peak returns" : (m5_peak_returns if m5_peak_returns !=None else "N/A") ,
                "m5 trough returns" : (m5_trough_returns if m5_trough_returns !=None else "N/A") ,

                "win rr" : rr,
                "loss rr" : loss_rr,  

                "Number of Peaks" : len(valid_peaks_m5),
                "Number of Troughs" : len(valid_troughs_m5),                
    
                "m5 ema distance" : (m5_ema_difference if m5_ema_difference != None else "N/A"),
                "m5 peak distance" : m5_peak_difference  if m5_peak_difference is not None else "N/A",
                "m5 trough distance" : m5_trough_difference if m5_trough_difference is not None else "N/A",  
                
                "Trade Size" : trade_size if trade_size is not None else 0,
                "std" : (std / r["close"].iloc[-1]) if std is not None else "N/A",
                "atr_std" : (sdv / r["close"].iloc[-1]) if sdv is not None else "N/A",
                "Trade Result" : (1 if min([tp_idx, sl_idx]) == tp_idx else 0)
            }
            log = pd.DataFrame([log_row]) 
            log.to_csv(file, mode = "w", index = False)
            print(f"Length of CSV = {len(log)}")
            print(f"Trade {log_row.get("Trade id")} Logged!")


# In[ ]:


def momentum_log_trade(r):

    file = f"{symbol} MTTF Magnitude Vector Backtest Log.csv"
    
    if os.path.exists(file):
        
        if type(r) == pd.core.frame.DataFrame:
    
            log_row = {
                "Trade id" : gen_trade_id(),
                "Datetime" : r["timestamp"].iloc[last_vector_trough] if macro_slope_m5 > 0 else r["timestamp"].iloc[last_vector_peak],
                "Instrument" : symbol,
                "Session" : check_session(str(r["timestamp"].iloc[last_vector_peak])) if macro_slope_m5 > 0 else check_session(str(r["timestamp"].iloc[last_vector_peak])) ,
                "Trade Mode" : "MOMENTUM",
                "Type of Trade":(
                    "Long" if macro_slope_m5 > 0 else
                    "Short" if macro_slope_m5 < 0 else
                    "N/A"
                ),
                "M5 Trend": (
                    "Uptrend" if macro_slope_m5 > 0 else
                    "Downtrend" if macro_slope_m5 < 0 else
                    "N/A"
                ),
                "macro_slope_m5": macro_slope_m5,
                "RSI at peak/trough" : (
                    round((r["rsi"].iloc[last_vector_peak]),2) if macro_slope_m5 < 0 else
                    round((r["rsi"].iloc[last_vector_trough]),2) if macro_slope_m5 > 0 else
                    "N/A"
                ),
                "M5 200 EMA" : (
                    "Above" if r["close"].iloc[-1] > r["moving average"].iloc[-1] else
                    "Below" if r["close"].iloc[-1] < r["moving average"].iloc[-1] else
                    "N/A"
                ),
                "Peak/Trough Z-Score" : (
                    round((r["z_score"].iloc[last_vector_peak]),2) if macro_slope_m5 < 0 else
                    round((r["z_score"].iloc[last_vector_trough]),2) if macro_slope_m5 > 0 else
                    "N/A"
                ),

                "win rr" : rr,
                "loss rr" : loss_rr,
                
                "Number of Peaks" : len(valid_peaks_m5),
                "Number of Troughs" : len(valid_troughs_m5),

                "Trade Size" : trade_size if trade_size is not None else 0,
                "std" : (std / r["close"].iloc[-1]) if std is not None else "N/A",
                "atr_std" : (sdv / r["close"].iloc[-1]) if sdv is not None else "N/A",
                "Trade Result" : (1 if min([tp_idx, sl_idx]) == tp_idx else 0)
            }
            log = pd.DataFrame([log_row])
            data = pd.read_csv(file)
            combined = pd.concat([data,log])
            combined.to_csv(file, mode = "w", index = False)
            print(f"Length of CSV = {len(combined)}")
            print(f"Trade {log_row.get("Trade id")} Logged!")

    else:
        
        if type(r) == pd.core.frame.DataFrame:
    
            log_row = {
                "Trade id" : gen_trade_id(),
                "Datetime" : r["timestamp"].iloc[last_vector_trough] if macro_slope_m5 > 0 else r["timestamp"].iloc[last_vector_peak],
                "Instrument" : symbol,
                "Session" : check_session(str(r["timestamp"].iloc[last_vector_peak])) if macro_slope_m5 > 0 else check_session(str(r["timestamp"].iloc[last_vector_peak])),
                "Trade Mode" : "MOMENTUM",
                "Type of Trade":(
                    "Long" if macro_slope_m5 > 0 else
                    "Short" if macro_slope_m5 < 0 else
                    "N/A"
                ),
                "M5 Trend": (
                    "Uptrend" if macro_slope_m5 > 0 else
                    "Downtrend" if macro_slope_m5 < 0 else
                    "N/A"
                ),
                "macro_slope_m5": macro_slope_m5,

                "RSI at peak/trough" : (
                    round((r["rsi"].iloc[last_vector_peak]),2) if macro_slope_m5 < 0 else
                    round((r["rsi"].iloc[last_vector_trough]),2) if macro_slope_m5 > 0 else
                    "N/A"
                ),
                "M5 200 EMA" : (
                    "Above" if r["close"].iloc[-1] > r["moving average"].iloc[-1] else
                    "Below" if r["close"].iloc[-1] < r["moving average"].iloc[-1] else
                    "N/A"
                ),
                "Peak/Trough Z-Score" : (
                    round((r["z_score"].iloc[last_vector_peak]),2) if macro_slope_m5 < 0 else
                    round((r["z_score"].iloc[last_vector_trough]),2) if macro_slope_m5 > 0 else
                    "N/A"
                ),
                "win rr" : rr,
                "loss rr" : loss_rr,  

                "Number of Peaks" : len(valid_peaks_m5),
                "Number of Troughs" : len(valid_troughs_m5),                
    
                "Trade Size" : trade_size if trade_size is not None else 0,
                "std" : (std / r["close"].iloc[-1]) if std is not None else "N/A",
                "atr_std" : (sdv / r["close"].iloc[-1]) if sdv is not None else "N/A",
                "Trade Result" : (1 if min([tp_idx, sl_idx]) == tp_idx else 0)
            }
            log = pd.DataFrame([log_row]) 
            log.to_csv(file, mode = "w", index = False)
            print(f"Length of CSV = {len(log)}")
            print(f"Trade {log_row.get("Trade id")} Logged!")


# In[ ]:


def compute_timeframe_returns(r, x1, x2):

    if type(r) == pd.core.series.Series:
        
            
        y2 = r.iloc[x2]
            
        y1 = r.iloc[x1]
    
        m5_log_returns = 100*math.log(y2/y1)
        m5_log_returns = round(m5_log_returns,7)
    return m5_log_returns


# In[ ]:


def check_session(x):
    format_time = "%Y-%m-%d %H:%M:%S"
    if type(x) != datetime:
        x = datetime.strptime(x, format_time)
    if 6 <= x.hour < 15:
        return "ASIAN"
    elif 15 <= x.hour <= 20:
        return "EU"
    else:
        return "US"


# In[ ]:


def dec(a,symbol):
    if symbol=="GBPUSD" or symbol=="EURUSD" or symbol=="AUDUSD" or symbol=="USDCAD" or symbol=="EURGBP":
        return round(a,4) 
    elif symbol=="GBPJPY" or symbol=="CL=F"or symbol=="USDJPY" or symbol=="EURJPY":
        return round(a,3)
    else:
        return round(a,2)


# In[ ]:


def risk_reward(a,b,c): # a = sl, b= entry level, c= tp
     
    try:
        return round(abs((c-b)/(b-a)),2)
        
    except ZeroDivisionError as e:
        
        send_telegram_message(f"{e}.")
        
    except Exception as e:
        
        send_telegram_message(f"{e}.")
    
    else:
        return False
        


# In[ ]:


def pip(a, symbol):
    if symbol=="GBPUSD" or symbol=="EURUSD" or symbol=="AUDUSD" or symbol=="USDCAD" or symbol=="EURGBP":
        return a * 10000
    elif symbol=="GBPJPY" or symbol=="CL=F"or symbol=="USDJPY" or symbol=="EURJPY":
        return a * 100  
    elif symbol=="XAUUSD" or symbol=="DX-Y.NYB" or symbol=="MSFT" or symbol== "BTC-USD":
        return a * 10
    else:
        return a


# In[ ]:


def risk_reward(a,b,c): # a = sl, b= entry level, c= tp
     
    try:
        return round(abs((c-b)/(b-a)),2)
        
    except ZeroDivisionError as e:
        
        send_telegram_message(f"{e}.")
        
    except Exception as e:
        
        send_telegram_message(f"{e}.")
    
    else:
        return False
        


# In[ ]:


def compute_fib(r): ### Fibonacci Function ✅
    global lvl_1,lvl_2,lvl_3,tp1,tp4, entry_lvl, fibo_end, fibo_start
    if type(r) == pd.core.frame.DataFrame:
            
        emer_recom_fib=[]
        entry_lvl=None
        ## Assuming trough has already been established (Most optimum)
        
        if macro_slope_m5 < 0:
            
            if last_peak_idx_m5 < last_trough_idx_m5:
                
                if len(valid_peaks_m5[-6:-1]) > 0:
                       
                    for i in range (-6,-1):
                        if (r["high"].iloc[valid_peaks_m5[i]] > r["trend"].iloc[last_peak_idx_m5]
                        and r["high"].iloc[valid_peaks_m5[i]] > r["high"].iloc[last_peak_idx_m5]
                        and (last_peak_idx_m5 - valid_troughs_m5[i]) < 288
                        ):
                            emer_recom_fib.append(r["high"].iloc[valid_peaks_m5[i]])
          
                if len(emer_recom_fib) > 0:
                    
                    fibo_start= dec(max(emer_recom_fib),symbol)
    
                elif len(emer_recom_fib) == 0:
                    
                    fibo_start=r["high"].iloc[last_peak_idx_m5] 
           
            # Lowest Low has to be detected from point of last_peak_idx to the point of signal being sent, which would be r["low"].iloc[-1]
            
                fibo_end=r["low"].iloc[last_trough_idx_m5]
            
            # Distance
                distance= fibo_start-fibo_end
                lvl1=0.382*distance
                lvl2=0.5*distance
                lvl3=0.681*distance
                lvl4=1.618*distance
                lvl5=2.618*distance
            
                lvl_1=dec((fibo_start-lvl1),symbol)
                lvl_2=dec((fibo_start-lvl2),symbol)
                lvl_3=dec((fibo_start-lvl3),symbol)
                tp1=dec((fibo_start-lvl4),symbol)
                tp4=dec((fibo_start-lvl5),symbol)
        
            if last_peak_idx_m5 > last_trough_idx_m5:
                
                ## Assumes trough has not been formed yet
                lowest_since_peak=(r["low"].iloc[last_peak_idx_m5:]).idxmin()
                
            
            # Lowest Low has to be detected from point of last_peak_idx to the point of signal being sent, which would be r["low"].iloc[-1]
            
                if len(valid_peaks_m5[-6:-1]) > 0:
                       
                    for i in range (-6,-1):
                        if (r["high"].iloc[valid_peaks_m5[i]] > r["trend"].iloc[last_peak_idx_m5]
                        and r["high"].iloc[valid_peaks_m5[i]] > r["high"].iloc[last_peak_idx_m5]
                        and (lowest_since_peak - valid_troughs_m5[i]) < 288
                        ):
                            emer_recom_fib.append(r["high"].iloc[valid_peaks_m5[i]])
          
                if len(emer_recom_fib) > 0:
                    
                    fibo_start= dec(max(emer_recom_fib),symbol)
    
                elif len(emer_recom_fib) == 0:
                    
                    fibo_start=r["high"].iloc[last_peak_idx_m5]
                    
                fibo_end=r["low"].iloc[lowest_since_peak]
            
            # Distance
                distance= fibo_start-fibo_end
                lvl1=0.382*distance
                lvl2=0.5*distance
                lvl3=0.681*distance
                lvl4=1.618*distance
                lvl5=2.618*distance
            
                lvl_1=dec((fibo_start-lvl1),symbol)
                lvl_2=dec((fibo_start-lvl2),symbol)
                lvl_3=dec((fibo_start-lvl3),symbol)
                tp1=dec((fibo_start-lvl4),symbol)
                tp4=dec((fibo_start-lvl5),symbol)

            if lvl_2 is not None:
                
                if r["close"].iloc[-1] > lvl_2:
                    entry_lvl = r["close"].iloc[-1]
                else:
                    entry_lvl = lvl_2
    
        if macro_slope_m5 > 0:
            
            if last_peak_idx_m5 > last_trough_idx_m5:
            
                if len(valid_troughs_m5[-6:-1]) > 0:
                    
                    for i in range (-6,-1):
                        if (r["low"].iloc[valid_troughs_m5[i]] < r["trend"].iloc[last_trough_idx_m5]
                        and r["low"].iloc[valid_troughs_m5[i]] < r["low"].iloc[last_trough_idx_m5]
                        and (last_trough_idx_m5 - valid_troughs_m5[i]) < 288
                        ):
                            emer_recom_fib.append(r["low"].iloc[valid_troughs_m5[i]])
                  
                if len(emer_recom_fib) > 0:
                    len(valid_troughs_m5[-6:-1]) > 0
                    fibo_start=dec(min(emer_recom_fib),symbol)
    
                elif len(emer_recom_fib) == 0:
                    
                    fibo_start=r["low"].iloc[last_trough_idx_m5]      
            
            # Lowest Low has to be detected from point of last_peak_idx to the point of signal being sent, which would be r["low"].iloc[-1]
            
                fibo_end=r["high"].iloc[last_peak_idx_m5]
            
            # Distance
                distance= fibo_end-fibo_start
                lvl1=0.382*distance
                lvl2=0.5*distance
                lvl3=0.681*distance
                lvl4=1.618*distance
                lvl5=2.618*distance
                
                ## Fibonacci retracements
            
                lvl_1=dec((fibo_start+lvl1),symbol)
                lvl_2=dec((fibo_start+lvl2),symbol)
                lvl_3=dec((fibo_start+lvl3),symbol)
                tp1=dec((fibo_start+lvl4),symbol)
                tp4=dec((fibo_start+lvl5),symbol)
        
            if last_trough_idx_m5 > last_peak_idx_m5:
                        
                highest_since_trough=(r["high"].iloc[last_trough_idx_m5:]).idxmax()
    
                if len(valid_troughs_m5[-6:-1]) > 0:
                    
                    for i in range (-6,-1):
                        if (r["low"].iloc[valid_troughs_m5[i]] < r["trend"].iloc[last_trough_idx_m5]
                        and r["low"].iloc[valid_troughs_m5[i]] < r["low"].iloc[last_trough_idx_m5]
                        and (highest_since_trough - valid_troughs_m5[i]) < 288
                        ):
                            emer_recom_fib.append(r["low"].iloc[valid_troughs_m5[i]])
                  
                if len(emer_recom_fib) > 0:
                    
                    fibo_start=dec(min(emer_recom_fib),symbol)
    
                elif len(emer_recom_fib) == 0:
                    
                    fibo_start=r["low"].iloc[last_trough_idx_m5]      
    
                fibo_end=r["high"].iloc[highest_since_trough]
            
            # Distance
                distance= fibo_end-fibo_start
                lvl1=0.382*distance
                lvl2=0.5*distance
                lvl3=0.681*distance
                lvl4=1.618*distance
                lvl5=2.618*distance
                
                ## Fibonacci retracements
            
                lvl_1=dec((fibo_start+lvl1),symbol)
                lvl_2=dec((fibo_start+lvl2),symbol)
                lvl_3=dec((fibo_start+lvl3),symbol)
                tp1=dec((fibo_start+lvl4),symbol)
                tp4=dec((fibo_start+lvl5),symbol)

            if lvl_2 is not None:
                
                if r["close"].iloc[-1] < lvl_2:
                    entry_lvl = r["close"].iloc[-1]
                else:
                    entry_lvl = lvl_2
    
    return lvl_1,lvl_2,lvl_3,tp1,tp4, entry_lvl, fibo_start


# In[ ]:


def compute_sl(r): ### In the process (18/08)
    ### Completely unbothered by other technicals
    global sl, entry_lvl
    if type(r) == pd.core.frame.DataFrame:
        
    
        sl=None
        #entry_lvl = lvl_2
        recom_sl=[]
        emer_recom_sl=[]
        stop_loss=[]
        sl_dist = data["atr"].mean() *5
        if entry_lvl is not None:
            sl1=dec((entry_lvl+sl_dist),symbol)
            stop_loss.append(sl1)
        m5atr=(data["atr"]).mean()
    
        if (last_peak_idx_m5 is not None
            and entry_lvl is not None
            and last_trough_idx_m5 is not None
             ):
        ## Most optimum
        # Uptrend
            if macro_slope_m5 > 0:
    
            ## Main SL mechanism
    
                if (data["low"].iloc[valid_troughs_m5[-6:-1]] is not None
                and len(valid_troughs_m5[-6:-1])>0 
        
                   ):
                    for i in range(-6,-1):
                        if last_trough_idx_m5 - valid_troughs_m5[i] < 288:
                            if (data["low"].iloc[valid_troughs_m5[i]]< data["lower band"].iloc[last_trough_idx_m5]
                            and ((data["low"].iloc[valid_troughs_m5[i]] < data["oversold"].iloc[last_trough_idx_m5])
                            or (data["low"].iloc[valid_troughs_m5[i]] < data["lower band"].iloc[last_trough_idx_m5]
                                
                                ))):
        
                                emer_recom_sl.append(data["low"].iloc[valid_troughs_m5[i]] - m5atr)
                                
                           
                            elif data["low"].iloc[valid_troughs_m5[i]]< data["low"].iloc[last_trough_idx_m5]:
                                
                                recom_sl.append(data["low"].iloc[valid_troughs_m5[i]] - m5atr)
                        
    
                if ( ## Classifier 
                ((data["low"].iloc[last_trough_idx_m5] < data["lower band"].iloc[last_trough_idx_m5])
                or (data["close"].iloc[last_trough_idx_m5] < data["lower band"].iloc[last_trough_idx_m5]))
                ):
        
                    recom_sl.append((data["low"].iloc[last_trough_idx_m5] - m5atr))  
                    
    
                elif ( ## Classifier 
                    ((r["lower midline"].iloc[last_trough_idx_m5] > r["low"].iloc[last_trough_idx_m5] > r["lower band"].iloc[last_trough_idx_m5])
                    or (r["lower midline"].iloc[last_trough_idx_m5] > r["close"].iloc[last_trough_idx_m5] > r["lower band"].iloc[last_trough_idx_m5]))
                ):
                    recom_sl.append((r["lower band"].iloc[last_trough_idx_m5] - m5atr)) 
        
                  ### Risk On
    
                if last_peak_idx_m5 > last_trough_idx_m5:
    
                    if len(r["low"].iloc[-6:-1])>0:      
                        max_5_idx = r["low"].iloc[-6:-1].idxmin()
                        for i in range(-6,-1):
                            if max_5_idx - valid_troughs_m5[i] < 288:
                                if (r["low"].iloc[valid_troughs_m5[i]]< r["low"].iloc[max_5_idx]
                                and ((r["low"].iloc[valid_troughs_m5[i]] < r["oversold"].iloc[max_5_idx])
                                or (r["close"].iloc[valid_troughs_m5[i]] < r["oversold"].iloc[max_5_idx]
                                    
                                    ))):
                                    
                                    emer_recom_sl.append(r["low"].iloc[valid_troughs_m5[i]] - m5atr)
                                    
                                elif r["low"].iloc[valid_troughs_m5[i]]< r["low"].iloc[max_5_idx]:
                                    
                                    recom_sl.append(r["low"].iloc[valid_troughs_m5[i]] - m5atr)
    
                        if len(emer_recom_sl) ==0 and len(recom_sl) ==0: ## Backup
                                            
                            for i in range(-6,-1):
        
                                    if (r["low"].iloc[valid_troughs_m5[i]]< r["low"].iloc[max_5_idx]
                                    and ((r["low"].iloc[valid_troughs_m5[i]] < r["oversold"].iloc[max_5_idx])
                                    or (r["close"].iloc[valid_troughs_m5[i]] < r["oversold"].iloc[max_5_idx]
                                        
                                        ))):
                                        
                                        emer_recom_sl.append(r["low"].iloc[valid_troughs_m5[i]] - m5atr)
                                        
                                    elif r["low"].iloc[valid_troughs_m5[i]]< r["low"].iloc[max_5_idx]:
                                        
                                        recom_sl.append(r["low"].iloc[valid_troughs_m5[i]] - m5atr)
    
                ## vector sl
                if r["low"].iloc[last_trough_idx_m5] > r["low"].iloc[last_vector_trough]:
                    emer_recom_sl.append(r["lower band"].iloc[last_vector_trough] - m5atr)
                ###
                
                if sl==None:
                    sl=recom_sl.append((r["low"].iloc[last_trough_idx_m5] - m5atr))
                
                # because uptrend, 2nd rated SLs should be MIN, and 1st rated SLs should be MAX
                
    
                if (len(recom_sl) > 0) and (len(emer_recom_sl) > 0) :
                    sl1=dec(min(emer_recom_sl),symbol)
                    sl2=dec(min(recom_sl),symbol)
                    if sl2 < sl1:
                        sl=sl2
                    else:
                        sl=sl1
    
                elif len(emer_recom_sl) > 0:
                    sl1=dec(min(emer_recom_sl),symbol)
                    
                elif len(emer_recom_sl) == 0 and len(recom_sl)>0:
                    
                    sl=dec(min(recom_sl),symbol)      
    
                if fibo_start != None:
                    
                    if sl > entry_lvl:
                        
                        sl = dec((fibo_start - m5atr), symbol)
    
        ## Downtrend ##
                
            if macro_slope_m5 < 0 :
    
            ## Main SL mechanism
                m5atr=(r["atr"].mean())
                vital_highs=[]
                if (r["high"].iloc[valid_peaks_m5[-6:-1]] is not None
                and len(valid_peaks_m5[-6:-1])>0
    
                   ):
                    for i in range(-6,-1):
                        if last_peak_idx_m5 - valid_peaks_m5[i] < 288:
                            if (r["high"].iloc[valid_peaks_m5[i]]> r["high"].iloc[last_peak_idx_m5]
                            and ((r["high"].iloc[i] > r["overbought"].iloc[last_peak_idx_m5])
                            or (r["high"].iloc[i] > r["upper band"].iloc[last_peak_idx_m5]
                                
                                ))):
                                
                                emer_recom_sl.append(r["high"].iloc[valid_peaks_m5[i]] + m5atr)
                                                            
                            elif r["high"].iloc[valid_peaks_m5[i]]> r["high"].iloc[last_peak_idx_m5]:
                                
                                recom_sl.append(r["high"].iloc[valid_peaks_m5[i]] + m5atr)
                               
                
                if ( ## Classifier 
                ((r["high"].iloc[last_peak_idx_m5] > r["upper band"].iloc[last_peak_idx_m5])
                or (r["close"].iloc[last_peak_idx_m5] > r["upper band"].iloc[last_peak_idx_m5]))
                ):
                    recom_sl.append((r["high"].iloc[last_peak_idx_m5] + m5atr))  
                                          
                elif ( ## Classifier 
                    ((r["upper midline"].iloc[last_peak_idx_m5] < r["high"].iloc[last_peak_idx_m5] < r["upper band"].iloc[last_peak_idx_m5])
                    or (r["upper midline"].iloc[last_peak_idx_m5] < r["close"].iloc[last_peak_idx_m5] < r["upper band"].iloc[last_peak_idx_m5]))
                ):
                    recom_sl.append((r["upper band"].iloc[last_peak_idx_m5] + m5atr))
    
                if last_trough_idx_m5 > last_peak_idx_m5:
    
            ### Risk On
                    if len(r["high"].iloc[-6:-1])> 0:      
                        max_5_idx = r["high"].iloc[-6:-1].idxmax()
                        for i in range(-6,-1):
                            if max_5_idx - valid_peaks_m5[i] < 288:
                                if (r["high"].iloc[valid_peaks_m5[i]]> r["high"].iloc[max_5_idx]
                                and ((r["high"].iloc[valid_peaks_m5[i]] > r["overbought"].iloc[max_5_idx])
                                or (r["close"].iloc[valid_peaks_m5[i]] > r["overbought"].iloc[max_5_idx]
                                    
                                    ))):
                                    emer_recom_sl.append(r["high"].iloc[valid_peaks_m5[i]] + m5atr)
                                    
                                elif r["high"].iloc[valid_peaks_m5[i]]> r["high"].iloc[max_5_idx]:
                                    recom_sl.append(r["high"].iloc[valid_peaks_m5[i]] + m5atr)
    
                        if len(emer_recom_sl)==0 and len(recom_sl) ==0:
                        
                            for i in range(-6,-1):
                                
                                if (r["high"].iloc[valid_peaks_m5[i]]> r["high"].iloc[max_5_idx]
                                and ((r["high"].iloc[valid_peaks_m5[i]] > r["overbought"].iloc[max_5_idx])
                                or (r["close"].iloc[valid_peaks_m5[i]] > r["overbought"].iloc[max_5_idx]
                                    
                                    ))):
                                    emer_recom_sl.append(r["high"].iloc[valid_peaks_m5[i]] + m5atr)
                                    
                                elif r["high"].iloc[valid_peaks_m5[i]]> r["high"].iloc[max_5_idx]:
                                    
                                    recom_sl.append(r["high"].iloc[valid_peaks_m5[i]] + m5atr)                        
                                    
            
                if sl==None:
                    
                    recom_sl.append((r["high"].iloc[last_peak_idx_m5] + m5atr))

                ## vector sl
                if r["high"].iloc[last_peak_idx_m5] < r["high"].iloc[last_vector_peak]:
                    emer_recom_sl.append(r["lower band"].iloc[last_vector_peak] + m5atr)
                ###
    
                # because downtrend, sl computed should be at MAX for 2nd rated SLs, and MIN for 1st rated SLs
                if (len(recom_sl) > 0) and (len(emer_recom_sl) > 0) :
                    sl1=dec(max(emer_recom_sl),symbol)
                    sl2=dec(max(recom_sl),symbol)
                    if sl1 < sl2:
                        sl=sl2
                    else:
                        sl=sl1             
                elif len(emer_recom_sl) > 0:
                    sl=dec(max(emer_recom_sl),symbol)
                    
                elif len(emer_recom_sl) == 0 and len(recom_sl)>0 :            
                    sl=dec(max(recom_sl),symbol) 
    
                ### Check if sl < entry_lvl
                if fibo_start != None:
                    
                    if sl < entry_lvl:
                        
                        sl = dec((fibo_start + m5atr), symbol)
                
    if sl is not None:
        return emer_recom_sl, recom_sl, sl

    else:
        return False


# In[ ]:


def compute_tp(r): ## Defining tp, we need to establish values for fibo up or fibo down first
    ## Assuming fibo_up() only runs upon the cond. of macro_slope_m5 > 0,
    global tp, rr
    if type(r) == pd.core.frame.DataFrame:
        
    
        if macro_slope_m5 > 0:
            tp_list=[]
            tp_recom=[]
            tp_norm=[]
            if tp1 is not None:
                tp_list.append(tp1)
    
            try :
                
                if len(jeblon)> 0:
                    for val in jeblon:
                        if val > r["upper midline"].iloc[last_trough_idx_m5]:
                            tp_list.append(val)
    
            except NameError:
                pass
                
            if r["high"].iloc[last_peak_idx_m5] > r["upper midline"].iloc[last_peak_idx_m5]:
                tp_list.append(r["high"].iloc[last_peak_idx_m5] + m5atr)
            if tp4 is not None:
                if tp4 <= r["upper band"].iloc[last_trough_idx_m5]:
                    tp_list.append(tp4)
            
            ### Risk reward bigger than 1.0
            
            for tp in tp_list:
                
                try :
                    
                    rr=risk_reward(sl,entry_lvl,tp)
                    if rr > 1.0:
                        tp_recom.append(tp)
                    else:
                        tp_norm.append(tp)
                        
                    if len(tp_recom) > 0:
                        
                        tp=dec(min(tp_recom),symbol)
                        rr = risk_reward(sl,entry_lvl,tp)
                        rr = round(rr,2)
                        
                    elif tp_recom==0 and len(tp_norm)>0:
                        
                        tp=dec(max(tp_norm),symbol)
                        rr = risk_reward(sl,entry_lvl,tp)
                        rr = round(rr,2)
    
                except ZeroDivisionError as e :
                    
                    rr = 0.0
                    
                except Exception as e:
                    rr= 0.0
        
                if rr < 1.0:
                    
                    try:
                        
                        tp = dec((lvl_2+(lvl_2-sl)),symbol)
                        ## reassign rr
                        rr=risk_reward(sl,lvl_2,tp)
    
                    except ZeroDivisionError as e: 
                        rr = 0.0
                        
                    except Exception as e:
                        print(e)
                        rr = 0.0
                        
            
        if macro_slope_m5 < 0:
            
            tp_list=[]
            tp_recom=[]
            tp_norm=[]
            if tp1 is not None:
                tp_list.append(tp1)
    
            try :
                if len(jeblon)> 0:
                    for val in jeblon:
                        if val < r["lower midline"].iloc[last_peak_idx_m5]:
                            tp_list.append(val)
    
            except NameError:
                pass
                
            if r["low"].iloc[last_trough_idx_m5] < r["lower midline"].iloc[last_trough_idx_m5]:
                tp_list.append(r["high"].iloc[last_trough_idx_m5] - m5atr)
            if tp4 is not None:
                if tp4 >= r["lower band"].iloc[last_peak_idx_m5]:
                    tp_list.append(tp4)
            
            ### Risk reward bigger than 1.0
            
            for tp in tp_list:
    
                try : 
                    
                    rr=risk_reward(sl,entry_lvl,tp)
                    if rr > 1.0:
                        tp_recom.append(tp)
                    else:
                        tp_norm.append(tp)
                        
                    if len(tp_recom) > 0 :
                        
                        tp=dec(max(tp_recom),symbol)
                        rr = risk_reward(sl,entry_lvl,tp)
                        rr = round(rr,2)
                        
                    elif len(tp_recom)==0 and len(tp_norm)>0 :
                        
                        tp=dec(min(tp_norm),symbol)
                        rr = risk_reward(sl,entry_lvl,tp)
                        rr = round(rr,2)
    
                except ZeroDivisionError as e:
                    rr = 0.0
    
                except Exception as e:
                    print(e)
                    rr = 0.0
                    
            if rr < 1.0: ## establish new rr
                
                try:
                    
                    tp=dec((lvl_2-(sl-lvl_2)),symbol)
                    rr=risk_reward(sl,lvl_2,tp)
    
                except ZeroDivisionError as e:
                    rr = 0.0
                except Exception as e:
                    rr = 0.0
    
    if tp is not None :
        return tp, tp_recom, tp_norm
    else:
        return False


# In[ ]:


def generate_dummy_regression(x,y):
    global test_trend, test_upper_band, test_lower_band, test_slope
    x_model= x.values.reshape(-1,1)
    y_model= y.values
    
    lig_model=LinearRegression().fit(x_model,y_model)
    test_trend=lig_model.predict(x_model)
    test_slope=lig_model.coef_[0]

    
    test_residuals = y_model - test_trend 
    test_std = test_residuals.std()
    

    test_upper_band =test_trend + 2*test_std
    test_lower_band = test_trend - 2*test_std

    return test_trend, test_upper_band, test_lower_band, test_slope


# In[ ]:


def conservative_tp():
    global csv_tp
    if macro_slope_m5 > 0:
        
        csv_tp = entry_lvl + (entry_lvl - sl)
        csv_tp = dec(csv_tp,symbol)
        
    if macro_slope_m5 < 0:
        
        csv_tp = entry_lvl - (sl - entry_lvl)
        csv_tp = dec(csv_tp,symbol)

    return csv_tp


# In[ ]:


def change_timezone(x):

    format_df = "%Y-%m-%d %H:%M:%S%z"
    new_format = "%Y-%m-%d %H:%M:%S"
    x = datetime.strptime(x, format_df).astimezone()     
    x = datetime.strftime(x, new_format)
    return x


# In[ ]:


def convert_unix_timestamp(x):
    format_df = "%Y-%m-%d %H:%M:%S%z"
    try:
       return int(datetime.strptime(x, format_df).timestamp())
        
    except Exception as e:
        format_df = "%Y-%m-%d %H:%M:%S" #'2018-01-02 06:10:00'
        return int(datetime.strptime(x, format_df).timestamp())
        print(e)


# In[ ]:


def plot_module():
    plt.figure(figsize = (12,6))
    plt.plot(numbers, data["close"], color = "green", alpha = 0.8)
    plt.plot(numbers, trend, color = "blue", linestyle = "-")
    plt.plot(numbers, upper_band, color = "blue", linestyle = "--")
    plt.plot(numbers, lower_band, color = "blue", linestyle = "--")
    plt.plot(numbers, data["moving average 50"], color = "black", linestyle = "--", alpha = 0.5)
    plt.scatter(numbers[valid_peaks_m5], data["close"].iloc[valid_peaks_m5], marker = "v", color = "red")
    plt.scatter(numbers[valid_troughs_m5], data["close"].iloc[valid_troughs_m5], marker = "^", color = "purple")
    plt.scatter(numbers[last_vector_peak], data["close"].iloc[last_vector_peak], marker = "v", color = "purple")
    plt.fill_between(x = numbers, y1 = overbought, y2 = upper_band, color = "red", alpha = 0.3)
    plt.fill_between(x = numbers, y1 = oversold, y2 = lower_band, color = "green", alpha = 0.3)
    plt.hlines(y = sl ,xmin = last_vector_peak, xmax = 2200, color ="red", alpha = 0.8)
    plt.hlines(y = tp ,xmin = last_vector_peak, xmax = 2200, color ="green", alpha = 0.8)
    plt.hlines(y = entry_lvl, xmin = last_vector_peak, xmax = 2200, color ="blue", alpha = 0.8)

    plt.plot(numbers, peak_trend, alpha = 0.5, color = "red", linestyle = "--")
    plt.plot(numbers, trough_trend, alpha = 0.5, color = "green", linestyle = "--")
    
    plt.grid()
    plt.show()    


# ### Instrument List

# In[ ]:


eu = {"id": "EURUSD",
      "telegram token" : "7655061560:AAF3ZhFIw1fDopS3Ubiq4d3BZC31vU9JQDM",
      "telegram chat id" : "-1002994239464"
    }

gu = {"id": "GBPUSD",
      "telegram token" : "7726925813:AAEVeGL3WqAHG3EWIpCEbPAt9cmzKUHkl4E",
      "telegram chat id" : "-1002689119862"
    }
gj = {"id": "GBPJPY",
      "telegram token" : "7724186584:AAH0IF168B0IIB3TG7VDk-XJr0S1hZB1ofk",
      "telegram chat id" : "-1003096573575"
    }
gold = {"id": "XAUUSD",
      "telegram token" : "7739329449:AAHtbnTQl1rM2z0G6LHXtSRETKxXz8V3NJM",
      "telegram chat id" : "-1003060664508"
    }
dow = {"id": "US30.cash",
      "telegram token" : "8090148960:AAEVim7HD9dIz6KZqP6jI2gkpFGbZDk6q7A",
      "telegram chat id" : "-1003044860973"
    }
dax = {"id": "GER40.cash",
      "telegram token" : "7814087673:AAE9zcAxk9Yio3MmmrC8VESDb1rkxEIdB-o",
      "telegram chat id" : "-1002934706394"
    }
uj = {"id": "USDJPY",
      "telegram token" : "7590221480:AAENzVa_IxQ23soCRYPebkPcPBLRtjRCL3w",
      "telegram chat id" : "-1003050413624"
    }
ej = {"id": "EURJPY",
      "telegram token" : "7997307038:AAGTk6xSqzJXQnNgprUdyA2P6yEgW0hatm4",
      "telegram chat id" : "-1003001392228"
    }
au = {"id": "AUDUSD",
      "telegram token" : "7968282303:AAFaraUeBUmDuD_msQe2AAZl31S3LdHxoYo",
      "telegram chat id" : "-1003095401611"
    }


# #### Momentum Functions

# In[ ]:


def check_priority_trade(file):

    if file.endswith("json"):
        with open(file, "r") as f:
            data = json.load(f)
            if len(data) > 0:
                start_momentum = []
                for idx, item in enumerate(data):
                    start_index = item.get("Momentum Start Index")
                    start_momentum.append(start_index)
                priority_trade = min(start_momentum)
                for idx, item in enumerate(data):
                    if item.get("Momentum Start Index") == priority_trade:
                        data.pop(idx)
                        print("Momentum Trade Renewed.")


# In[ ]:


### creating Class for momentum trade
class MOMENTUM_TRADE:
    file = "momentum_trade.json"
    def __init__(self, type_of_trade):
        self.trade_type = type_of_trade
        
    def entry(self):
        
        with open(self.file, "r") as f:
            dt = json.load(f)
            if len(dt) > 0:
                    
                for number, item in enumerate(dt):
                ## acquire index where SL was hit (master idx)
                    
                    if item.get("Momentum Start Index") == idx:
                        ### acquire highs if SHORTING
                        entry_lvl = (data["high"].iloc[-1] + data["low"].iloc[-1]) / 2
                        self.entry = entry_lvl
                        try:
                            return self.entry
                        except NameError as e:
                            print(e) 
            
    def sl(self):
        
        with open(self.file, "r") as f:
            dt = json.load(f)
            if len(dt) > 0:
                
                for number, item in enumerate(dt):
                    
                ## acquire index where SL was hit (master idx)
                    if item.get("Momentum Start Index") == idx:            
                ## acquire index where SL was hit (master idx)
                        sl_index = item.get("Hit SL Index")
                        ### acquire highs if SHORTING
                        if self.trade_type == "SHORT":
                            
                            sl = data["high"].iloc[last_peak_idx_m5] + m5atr
                            
                            
                        elif self.trade_type == "LONG":
                            
                            sl = data["low"].iloc[last_trough_idx_m5] - m5atr
                            
                        try:
                            self.stop_loss = sl
                            return self.stop_loss
                        except NameError as e:
                            print(e)
            
    def tp(self):
    
        with open(self.file, "r") as f:
            dt = json.load(f)
            if len(dt) > 0:
                
                for number, item in enumerate(dt):
                    
                ## acquire index where SL was hit (master idx)
                    if item.get("Momentum Start Index") == idx:              
                ## acquire index where SL was hit (master idx)
                        ### acquire highs if SHORTING
                        if self.trade_type == "SHORT":
                            
                            distance = self.stop_loss - self.entry
                            tp = self.entry - distance
                            
                        elif self.trade_type == "LONG":
                            
                            distance = self.entry - self.stop_loss
                            tp = self.entry + distance
                        try:
                            self.take_profit = tp
                            return self.take_profit
                        except NameError as e:
                            print(e)
        


# ### Notes

# 1. Range must be defined, starting index must be at 2000, ends at 48000,
#    - so for example, for each candle, (2000, 4000), (2001, 4001), (2002, 4002)
# 2. 19:09 - Template for calculating regression slopes completed
# 3. TBC - 07/12 onwards
#    - import compute_sl(), compute_fib(), compute_tp() functions
#    - Calc. Max Price deviation from signal price
#    - Emulate csv logging features in template
#    - Run for all symbols
#    
# 4. Ammended sl logic for vector indicator, might need fine tuning, basic sl implemented.

# ### Backtest Template

# In[ ]:


#### For all instruments
total_cum = []
total_short_sl = []
total_short_tp = []

total_long_tp = []
total_long_sl = []
signal_delay=[]

symbol = "GBPUSD"

file = f"{symbol} M5 DATA 2018-NOW.csv"

df = pd.read_csv(file)

if "timestamp" in df.columns:
    df["timestamp"] = df["timestamp"].apply(change_timezone)
    print("Timezone changed")
    df["unix timestamp"] = df["timestamp"].apply(convert_unix_timestamp)


df["rsi"] = ta.rsi(df["close"], length=14)
df["moving average"] = df["close"].rolling(window = 200).mean()
df["moving average 50"] = df["close"].rolling(window = 50).mean()

init_bar = 4000
fin_bar = len(df)
bars = len(df)
progress = np.linspace(init_bar, bars, 11)
all_slopes = []
shorts = []
close_shorts = []
longs = []
close_longs = []
entry_hit = []

short_tp_hit = []
short_sl_hit = []

long_tp_hit = []
long_sl_hit = []
cum= []
momentum_trade = False

for idx in range(init_bar, fin_bar, 1):
    
## 1. Establish dataframe taken
    confirmation = "momentum_trade.json"
    check_priority_trade(confirmation) ## to constantly remove trades that take longer

    data = df.iloc[idx - 2000:idx].copy()
    data = data.reset_index(drop=True)
    numbers = np.arange(0,len(data))
    numbers = pd.Series(numbers)

    ## 2. Generate slope regression for each individual slice of data
    test = generate_dummy_regression(numbers, data["close"])
    trend = test[0]
    data["trend"] = trend
    upper_band = test[1]
    lower_band = test[2]
    data["upper midline"] = (test[1] + test[0])/2
    data["lower midline"] = (test[2] + test[0])/2#+ lower_band
    
    data["overbought"] = (data["upper midline"] + upper_band)/2
    overbought = data["overbought"].values
    
    data["oversold"] = (data["lower midline"] + lower_band)/2
    oversold = data["oversold"].values
    
    data["upper band"] = upper_band
    data["lower band"] = lower_band
    macro_slope_m5 = test[3]

    data["atr"] = ta.atr(high = data["high"], low= data["low"], close = data["close"], length = 14)
    m5atr = data["atr"].mean() 
    sdv= data["atr"].std() 
    data["z_score"]=(data["atr"]-data["atr"].mean())/sdv

    std = data["close"].std() 

    ## 3. Generate peaks/ troughs with find_peaks()
    const = 0.5
    peaks, _ = find_peaks(data["close"], prominence = const*data["close"].std(), distance = 3)
    troughs, _ = find_peaks(-data["close"], prominence = const*data["close"].std(), distance = 3)
    valid_peaks_m5 = peaks
    valid_troughs_m5 = troughs
  
    if len(valid_peaks_m5) < 10 and len(valid_troughs_m5) < 10:
        const = 0.5*0.5
        peaks, _ = find_peaks(data["close"], prominence = const*data["close"].std(), distance = 3)
        troughs, _ = find_peaks(-data["close"], prominence = const*data["close"].std(), distance = 3)
        valid_peaks_m5 = peaks
        valid_troughs_m5 = troughs
        
        if len(valid_peaks_m5) < 10 and len(valid_troughs_m5) < 10:
            const = 0.5*0.5*0.5
            peaks, _ = find_peaks(data["close"], prominence = const*data["close"].std(), distance = 3)
            troughs, _ = find_peaks(-data["close"], prominence = const*data["close"].std(), distance = 3)
            valid_peaks_m5 = peaks
            valid_troughs_m5 = troughs

### TBA

        
    last_peak_idx_m5 = peaks[-1]
    last_trough_idx_m5 = troughs[-1]    

    last_vector_peak = peaks[-1]
    last_vector_trough = troughs[-1]

    ## generate linear regression for peaks and troughs
    numbers = np.arange(len(data)).reshape(-1,1)
    x = valid_peaks_m5.reshape(-1,1)
    y = data["high"].iloc[valid_peaks_m5]
    model = LinearRegression().fit(x,y)
    peak_trend = model.predict(numbers)
    data["peak trend"] = peak_trend
    
    x = valid_troughs_m5.reshape(-1,1)
    y = data["low"].iloc[valid_troughs_m5]
    model = LinearRegression().fit(x,y)
    trough_trend = model.predict(numbers)   
    data["trough trend"] = trough_trend

    numbers = np.arange(len(data))
    ## momentum trade

    try:
        with open(confirmation, "r") as f:
            active_trades = json.load(f)
            for number, momentum_truth in enumerate(active_trades):
                
                if momentum_truth.get("Momentum Trade") == True:
                    
                    if momentum_truth.get("Momentum Start Index") == idx:
                        trade_id = momentum_truth.get("Trade ID")
                        
                        if momentum_truth.get("trend") < 0: ## initiate short
                            short = MOMENTUM_TRADE("SHORT")
                            entry_lvl = short.entry()
                            tp = short.sl()
                            sl = short.tp()
                            rr = risk_reward(sl, entry_lvl,tp)
                            loss_rr = 2 - rr
                            
                            hit_tp = np.argwhere(df["low"].iloc[idx:].values > tp)
                            if len(hit_tp) > 0:
                                tp_idx = hit_tp[0][0]
                            else:
                                tp_idx = 9999

                            hit_sl = np.argwhere(df["high"].iloc[idx:].values < sl)
                            if len(hit_sl) > 0:
                                sl_idx = hit_sl[0][0]
                            else:
                                sl_idx = 9999
                                        
                            if sl_idx  != None and tp_idx != None:
                                print(f"sl_idx = {sl_idx}, tp_idx = {tp_idx}")
                                
                                if min([tp_idx, sl_idx]) == sl_idx:
                                    short_sl_hit.append(sl_idx)
                                    total_short_sl.append(sl_idx)
                                    cum.append(loss_rr*-1)
                                    total_cum.append(loss_rr*-1)
                                    
                                    print("SL Hit!")   
                                    
                                elif min([tp_idx, sl_idx]) == tp_idx:
                                     
                                    short_tp_hit.append(tp_idx)
                                    total_short_tp.append(tp_idx)
                                    total_cum.append(rr*1)
                                    cum.append(rr*1)
                                    print("TP Hit!")  
                                    
                                print(len(active_trades))
                                active_trades.clear()

                                with open(confirmation, "w") as f:
                                    json.dump(active_trades, f)
                                
                                trade_size = normalised_difference(tp,sl)
                                print("MOMENTUM TRADE ACTIVE")
                                print(f"RR = {rr}") if rr is not None else print("RR failed to be computed")
                                print(f"Loss RR = {-loss_rr}") if loss_rr is not None else print("Loss RR failed to be computed")
    
                                print(f"sl == {sl}") if entry_lvl != None else print("None")
                                print(f"entry = {entry_lvl}") if entry_lvl != None else print("None")
                                print(f"tp == {tp}") if tp != None else print("None")                                       
                                momentum_log_trade(data)
                                plot_module()
                                break
                                        
                                       
                        elif momentum_truth.get("trend") > 0: ## initiate long
                            
                            long = MOMENTUM_TRADE("LONG")
                            entry_lvl = long.entry()
                            tp = long.sl()
                            sl = long.tp()
                            rr = risk_reward(sl, entry_lvl,tp)
        
                            loss_rr = 2 - rr
                            hit_tp = np.argwhere(df["high"].iloc[idx:].values < tp)
                            if len(hit_tp) > 0:
                                tp_idx = hit_tp[0][0]
                            else:
                                tp_idx = 9999

                            hit_sl = np.argwhere(df["low"].iloc[idx:].values > sl)
                            if len(hit_sl) > 0:
                                sl_idx = hit_sl[0][0]
                            else:
                                sl_idx = 9999

                            if sl_idx  != None and tp_idx != None:
                                print(f"sl_idx = {sl_idx}, tp_idx = {tp_idx}")
                                
                                if min([tp_idx, sl_idx]) == sl_idx:
                                    short_sl_hit.append(sl_idx)
                                    total_short_sl.append(sl_idx)
                                    cum.append(loss_rr*-1)
                                    total_cum.append(loss_rr*-1)
                                    
                                    print("SL Hit!")   
                                    
                                elif min([tp_idx, sl_idx]) == tp_idx:
                                     
                                    short_tp_hit.append(tp_idx)
                                    total_short_tp.append(tp_idx)
                                    total_cum.append(rr*1)
                                    cum.append(rr*1)
                                    print("TP Hit!") 

                                print(len(active_trades))
                                active_trades.clear()

                                with open(confirmation, "w") as f:
                                    json.dump(active_trades, f)
                                
                                trade_size = normalised_difference(tp,sl)
                                print("MOMENTUM TRADE ACTIVE")
                                print(f"RR = {rr}") if rr is not None else print("RR failed to be computed")
                                print(f"Loss RR = {-loss_rr}") if loss_rr is not None else print("Loss RR failed to be computed")
    
                                print(f"sl == {sl}") if entry_lvl != None else print("None")
                                print(f"entry = {entry_lvl}") if entry_lvl != None else print("None")
                                print(f"tp == {tp}") if tp != None else print("None")                                       
                                momentum_log_trade(data)
                                plot_module()
                                break

                                    
                    ### Implement momentum strategy here
                    else:
                        pass
                    
                else:
                    pass
                            
    except FileNotFoundError as e:
        with open(confirmation, "w") as f:
            empty = []
            json.dump(empty, f)
    except JSONDecodeError as e:
        if confirmation in os.listdir():
            os.remove(confirmation)
    
    ## establishing logic
    
    if macro_slope_m5 < 0:
        
        if upper_band[last_vector_peak] >= data["high"].iloc[last_vector_peak] >= overbought[last_vector_peak]:
            
            if shorts == []:

                ## Trade metrics
                compute_fib(data)
                # print(f"Entry lvl is {entry_lvl}")
                compute_sl(data)
                compute_tp(data)
                tp = conservative_tp()

                ### Establish risk reward, regardless of hit entry or not, assume market order

                rr = risk_reward(sl, data["close"].iloc[-1],tp)

                if rr > 0.8:
                    
                
                    if rr < 1:
                        loss_rr = 2-rr
                    else:
                        loss_rr = 1
                    
                    if data["close"].iloc[-1] < sl:
                        
                        
                        ## global df
                        ## for tp,
                        df_idx = (idx + last_vector_peak) - 2000
        
                        if df_idx < bars - 2000:
    
                            print(f"RR = {rr}") if rr is not None else print("RR failed to be computed")
                            print(f"Loss RR = {-loss_rr}") if loss_rr is not None else print("Loss RR failed to be computed")

                            print(f"sl == {sl}") if entry_lvl != None else print("None")
                            print(f"entry = {entry_lvl}") if entry_lvl != None else print("None")
                            print(f"tp == {tp}") if tp != None else print("None")                     
                            
                            hit_tp = np.argwhere(df["close"].iloc[df_idx:].values < tp)
                            if len(hit_tp) > 0:
                                tp_idx = hit_tp[0][0]
                            else:
                                tp_idx = 9999

                            hit_sl = np.argwhere(df["close"].iloc[df_idx:].values > sl):
                            if len(hit_sl) > 0:
                                sl_idx = hit_sl[0][0]
                            else:
                                sl_idx = 9999
                                    
                            if sl_idx  != None and tp_idx != None:
                                print(f"sl_idx = {sl_idx}, tp_idx = {tp_idx}")
                                
                                if min([tp_idx, sl_idx]) == sl_idx:
                                    short_sl_hit.append(sl_idx)
                                    total_short_sl.append(sl_idx)
                                    cum.append(loss_rr*-1)
                                    total_cum.append(loss_rr*-1)
                                    
                                    print("SL Hit!")

                                    ### Implement Momentum trade here
                                    momentum_idx = idx + sl_idx + 1
                                    opposite_trend = -macro_slope_m5
                                    confirmation = "momentum_trade.json"
                                    with open(confirmation, "r") as f:
                                        
                                        json_data = json.load(f)
                                        # if len(json_data) == 0:
                                            
                                        upload = {"Momentum Trade" : True,
                                                  "Trade ID" : np.random.randint(1000000,9999999),
                                                 "Momentum Start Index" : momentum_idx,
                                                  "Hit SL Index" : sl_idx + idx,
                                                 "trend" : opposite_trend}
                                        json_data.append(upload)
                                        
                                        print("Momentum Trade Details Saved.")
                                        
                                        with open(confirmation, "w") as f:
                                            
                                            json.dump(json_data, f)
       
                                elif min([tp_idx, sl_idx]) == tp_idx:
                                    short_tp_hit.append(tp_idx)
                                    total_short_tp.append(tp_idx)
                                    total_cum.append(rr*1)
                                    cum.append(rr*1)
                                    print("TP Hit!")
                                
                            # if data["close"].iloc[-1] == entry_lvl:
                            #     entry_hit.append(1)
                            # if data["close"].iloc[-1] < entry_lvl:
                            #     for index, boolean in data["close"].iloc[df_idx:] > entry_lvl:
                                    
                            #         if index < 2000:
                                        
                            #             if boolean == True:
                            #                 if min([tp_idx, sl_idx]) == tp_idx:
                            #                     entry_idx = index
                            #                     if entry_idx < tp_idx:
                            #                         entry_hit.append(entry_idx)
                            #                         break
                                                    
                            #                 elif min([tp_idx, sl_idx]) == sl_idx:
                            #                     entry_idx = index
                            #                     entry_idx = index
                            #                     if entry_idx < sl_idx:
                            #                         entry_hit.append(entry_idx)
                            #                         break                  
                                                
                            ############## Metrics ###########
                            
                            m5_ema_difference = normalised_difference(data["trend"].iloc[last_vector_peak],data["moving average"].iloc[last_vector_peak])
                            m5_peak_difference = normalised_difference(data["peak trend"].iloc[last_vector_peak], data["close"].iloc[last_vector_peak]) 
                            m5_trough_difference = normalised_difference(data["trough trend"].iloc[last_vector_peak], data["close"].iloc[last_vector_peak]) 

                            m5_log_returns = compute_timeframe_returns(data["trend"],0,1000)
                            m5_peak_returns = compute_timeframe_returns(data["peak trend"],0,1000)
                            m5_trough_returns = compute_timeframe_returns(data["trough trend"],0,1000) 

                            trade_size = normalised_difference(tp,sl)
                            
                            log_trade(data)                                
                        
                            shorts.append(last_peak_idx_m5)
                            signal_delay.append(len(data) - last_vector_peak)
                            
                            close_shorts.append(data["close"].iloc[last_peak_idx_m5])
                            
                            print(f"Short at idx {df_idx}")
                            plot_module()
                
            if len(shorts) > 0:
                
                latest_short_idx = shorts[-1]
                
                if data["close"].iloc[last_peak_idx_m5] != close_shorts[-1]: #data["close"].iloc[close_shorts[-1]]:

                    compute_fib(data)
                    # print(f"Entry lvl is {entry_lvl}")
                    compute_sl(data)
                    compute_tp(data)    
                    tp = conservative_tp()
                    
                    ### Establish risk reward, regardless of hit entry or not, assume market order

                    rr = risk_reward(sl, data["close"].iloc[-1],tp)

                    if rr > 0.8:

                        if rr < 1:
                            loss_rr = 2-rr
                        else:
                            loss_rr = 1
                        
                        if data["close"].iloc[-1] < sl:
                            
                    
                            df_idx = idx + last_vector_peak - 2000
        
                            if df_idx < bars - 2000:
    
                                print(f"RR = {rr}") if rr is not None else print("RR failed to be computed")
                                print(f"Loss RR = {-loss_rr}") if loss_rr is not None else print("Loss RR failed to be computed")
    
                                print(f"sl == {sl}") if entry_lvl != None else print("None")
                                print(f"entry = {entry_lvl}") if entry_lvl != None else print("None")
                                print(f"tp == {tp}") if tp != None else print("None")                          
                                
                            
                                hit_tp = np.argwhere(df["close"].iloc[df_idx:].values < tp)
                                if len(hit_tp) > 0:
                                    tp_idx = hit_tp[0][0]
                                else:
                                    tp_idx = 9999

                                hit_sl = np.argwhere(df["close"].iloc[df_idx:].values > sl)
                                if len(hit_sl) > 0:
                                    sl_idx = hit_sl[0][0]
                                else:
                                    sl_idx = 9999
            
                                if sl_idx  != None and tp_idx != None:
                                    print(f"sl_idx = {sl_idx}, tp_idx = {tp_idx}")
                                    if min([tp_idx, sl_idx]) == sl_idx:
                                        short_sl_hit.append(sl_idx)
                                        total_short_sl.append(sl_idx)
                                        cum.append(loss_rr*-1)
                                        total_cum.append(loss_rr*-1)
                                        
                                        print("SL Hit!")
    
                                        confirmation = "momentum_trade.json"
                                        momentum_idx = idx + sl_idx + 1
                                        opposite_trend = -macro_slope_m5
                                        with open(confirmation, "r") as f:
                                            
                                            json_data = json.load(f)
                                            # if len(json_data) == 0:
                                                
                                            upload = {"Momentum Trade" : True,
                                                      "Trade ID" : np.random.randint(1000000,9999999),
                                                     "Momentum Start Index" : momentum_idx,
                                                      "Hit SL Index" : sl_idx + idx,
                                                     "trend" : opposite_trend}
                                            json_data.append(upload)
                                            print("Momentum Trade Details Saved.")
                                            with open(confirmation, "w") as f:
                                                
                                                json.dump(json_data, f)
                                        
                                    elif min([tp_idx, sl_idx]) == tp_idx:
                                        short_tp_hit.append(tp_idx)
                                        total_short_tp.append(tp_idx)
                                        total_cum.append(rr*1)
                                        cum.append(rr*1)
                                        print("TP Hit!")
                                        
                                # if data["close"].iloc[-1] == entry_lvl:
                                #     entry_hit.append(1)
                                    
                                # if data["close"].iloc[-1] < entry_lvl:
                                #     for index, boolean in data["close"].iloc[df_idx:] > entry_lvl:
                                #         if index < 2000:
                                            
                                #             if boolean == True:
                                #                 if min([tp_idx, sl_idx]) == tp_idx:
                                #                     entry_idx = index
                                #                     if entry_idx < tp_idx:
                                #                         entry_hit.append(entry_idx)
                                #                         break
                                                        
                                #                 elif min([tp_idx, sl_idx]) == sl_idx:
                                #                     entry_idx = index
                                #                     entry_idx = index
                                #                     if entry_idx < sl_idx:
                                #                         entry_hit.append(entry_idx)
                                #                         break    

                                ############## Metrics ###########
                                    
                                m5_ema_difference = normalised_difference(data["trend"].iloc[last_vector_peak],data["moving average"].iloc[last_vector_peak])
                                m5_peak_difference = normalised_difference(data["peak trend"].iloc[last_vector_peak], data["close"].iloc[last_vector_peak]) 
                                m5_trough_difference = normalised_difference(data["trough trend"].iloc[last_vector_peak], data["close"].iloc[last_vector_peak]) 

                                m5_log_returns = compute_timeframe_returns(data["trend"],0,1000)
                                m5_peak_returns = compute_timeframe_returns(data["peak trend"],0,1000)
                                m5_trough_returns = compute_timeframe_returns(data["trough trend"],0,1000)  

                                trade_size = normalised_difference(tp,sl)
                                
                                log_trade(data)
                                                    
            
                                signal_delay.append(len(data) - last_vector_peak)
                                shorts.append(last_peak_idx_m5)
                                
                                close_shorts.append(data["close"].iloc[last_peak_idx_m5])
                                
                                print(f"Short at idx {df_idx}")
                                plot_module()
                    
                    
            
    if macro_slope_m5 > 0:
        
        if  lower_band[last_vector_trough] <= data["low"].iloc[last_vector_trough] <= oversold[last_vector_trough]:

            if longs == []:

                compute_fib(data)
                # print(f"Entry lvl is {entry_lvl}")
                compute_sl(data)
                compute_tp(data)
                tp = conservative_tp()

                ### Establish risk reward, regardless of hit entry or not, assume market order

                rr = risk_reward(sl, data["close"].iloc[-1],tp)
                if rr > 0.8:
                    
                    if rr < 1:
                        loss_rr = 2-rr
                    else:
                        loss_rr = 1
                     
                    df_idx = idx + last_vector_trough - 2000
    
                    if data["close"].iloc[-1] > sl:
                        
    
                        if df_idx < bars - 2000:
                            
                            
                            print(f"RR = {rr}") if rr is not None else print("RR failed to be computed")
                            print(f"Loss RR = {-loss_rr}") if loss_rr is not None else print("Loss RR failed to be computed")

                            print(f"sl == {sl}") if entry_lvl != None else print("None")
                            print(f"entry = {entry_lvl}") if entry_lvl != None else print("None")
                            print(f"tp == {tp}") if tp != None else print("None")                     
                            
                            hit_tp = np.argwhere(df["close"].iloc[df_idx:].values > tp)
                            if len(hit_tp) > 0:
                                tp_idx = hit_tp[0][0]
                            else:
                                tp_idx = 9999

                            hit_sl = np.argwhere(df["close"].iloc[df_idx:].values < sl)
                            if len(hit_sl) > 0:
                                sl_idx = hit_sl[0][0]
                            else:
                                sl_idx = 9999

                            if sl_idx  != None and tp_idx != None:
                                print(f"sl_idx = {sl_idx}, tp_idx = {tp_idx}")
                                
                                if min([tp_idx, sl_idx]) == sl_idx:
                                    long_sl_hit.append(sl_idx)
                                    total_long_sl.append(sl_idx)
                                    cum.append(loss_rr*-1)
                                    total_cum.append(loss_rr*-1)
                                    
                                    print("SL Hit!")
                                    
                                    momentum_idx = idx + sl_idx + 1
                                    opposite_trend = -macro_slope_m5
                                    confirmation = "momentum_trade.json"
                                    with open(confirmation, "r") as f:
                                        json_data = json.load(f)
                                        # if len(json_data) == 0:
                                            
                                        upload = {"Momentum Trade" : True,
                                                  "Trade ID" : np.random.randint(1000000,9999999),
                                                 "Momentum Start Index" : momentum_idx,
                                                  "Hit SL Index" : sl_idx + idx,
                                                 "trend" : opposite_trend}
                                        json_data.append(upload)
                                        print("Momentum Trade Details Saved.")
                                        with open(confirmation, "w") as f:
                                            
                                            json.dump(json_data, f)
                                        
                                elif min([tp_idx, sl_idx]) == tp_idx:
                                    long_tp_hit.append(tp_idx)
                                    total_long_tp.append(tp_idx)
                                    total_cum.append(rr*1)
                                    cum.append(rr*1)
                                    print("TP Hit!")
                                
                            # if data["close"].iloc[-1] == entry_lvl:
                            #     entry_hit.append(1)
                            # if data["close"].iloc[-1] < entry_lvl:
                            #     for index, boolean in data["close"].iloc[df_idx:] > entry_lvl:
                            #         if boolean == True:
                            #             if index < 2000:
                                            
                            #                 if min([tp_idx, sl_idx]) == tp_idx:
                            #                     entry_idx = index
                            #                     if entry_idx < tp_idx:
                            #                         entry_hit.append(entry_idx)
                            #                         break
                                                    
                            #                 elif min([tp_idx, sl_idx]) == sl_idx:
                            #                     entry_idx = index
                            #                     entry_idx = index
                            #                     if entry_idx < sl_idx:
                            #                         entry_hit.append(entry_idx)
                            #                         break                                 

                            ############## Metrics ###########
                            
                            m5_ema_difference = normalised_difference(data["trend"].iloc[last_vector_trough],data["moving average"].iloc[last_vector_trough])
                            m5_peak_difference = normalised_difference(data["peak trend"].iloc[last_vector_trough], data["close"].iloc[last_vector_trough]) 
                            m5_trough_difference = normalised_difference(data["trough trend"].iloc[last_vector_trough], data["close"].iloc[last_vector_trough]) 

                            m5_log_returns = compute_timeframe_returns(data["trend"],0,1000)
                            m5_peak_returns = compute_timeframe_returns(data["peak trend"],0,1000)
                            m5_trough_returns = compute_timeframe_returns(data["trough trend"],0,1000)    

                            trade_size = normalised_difference(tp,sl)
                            
                            log_trade(data)
                            
                            signal_delay.append(len(data) - last_vector_trough)
                            longs.append(last_vector_trough)
                            close_longs.append(data["close"].iloc[last_trough_idx_m5])
                            print(f"Long at idx {df_idx}")
                            plot_module()
                
            if len(longs) > 0:
                latest_long_idx = longs[-1]
                
                if data["close"].iloc[last_trough_idx_m5] != close_longs[-1]: #data["close"].iloc[close_longs[-1]]:

                    compute_fib(data)
                    # print(f"Entry lvl is {entry_lvl}")
                    compute_sl(data)
                    compute_tp(data) 
                    tp = conservative_tp()


                    

                    ### Establish risk reward, regardless of hit entry or not, assume market order

                    rr = risk_reward(sl, data["close"].iloc[-1],tp)

                    if rr > 0.8:
                        

                        if rr < 1:
                            loss_rr = 2-rr
                        else:
                            loss_rr = 1
                        
                        if data["close"].iloc[-1] > sl:
    
                        
                            df_idx = idx + last_vector_trough - 2000
                            
                            if df_idx < bars - 2000:

                                print(f"RR = {rr}") if rr is not None else print("RR failed to be computed")
                                print(f"Loss RR = {-loss_rr}") if loss_rr is not None else print("Loss RR failed to be computed")
    
                                print(f"sl == {sl}") if entry_lvl != None else print("None")
                                print(f"entry = {entry_lvl}") if entry_lvl != None else print("None")
                                print(f"tp == {tp}") if tp != None else print("None")
                                
                                hit_tp = np.argwhere(df["close"].iloc[df_idx:].values > tp)
                                if len(hit_tp) > 0:
                                    tp_idx = hit_tp[0][0]
                                else:
                                    tp_idx = 9999

                                hit_sl = np.argwhere(df["close"].iloc[df_idx:].values < sl)
                                if len(hit_sl) > 0:
                                    sl_idx = hit_sl[0][0]
                                else:
                                    sl_idx = 9999

                                        
                                if sl_idx  != None and tp_idx != None:
                                    print(f"sl_idx = {sl_idx}, tp_idx = {tp_idx}")
                                    
                                    if min([tp_idx, sl_idx]) == sl_idx:
                                        long_sl_hit.append(sl_idx)
                                        total_long_sl.append(sl_idx)
                                        cum.append(loss_rr*-1)
                                        total_cum.append(loss_rr*-1)
                                        
                                        print("SL Hit!")
                                                                                    
                                        momentum_idx = idx + sl_idx + 1
                                        opposite_trend = -macro_slope_m5
                                        confirmation = "momentum_trade.json"
                                        with open(confirmation, "r") as f:
                                            
                                            json_data = json.load(f)
                                            # if len(json_data) == 0:
                                                
                                            upload = {"Momentum Trade" : True,
                                                      "Trade ID" : np.random.randint(1000000,9999999),
                                                     "Momentum Start Index" : momentum_idx,
                                                      "Hit SL Index" : sl_idx + idx,
                                                     "trend" : opposite_trend}
                                            json_data.append(upload)
                                            print("Momentum Trade Details Saved.")
                                            with open(confirmation, "w") as f:
                                                
                                                json.dump(json_data, f)
                                        
                                    elif min([tp_idx, sl_idx]) == tp_idx:
                                        long_tp_hit.append(tp_idx)
                                        total_long_tp.append(tp_idx)
                                        total_cum.append(rr*1)
                                        cum.append(rr*1)
                                        print("TP Hit!")
                                    
                                # if data["close"].iloc[-1] == entry_lvl:
                                #     entry_hit.append(1)
                                    
                                # if data["close"].iloc[-1] > entry_lvl:
                                #     for index, boolean in data["close"].iloc[df_idx:] < entry_lvl:
                                #         if index < 2000:
                                            
                                #             if boolean == True:
                                #                 if min([tp_idx, sl_idx]) == tp_idx:
                                #                     entry_idx = index
                                #                     if entry_idx < tp_idx:
                                #                         entry_hit.append(entry_idx)
                                #                         break
                                                        
                                #                 elif min([tp_idx, sl_idx]) == sl_idx:
                                #                     entry_idx = index
                                #                     entry_idx = index
                                #                     if entry_idx < sl_idx:
                                #                         entry_hit.append(entry_idx)
                                #                         break    

                                ############## Metrics ###########
                                
                                m5_ema_difference = normalised_difference(data["trend"].iloc[last_vector_trough],data["moving average"].iloc[last_vector_trough])
                                m5_peak_difference = normalised_difference(data["peak trend"].iloc[last_vector_trough], data["close"].iloc[last_vector_trough]) 
                                m5_trough_difference = normalised_difference(data["trough trend"].iloc[last_vector_trough], data["close"].iloc[last_vector_trough]) 

                                m5_log_returns = compute_timeframe_returns(data["trend"],0,1000)
                                m5_peak_returns = compute_timeframe_returns(data["peak trend"],0,1000)
                                m5_trough_returns = compute_timeframe_returns(data["trough trend"],0,1000)   

                                trade_size = normalised_difference(tp,sl)
                                
                                log_trade(data)
                                                
                                signal_delay.append(len(data) - last_vector_trough)
                                longs.append(last_vector_trough)
                                close_longs.append(data["close"].iloc[last_trough_idx_m5])
                                print(f"Long at idx {df_idx}")
                                plot_module()
    
    if idx in progress:
        print(F"###################### PROGRESS {100*idx/bars}% TO COMPLETION ######################")
    #all_slopes.append(macro_slope_m5)
        #print(f"Slope from {idx - 2000} to {idx} computed successfully") if macro_slope_m5 != None else None
    
length = [len(short_tp_hit), len(short_sl_hit)]
labels = ["TP Hit", "SL Hit"]

fig = plt.figure(figsize = (12,12))
from matplotlib import gridspec
gs = gridspec.GridSpec(2,2)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])

ax1.set_title(f"Short Trade Summary for {symbol}")

length = [len(short_tp_hit), len(short_sl_hit)]
labels = ["TP Hit", "SL Hit"]
bars = ax1.bar(labels, length, color = ["green","red"], alpha = 0.6)
ax1.bar_label(bars, padding = 1)

ax2.set_title(f"Long Trade Summary for {symbol}")
length = [len(long_tp_hit), len(long_sl_hit)]
labels = ["TP Hit", "SL Hit"]
bars = ax2.bar(labels, length, color = ["green","red"], alpha = 0.6)
ax2.bar_label(bars, padding = 1)

ax3.set_title(f"Total Trade Summary for {symbol}")
total_trades = len(shorts) + len(longs)
total_tp_hit = len(short_tp_hit) + len(long_tp_hit)
total_sl_hit = len(short_sl_hit) + len(long_sl_hit)
length = [round(100*total_tp_hit/total_trades,2), round(100*total_sl_hit/ total_trades,2)]
labels = ["TP Hit %", "SL Hit %"]
bars = ax3.bar(labels, length, color = ["blue","red"], alpha = 0.6)
ax3.bar_label(bars, padding = 1)

profit = np.array(cum).reshape(-1,1)
profit = profit * 500
profit = profit.cumsum()
starting_balance = 200000
equity = starting_balance + profit
numbers = np.arange(0, len(equity))
ax4.set_title(f"Running equity with start.equity of {starting_balance}")
ax4.plot(numbers, equity, color= "red", alpha = 0.8)
#plt.text(f"No. of trades taken = {len(shorts) + len(longs)}")
ax4.scatter(numbers[-1], equity[-1], color="black", marker="*", label = f"Last equity value = {equity[-1]}")
ax4.grid()
ax4.legend()

plt.savefig(f"MV Strategy Summary for {symbol}.png", bbox_inches="tight")
    
    


# ### Backtest Results

# #### Summary

# In[ ]:


sd_delay = []
for i in signal_delay:
    sd_delay.append(int(i))


# In[ ]:


#files = [long_sl_hit, long_tp_hit, short_sl_hit, short_tp_hit]
import json
data = {"Long" : {"long_sl_hit" : long_sl_hit,
                "long_tp_hit" : long_tp_hit },
       "Short" : {"short_sl_hit" : short_sl_hit,
                 "short_tp_hit" : short_tp_hit},
        "Signal Delay" : sd_delay
       }

with open(f"{symbol} Backtest Metrics.json", "w") as f:
    json.dump(data,f)
    


# In[ ]:


with open(f"{symbol} Backtest Metrics.json", "r") as f:
    dfee = json.load(f)
    


# In[ ]:


length = [len(short_tp_hit), len(short_sl_hit)]
labels = ["TP Hit", "SL Hit"]

fig = plt.figure(figsize = (12,12))
from matplotlib import gridspec
gs = gridspec.GridSpec(2,2)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])

ax1.set_title(f"Short Trade Summary for {symbol}")

length = [len(short_tp_hit), len(short_sl_hit)]
labels = ["TP Hit", "SL Hit"]
bars = ax1.bar(labels, length, color = ["green","red"], alpha = 0.6)
ax1.bar_label(bars, padding = 1)

ax2.set_title(f"Long Trade Summary for {symbol}")
length = [len(long_tp_hit), len(long_sl_hit)]
labels = ["TP Hit", "SL Hit"]
bars = ax2.bar(labels, length, color = ["green","red"], alpha = 0.6)
ax2.bar_label(bars, padding = 1)

ax3.set_title(f"Long Trade Summary for {symbol}")
length = [(len(short_tp_hit) + len(long_tp_hit))/(len(longs) + len(shorts)), (len(short_sl_hit) + len(long_sl_hit))/ len(longs) + len(shorts)]
labels = ["TP Hit", "SL Hit"]
bars = ax3.bar(labels, length, color = ["green","red"], alpha = 0.6)
ax3.bar_label(bars, padding = 1)

profit = np.array(cum).reshape(-1,1)
profit = profit * 500
profit = profit.cumsum()
starting_balance = 200000
equity = starting_balance + profit
numbers = np.arange(0, len(equity))
ax4.set_title(f"Running equity with start.equity of {starting_balance}")
ax4.plot(numbers, equity, color= "red", alpha = 0.8)
#plt.text(f"No. of trades taken = {len(shorts) + len(longs)}")
ax4.scatter(numbers[-1], equity[-1], color="black", marker="*", label = f"Last equity value = {equity[-1]}")
ax4.grid()
ax4.legend()

plt.savefig(f"MV Strategy Summary for {symbol}.png", bbox_inches="tight")


# ### Total Summary Template

# In[ ]:


total_long = len(total_long_tp) +  len(total_long_sl)
total_short = len(total_short_sl) +  len(total_short_tp)
total_trades = total_long + total_short
total_tp = len(total_long_tp) + len(total_short_tp)
total_sl = len(total_long_sl) + len(total_short_sl)

total_win_pc = round(100*total_tp/total_trades,2)
total_loss_pc = round(100*total_sl/total_trades,2)

from matplotlib import gridspec

gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize = (16,12))

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])

ax1.set_title(f"MV Logic Summary - Across 9 Instruments {9*90000} Bars")

labels = ["Win %", "Loss %"]
length = [total_win_pc, total_loss_pc]
bars = ax1.bar(labels, length, color = ["blue", "red"], alpha = 0.6)
ax1.bar_label(bars, padding = 1)
ax1.text(0.7,60, f"Entry Rate = 100%", bbox=dict(facecolor='white', alpha=0.5))
#ax1.grid()

sd = signal_delay
sd = np.array(sd)
sd_realized = sd*5

## Plot normal distribution graph

mu, sigma = sd_realized.mean(), sd_realized.std()
x = np.linspace(sd_realized.min(), sd_realized.max(), 100 )
mu, loc, sigma = skewnorm.fit(sd_realized)
skew_pdf = skewnorm.pdf(x, mu, loc, sigma)

ax2.set_title("Skewnormal Distrubution of Signal Delay (mins)")
ax2.plot(x, skew_pdf, color = "red", alpha = 0.5)
ax2.hist(sd_realized, bins = 200, density=True)
ax2.axvline(loc, label=f"Avg signal delay = {round(loc,2)} mins \n\nTotal Signals = {total_trades}", color = "red")
ax2.set_xlabel("Time in Minutes")
#ax2.text(1000, 0.09, f"Total Signals = {total_trades}", bbox=dict(facecolor='white', alpha=0.5))
ax2.grid()
ax2.legend()


cumul = total_cum[0:len(total_cum) - 2]
cumul = np.array(cumul).reshape(-1,4)
profit = cumul.sum(axis = 1)
profit = profit * 500
profit = profit.cumsum()
starting_balance = 200000
equity = starting_balance + profit
numbers = np.arange(0, len(equity))

ax3.set_title(f"Running equity with start.equity of {starting_balance}")
ax3.plot(numbers, equity, color= "red", alpha = 0.8)
#plt.text(f"No. of trades taken = {len(shorts) + len(longs)}")
ax3.scatter(numbers[-1], equity[-1], color="black", marker="*", label = f"Last equity value = {equity[-1]}")
ax3.grid()
ax3.set_xlabel("Days")
ax3.legend()

### ax4

cumul = total_cum[0:len(total_cum) - 2]
cumul = np.array(cumul).reshape(-1,4)
profit = cumul.sum(axis = 1)
runs = []
start_balance = 200000
risk = 500
while len(runs) != 1000:
    cumul = total_cum[0:len(total_cum) - 2]
    cumul = np.random.permutation(cumul)
    cumul = np.array(cumul).reshape(-1,4)
    profit = cumul.sum(axis = 1) * 500
    equity_run = start_balance + profit.cumsum()
    runs.append(equity_run)
print(f"Monte Carlo Success, len(runs) = {len(runs)}")

cumul = total_cum[0:len(total_cum) - 2]
cumul = np.array(cumul).reshape(-1,4)
profit = cumul.sum(axis = 1)
runs = []
start_balance = 200000
risk = 500
while len(runs) != 1000:
    cumul = total_cum[0:len(total_cum) - 2]
    cumul = np.random.permutation(cumul)
    cumul = np.array(cumul).reshape(-1,4)
    profit = cumul.sum(axis = 1) * 500
    equity_run = start_balance + profit.cumsum()
    runs.append(equity_run)
print(f"Monte Carlo Success, len(runs) = {len(runs)}")

drawdown_list = []
for equity_run in runs:
    numbers = np.arange(len(equity_run))
    drawdown = -100*(start_balance - equity_run.min())/start_balance
    if drawdown < 0 :
        drawdown_list.append(drawdown)
print("Drawdown Successfully Calculated")
dd = np.array(drawdown_list)
j = np.linspace(dd.min(), dd.max(), 500)

from scipy.stats import jf_skew_t
mu, loc, sigma, k = jf_skew_t.fit(dd)
gamma_pdf = jf_skew_t.pdf(j, mu, loc, sigma,k)
ax4.set_title("Monte Carlo Sim 1000 Trials - Skewed-T Dist. of VaR% with MV")
ax4.plot(j, gamma_pdf, color = "red", alpha = 1)
ax4.hist(dd, bins = 500, density = True, color="blue", alpha = 1 )
fifty_percentile = round(jf_skew_t.ppf(0.8, mu, loc, sigma, k),4)

ax4.axvline(fifty_percentile, label=f"Avg DD/Risk = {fifty_percentile}%", color = "red")
ninety_percentile = round(jf_skew_t.ppf(0.1, mu, loc, sigma, k),4)
print(ninety_percentile)
ax4.axvline(ninety_percentile, label=f"90th Percentile = {ninety_percentile}%", color = "red", linestyle = "--", alpha = 0.8)

ax4.set_xlabel("DD Risk %")
ax4.grid()

ax4.legend()  
plt.savefig("MV Logic Summary Across All Instruments.png", bbox_inches = "tight")


# In[ ]:


jf_skew_t.fit(dd)


# ### Time to TP/SL Plots

# In[ ]:


fig = plt.figure(figsize = (20,12))
gs = gridspec.GridSpec(2,2)
ax5 = fig.add_subplot(gs[0,0])
ax6 = fig.add_subplot(gs[0,1])
ax7 = fig.add_subplot(gs[1,0])
ax8 = fig.add_subplot(gs[1,1])



weibull_plot(total_long_tp)
ax5.set_title("Longs - Time to TP")
ax5.plot(x, skew_pdf, color = "red", alpha = 1)
ax5.hist(time_to_tp_long, bins = 200, density= True, color = "blue", alpha = 0.8)
ax5.axvline(round(loc,2), label = f"Avg Time = {5*round(loc,2)} minutes", color = "red", alpha = 0.6)
ax5.axvline(round(ninety_percentile,2), label = f"90th Percentile = {5*round(ninety_percentile,2)} minutes", color = "red", alpha = 0.6, linestyle = "--")
ax5.set_xlabel("Bars")
ax5.legend()
ax5.grid()


weibull_plot(total_long_sl)
ax6.set_title("Longs - Time to SL")
ax6.plot(x, skew_pdf, color = "red", alpha = 1)
ax6.hist(total_long_sl, bins = 200, density= True, color = "blue", alpha = 0.8)
ax6.axvline(round(loc,2), label = f"Avg Time = {5*round(loc,2)} minutes", color = "red", alpha = 0.6)
ax6.axvline(round(ninety_percentile,2), label = f"90th Percentile = {5*round(ninety_percentile,2)} minutes", color = "red", alpha = 0.6, linestyle = "--")
ax6.set_xlabel("Bars")
ax6.legend()
ax6.grid()

weibull_plot(total_short_tp)
ax7.set_title("Short - Time to TP")
ax7.plot(x, skew_pdf, color = "red", alpha = 1)
ax7.hist(total_short_tp, bins = 200, density= True, color = "blue", alpha = 0.8)
ax7.axvline(round(loc,2), label = f"Avg Time = {5*round(loc,2)} minutes", color = "red", alpha = 0.6)
ax7.axvline(round(ninety_percentile,2), label = f"90th Percentile = {5*round(ninety_percentile,2)} minutes", color = "red", alpha = 0.6, linestyle = "--")
ax7.set_xlabel("Bars")
ax7.legend()
ax7.grid()

weibull_plot(total_short_sl)
ax8.set_title("Short - Time to SL")
ax8.plot(x, skew_pdf, color = "red", alpha = 1)
ax8.hist(total_short_sl, bins = 200, density= True, color = "blue", alpha = 0.8)
ax8.axvline(round(loc,2), label = f"Avg Time = {5*round(loc,2)} minutes", color = "red", alpha = 0.6)
ax8.axvline(round(ninety_percentile,2), label = f"90th Percentile = {5*round(ninety_percentile,2)} minutes", color = "red", alpha = 0.6, linestyle = "--")
ax8.set_xlabel("Bars")
ax8.legend()
ax8.grid()

plt.savefig("Time to TP-SL.png", bbox_inches = "tight")


# ### Signal Delay Analysis

# In[ ]:


### Plotting signal delay

sd = signal_delay
sd = np.array(sd)
sd_realized = sd*5

## Plot normal distribution graph

mu, sigma = sd_realized.mean(), sd_realized.std()
x = np.linspace(sd_realized.min(), sd_realized.max(), 50 )
mu, loc, sigma = skewnorm.fit(sd_realized)
skew_pdf = skewnorm.pdf(x, mu, loc, sigma)
plt.title("Normal Distrubution of Signal Delay (mins)")
plt.plot(x, skew_pdf, color = "red", alpha = 0.5)
plt.hist(sd_realized, bins = 50, density=True)
plt.axvline(loc, label=f"Avg signal delay = {round(loc,2)} mins", color = "red")
plt.xlabel("Time in Minutes")
plt.grid()
plt.legend()


# ### Running Equity Based on Results - Vectorised (4 trades per day)

# #### Monte-Carlo Sim - 1000 Trials (VaR)

# In[ ]:


cumul = total_cum[0:len(total_cum) - 2]
cumul = np.array(cumul).reshape(-1,4)
profit = cumul.sum(axis = 1)
runs = []
start_balance = 200000
risk = 500
while len(runs) != 1000:
    cumul = total_cum[0:len(total_cum) - 2]
    cumul = np.random.permutation(cumul)
    cumul = np.array(cumul).reshape(-1,4)
    profit = cumul.sum(axis = 1) * 500
    equity_run = start_balance + profit.cumsum()
    runs.append(equity_run)
print(f"Monte Carlo Success, len(runs) = {len(runs)}")

drawdown_list = []
for equity_run in runs:
    numbers = np.arange(len(equity_run))
    drawdown = -100*(start_balance - equity_run.min())/start_balance
    if drawdown < 0 :
        drawdown_list.append(drawdown)
print("Drawdown Successfully Calculated")
dd = np.array(drawdown_list)
j = np.linspace(dd.min(), dd.max(), 500)

mu, loc, sigma = skewnorm.fit(dd)
gamma_pdf = skewnorm.pdf(j, mu, loc, sigma)
plt.title("Skewnormal Dist. of Risk% with MV")
plt.plot(j, gamma_pdf, color = "red", alpha = 0.8)
plt.hist(dd, bins = 500, density = True, color="blue", alpha = 1 )
plt.axvline(round(loc,4), label=f"Avg DD/Risk = {round(loc,4)}%", color = "red")
ninety_percentile = round(skewnorm.ppf(0.1, mu, loc, sigma),4)
print(ninety_percentile)
plt.axvline(ninety_percentile, label=f"90th Percentile = {ninety_percentile}%", color = "red", linestyle = "--", alpha = 0.8)

plt.xlabel("DD Risk %")
plt.grid()

plt.legend()


# #### Cumulative RR of Backtest

# In[ ]:


cumul = total_cum[0:len(total_cum) - 2]
cumul = np.array(cumul).reshape(-1,4)
profit = cumul.sum(axis = 1)
profit = profit * 500
profit = profit.cumsum()
starting_balance = 200000
equity = starting_balance + profit
numbers = np.arange(0, len(equity))
plt.title(f"Running equity with start.equity of {starting_balance}")
plt.plot(numbers, equity, color= "red", alpha = 0.8)
#plt.text(f"No. of trades taken = {len(shorts) + len(longs)}")
plt.scatter(numbers[-1], equity[-1], color="black", marker="*", label = f"Last equity value = {equity[-1]}")
plt.grid()
plt.xlabel("Days")
plt.legend()


# ### Testing
# 

# In[ ]:




