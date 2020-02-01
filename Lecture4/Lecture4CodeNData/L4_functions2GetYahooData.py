# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 18:52:20 2019

@author: Steve Xia
"""

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

#pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
#import fix_yahoo_finance as yf
import yfinance as yf 
yf.pdr_override() 
import datetime 

start_dt = datetime.datetime(2008, 3, 19)
end_dt = datetime.datetime(2017, 12, 31)


# function to get the price data from yahoo finance 
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))


# function to get the return data calculated from price data 
# retrived from yahoo finance 
def getReturns(tickers, start_dt, end_dt, freq='monthly'): 
    px_data = getDataBatch(tickers, start_dt, end_dt)
    # Isolate the `Adj Close` values and transform the DataFrame
    px = px_data[['Adj Close']].reset_index().pivot(index='Date', 
                           columns='Ticker', values='Adj Close')
    if (freq=='monthly'):
        px = px.resample('M').last()
        
    # Calculate the daily/monthly percentage change
    ret = px.pct_change().dropna()
    
    ret.columns = tickers
    return(ret)
    

DJData = pdr.get_data_yahoo('^DJI', start=start_dt, end=end_dt)
#
Ticker_List = ['GE','MS']
stock_data = getDataBatch(Ticker_List, start_dt, end_dt)
stock_data_AjdPr = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
#
return_data = getReturns(Ticker_List, start_dt, end_dt)
