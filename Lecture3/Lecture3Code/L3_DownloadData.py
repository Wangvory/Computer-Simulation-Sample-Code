# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:15:36 2018

@author: Steve Xia

This code gives examples of how to download financial data from soures such as
 - Yahoo finance
 - St. Louis Fed
 - Ken French Website
"""
import pandas_datareader.fred as fred
import pandas as pd  
#pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
#import fix_yahoo_finance as yf 
import yfinance as yf 
yf.pdr_override() 
import datetime 

import warnings
warnings.filterwarnings("ignore")

# function to get the price data from yahoo finance 
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
# define the time period 
start_dt = datetime.datetime(2000, 1, 1)
end_dt = datetime.datetime(2017, 12, 31)


#----------------------------------------------
# get stock price from Yahoo Finance 
# for one stock/index case, 
# ^GSPC is the ticker for S&P500
SP500 = pdr.get_data_yahoo('^GSPC', start=start_dt, end=end_dt)
# calculate returns
ret_Stock = SP500['Adj Close'].pct_change().dropna()

# for multiple stock cases 
tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG', '^GSPC', 'C', 'GE', 'PG', 'CELG', 'DIOD', 'FCNTX']
stock_data = getDataBatch(tickers, start_dt, end_dt)
#%%
# Isolate the `Adj Close` values and transform the DataFrame
# Note the tickers are sorted after the operation below
daily_close_px = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
# Calculate the daily percentage change for `daily_close_px`
daily_pct_change = daily_close_px.pct_change().dropna()
# Error
#daily_pct_change.columns = ['AAPL', 'MSFT', 'IBM', 'GOOG', 'SP500', 'C', 'GE', 'PG', 'CELG', 'DIOD', 'Contra']
# Fixed
#daily_pct_change.rename({'^GSPC':'SP500'}, axis='columns')
names = daily_pct_change.columns.tolist()
names[names.index('^GSPC')] = 'SP500'
daily_pct_change.columns = names

# 
# Get data from Fred
#
start = datetime.datetime(1990, 1, 1)
# old: end = datetime.datetime(2018, 4, 27)
end = datetime.datetime(2018, 4, 30)

fred_cpi = fred.FredReader(symbols="CPIAUCSL", start=start, end=end, 
                retry_count=3, pause=0.1, timeout=30, session=None, freq=None)
cpidata = fred_cpi.read()
fred_cpi.close()
fred_Baa_Yield = fred.FredReader(symbols="BAA10Y", start=start, end=end, 
                retry_count=3, pause=0.1, timeout=30, session=None, freq=None)
Baa_Yield = fred_Baa_Yield.read()
fred_Baa_Yield.close()
#cpi = web.DataReader("CPIAUCSL", "fred", start, end) # monthly 
#Baa_Yield = web.DataReader("BAA10Y", "fred", start, end)

