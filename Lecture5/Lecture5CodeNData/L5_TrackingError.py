# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 07:57:40 2018

This code performs the following tasks:
    1. load or download fund(fidelity contra fund) and benchmark(SP500) from Yahoo 
    2. Calculate Tracking Error fund vs. benchmark
    3. Plot the TE together with cumulative returns of fund/benchmark
    
@author: Steve Xia 
"""

#from pandas_datareader.famafrench import get_available_datasets
#dsa = get_available_datasets()
import pandas as pd  
#pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() 
import datetime 
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import numpy as np

# function to get the price data from yahoo finance 
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

#---------------------------------------------------------
flag_downloadData = False # if True donwload data from the web. Otherwise read in from excel
if flag_downloadData:
    # define the time period 
    start_dt = datetime.datetime(2000, 1, 1)
    end_dt = datetime.datetime(2017, 12, 31)

    # for multiple stock cases 
    tickers = ['FCNTX', '^GSPC']
    all_data = getDataBatch(tickers, start_dt, end_dt)
    # Isolate the `Adj Close` values and transform the DataFrame
    Px_daily_close = all_data.reset_index().pivot('Date', 'Ticker', 'Adj Close')
    # Calculate the daily percentage change for `daily_close_px`
    Ret_All = Px_daily_close.pct_change().dropna()
    Ret_All.columns = ['Contra', 'SP500']
    # Save data to excel
    writer = pd.ExcelWriter('InputForTrackingError.xlsx')
    Ret_All.to_excel(writer,'Return')
    Px_daily_close.to_excel(writer,'Price')
    writer.save()
else:
    # Changed from sheetname to sheet_name for nwer version to rid of the warnings
    Ret_All = pd.read_excel('InputForTrackingError.xlsx', sheet_name='Return',index_col=0)
    Px_daily_close = pd.read_excel('InputForTrackingError.xlsx', sheet_name='Price',index_col=0)

#--------------------------------------------------
#%%
# calculate tracking error
#
roll_window = 250# 250 rolling days
Ret_relative = Ret_All['Contra'] - Ret_All['SP500']
TE = Ret_relative.rolling(roll_window).std()*np.sqrt(roll_window)
# only take the nonan portion
TE1 = TE[~np.isnan(TE)]
# plot TE curve
figure_count = 1
plt.figure(figure_count)

fig, ax1 = plt.subplots(figsize=(10,8))
#dates4plot = mdates.datestr2num(Ret_relative.index)
dates4plot = Ret_relative.index
Px_Plot = Px_daily_close.iloc[1:,:]
Ret_All_Cumulative = (1 + Ret_All).cumprod() - 1


colors = ['red','black']
ax1.set_xlabel('date', fontsize=18)
ax1.set_ylabel('Fund/Bench Cumulative Return', color='red', fontsize=18)
ax1.plot(dates4plot, Ret_All_Cumulative['Contra'], color='red',label='Contra')
#add_rec_bars(ax1, dates=None)
# # 1pt line, 2pt break dash
ax1.plot(dates4plot, Ret_All_Cumulative['SP500'], dashes=[1, 2], color='red',label='SP500')
ax1.tick_params(axis='both', which='major', labelsize=16, rotation=0)
ax1.tick_params(axis='y', labelcolor='red')
plt.xticks(fontsize=16, rotation=0)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color1 = 'tab:blue'
ax2.set_ylabel('Tracking Error', color=color1, fontsize=18)  # we already handled the x-label with ax1
ax2.plot(dates4plot, TE, color=color1,label='Contra TE')
ax2.tick_params(axis='y', labelcolor=color1)
ax2.tick_params(axis='both', which='major', labelsize=16, rotation=0)


#ax2.legend()
fig.legend(loc=9, prop={'size': 14})

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

