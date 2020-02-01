import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()
import datetime

import warnings

warnings.filterwarnings("ignore")

# load module with utility functions, including optimization
import risk_opt_2Student as riskopt


def tracking_error(wts_active, cov):
    TE = np.sqrt(np.transpose(wts_active) @ cov @ wts_active)
    return TE

# function to get the price data from yahoo finance
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  #return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])) # old
  # new - force it not to sort by the ticker alphabetically
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date'],sort=False))


def getReturns(tickers, start_dt, end_dt, freq='monthly'):
    px_data = getDataBatch(tickers, start_dt, end_dt)
    # Isolate the `Adj Close` values and transform the DataFrame
    px = px_data[['Adj Close']].reset_index().pivot(index='Date',
                                                    columns='Ticker', values='Adj Close')
    if (freq == 'monthly'):
        px = px.resample('M').last()

    # Calculate the daily/monthly percentage change
    ret = px.pct_change().dropna()

    ret.columns = tickers
    return (ret)


# %%

if __name__ == "__main__":
    TickerNWeights = pd.read_csv('', sheet_name='DowJones', header=2, index_col=0)
    Ticker_AllStock_DJ = TickerNWeights['Symbol']
    wts_AllStock_DJ = 0.01 * TickerNWeights['Weight']