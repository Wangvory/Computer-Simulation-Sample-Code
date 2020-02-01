# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:29:43 2018

In this code we perform the following tasks
 - demonstrate historical returns have fat tails
 - statistically show financial returns are not normally distributed
 - fit financial data to normal and student t distribution with different degree of freedom
 - demonstrate that returns has no staistically significant autocorrelation
 - demonstrate that risk (return squred) has statisically signficant autocorrelation
 - demonstrate risk has regimes
 
@author: Steve Xia 
"""

import numpy as np 
import scipy.io as spio
import pandas as pd 
import datetime as dt
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm


import warnings
warnings.filterwarnings("ignore")
#--------------------------------------------------------
# 
plt.close("all")# close all opened figures

# load in the input file, which contains US equity return data
dow = pd.DataFrame.from_csv('dow.csv')

dow['ret'] = dow['price'].pct_change()

Ret_Dow = dow['ret'].dropna()

T = Ret_Dow.shape[0]


plt.figure()
bins =200
# Hit Histogram - normal
dist = stats.norm
mu_norm, sigma_norm = dist.fit(data=Ret_Dow)
y, x = np.histogram(Ret_Dow, bins=bins, density=True)
a = np.roll(x, -1)
#a1 = x + a 
#a2 = a1[:,-1]
x = (x + np.roll(x, -1))[:-1] / 2.0
pdf = pd.Series(dist.pdf(x, loc=mu_norm, scale=sigma_norm), x)
pdf.plot(lw=2, label='normal', legend=True,color='red')
Ret_Dow.plot(kind='hist', bins=bins, normed=True, 
                  alpha=0.5, color='blue')
plt.xlabel('Daily Return',fontSize=12,fontWeight='bold',Color='k')
plt.ylabel('Frequency',fontSize=12,fontWeight='bold',Color='k')

# QQ plot vs. Normal Distribution
plt.figure()
stats.probplot(Ret_Dow, dist="norm", sparams=(mu_norm, sigma_norm), plot=plt)
plt.title('QQ plot vs. Normal Distribution')
plt.xlabel('Quantiles of Normal Distribution',fontSize=12,fontWeight='bold',Color='k')
plt.ylabel('Quantiles of Input Sample',fontSize=12,fontWeight='bold',Color='k')

# QQ plot decomposition
Ret_Dow_sorted = np.sort(Ret_Dow)
Ret_Dow_min = Ret_Dow_sorted[0]
cdf_Ret_Dow_min = 1/T
normZscore = norm.ppf(cdf_Ret_Dow_min,loc=mu_norm, scale=sigma_norm)
print('x,y value of the first data point on the QQ plot is',normZscore, Ret_Dow_min)
Ret_Dow_2ndmin = Ret_Dow_sorted[1]
cdf_Ret_Dow_2ndmin = 2/T
normZscore_2ndmin = norm.ppf(cdf_Ret_Dow_2ndmin,loc=mu_norm, scale=sigma_norm)
print('x,y value of the second data point on the QQ plot is',normZscore_2ndmin, Ret_Dow_2ndmin)


# Hit Histogram - student t
dist = stats.t
deg_f, mu_t, sigma_t = dist.fit(data=Ret_Dow)
y, x = np.histogram(Ret_Dow, bins=bins, density=True)
x = (x + np.roll(x, -1))[:-1] / 2.0
pdf = pd.Series(dist.pdf(x, loc=mu_t, scale=sigma_t, df=deg_f), x)
plt.figure()
pdf.plot(lw=2, label='student-t', legend=True,color='red')
Ret_Dow.plot(kind='hist', bins=bins, normed=True, 
                  alpha=0.5, color='blue')
plt.xlabel('Daily Return',fontSize=12,fontWeight='bold',Color='k')
plt.ylabel('Frequency',fontSize=12,fontWeight='bold',Color='k')

# QQ plot vs. 4 DOF Student t Distribution
ax1 = plt.figure()
stats.probplot(Ret_Dow, dist=stats.t, sparams=(4, mu_t, sigma_t), plot=plt)
plt.title('QQ plot vs. 4 DOF Student t Distribution')
plt.xlabel('Quantiles of 4 DOF Student t Distribution',fontSize=12,fontWeight='bold',Color='k')
plt.ylabel('Quantiles of Input Sample',fontSize=12,fontWeight='bold',Color='k')

# QQ plot vs. 3 DOF Student t Distribution
plt.figure()
stats.probplot(Ret_Dow, dist=stats.t, sparams=(3, mu_t, sigma_t), plot=plt)
plt.title('QQ plot vs. 3 DOF Student t Distribution')
plt.xlabel('Quantiles of 3 DOF Student t Distribution',fontSize=12,fontWeight='bold',Color='k')
plt.ylabel('Quantiles of Input Sample',fontSize=12,fontWeight='bold',Color='k')

# QQ plot vs. fitted DOF Student t Distribution
plt.figure()
stats.probplot(Ret_Dow, dist=stats.t, sparams=(deg_f, mu_t, sigma_t), plot=plt)
plt.title(('QQ plot vs.fitted Student t Distributionf with ' + str(deg_f) + ' DoF'))
plt.xlabel('Quantiles of fitted Student t Distribution',fontSize=12,fontWeight='bold',Color='k')
plt.ylabel('Quantiles of Input Sample',fontSize=12,fontWeight='bold',Color='k')

# Compare proability of tail negative returns for DOW with those from a Normal
# distriubtion with the same mean and std    
TailThresh = [-0.01, -0.02, -0.03, -0.05]
mu_Dow = np.mean(Ret_Dow)
sigma_Dow = np.std(Ret_Dow)
Prob_Tail_Dow = np.zeros(4)
Prob_Tail_Normal = np.zeros(4)
for i in range(0, 4):
    a1 = Ret_Dow<TailThresh[i]
    a2 = np.mean(a1)
    Prob_Tail_Dow[i] = np.mean(Ret_Dow<TailThresh[i])
    Prob_Tail_Normal[i] = stats.norm.cdf(TailThresh[i],mu_Dow,sigma_Dow)

#
# -- JB test
h, p = stats.jarque_bera(Ret_Dow)

lags = 100
# Calc Autocorrelation np.corrcoef return matrix 
ACorr = np.full((lags+1, 1), np.nan)
ACorr[0, :] = stats.pearsonr(Ret_Dow,Ret_Dow)[0]
for i in range(1, lags+1) :
    ACorr[i,:] = stats.pearsonr(Ret_Dow[i:],Ret_Dow[0:(-i)])[0]

# pandas way to calculate lag
ACorr2 = [Ret_Dow.autocorr(lag) for lag in range(1, lags+1)]

# Use the Ljung-Box (LB) test to verify whether the autocorrelation is statistically significant
import statsmodels.stats.diagnostic as tsd
import statsmodels.graphics.tsaplots as tgs

# high tstat or low p value suggest we should reject the null hypothesis, which is the data has NO autocorrelation
t0, p0 = tsd.acorr_ljungbox(Ret_Dow, lags=10) # this produces the tstat and pvalue of the first 10 lags

plt.figure()
confidenceInterval = 0.05
tgs.plot_pacf(Ret_Dow, alpha=confidenceInterval, lags=100, zero=False)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Autocorrelation of returns', fontsize=12)
plt.title('pacf autocorr with 95% confidence interval')

plt.figure()
#plt.plot(range(0, lags+1), ACorr,'-r')
plt.plot(range(1, lags+1), ACorr[1:],'-r')
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Autocorrelation of Returns', fontsize=12)

#
# -- Squared Daily Return autocorrelation
#
lags =100 
Ret_Dow_square = np.square(Ret_Dow)
RSqaureCorr = np.full((lags+1, 1), np.nan)
RSqaureCorr[0, :] = stats.pearsonr(Ret_Dow_square,Ret_Dow_square)[0]
for i in range(1, lags+1):
    RSqaureCorr[i,:] = stats.pearsonr(Ret_Dow_square[i:],Ret_Dow_square[0:(-i)])[0]

plt.figure()
plt.plot(range(1, lags+1), RSqaureCorr[1:],'-r')
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Autocorrelation-Squared Return', fontsize=12)

plt.figure()
confidenceInterval = 0.05
tgs.plot_pacf(Ret_Dow_square, alpha=confidenceInterval, lags=100, zero=False)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Autocorrelation-Squared Return', fontsize=12)
plt.title('pacf autocorr with 95% confidence interval')

# -- Squared Daily Return scatter
#
Ret_Dow_Demean = Ret_Dow - np.mean(Ret_Dow)
window = 92
TT = Ret_Dow.shape[0]-window
Std_Roll = np.full((TT, 1), np.nan)
Ret_Roll = np.full((TT, 1), np.nan)
for i in range(0, TT, window):
    Std_Roll[i,:] = np.std(Ret_Dow_Demean[i:i+window])*np.sqrt(250)
    Ret_Roll[i,:] = np.prod(1+Ret_Dow[i:i+window])-1

plt.figure()
plt.scatter(Ret_Roll,Std_Roll)
plt.xlabel('92D cumulative return',fontSize=12,fontWeight='bold',Color='k')
plt.ylabel('Vol. of 92D return',fontSize=12,fontWeight='bold',Color='k')

#-- Dow Daily Return volatility regimes
#
plt.figure()
Ret_Dow.plot()
plt.xlabel('Year',fontSize=12,fontWeight='bold',color='k')
plt.ylabel('Daily Return',fontSize=12,fontWeight='bold',color='k')

plt.figure()
Ret_Dow[-6000:].plot()
plt.xlabel('Year',fontSize=12,fontWeight='bold',color='k')
plt.ylabel('Daily Return',fontSize=12,fontWeight='bold',color='k')

#-----------------------------------------------------------------------
# need install pandas_datareader and fix_yahoo_finance using pip install 
from pandas_datareader import data as pdr
#import fix_yahoo_finance as yf
import yfinance as yf 
yf.pdr_override() # <== that's all it takes :-)
import datetime 

Flag_downloadData = False
if Flag_downloadData:
    stocks = pdr.get_data_yahoo('^GSPC', 
                          start=datetime.datetime(1980, 1, 1), 
                          end=datetime.datetime(2018, 5, 30))
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('Input_S&P.xlsx', engine='xlsxwriter')
    stocks.to_excel(writer, sheet_name='Returns',
             startrow=0, startcol=0, header=True, index=True)
else:
    stocks = pd.read_excel('Input_S&P.xlsx', sheet_name='Returns',
                    header=0, index_col = 0)


#stocks.to_csv('GSPC.csv')
#stocks = pd.DataFrame.from_csv('GSPC.csv')

#key symbols '^GSPC', '^DJI', '^VIX' 'DNKN''AAPL'

# find returns
ret_Stock = stocks['Adj Close'].pct_change().dropna()

# ACF (autocorrelation function) 
# acf and pacf from statsmodels
from statsmodels.tsa.stattools import acf
ac = acf(ret_Stock,nlags=250)
ac2 = acf(np.square(ret_Stock),nlags=200)

#from statsmodels.graphics.tsaplots import plot_acf
#plot_acf(ret_Stock)

# Ljung-Box Test for 20 lags
# statsmodels.stats.diagnostic.acorr_ljungbox
import statsmodels.stats.diagnostic as ssd 
lags = 10;
qstat, pval = ssd.acorr_ljungbox(ret_Stock,lags)
qstat2, pval2 =ssd.acorr_ljungbox(np.square(ret_Stock),lags)

# Engle LM test for heteroskedasticity consistent
# statsmodels.sandbox.stats.diagnostic.acorr_lm het_arch
lmstat = ssd.het_arch(ret_Stock,lags)
lmstat2 = ssd.het_arch(np.square(ret_Stock),lags)
