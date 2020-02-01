# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:56:22 2018

@author: Steve Xia
# -- Fit GARCH(1,1) Model
#    Forecast Variance using fitted GARCH(1,1) model
#    Uuse forecasted varince to calculate Value at Risk
"""
# go to command prompt from your computer and type in "pip install arch"
# conda install arch -c bashtage
from arch import arch_model
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import statsmodels.graphics.tsaplots as tgs
from collections import OrderedDict

#def ewaold(values, lamda, window):
#    wgts = np.power(lamda, np.arange(window-1))
#    wgts = wgts/wgts.sum()
#    ewas = np.convolve(values, wgts, mode='full')[:len(values)]
#    #ewas[:window] = ewas[window]
#    ewas[:window] = np.nan
#    return ewas

def ewa(values, lamda, window):
    wgts = np.power(lamda, np.arange(window))# Mod, windoew was window-1
    wgts = wgts/wgts.sum()
    ewas = np.convolve(values, wgts, mode='full')[:len(values)]
    #ewas[:window] = ewas[window]
    ewas[:window-1] = np.nan# Mod: window -1 was window
    return ewas
#%%
figure_count = 1

retUS = pd.read_excel('retUS.xlsx', sheet_name='Sheet1',index_col =0)
# Scale return data by a factor of 100. It seems that the optimizer fails when the values are too close to 0
scale = 100
ret1 = scale*retUS['Ret'] #
ret1.index = pd.to_datetime(ret1.index,format='%Y%m%d')
# take only a portion of the data
ret1=ret1.loc['2006-12-31':]
ret1 = ret1 - ret1.mean() #demean the series
n = len(ret1)

#%%
#--------------------------------------------------------
# rolling window size in number of days
win_size = 100 

# calculate realized vol.
k = len(ret1)-win_size+1
std_Realized = np.empty(k)
for i in range(k):
    std_Realized[i] = np.std(ret1[i:(i+win_size-1)])/scale # old window size only 99
    std_Realized[i] = np.std(ret1[i:(i+win_size)])/scale # New window size 100
    
# calculate forward-realized vol.
std_Realized_fwd = np.empty(k)
for i in range(k):
    std_Realized_fwd[i] = np.std(ret1[i+win_size:(i+2*win_size-1)])/scale
    std_Realized_fwd[i] = np.std(ret1[i+win_size:(i+2*win_size)])/scale# New window size 100-200

# Convert data to a dataframe for ease of plotting later
std_Realized = pd.DataFrame(data=std_Realized,index=ret1.index[win_size-1:])# Note the dates used here
std_Realized_fwd = pd.DataFrame(data=std_Realized_fwd,index=ret1.index[win_size-1:])

# calculated ewma variance to be compared with Garch results
lamda = 0.94
Variance_Ema = ewa(np.square(ret1), lamda, win_size)/scale**2 # it was win_size-1
# take out Na
Variance_Ema = Variance_Ema[~np.isnan(Variance_Ema)]
# Mod - it was win_size-2
std_EWMA = pd.DataFrame(data=np.sqrt(Variance_Ema),index=ret1.index[win_size-1:]) # Mod: win_size-1 before

#%% 
# -- Fit GARCH(1,1) Model
#    Forecast Variance using fitted GARCH(1,1) model
#    Uuse forecasted varince to calculate Value at Risk
varLevel = 0.01

# ret1 need to be demeaned
ret1_demeaned = ret1 - np.mean(ret1)
#rets = ret1
# A basic GARCH(1,1) with a constant mean can be constructed using only the return data
#garch11 = arch_model(rets, p=1, q=1)
garch11 = arch_model(ret1_demeaned, mean='zero',p=1, q=1) 
#
# estimate GARCH model
#
# 1. normal model
res_normal = garch11.fit(disp='off') 
# 2. student-t model
res_t = arch_model(ret1_demeaned, dist='t').fit(disp='off')

lls = pd.Series(OrderedDict((('normal', res_normal.loglikelihood),
                 ('t', res_t.loglikelihood),
                 )))
print('Loglikihood of different models:\n',lls)
params = pd.DataFrame(OrderedDict((('normal', res_normal.params),
                 ('t', res_t.params),
                 )))
print('Parameters of different models:\n',params)

print(res_normal.summary())
print(res_t.summary())
# take the residuals
residual_norm = res_normal.resid/scale
residual_t = res_t.resid/scale

# forecasted volatility - also called conditional volatility
vol_conditional = res_normal.conditional_volatility /scale
vol_conditional_t = res_t.conditional_volatility/scale
# generate conditional variances
variance = np.square(vol_conditional) 
# standardized residuals can be computed by dividing the residuals by the conditional volatility
#residual_standardized = ret1_demeaned/vol_conditional/scale
#residual_t_standardized = ret1_demeaned/vol_conditional_t/scale
residual_standardized = residual_norm/vol_conditional
residual_t_standardized = residual_t/vol_conditional_t

# fit the residual
(mu_resiStdized, sigma_resiStdized) = norm.fit(residual_standardized)
dist = stats.t
(deg_f, mu_resiStdized_t, sigma_resiStdized_t) = dist.fit(residual_t_standardized)
# model parameters 
#change SX mu, omega, alpha1, beta1 = res_normal.params
omega, alpha1, beta1 = res_normal.params
# find forecast of next period variance
#  -----error corrected ------- need to be divided by scale**2 etc
Variance_next_Forecasted = omega/scale**2 + alpha1*np.square(ret1_demeaned.iloc[-1]/scale)+beta1*variance.iloc[-1]

#Variance_next_Forecasted = Variance_next_Forecasted/10000
print('Variance forecast {0:.6f}'.format(Variance_next_Forecasted))
# VaR(p) for security starting at value
VaR = -stats.norm.ppf(varLevel, np.mean(ret1/scale),
                      np.sqrt(Variance_next_Forecasted)) 
print('1% VaR forecasted by GARCH {0:.5f}'.format(VaR))    

VaR_EWMA = -stats.norm.ppf(varLevel, np.mean(ret1/scale),std_EWMA.iloc[-1,])
print('1% VaR forecasted by EWMA {0:.5f}'.format(VaR_EWMA[0]))    
                      
#%%
# -----  figure 2  -----------
# plot the fitted residual
fig0 = plt.figure(figure_count)
figure_count = figure_count+1
plt.plot(residual_norm)
plt.ylabel('Garch fit residual', fontweight = 'bold')

# -----  figure 3  -----------
# plot histogram of residuals
plt.figure(figure_count)
figure_count = figure_count+1 # this name the figure by setting numbers
counts, bins, patches = plt.hist(residual_norm, density='norm', bins=200, 
                 histtype='stepfilled', alpha=0.5, label='residual')
# add a 'best fit' line
(mu, sigma) = norm.fit(residual_norm)
#y = mlab.normpdf(bins, mu, sigma)
y = stats.norm.pdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1, label='normal fit')
plt.legend(loc='upper left',  bbox_to_anchor=(0.05, 0.9), shadow=True, ncol=1)
#xmin, xmax = -0.06, 0.06
#plt.xlim( (xmin, xmax) )
plt.xlabel('residual')
plt.ylabel('pdf')
#
# -----  figure 4 & 4a -----------
start = win_size-1
stop = len(ret1)

fig1=plt.figure(figure_count)
figure_count = figure_count+1
#vol_conditional[start:stop].plot()
#std_Realized.plot()
plt.plot(vol_conditional.index[start:stop], vol_conditional[start:stop],'g', label='GARCH Model Vol')
plt.plot(std_Realized.index, std_Realized, 'r',label='Simple Average Realized 50 day Vol.')
plt.plot(std_Realized_fwd.index, std_Realized_fwd, 'b',label='Fwd Realized 50 day Vol.')
plt.legend();
plt.ylabel('Return Volatility');
plt.grid(True)


fig1=plt.figure(figure_count)
figure_count = figure_count+1
plt.plot(vol_conditional.index[start:stop], vol_conditional[start:stop],'g', label='GARCH Model Vol')
plt.plot(std_Realized.index, std_EWMA, 'r',label='EWMA Vol')
plt.plot(std_Realized_fwd.index, std_Realized_fwd, 'b',label='Fwd Realized 50 day Vol.')
plt.legend();
plt.ylabel('Return Volatility');
plt.grid(True)

# ---  figure 5
plt.figure(figure_count)
figure_count = figure_count+1
residual_standardized.plot()
plt.ylabel('Standardized Residual');


# -----  figure 6  -----------
# plot histogram of standardized residuals
plt.figure(figure_count)
figure_count = figure_count+1 # this name the figure by setting numbers
counts1, bins1, patches1 = plt.hist(residual_standardized, density='norm', bins=200, 
        histtype='stepfilled', alpha=0.5, label='standardized residual')
# add a 'best fit' line
y1 = stats.norm.pdf(bins1, mu_resiStdized, sigma_resiStdized)
l1 = plt.plot(bins1, y1, 'r--', linewidth=1, label='normal fit')
plt.legend(loc='upper left',  bbox_to_anchor=(0.05, 0.9), shadow=True, ncol=1)
#xmin, xmax = -0.06, 0.06
#plt.xlim( (xmin, xmax) )
plt.xlabel('standardized residual')
plt.ylabel('pdf')

# ---  figure 7 
# QQ plot of Standardized Residual vs. Normal Distribution
plt.figure()
stats.probplot(residual_standardized, dist="norm", sparams=(mu_resiStdized, sigma_resiStdized), plot=plt)
plt.title('QQ plot vs. Normal Distribution')
plt.xlabel('Quantiles of Normal Distribution',fontSize=12,fontWeight='bold',Color='k')
plt.ylabel('Quantiles of Standardized Residual',fontSize=12,fontWeight='bold',Color='k')

# ---  figure 7 a
# QQ plot of Standardized Residual vs. t Distribution


plt.figure()
stats.probplot(residual_t_standardized, dist=stats.t, 
               sparams=(deg_f, mu, sigma), plot=plt)
plt.title('QQ plot vs. 6 DOF Student t Distribution')
plt.xlabel('Quantiles of 6 DOF Student t Distribution',fontSize=12,fontWeight='bold',Color='k')
plt.ylabel('Quantiles of Standardized T Residual',fontSize=12,fontWeight='bold',Color='k')
ymin, ymax = -6,6
plt.ylim( (ymin, ymax) )


# ---  figure 8
# ACF plot for autocorrelation of Standardized Residual
plt.figure()
confidenceInterval = 0.05
tgs.plot_pacf(residual_standardized, alpha=confidenceInterval, lags=100, zero=False)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Autocorrelation of residual_standardized', fontsize=12)
plt.title('pacf autocorr with 95% confidence interval')

# ---  figure 9
# ACF plot for autocorrelation of Standardized Residual
plt.figure()
tgs.plot_pacf(residual_t_standardized, alpha=confidenceInterval, lags=100, zero=False)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Autocorrelation of t residual_standardized', fontsize=12)
plt.title('pacf autocorr with 95% confidence interval')
## ---  figure 6 default plot on standardized residuals and conditional volatility
#plt.figure(figure_count)
#figure_count = figure_count+1
#res.plot()
#%%
