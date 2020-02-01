# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:12:14 2018
@author: Steve Xia 

In this code, we perform the following tasks
    1. Use Exponentil moving average model to calculate multi-variate portfolio level covariance matrix and correlatoin
    2. Use forecasted variance matrix to calculate VaR and CVaR
    
"""

import numpy as np 
from scipy import stats
import scipy.io as spio
import pandas as pd 
import matplotlib.pyplot as plt
#--------------------------------------------------------

#writer = pd.ExcelWriter('msft_ibm.xlsx', engine='xlsxwriter')
#df.to_excel(writer, sheet_name='Sheet1',startrow=0, startcol=0)
#writer.save()
df = pd.read_excel('msft_ibm.xlsx', sheet_name='Sheet1',index_col =0)
#df = df.loc[:'2008-12-31',:]
figure_count = 1

fig1=plt.figure(figure_count, figsize=(12, 10), edgecolor='k')
figure_count = figure_count+1
df.plot()

#----------

rets = df.pct_change().dropna()
#rets2 = df / df.shift(1) - 1
#log_rets = np.log(df.pct_change()+1)
#log_rets2 = np.log(df / df.shift(1))

vols = rets.std(axis=0)
# demean returns 
rets = rets.sub(rets.mean(axis=0), axis=1)

T = len(rets)
Wts_Current = [0.5, 0.5]
num_assets = len(Wts_Current)

ret_mat = rets.values

# Listing 3.4 : EWMA
lamda = 0.94
EWMA = np.zeros((T+1,3))#构建一个3变量数组，第一个变量是a的标准差，第二个变量是b的标准差，第三个变量是ab的相关系数
Variance_EWMA_Hist = np.zeros((T+1,1))#这个是最后的Voltality数组
S = np.cov(ret_mat[:,0], ret_mat[:,1])  #rets.cov()Stock1&Stock2
S1 = rets.cov().values  #Corr 矩阵出现
S2 = rets.cov()
EWMA[0,:] = (S[0,0], S[1,1], S[0,1])
Variance_EWMA_Hist[0,:] = np.matmul(np.matmul(Wts_Current,S), np.transpose(Wts_Current))#计算组成的Portfolio的方差
for i in range(1, T+1) :
    S = lamda * S  + (1-lamda) * np.matmul(ret_mat[i-1,:].reshape((-1,1)), 
                      ret_mat[i-1,:].reshape((1,-1)))
    EWMA[i,:] = (S[0,0], S[1,1], S[0,1])
    # Calc Portfolio level Variance for each time  using EWMA
    for j in range(0, num_assets):
        for k in range(0, num_assets):
            Variance_EWMA_Hist[i,:] = Variance_EWMA_Hist[i,:] + Wts_Current[j]*Wts_Current[k]*S[j,k]

# Calculate correlaton
EWMArho = np.divide(EWMA[:,2], np.sqrt(np.multiply(EWMA[:,0], EWMA[:,1])))
#%%
# chart the coefficients
coeff = np.zeros((T+1,1))
for i in range(0, T+1) :
    coeff[i] = (1-lamda)*lamda**(i)
    
print('sum of coeff=',np.sum(coeff))

# produce the weighting coefficient as a function of lembda to be plotted 
num_lamda = 5
coeff_plot = np.zeros((T+1,num_lamda))
lamda_c = np.zeros((num_lamda,1))
for j in range(0,num_lamda):
    lamda_c[j] = 1 - (j+1)*0.06
    for i in range(0, T+1) :
        coeff_plot[i,j] = (1-lamda_c[j])*lamda_c[j]**(i)
        
Name_Lamda = ['Lamda='+str(round(float(lamda_c[i]), 2)) for i in range(num_lamda)]
df_coeff = pd.DataFrame(coeff_plot, columns=Name_Lamda)

# weight coefficient plot
plt.figure(figure_count, figsize=(20, 16), edgecolor='k')
figure_count = figure_count+1
df_coeff.iloc[0:20,:].plot(color = ['b', 'r','k','y','c'])
plt.xlabel(' nth coefficient')
plt.ylabel('effective wights')
plt.show()
#%%
#
# Calc the Portfolio Variance (at the last date) for VaR Calculation
#
S = S/np.sum(coeff)
Var_Forecast = 0;
for i in range(0, num_assets):
    for j in range(0, num_assets):
        Var_Forecast = Var_Forecast+Wts_Current[i]*Wts_Current[j]*S[i,j]

# Forecasted vol. for the next period (the last element of the array)
Sigma_Forecast = np.sqrt(Var_Forecast)
Mu_Forecast = 0
pthrshold = 0.01
PortValue_Current = 100
# VaR
R_star = stats.norm.ppf(pthrshold, Mu_Forecast, Sigma_Forecast)
VaR_Rstar = PortValue_Current*R_star
# CVaR
R_Tilde = -Sigma_Forecast*stats.norm.pdf((R_star-Mu_Forecast)/Sigma_Forecast)/pthrshold+Mu_Forecast
ES = PortValue_Current*R_Tilde

#%%
#--------------------------------------------------------
# 

# old ewma_df = pd.DataFrame(EWMA, index=df.index[1:])
ewma_df = pd.DataFrame(EWMA, index=df.index)
ewma_df.columns = ('msft', 'ibm', 'cov')
ewma_df['Portfolio'] = Variance_EWMA_Hist

fig2=plt.figure(figure_count, figsize=(12, 10), edgecolor='k')
figure_count = figure_count+1
ax = ewma_df.loc[:,['msft', 'ibm', 'Portfolio']].plot(color = ['b', 'r','k'])
#The df.plot() function returns a matplotlib.axes.AxesSubplot object. You can set the labels on that object.
ax.set_ylabel('Variance',size=12)
plt.tight_layout()

start, end = ax.get_xlim()
num_tick = 10
interval = (end - start)/10
ax.xaxis.set_ticks(np.arange(start-interval, end+interval, interval))
#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
plt.show()

#
ewma_df['Msft-LongTermAverage'] = S1[0,0]*np.ones(ewma_df['msft'].shape)
fig3=plt.figure(figure_count, figsize=(12, 10), edgecolor='k')
figure_count = figure_count+1
ax3 = ewma_df.loc[:,['msft', 'ibm', 'Msft-LongTermAverage']].plot(color = ['b', 'r','k'])
#The df.plot() function returns a matplotlib.axes.AxesSubplot object. You can set the labels on that object.
ax3.set_ylabel('Variance',size=12)
plt.tight_layout()

start, end = ax3.get_xlim()
num_tick = 10
interval = (end - start)/10
ax3.xaxis.set_ticks(np.arange(start-interval, end+interval, interval))
