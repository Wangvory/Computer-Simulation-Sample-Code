# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 17:08:35 2018

@author: Steve Xia
"""

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats

#pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() 
import datetime 

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
        
    # get data 
    Flag_downloadData = False
    # define the time period 
    start_dt = datetime.datetime(1994, 2, 11)
    end_dt = datetime.datetime(2006, 1, 10)
    #start_dt = datetime.datetime(1989, 12, 31)
    end_dt = datetime.datetime(2017, 12, 31)
    
    if Flag_downloadData:
        SPData = pdr.get_data_yahoo('^gspc', start=start_dt, end=end_dt)
        NikkeiData = pdr.get_data_yahoo('^N225', start=start_dt, end=end_dt)
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('Input_SPNK.xlsx', engine='xlsxwriter')
        SPData.to_excel(writer, sheet_name='PriceSP',startrow=0, startcol=0, header=True, index=True)
        NikkeiData.to_excel(writer, sheet_name='PriceNK',startrow=0, startcol=0, header=True, index=True)
    else:
        SPData = pd.read_excel('Input_SPNK.xlsx', sheet_name='PriceSP',
                        header=0, index_col = 0)
        NikkeiData = pd.read_excel('Input_SPNK.xlsx', sheet_name='PriceNK',
                        header=0, index_col = 0)
    
    # Returns
    price_SP = SPData['Adj Close']
    price_NK = NikkeiData['Adj Close']
    # merging/joining dataframe
    frames = [price_SP, price_NK]# this creats a list of two frames
    # merge the two frames by simply combing them
    price_SPnNK = pd.concat(frames, axis=1, join='inner') # merge only the common rows
    price_SPnNK.columns = ['SP500','Nikkei']
    
    # update the data to take only the common data
    price_SP = price_SPnNK['SP500']
    price_NK = price_SPnNK['Nikkei']
    #
    ret_SP = price_SP.pct_change().dropna().to_frame('Return')
    ret_All = price_SPnNK.pct_change().dropna()
    # demean the return
    ret_SP_mean = np.mean(ret_SP)
    ret_SP = ret_SP - ret_SP_mean
    
    
    #%%
    # Set up backtest
    T = len(ret_SP) 
    WE = 1000	# Estimation window length				
    p = 0.05	# Threshold					
    value = 1								
    VaR = np.full([T,4],np.nan)			
    lamda = 0.94
    s11 = ret_SP[:WE].var() # initial guess of covariance
    
    # calc EWMA cov till t=window
    for t in range(1,WE):
        s11 = lamda * s11  + (1-lamda) * ret_SP.iloc[t-1]**2
    
    # Listing 8.6 : Running backtest 
    VaR = np.full([T,4],np.nan)#列一个四维向量，长度为T，放四种condition算出来的Var
    for t in range(WE,T):
        t1 = t-WE # only use data in the window, will leads to more volatile VaR
        #t1 = 0 # use all data available, instead of those in the window
        t2 = t
        ret_inWindow = ret_SP[t1:t2]#计算时间周期内的Return
        s11 = lamda * s11  + (1-lamda) * ret_SP.iloc[t-1]**2
        # 1 - VaR based on EWMA method
        VaR[t,0] = -stats.norm.ppf(p,0,1) * np.sqrt(s11)*value
        # 2 - VaR based on simple MA method
        VaR[t,1] = -ret_inWindow.std()*stats.norm.ppf(p,0,1)*value
        # 3 - VaR based on HS method
        VaR[t,2] = -np.percentile(ret_inWindow,p*100)*value 
        # 4 - VaR based on GARCH(1,1)
        # Scale return data by a factor of 100. It seems that the optimizer fails when the values are too close to 0
        scale = 100
        garch11 = arch_model(ret_inWindow*scale, p=1, q=1)
        res_normal = garch11.fit(disp='off') 
        # forecasted volatility - also called conditional volatility
        vol_conditional = res_normal.conditional_volatility
        # generate conditional variances
        variance = np.square(vol_conditional) 
        # model parameters 
        mu, alpha0, alpha1, beta = res_normal.params
        # find forecast of next period variance
        Variance_next_Forecasted = alpha0 + alpha1*np.square(ret_inWindow.iloc[-1])+beta*variance.iloc[-1]
        Variance_next_Forecasted = Variance_next_Forecasted/scale**2
        # VaR based on GARCH method
        VaR[t,3] = -stats.norm.ppf(p,0,1) * np.sqrt(Variance_next_Forecasted) * value
    
    label_m = ['EWMA','MA','Historical','GARCH']
    VaR = pd.DataFrame(data=VaR, index=ret_SP.index, columns=label_m)
    VaR = VaR.dropna()
    #%%
    # Listing 8.8 : Backtesting analysis
    VR_results = np.zeros([4,2])
    num_ViolationTest = len(VaR)-1
    Flag_Violation = np.zeros([num_ViolationTest,4])
    for i in range(4):
        # create flag to check whether forward realized return is equal or less than Ex-ante VaR
        #sum_violation = 0
        for t in range(num_ViolationTest):
            # if next period return is more negative than the projected VaR, then one violation occurs
            Flag_Violation[t,i] = ret_SP.iloc[WE+t+1]<=-VaR.iloc[t,i]
            #sum_violation += np.sum(Flag_Violation)
        sum_violation = np.sum(Flag_Violation[:,i])
        print('i,sum_violation=')
        print(i,sum_violation)
        # Calculate violation ratio
        VR = sum_violation/(p*(T-WE-1))
        # calculate volatility of VaR for each methods
        s = VaR.iloc[WE:,i].std()
        #WE是一个时间窗口， 在已有Train Test Split的时候WE可以看成-1
        # Save results
        VR_results[i,] = [VR, s]
    
    # create data frames for results
    Flag_Violation = pd.DataFrame(data=Flag_Violation, index=ret_SP[WE:-1].index, columns=label_m)
    VR_results = pd.DataFrame(data=VR_results, index=label_m, columns=['Violation Ratio','Std of VaR'])
    print(VR_results)

    #%%
    
    # -----  plotting 
    figure_count = 1
    
    # --------plot the VaR with violation and returns ------
    plt.figure(figure_count)
    figure_count = figure_count+1
    fig, ax1 = plt.subplots(figsize=(12,8))
    
    ax1.plot(ret_SP.index[WE:], ret_SP[WE:], 'c--')
    ax1.plot(VaR.index, -VaR)
    
    ax1.set_xlabel('Date',fontsize=16)
    ax1.set_ylabel('Daily Return/VaR', color='k',fontsize=16)
    
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=18)
    fig.legend(fontsize=13, labels=['Daily return','EWMA VaR','MA VaR','Historical VaR','GARCH VaR'],
               bbox_to_anchor=(0.85, 0.8))
    
    plt.show()
    # --------plot the VaR with violation and returns for each of the VaR methods ------
    for i in range(4):
        plt.figure(figure_count)
        figure_count = figure_count+1
        fig, ax1 = plt.subplots(figsize=(12,6))
        ax2 = ax1.twinx()
        
        ax1.plot(VaR.iloc[:-1,:].index, Flag_Violation.iloc[:,i], 'y--', label='Violation')
        ax2.plot(ret_SP.index[WE:-1], ret_SP[WE:-1], 'c--',label='SP return')
        ax2.plot(VaR.iloc[:-1,:].index, -VaR.iloc[:-1,i], 'k', label=label_m[i]+' VaR')
        
        ax1.set_xlabel('Date',fontsize=16)
        ax1.set_ylabel('Violation', color='k',fontsize=16)
        ax2.set_ylabel('Daily Return/VaR', color='k',fontsize=16)
        
        ax1.xaxis.set_tick_params(labelsize=16)
        ax1.yaxis.set_tick_params(labelsize=18)
        ax2.yaxis.set_tick_params(labelsize=18)
        fig.legend(fontsize=16, bbox_to_anchor=(0.85, 0.8))
        
