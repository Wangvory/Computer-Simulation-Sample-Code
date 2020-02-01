# code_S2L2_MAEMAFinal
"""
code_S2L2_MAEMAFinal

Created on Tue Feb  6 20:52:45 2018

@author: Steve Xia


In this code, we perform the following tasks
    1. Use Moving Average Exponentil moving average model to calculate Variance
    2. Compare different ways of calcuating volatilities
"""
import numpy as np 
from scipy import stats
import scipy.io as spio
import matplotlib.pyplot as plt
import pandas as pd 
#--------------------------------------------------------

# roll your own functions 

def ma(values, window):
    wgts = np.repeat(1.0, window+1)/(window+1)
    #smas = np.convolve(values, wgts, mode='full')[:len(values)]
    smas = np.zeros(len(values))
    for i in range(window, len(values)):
        currentdata = values[i-window:i+1]
        smas[i] = np.dot(currentdata,wgts)
        del currentdata
        
    #smas[:window] = smas[window]
    smas[:window] = np.nan
    return smas

# lamda here is the same as in MATLAB function 
def ewa(values, lamda, window):
    wgts = np.power(lamda, np.arange(window-1))
    wgts = wgts/wgts.sum()
    ewas = np.convolve(values, wgts, mode='full')[:len(values)]
    #ewas[:window] = ewas[window]
    ewas[:window] = np.nan
    return ewas

    
#----------------------------------------------------
if __name__ == "__main__":
    # Read in data from Excel
    df = pd.read_excel('/Users/zhoujiawang/Desktop/Brandeis Life/Computer Simulation/Lecture7/Lecture7CodeNData/Input_MAEMA.xlsx', index_col=0, sheet_name='Returns')
    
    n_period = 250 
    figure_count = 1
    ret1 = df['ret'] # US val weighted return_returnss with dividends
    #demean returns
    ret1 = ret1 - ret1.mean()
    
    n_returns = len(ret1)
    
    #%%
    df_bs = df.sample(n_returns).set_index(df.index) #  reset index to index from df 
    
    # randomly sample the given return column to create a return vector of the same size as the original dataset
    ret1bs = ret1[np.random.choice(n_returns, n_returns)] # re-sampled returns
    #
    # calc square of return 
    df['ret_square'] = np.square(df['ret'])
    
    # Calculate standard vol. we use regularly using the standard std function on historical back data
    #Std_Standard = np.zeros((n_returns,1))
    Std_Standard = np.full((n_returns,1), np.nan)
    for t in range(n_period-1, n_returns):
        a=df['ret'][t-n_period+1 : t+1]
        Std_Standard[t] = np.std(a,ddof=1)
        del a
    df['std_standard'] = Std_Standard    

    # Or the same results can be calculated using the rolling method
    df['std_standard1'] = df['ret'].rolling(n_period).std()
    
    
    # Calculate simple moving average variance, using original data
    Variance_ma = ma(df['ret_square'], n_period-1)
    df['Variance_ma'] = Variance_ma
    # calculate rolling 250 days mean of return squared. The first Non-nan element equals np.mean(df['ret_square'][0:250])
    df['Variance_ma1'] = df['ret_square'].rolling(n_period).mean() 
    
    #%%
    #
    # Calculate exponentially weighted average variance
    #
    lamda = 0.94
    Variance_Ema = ewa(df['ret_square'], lamda, n_period)
    #上下两种都可以好像没啥区别？
    Variance_Ema1=np.zeros((n_returns,1))
    for t in range(n_period-1, n_returns):
        #print(t)
        #print(t-n_period+1)
        a=df['ret_square'][t-n_period+1 : t+1]
        b = ewa(a, lamda, n_period-1)
        Variance_Ema1[t] = b[-1]
        del a, b
    
    #df['vols_ewma'] = df['ret_square'].ewm(span=n_period).mean()
    df['Varaince_ewma'] = Variance_Ema
    #Compare = np.concatenate((Variance_Ema,Variance_Ema1),axis=0)
    Compare1 = np.column_stack([Variance_Ema,Variance_Ema1])
    
    
    # Convert Variance to Standard Deviation
    df['std_ma'] = np.sqrt(df['Variance_ma'])
    df['std_ewma'] = np.sqrt(df['Varaince_ewma'])
    #%%
    # calculate the forward-realized standard deviation of returns
    Std_FwdRealized = np.full((n_returns,1), np.nan)
    # Note we only care about the forward realized std, starting from period 250, because we intend to compare 
    # them with the ones based on backward-looking ma and ema models
    for t in range(n_period, n_returns-n_period+1):
        a=df['ret'][t : t+n_period]
        Std_FwdRealized[t] = np.std(a,ddof=1)
        del a
    df['std_realized_fw'] = Std_FwdRealized
    # Use the shift method to calculate std. The first Non-nan element equals np.std(df['ret_square'][0:250])
    # a1=df['ret'][np.isnan(df['realized_fw'])]
    #df['std_realized_fw1'] = df['ret'].rolling(n_period).std().shift(-n_period)
    
    df_bs['ret_square'] = np.square(df_bs['ret'])
    # Calculate simple moving average variance, using sampled return data
    Variance_ma_sampledRet = ma(np.square(ret1bs), n_period-1)
    df_bs['Variance_SampledRet'] = Variance_ma_sampledRet
    df_bs['Variance_SampledRet1'] = df_bs['ret_square'].rolling(n_period).mean()
    df_bs['std_SampledRet'] = np.sqrt(df_bs['Variance_SampledRet'])
    
    #%%%
    #
    # ---------   plotting ----------------
    #
    
    import matplotlib.dates as mdates
    
    
    fig2=plt.figure(figure_count, figsize=(12, 10), edgecolor='k')
    figure_count = figure_count+1
    
    ax1 = plt.subplot(311, facecolor='w')
    plt.plot(df['date'], df['std_standard'],'k-', linewidth=2, label = 'std function')
    plt.plot(df['date'], df['std_ma'],'r-.', linewidth=1, label = 'simple moving average')
    
    xfmt = mdates.DateFormatter('%Y')
    ax1.xaxis.set_major_formatter(xfmt)
    
    #ax1.legend(loc='upper center', ncol=2)
    ax1.legend(loc='upper left', ncol=1)
    plt.ylabel('Volatility', fontweight = 'bold')
    
    plt.setp(ax1.get_xticklabels(), fontsize=12)
    
    # subplot 2
    ax2 = plt.subplot(312, sharex=ax1,  facecolor='w')
    plt.plot(df['date'], df['std_standard'],'k-', linewidth=2, label = 'std function')
    plt.plot(df['date'], df['std_ewma'],'r-.', linewidth=1, label = 'exponential moving average')
    
    ax2.legend(loc='upper left', ncol=1)
    plt.ylabel('Volatility', fontweight = 'bold')
    # make these tick labels invisible
    #plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), fontsize=12)
    
    # subplot 3
    ax3 = plt.subplot(313, sharex=ax1, sharey=ax1, facecolor='w')
    plt.plot(df['date'], df['std_standard'],'k-', linewidth=2, label = 'std function')
    plt.plot(df['date'], df_bs['std_SampledRet'],'r-.', linewidth=1, label = 'Boot Strap moving average')
    plt.setp(ax3.get_xticklabels(), fontsize=12)
    
    xfmt = mdates.DateFormatter('%Y')
    ax3.xaxis.set_major_formatter(xfmt)
    
    ax3.legend(loc='upper left', ncol=1)
    plt.ylabel('Volatility', fontweight = 'bold')
    
    #
    #-------------figure 2
    #
    fig3=plt.figure(figure_count, figsize=(12, 10), edgecolor='k')
    figure_count = figure_count+1
    
    ax1 = plt.subplot(311, facecolor='w')
    plt.plot(df['date'], df['std_realized_fw'],'k-', linewidth=2, label = 'forward realized vol.')
    plt.plot(df['date'], df['std_ma'],'r-.', linewidth=1, label = 'simple moving average')
    
    xfmt = mdates.DateFormatter('%Y')
    ax1.xaxis.set_major_formatter(xfmt)
    
    #ax1.legend(loc='upper center', ncol=2)
    ax1.legend(loc='upper left', ncol=1)
    plt.ylabel('Volatility', fontweight = 'bold')
    
    plt.setp(ax1.get_xticklabels(), fontsize=12)
    
    # subplot 2
    ax2 = plt.subplot(312, sharex=ax1,  facecolor='w')
    plt.plot(df['date'], df['std_realized_fw'],'k-', linewidth=2, label = 'forward realized vol.')
    plt.plot(df['date'], df['std_ewma'],'r-.', linewidth=1, label = 'exponential moving average')
    
    ax2.legend(loc='upper left', ncol=1)
    plt.ylabel('Volatility', fontweight = 'bold')
    # make these tick labels invisible
    #plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), fontsize=12)
    
    # subplot 3
    ax3 = plt.subplot(313, sharex=ax1, sharey=ax1, facecolor='w')
    plt.plot(df['date'], df['std_realized_fw'],'k-', linewidth=2, label = 'forward realized vol.')
    plt.plot(df['date'], df_bs['std_SampledRet'],'r-.', linewidth=1, label = 'Boot Strap moving average')
    plt.setp(ax3.get_xticklabels(), fontsize=12)
    
    xfmt = mdates.DateFormatter('%Y')
    ax3.xaxis.set_major_formatter(xfmt)
    
    ax3.legend(loc='upper left', ncol=1)
    plt.ylabel('Volatility', fontweight = 'bold')

