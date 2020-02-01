# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 12:57:36 2018

@author: Steve Xia
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.dates as mdates

plt.close("all") # close all existing figures

df = pd.read_csv('Lecture2/Lecture2Code/retUS_Short.csv', index_col=0, parse_dates=True)

ret1 = df['ret']
n_period = 200 # rolling period
# calculate rolling 200 days return standard deviation
#大概意思就是选在该日期的Std是该日期前n_period天的
df['std'] = df['ret'].rolling(n_period).std()

#%%
#
# ---------   plotting ----------------
#
figure_count = 1# this name the figure by setting numbers
plt.figure(figure_count)# this name the figure by setting numbers

plt.plot(ret1)
plt.ylabel('R(t)')

#

figure_count = figure_count+1 # this name the figure by setting numbers
plt.figure(figure_count)
plt.hist( ret1, normed=True, bins=50, histtype='stepfilled', alpha=0.5, label='ret')
plt.legend(loc='upper left',  bbox_to_anchor=(0.05, 0.9), shadow=True, ncol=1)
xmin, xmax = -0.1, 0.1
plt.xlim( (xmin, xmax) )
plt.xlabel('returns')
plt.ylabel('frequency')
  
#%%
#
# plot the historical returns as a function of time
#
# Change the X Axis to dates
figure_count = figure_count+1
fig=plt.figure(figure_count,figsize=(20, 10))

ax=plt.subplot(111)#这个111代表了图片比例
# fig, ax = plt.subplots() - another way to return figure and axis subject
dates4plot = mdates.datestr2num(df['date'])#把Date变成数字
line = plt.plot(dates4plot, df['ret'],'k-', linewidth=2, label = 'return')

plt.locator_params(axis='y', nbins=6)
plt.locator_params(axis='x', nbins=8)
start, end = ax.get_xlim() # get the min and max of the x axis value
num_x_tick = 8
stepsize = (end-start)/num_x_tick
ax.xaxis.set_ticks(np.arange(start, end, stepsize))#设置八等份的X横轴

# Change dates format into years only without month and day
xfmt = mdates.DateFormatter('%Y%M')
ax.xaxis.set_major_formatter(xfmt)
#这里 出来的东西好像有点问题，出来的是年份后面是00-30？
# define legend location
ax.legend(loc='upper left', ncol=1)
plt.ylabel('return', fontweight = 'bold') #define y axis label

plt.setp(ax.get_xticklabels(), fontsize=12) #make x tick label font larger
#%%
# Scatter Plot
#这个散点图与金融知识无关了
figure_count = figure_count+1
plt.figure(figure_count)
N = 50
# Fixing random state for reproducibility
np.random.seed(1000)

# normal random numbers
x = np.random.normal(0.0, 1.0, N)
y = np.random.rand(N)
colors = np.random.rand(N)
# variable to control the size of the circles
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()