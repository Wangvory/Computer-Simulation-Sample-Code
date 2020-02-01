# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:53:05 2019

https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced
@author: Steve Xia
"""
import pandas as pd
import numpy as np

arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
           ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']] 

tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
index
s = pd.Series(np.random.randn(8), index=index)
s
s['bar']
#%%%
#multiIndex = [('3/1/1994', 'JPM'), ('9/1/1994','JPM'),('3/1/1994', 'GooG'), ('9/1/1994', 'GooG')]
arrays1 = [['3/1/1994', '3/1/1994', '3/1/1994', '9/1/1994', '9/1/1994', '9/1/1994'], 
           ['JPM', 'GooG', 'GS', 'JPM', 'GooG', 'GS']]
multiIndex = list(zip(*arrays1))
index1 = pd.MultiIndex.from_tuples(multiIndex,names=['date', 'ticker'])

#df = pd.DataFrame(data=[100., 110., 30., 40.], index=index1,columns=['return1','return2'])

#pricedata = [100., 110., 30., 40.]
pricedata = np.random.randn(6,2)
df = pd.DataFrame(data=pricedata, index=index1,columns=['return1','return2'])
#print(df.index.get_level_values(level=1).dtype)
# object
#print(df.index.get_level_values(level=1).dtype)

index1.get_level_values(0)
index1.get_level_values(1)

print(df)

# Basic indexing on axis with MultiIndex
df.loc['3/1/1994']
df.loc['3/1/1994', 'JPM']
df.loc['3/1/1994', 'JPM']['return1']
np.mean(df.loc['3/1/1994', 'JPM'])
df2 = df.mean(level=0)
df3 = df.mean(level=1)
# “Partial” slicing
df.loc['3/1/1994':'9/1/1994']
df.loc[['9/1/1994'][0:2]]
df.loc[['9/1/1994'][0:2],'return1']
# The xs() method of DataFrame additionally takes a level argument to make selecting data at a particular level of a MultiIndex easier.
# This take all of the data from the second level (ticker)
df.xs('JPM', level='ticker')
df.xs(('GS', '3/1/1994'), level=('ticker', 'date'), axis=0)
