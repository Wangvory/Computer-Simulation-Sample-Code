# -*- coding: utf-8 -*-
"""
Spyder Editor

Created by Steve Xia 
"""
import numpy as np
import pandas as pd # load the pandas package

# turning off warning messages type = FutureWarning
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#import warnings
#warnings.filterwarnings("ignore")
#%%
# numbers
str1 = 'helloWorld'
num = 888
pi = 3.14159

print(type(str1))  # This will return a string
print(type(num))  # This will return an integer
print(type(pi))  # This will return a float

# referencing parts of a string
print(str1[5]) # this will print char W, which is the six character in the string 
print(str1[1:5]) # this will print the 2-5 characters in str1 

# creating  and define strings array
arr = np.chararray((3, 2),itemsize=2)
arr[0,0] ='b'
#%%
#
# Tuples
#
t = (1, 'hello ' , True , 2.5)
t1 = (1, 'hello ', (3.14 , True ), 2.5) # tuple contains int, str, and a tuple as its elements
tup1 = () # empty tuple
t1[2][1] #  referencing tuples - gives "True"
my_tuples = ( (1, 2, 3), ('a', 'b', 'c', 'd', 'e'), (True, False), 'Hello' )
#%%
#
# List - List is comma separated and enclosed in sqaure brackets
#
# a list containing tuples
my_list = [(1, 2, 3), ('a', 'b', 'c', 'd', 'e'), (True, False), 'Hello']
my_list.append ((0.5,1))
my_list.insert(0, 'xxx') ## insert elem 'xxx' at index 0
my_list.extend(['yyy', 'zzz']) ## add two elements at the end
del my_list[-1] # delete the last element
A = [ ] # This is a blank list variable
# a list containing lists
my_list1 = [[1, 2], ['a', 'b', 'c'], [True, False, True], ['Hello', 'goodbye']]
my_list1[2][0] = False # modifying the content of a list
list_oflists = [my_list, my_list1]# Two element list with each element a list itself
# another way to create a list
l = [i for i in range(10)]
l1 = [i for i in np.linspace(12,16,8)]
l1a = list(np.linspace(12,16,8))


#
# Dictionaries - Dictionaries is comma separated and enclosed in {}
#
d = {'Name' : 'David G', 'Gender' : 'M', 'Age' : 30} # define a dictionary
d1 = {'Name' : 'Steve A', 'Gender' : 'M', 'Age' : 40}
d2 = {'Name' : ('Steve A', 'David G'), 'Gender' : ('M','M'), 'Age' : (40, 30)} # dict with tuples
d3 = {'Name' : ['Steve A', 'David G'], 'Gender' : ['M','M'], 'Age' : [40, 30]} # dict with list
d3['Name']
d['Age'] = 40 # change element 
d['School'] = 'MIT'; # Add new entry
d.keys() #Returns list of dictionary dict's keys
d.values() #Returns list of dictionary dict's values
d.items() # a list of (key , value ) tuple pairs
d.update(d1) #Update dictionary d's key-values pairs using those of d1
del d['Name']; # remove entry with key 'Name'
d.clear();     # remove all entries in dict
del d ;        # delete entire dictionary
d3['Name'].append('John G') # add elements
d3['Gender'].append('M')
d3['Age'].append(44)
# 
# Series - 
#
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
s.index

# convert a dictionary to series
d = {'b' : 1, 'a' : 0, 'c' : 2}
s1 = pd.Series(d)


#
# Dataframe
#
# convert a dictionary to dataframe
df3 = pd.DataFrame(d3)
# generate random number
np.random.seed(1234)

data = np.random.randn(5, 2)  # 5x2 matrix of N(0, 1) random draws
data1 = np.random.randn(5, 2)  # 5x2 matrix of N(0, 1) random draws
dates = pd.date_range('28/12/2010', periods=5)
dates1 = pd.date_range('01/01/2011', periods=5)

df = pd.DataFrame(data, columns=('price', 'weight'), index=dates)
df0 = pd.DataFrame(data1, columns=('price', 'weightNew'), index=dates1)
a=df.index # the index of df, which is a datetime
b= df.columns # name of the columns
print(df)
# manipulating dataframes
df1=pd.DataFrame(df, columns=['price']) # creating a new dataframe, taking only the price column from df
df2 = df.loc['2010-12-30':] # creating a new dataframe, taking only a portion of the rows
df3 = df.loc[df.index[0:2]] # taking only the first two rows of df, but use index
df3a = df.loc[[df.index[0],df.index[2]]]
df5 = df.iloc[0:2] # taking only the first two rows of df, use integer row number. iloc's i means using integer number for row/column
df5a = df.iloc[[2]] # force output to be a Dataframe, use double [[ ]] for indexing
df6 = df.iloc[:,0:1] # take only the first column of df, using integer column numbers
df7 = df.iloc[0:2,0:2] # take only the first two row and first two column of df
#df6 = df[df.columns[2:4]] 
# df7 = df[['price','b']]
# Column selection, addition, deletion
df4 = df.copy() # creates a copy
df4['weightedPrice'] = df['price'] * df['weight'] #creates a new column
df4['Hiflag'] = df['weight'] < -0.1 # add a boolean flag column
df4b = df4.loc['2010-12-30':,'weight':'Hiflag']
df4c = df4.loc['2010-12-30':,['weight','Hiflag']]
del df4['Hiflag'] # delete the added flag column
df4.drop(df4.index[0:2], axis=0) # delete the first two rows
#df4.drop(['price', 'weight'], axis=1) # delete the two columns
df4.drop(df4.columns[0:2], axis=1)  # delete the two columns
# boolean indexing
ID=df['weight'] < -1
df4e = df4[ID]

# slicing - boolean indexing using .loc
df4f = df4.loc[df.price>0,:]
df4d = df4.loc[df.price>0,'price']

# merging/joining dataframe
df_merged0 = df.append(df0, sort=False)
frames = [df, df0]# this creats a list of two frames
# merge the two frames by simply combing them
df_merged = pd.concat(frames, sort=True) # default axis is 0, which means merge the columns, keep all the rows
df_merged1 = pd.concat(frames, keys=['d1', 'd2'], sort=True) # by this this using 'keys', you introduce an  hierarchical index
df_merged1.loc['d1'] # this will give you back the dataframe df
df_merged1.iloc[7]

df_merged2 = pd.concat(frames, axis=0, join='inner', sort=True) # only take common columns
df_merged3 = pd.concat(frames, axis=0, join='outer', sort=True) # combing all the columns
df_merged4 = pd.concat(frames, axis=1, join_axes=[df.index], sort=True) # only take data based on first dataframe's rowsnames

df_merged5 = pd.concat(frames, axis=1, join='outer', sort=True) # merge the all rows
df_merged6 = pd.concat(frames, axis=1, join='inner', sort=True) # merge only the common rows
#%%
# 
#  Datetime data type
#
from datetime import datetime
datetime.now()
datetime.now().isoformat(timespec='minutes')
datetime.now().isoformat(timespec='hours')
dt = datetime(2015, 1, 1, 12, 30, 59, 0)
dt.isoformat(timespec='microseconds')

datetime(2018, 3, 18)

np.sum([1,2])