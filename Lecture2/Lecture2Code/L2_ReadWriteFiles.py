# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:46:16 2018

@author: Steve Xia
"""

# read in data from an csv file
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np


df = pd.read_csv('InputDataExample.csv', index_col=0, parse_dates=True)
df1 = pd.read_csv('InputDataExample.csv', parse_dates=True) # Date will become a column, instead of the index

# perform some calculationss then write out the new data to a file
Price_msft = df['msft']*2
df['msft'] = df['msft']*2
# write out data into an excel csv file
Price_msft.to_csv('OutputDataExample1.csv', header=True, index=True) # this will write the column header & index
df.to_csv('OutputDataExample2.csv')

# read in data from an excel xlsx file
df3 = pd.read_excel('InputDataExampleXlsx.xlsx', index_col=0, sheet_name='Sheet1')
print(df.columns)

df4 = df3.loc[df.index[0:2]] # taking only the first two rows of df, but use index

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('OutputDataExampleXlsx.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df3.to_excel(writer, sheet_name='OriginalData')
df4.to_excel(writer, sheet_name='ModifiedData')
# It is also possible to write the dataframe without the header and index.
df4.to_excel(writer, sheet_name='ModifiedData1',
             startrow=3, startcol=4, header=False, index=False)
#
# Close the Pandas Excel writer and output the Excel file.
writer.save()

# 
# read/write data from/to a text file.
#
file1 = open("myfile.txt","w")
L = ["This is Delhi \n","This is Paris \n","This is London \n"] 
 
# \n is placed to indicate EOL (End of Line)
file1.write("Hello \n")
file1.writelines(L)
file1.close() #to change file access modes
 
file1 = open("myfile.txt","r+") 
content_1st = file1.readline()
content_2nd = file1.readline()
content_rest = file1.readlines()
file1.close()
# 

 
