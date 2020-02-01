# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:51:20 2018

@author: SteveX

In this code, We
 - Read in data from two existing Excel files: 
 - The two files contain time series data that we will actually use in later parts of the course. 
     - One contains historical flags indicating each month in history whether it is a RiskOn (good for equity) or RiskOff (bad for equity) month
     - The other contains historical factor returns for US Equity
     - The dates formats are different, one is YYYYMM, one is YYYY-MM-DD
 - Merge the two data set together 
 - First convert the YYYYMM dates into the YYYY-MM-DD format, assuming the missing days are at month end
 - Calculate conditional mean and covariance of the factor returns, given the regime flags
     - Two regimes: Risk-on and risk-off
 - Convert the calculated results into Dataframes and export them into excel files to be used later in this course! And put some formatting into the excel file!

"""
import pandas as pd
#from pandas import ExcelWriter
#from pandas import ExcelFile
import numpy as np
import datetime

import warnings
warnings.filterwarnings("ignore")

# This function does two tasks:
# 1. Given  input Dates with the format of YYYYMM, identify the month end date for each of the dates
# 2. Converts the given dates into YYYYMMDD with the month end days added
def addMthEndDaytoYearMonth(x):
    import calendar
    #year=int(x[0:4])
    year = int(x/100)
    month = int(x-year*100)
    # Find what is the last day of the month for the given year and month
    day = calendar.monthrange(year, month)[1]
    #  Create new dates by adding the month end days to the known year and month
    DateWithMthEndDate = datetime.date(year,month,day)
    return DateWithMthEndDate

# read in data from an excel xlsx file
df_Factor = pd.read_excel('FamaFrenchFactorReturns.xlsx', sheet_name='FamaFrench4FactorHistData_Month',
                    header=3, index_col = 0)
print(df_Factor.columns)

#
df_Regime = pd.read_excel('InputDataRegimeFlag.xlsx', sheet_name='Flag',
                    header=0, index_col = 0)
print(df_Regime.columns)

#%%
FactorDate = pd.DataFrame(df_Factor.index)
FactorDateWDay = FactorDate.apply(addMthEndDaytoYearMonth, axis=1)
df_Factor.index = FactorDateWDay
# change the format of the index from datetime to dateindex so it can be matched with the index of the Regime dataframe
df_Factor.index = pd.to_datetime(df_Factor.index)
Name_Factors = df_Factor.columns

# merging/joining dataframe
frames = [df_Regime, df_Factor]# this creats a list of two frames
# merge the two frames by simply combing them
df_merged = pd.concat(frames) # default axis is 0, which means merge the columns, keep all the rows
df_merged_C = pd.concat(frames, axis=1, join='inner') # merge only the common rows
df_merged_C1 = pd.concat(frames, axis=1, join='outer') 

# Calculate conditional expected return and risk
a1 = df_merged_C['RiskOn Flag']
df_merged_C_RiskOn = df_merged_C[a1]
del df_merged_C_RiskOn['RiskOn Flag']#take out the flag
mean_ret_RiskOn = df_merged_C_RiskOn.mean()
cov_ret_RiskOn = df_merged_C_RiskOn.cov()
df_merged_C_RiskOff = df_merged_C[~df_merged_C['RiskOn Flag']]
del df_merged_C_RiskOff['RiskOn Flag']#take out the flag
# Calculate mean, covarinace and correlation
mean_ret_RiskOn = df_merged_C_RiskOn.mean()
cov_ret_RiskOn = df_merged_C_RiskOn.cov()
mean_ret_RiskOff = df_merged_C_RiskOff.mean()
cov_ret_RiskOff = df_merged_C_RiskOff.cov()
# create dataframes to be exported to excel
mean_data = np.vstack((mean_ret_RiskOn, mean_ret_RiskOff))
df_mean = pd.DataFrame(mean_data, columns=Name_Factors, index=np.transpose(['RiskOn','RiskOff']))
df_cov_RiskOn = pd.DataFrame(cov_ret_RiskOn, columns=Name_Factors, index=np.transpose(Name_Factors))
df_cov_RiskOff = pd.DataFrame(cov_ret_RiskOff, columns=Name_Factors, index=np.transpose(Name_Factors))

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('Output_RegimeBasedRetNRisk.xlsx', engine='xlsxwriter')
# Write each dataframe to a different worksheet.

#cell_format1.set_num_format('0.000')
# cell_format05.set_num_format('mm/dd/yy')
# cell_format03.set_num_format('#,##0.00')

df_mean.to_excel(writer, sheet_name='Returns',
             startrow=1, startcol=0, header=True, index=True)
df_cov_RiskOn.to_excel(writer, sheet_name='CovRiskOn',
             startrow=1, startcol=0, header=True, index=True)
df_cov_RiskOff.to_excel(writer, sheet_name='CovRiskOff',
             startrow=1, startcol=0, header=True, index=True)

# Get the xlsxwriter objects from the dataframe writer object.
workbook  = writer.book

# define desired formats for excel cells
cell_format_bold = workbook.add_format({'bold': True, 'italic': True, 'font_color': 'red'})
cell_format1 = workbook.add_format({'font_color': 'black', 'num_format':'0.00'})

# saving Mean returns to the first sheet
worksheet1 = writer.sheets['Returns']
worksheet1.write('A1', 'Mean Monthly Regime Returns',cell_format_bold)
worksheet1.conditional_format('B3:F4', {'type': '3_color_scale','format': cell_format1})
worksheet1.conditional_format('B3:F4', {'type':     'cell',
                                    'criteria': '>',
                                    'value':    -10000,
                                    'format':   cell_format1})
    
# saving CoVariance Matrix for RiskOn to the first sheet
worksheet2 = writer.sheets['CovRiskOn']    
worksheet2.write(0, 0, 'RiskOn Monthly Covariance Matrix',cell_format_bold)
worksheet2.conditional_format('B3:F8', {'type':     'cell',
                                    'criteria': '>',
                                    'value':    -10000,
                                    'format':   cell_format1})
worksheet2.conditional_format('B3:F8', {'type': '3_color_scale','format': cell_format1})
    
# saving CoVariance Matrix for RiskOff to the first sheet
worksheet3 = writer.sheets['CovRiskOff']
worksheet3.write(0, 0, 'RiskOff Monthly Covariance Matrix',cell_format_bold) 
worksheet3.conditional_format('B3:F8', {'type': '3_color_scale','format': cell_format1})
worksheet3.conditional_format('B3:F8', {'type':     'cell',
                                    'criteria': '>',
                                    'value':    -10000,
                                    'format':   cell_format1})
writer.save()

