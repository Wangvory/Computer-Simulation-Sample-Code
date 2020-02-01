# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:10:03 2018

in this code we demonstrate different ways of calculating Value at Risk
    1. historical
    2. Analytical
    3. Simulation
@author: Xia 
"""
from scipy import stats
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as spio
import pandas as pd
#from matplotlib.pyplot import imshow, pause

figure_count = 1

#%%
# inverse cdf function 
inversecdf_test1 = stats.norm.ppf(0.05,0,1)
inversecdf_test2 = stats.norm.ppf(0.01,0,1)
print('the inverse cdf at 5% conf. level (with mean=0, std=1) is {y}'.format(y=inversecdf_test1))
print('the inverse cdf at 1% conf. level (with mean=0, std=1) is {y}'.format(y=inversecdf_test2))

############## Analytical VAR example################# 
PortValue_Current = 100
mu=100
sigma = 10
p=0.05 # VaR confidence level

#  ----- Analytical Approach to calculate VaR ------
PortValue_At_p = stats.norm.ppf(p,mu,sigma) # Value at Risk following the equation
Loss_At_p_Analytical = PortValue_At_p - PortValue_Current

#%%
#  ------ Monte-Carlo Approach ------
N = 10000
# generate 10000 normally distributed random numbers
PortValue_MC = np.random.normal(mu, sigma, N)
# or stats.norm.rvs(mu, sigma, N)
PortValue_MC_Sorted = np.sort(PortValue_MC)
ID_At_p = round(p*N)-1 # matlab start from 1, python start from 0 
#这一步是在找出1000里的前5%
PortValue_At_p_MC = PortValue_MC_Sorted[ID_At_p]
# First way
Loss_At_p_MC = PortValue_At_p_MC - PortValue_Current
# Second Way
#Loss_At_p_MC1 = np.percentile(PortValue_MC, p*100) - PortValue_Current
#这个Second Way一步到位很方便

plt.figure(figure_count)
figure_count = figure_count+1
plt.hist(PortValue_MC, density=True, bins=50, histtype='stepfilled', alpha=0.5)
plt.axvline(x=PortValue_At_p_MC, ymax=0.5,linewidth=4, color='r')

#%%
#
#---------Historical VaR Method ---------------------
# load historical data
# Daily return for US equity
retUS = pd.read_excel('retUS.xlsx', sheet_name='Sheet1',index_col =0)
ret1 = retUS['Ret'] 
#因为给的是return，要转化成Value
# vector of portfolio value for 100 held for 1 day
#    all possible outcomes from the sample
port1day = 100*(ret1+1)
# find historical var using direct percentile on portfolio values
#    report the oposite of the loss
var05Hist = 100-np.percentile(port1day,5)
print("5% VaR Historical data {0:8.4f}".format(var05Hist) )

# find historical var using direct percentile on portfolio values
var01Hist = 100-np.percentile(port1day,1)
print("1% VaR Historical data {0:8.4f}".format(var01Hist) )

#%%
#
# --- Compare Historical VaR to Normal assumption based Analytical VaR

# estimate mean and std
s = np.std(ret1)
m = np.mean(ret1)

# VaR(0.05) 5% percentile  
RStar05 = stats.norm.ppf(0.05, m, s)
var05Norm = -100*RStar05
#print("5% VaR Normal"+ "%8.2f" % var05Norm)
print("5% VaR Normal {0:8.4f}".format(var05Norm))

# Normal Analytical VaR(0.01) 
RStar01 = stats.norm.ppf(0.01, m, s)
var01Norm = -100*RStar01
print("1% VaR Normal {0:8.4f}".format(var01Norm))

#%%
# 
################# Student t distribution & Expected Shortfall example ####################
#

# fit the return time series using a student-t distribution
tdf, tmean, tsigma = stats.t.fit(ret1)

p1 = 0.01 # VaR confidence level

# --- Use Student-t Inverse cdf function to calculate analytical VaR

R_star_t = stats.t.ppf(p, tdf, tmean, tsigma)
VaR_Rstar_t =  - PortValue_Current*R_star_t
print("5% student-t Analytical VaR {0:8.4f}".format(VaR_Rstar_t))
R_star_t1 = stats.t.ppf(p1, tdf, tmean, tsigma)
VaR_Rstar_t1 =  - PortValue_Current*R_star_t1
print("1% student-t Analytical VaR {0:8.4f}".format(VaR_Rstar_t1))
tdf_mod = 2 # artificially increasae the tail of the student-st distribution
R_star_t2 = stats.t.ppf(p1, tdf_mod, tmean, tsigma)
VaR_Rstar_t2 =  - PortValue_Current*R_star_t2
print("1% student-t Analytical VaR with fatter tail {0:8.4f}".format(VaR_Rstar_t2))

# show the fit
plt.figure(figure_count)
figure_count = figure_count+1

n, bins, patches  = plt.hist(ret1, density=True, bins=50, histtype='stepfilled', alpha=0.5)
# generate tpdf with x values set to equal the bins
tpdf_ret1 = stats.t.pdf(bins, tdf, tmean, tsigma)
l = plt.plot(bins, tpdf_ret1, 'r--', linewidth=2)

#%%
#-----------Monte Carlo VaR using student-t distribution -------
N = 10000; #number of simulations

# generate randrom returns using fitted student-t parameters from above,
PortRet_MC_t = stats.t.rvs(tdf, loc=tmean, scale=tsigma, size=N)
# Make sure loss doesn't exceed -100%
PortRet_MC_t[PortRet_MC_t<-1.0]=-1.0;

PortRet_MC_t_Sorted= np.sort(PortRet_MC_t)

ID_At_p = round(p*N)-1

PortRet_At_p_MC_t = PortRet_MC_t_Sorted[ID_At_p]
PortRet_MC_LeftTail_t = PortRet_MC_t_Sorted[0:(ID_At_p+1)]

VAR_At_p_MC_t = - PortValue_Current* PortRet_At_p_MC_t

print("5% student-t Monte Carlo VaR {0:8.4f}".format(VAR_At_p_MC_t))