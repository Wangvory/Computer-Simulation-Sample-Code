"""
Created on Tue Feb  6 19:34:03 2018
in this code we demonstrate different ways of calculating Expected Shortfall
    1. historical
    2. Analytical
    3. Simulation
    
@author: Steve Xia 
"""
#%%
from scipy import stats
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as spio
import pandas as pd
#%%
#-
#------------------------------
PortValue_Current = 100
mu=100
sigma = 10
p=0.05

# ----- Calculate VaR based on Portfolio Value or P(t+1) ------
PortValue_At_p = stats.norm.ppf(p,mu,sigma)
Loss_At_p_Analytical = PortValue_At_p - PortValue_Current
#  ----- Calculate VaR based on P&L or Q(t+1) ------
VaR_Analytical = stats.norm.ppf(p,mu-PortValue_Current,sigma)
#%%
#  ----- Calculate VaR based on Portfolio Return or R(t+1) ------
# --- Normal Distributed Return
mu_R = 0.12
sigma_R = 0.2
R_star = stats.norm.ppf(p, mu_R, sigma_R)
VaR_Rstar = PortValue_Current*R_star

# expected shortfall - analytical approach based on normal distribution
R_Tilde = -sigma_R*stats.norm.pdf((R_star-mu_R)/sigma_R)/p+mu_R
ES = PortValue_Current*R_Tilde

# --- Student-t Distributed Return
nu = 4
mu_R_t = 0.12
sigma_R_t = 0.2
R_star_t = stats.t.ppf(p, nu, mu_R_t, sigma_R_t)
VaR_Rstar_t = PortValue_Current*R_star_t
#%%
#-----------------------------
# Historical Expected Shortfall
#------------------------------
# Daily return for US equity
retUS = pd.read_excel('retUS.xlsx', sheet_name='Sheet1',index_col =0)
ret1 = retUS['Ret'] 

PortVal_0 = 100
v = np.var(ret1)
mu = np.mean(ret1)
sigma = np.std(ret1)
p = 0.05

# historical value
# method 1 with prices
port1day = PortVal_0*(1+ret1)
PnL = port1day - PortVal_0
P1Sigma = -PortVal_0*(mu-sigma)
PStar = np.percentile(port1day,p*100)
VaR = PortVal_0 - PStar
PTilde = np.mean( port1day[port1day<=PStar])
es1 = PortVal_0-PTilde
print("Historical ES: Prices {0:8.4f}".format(es1))

figure_count=1

plt.figure(figure_count)
figure_count = figure_count+1
plt.hist(PnL, density=True, bins=200, histtype='stepfilled', alpha=0.5)
plt.axvline(x=-P1Sigma, ymax=0.4,linewidth=3, color='pink')
plt.axvline(x=-VaR, ymax=0.3,linewidth=2, color='hotpink')
plt.axvline(x=-es1, ymax=0.2,linewidth=2, color='r')

plt.annotate('Vol.', fontweight = 'bold',xy=(-P1Sigma, 0.25), xytext=(-P1Sigma-2, 0.4),
            arrowprops=dict(facecolor='pink', shrink=0.05),
            )
plt.annotate('VaR', fontweight = 'bold',xy=(-VaR, 0.2), xytext=(-VaR-2, 0.3),
            arrowprops=dict(facecolor='hotpink', shrink=0.05),
            )
plt.annotate('CVaR', fontweight = 'bold',xy=(-es1, 0.14), xytext=(-es1-2.5, 0.25),
            arrowprops=dict(facecolor='r', shrink=0.05),
            )

plt.xlim(-8,8)
#%%
# method 2 with returns
RStar = np.percentile(ret1,p*100)
RTilde = np.mean(ret1[ret1<=RStar])
es2 = -PortVal_0*RTilde
print("Historical ES: Returns {0:8.4f}".format(es2))

# now go with analytic.  Formula from notes (section 5.1)
rStar = stats.norm.ppf(p, mu, sigma)
RTilde = -sigma* stats.norm.pdf( (rStar-mu)/sigma)/p + mu # Changed by SX
es = -PortVal_0*RTilde
print("Formula ES {0:8.4f}".format(es))
#%%
# 
################# Student t distribution & Expected Shortfall example ####################
#
# Note these are annual returns

retUS = pd.read_excel('USStockReturns.xlsx', sheet_name='Sheet1',index_col =0)
ret = retUS['retAnnual'] 
# fit the return time series using a student-t distribution
tdf, tmean, tsigma = stats.t.fit(ret)

p=0.05; # VaR threshold

# --- Student-t Distributed Return
nu = 4
mu_R_t = 0.12
sigma_R_t = 0.2
R_star_t = stats.t.ppf(p, tdf, tmean, tsigma)
VaR_Rstar_t = PortValue_Current*R_star_t
print("5% student-t Analytical VaR {0:8.4f}".format(VaR_Rstar_t))

# show the fit
plt.figure(figure_count)
figure_count = figure_count+1
plt.hist(ret, density=True, bins=50, histtype='stepfilled', alpha=0.5)

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

VAR_At_p_MC_t = PortValue_Current* PortRet_At_p_MC_t
ES_t = PortValue_Current*np.mean(PortRet_MC_LeftTail_t)

print("5% student-t Monte Carlo VaR {0:8.4f}".format(VAR_At_p_MC_t))
print("5% student-t Monte Carlo ES {0:8.4f}".format(ES_t))