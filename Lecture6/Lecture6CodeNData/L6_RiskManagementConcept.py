"""
Created on Tue Feb  6 19:34:03 2018
in this code we demonstrate different ways of calculating Expected Shortfall
    1. historical
    2. Analytical
    3. Simulation
    
@author: Steve Xia 
"""

from scipy import stats
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#%%
#-
#------------------------------
#
PortVal_0 = 100
mu = 0.0
sigma = 0.01
p = 0.05
# 
#  --- Probability of loss greater than mean - 1sigma for normal and student t
# Normal cdf at mu-1*sigma
Prob_LessThan1Sigma_normal = stats.norm.cdf(mu-sigma, loc=mu, scale=sigma)
print("Prob of loss greater than mu-1*sigma under normal = {0:8.4f} %".format(100*Prob_LessThan1Sigma_normal))
# Normal cdf at mu-1*sigma
nu = 3
Prob_LessThan1Sigma_t = stats.t.cdf(mu-sigma, nu, loc=mu, scale=sigma)
print("Prob of loss greater than mu-1*sigma under 3 dof t = {0:8.4f} %".format(100*Prob_LessThan1Sigma_t))

#%%
# 
# Compare different risk measures for a single variable
#   - Volatilities
#   - Var
#   - CVaR or ES
VaR1 = PortVal_0 - stats.norm.ppf(p,PortVal_0*(1+mu),sigma*PortVal_0)
R_star = stats.norm.ppf(p,mu,sigma)
VaR_normal = PortVal_0*R_star
# --- Student-t Distributed VaR
nu = 4
mu_R_t = mu
sigma_R_t = sigma
R_star_t = stats.t.ppf(p, nu, mu_R_t, sigma_R_t)
VaR_t = PortVal_0*R_star_t
# 
# Simulate Normal Returns
N = 50000
# generate 10000 normally distributed random numbers
PortValue_MC = np.random.normal(PortVal_0*(1+mu),sigma*PortVal_0, N)
PnL = PortValue_MC - PortVal_0
P1Sigma = PortVal_0*(mu-sigma)

R_Tilde = -sigma*stats.norm.pdf((R_star-mu)/sigma)/p+mu
ES_normal = PortVal_0*R_Tilde

figure_count=1

plt.figure(figure_count)
figure_count = figure_count+1
plt.hist(PnL, density=True, bins=200, histtype='stepfilled', alpha=0.5)
plt.axvline(x=P1Sigma, ymax=0.6,linewidth=3, color='pink')
plt.axvline(x=VaR_normal, ymax=0.44,linewidth=2, color='hotpink')
plt.axvline(x=ES_normal, ymax=0.3,linewidth=2, color='r')

plt.annotate('Vol.', fontweight = 'bold',xy=(P1Sigma, 0.27), xytext=(P1Sigma-1.5, 0.3),
            arrowprops=dict(facecolor='pink', shrink=0.05),
            )
plt.annotate('VaR', fontweight = 'bold',xy=(VaR_normal, 0.2), xytext=(VaR_normal-1.5, 0.21),
            arrowprops=dict(facecolor='hotpink', shrink=0.05),
            )
plt.annotate('CVaR', fontweight = 'bold',xy=(ES_normal, 0.13), xytext=(ES_normal-1.5, 0.15),
            arrowprops=dict(facecolor='r', shrink=0.05),
            )

plt.xlim(-4,4)
plt.xlabel('P&L')
plt.title('Different Risk Measures')
#%%
#
# tracking error calculation
#
wts_active = np.matrix([[-0.1], [0.1]])
cov = np.matrix([[0.001,0.0002], [0.0002, 0.002]])
#np.transpose(wts_active)@cov_end@wts_active
TE1 = np.sqrt(np.transpose(wts_active)@cov@wts_active)
#%%
#
# Portfolio Variance Calculation
#
wts = np.matrix([[0.6], [0.4]])
Variance_Port = 0
num_assets = len(wts)
for j in range(0, num_assets):
    for k in range(0, num_assets):
        Variance_Port = Variance_Port + wts[j]*wts[k]*cov[j,k]
        
#%%
# Correlation and covariance
#
import seaborn as sns

# read in data from an excel xlsx file
df_Factor = pd.read_excel('FamaFrenchFactorReturns.xlsx', sheet_name='FamaFrench4FactorHistData_Month',
                    header=3, index_col = 0)
df_Factor = df_Factor/100 # convert to the right units
labels = df_Factor.columns
# calculate the correlation matrix
corr = df_Factor.corr()

plt.figure(figure_count)
figure_count = figure_count+1
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="YlGnBu"        )
plt.title('Correlation of factors')

#
cov_factors=df_Factor.cov()
