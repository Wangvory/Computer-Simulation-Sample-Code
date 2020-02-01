# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 18:12:43 2018

In this code we perform the following tasks
 - generate random numbers following certain distribution assumptions
 - plot the and compare the normal distribution with the student t distribution
 - show how to calculate the variance and standard deviation of returns
 - simulate and plot the binomial distributions
@author: Xia & Shan
"""
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t
import matplotlib.mlab as mlab

figure_count = 1
# normal random numbers, pdf and fit
Randomdata = np.random.randn(10000, 1)  # 100x1 matrix of N(0, 1) random draws
# cdf
#cdf_Rand = mlab.normcdf(Randomdata_sorted, loc=0, scale=1)

# best fit of data
(mu, sigma) = norm.fit(Randomdata)

plt.figure(figure_count)
figure_count += 1
counts, bins, patches = plt.hist(Randomdata, density=True, bins=50, histtype='stepfilled', alpha=0.5)
#
# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
#
#

# Uniform distribution
figure_count += 1
plt.figure(figure_count)
plt.hist(np.random.rand(10000,1))
plt.title('Uniform distribution')

# Binomial Distribution
p=0.3 #Prob of success at each trial
n=1 #Number of trials
ks=range(n+1) #outcome
m=[stats.binom.pmf(k,n,p) for k in ks] # Bernoulli pdf

#figure_count = figure_count + 1
#plt.figure(figure_count)
figure_count += 1
plt.figure(figure_count)
fig, ax = plt.subplots(1, 1)
#ax.plot(ks, m, 'bo', ms=8, label='binom pmf')
#ax.vlines(ks, 0, m, colors='b', lw=5, alpha=0.5)
ax.bar(ks, m,)
plt.title('binomial distribution')
plt.ylabel('probability')
plt.xticks(ks)



