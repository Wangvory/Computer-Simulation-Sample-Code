# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 18:12:43 2018

In this code we perform the following tasks
 - generate random numbers following certain distribution assumptions
 - plot the and compare the normal distribution with the student t distribution
 - show how to calculate the variance and standard deviation of returns
 - simulate and plot the binomial distributions
@author: Steve Xia 
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
Randomdata_sorted = sorted(Randomdata)
# cdf
cdf_Rand = norm.cdf(Randomdata_sorted, loc=0, scale=1)
plt.figure(figure_count)
figure_count += 1
plt.plot(Randomdata_sorted, cdf_Rand, 'r-', lw=2, alpha=0.6, label='normal cdf')
plt.title('normal cdf')

# best fit of data
(mu, sigma) = norm.fit(Randomdata)

plt.figure(figure_count)
figure_count += 1
counts, bins, patches = plt.hist(Randomdata, density=True, bins=50, histtype='stepfilled', alpha=0.5)
#
# add a 'best fit' line
#y = mlab.normpdf( bins, mu, sigma)
y = norm.pdf(bins, mu, sigma) # new mlab.normpdf doesn't work anymore
l = plt.plot(bins, y, 'r--', linewidth=2)
#
#

mean = 0.2,
std = 0.3

#x = np.random.normal(mean, std, (10000,1))
x = np.linspace(-5,5,200)
n1 = norm.rvs(loc=mean, scale=std, size=10000) # normal random variable
num_data = len(x)
sample_mean_n = np.mean(n1)
sample_std_n = np.std(n1)
pdf_n = norm.pdf(x, loc=mean, scale=std) # normal probability distribution function

dof = 2.5 # degree of freedom for student t distribution
t1 = t.rvs(10, loc=mean, scale=std, size=10000) # generate student-t random variable
sample_mean_t = np.mean(t1)
sample_std_t = np.std(t1)
#x = np.linspace(t.ppf(0.01, dof, loc=mean, scale=std), t.ppf(0.99, dof, loc=mean, scale=std), 100)
pdf_t = t.pdf(x, dof, loc=mean, scale=std)

plt.figure(figure_count)
figure_count += 1
plt.plot(x, pdf_t, 'r-', lw=2, alpha=0.6, label='t pdf, dof=2.5')
plt.plot(x, pdf_n, 'k-', lw=2, alpha=0.6, label='normal pdf')
#ax.hist(t1, normed=True, histtype='stepfilled', alpha=0.2)
plt.legend(loc='best', frameon=False)
plt.show()

# Calculate mean
cumsum = 0;
for i in range(num_data):
    cumsum = cumsum + x[i]

mean_calc = cumsum/num_data

# Calculate variance and standard deviation
Var = 0
for i in range(num_data):
    Var = Var + (x[i]-mean_calc)**2

Var_calc = Var/(num_data-1)
Std_calc = np.sqrt(Var_calc)

# Uniform distribution
figure_count += 1
plt.figure(figure_count)
plt.hist(np.random.rand(10000,1))
plt.title('Uniform distribution')

# Binomial Distribution
p=0.3 #Prob of success at each trial
n=1 #Number of trials
ks=range(n+1) #outcome
m=[stats.binom.pmf(k,n,p) for k in ks] # Binomial pdf

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



