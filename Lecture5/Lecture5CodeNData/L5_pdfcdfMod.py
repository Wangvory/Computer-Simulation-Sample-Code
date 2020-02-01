# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:10:03 2018

In this code, we perform the following tasks
    1. demonstrate how to calculate pdf and cdf from raw data
    2. calculate the value of the x variable given cdf number
    3. Calculate the VaR and plot the VaR line
    4. Plot the cdf line together with pdf
    5. use normal random numbers to calculate VaR
@author: Steve Xia 
"""

#%reset

#from scipy import stats
from scipy.stats import norm
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#from matplotlib.pyplot import imshow, pause

figure_count = 1

#   ---- disbribution, pdf, cdf example ----
ReturnSample = np.array([-0.05, 
                         -0.04, 
                         -0.03, -0.03, -0.03,
                         -0.02, -0.02,-0.02, -0.02,
                         -0.01, -0.01, -0.01, -0.01, -0.01, -0.01,   
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      
                         0.01, 0.01, 0.01, 0.01, 0.01,0.01,
                         0.02, 0.02,0.02, 0.02,
                         0.03, 0.03,0.03,
                         0.04, 
                         0.05
                         ])

ConfidenceLevel = 0.05
ReturnSample_sorted = np.sort(ReturnSample)
num_datapoints = len(ReturnSample_sorted)

CutoffPointID1 = int(ConfidenceLevel*len(ReturnSample))-1
Return_AtThreshold = ReturnSample[CutoffPointID1]

# Calculate pdf
count_uniqRet = np.array(1) # this array tracks how many times each unique return show up in the sample
ret_uniq = np.array(ReturnSample_sorted[0]) # first element of the unique return array
cdf_calculated = np.array(0.0)

count = 1
for i in range(1,num_datapoints):
    #print(i)
    if ReturnSample_sorted[i]!=ReturnSample_sorted[i-1]:
        #cdf_calculated[-1] = cdf_calculated[-1] + count_uniqRet[-1]/num_datapoints
        if count_uniqRet.size==1:
            #count_uniqRet = 1
            pdf_calculated = np.array(count_uniqRet/num_datapoints)
            cdf_calculated = np.array(count_uniqRet/num_datapoints)     
        else:
            count_uniqRet[-1] = count_uniqRet[-1]+1 
            pdf_calculated[-1] = count_uniqRet[-1]/num_datapoints
            cdf_calculated[-1] = cdf_calculated[-2] + count_uniqRet[-1]/num_datapoints
            
        pdf_calculated = np.append(pdf_calculated,1/num_datapoints)
        cdf_calculated = np.append(cdf_calculated,1)
        count_uniqRet = np.append(count_uniqRet,0)
        ret_uniq = np.append(ret_uniq,ReturnSample_sorted[i])
    else:
#        if i==1:
#            count_uniqRet = 2
#            pdf_calculated = np.array(count_uniqRet/num_datapoints)
#            cdf_calculated = np.array(count_uniqRet/num_datapoints)  
        if count_uniqRet.size==1:
            count_uniqRet = count_uniqRet + 1
        else:
            count_uniqRet[-1] = count_uniqRet[-1]+1

        if i==num_datapoints-1:
            count_uniqRet[-1] = count_uniqRet[-1]+1 
            pdf_calculated[-1] = count_uniqRet[-1]/num_datapoints
            cdf_calculated[-1] = cdf_calculated[-2] + count_uniqRet[-1]/num_datapoints

#pdf_calculated1 = np.array(count_uniqRet)/num_datapoints  

# ---- find at which element of the cdf array that the cdf value equal to the 5% threshold
cdf_calculated_LeftTail = cdf_calculated[cdf_calculated <= ConfidenceLevel]
ID_cdf_Threhold = len(cdf_calculated_LeftTail)
# Typo, need to take the following line out
VaR = ret_uniq[ID_cdf_Threhold]

#
# ---- plot pdf with the VaR line -----
#
plt.figure(figure_count)
figure_count = figure_count+1
binWidth = 0.5*(ret_uniq[1] - ret_uniq[0])
#results, bins = np.histogram(myarray, density=True)
#results, bins, patches  = plt.hist(ReturnSample, density = 1, bins=50, histtype='stepfilled', alpha=0.5)
bar = plt.bar(ret_uniq, pdf_calculated,binWidth)
#plt.axvline(x=Return_AtThreshold, linewidth=2, color='r')
#plt.text(Return_AtThreshold+0.003,0.5*max(pdf_calculated),'VaR @ 5%', color='r',fontweight='bold')

plt.xlabel('Return')
plt.ylabel('Probability distribution function')
#plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)

# ---- Calculate Value At Risk for the 5% Threshold
VaR = ret_uniq[ID_cdf_Threhold-1]

# ---- plot pdf & cdf together

fig, ax1 = plt.subplots()
#ax1.plot(ret_uniq, pdf_calculated,'b-')
binWidth = 0.5*(ret_uniq[1] - ret_uniq[0])
bar = ax1.bar(ret_uniq, pdf_calculated, binWidth,label = 'pdf')
ax1.set_xlabel('Unique Returns')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Probability Density Function', color='b',fontweight='bold')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(ret_uniq, cdf_calculated, 'r-',label = 'cdf')
ax2.set_ylabel('Cumulative Distribution Function', color='r',fontweight='bold')
ax2.tick_params('y', colors='r')

#fig.legend(loc=2)
fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.92), shadow=True, ncol=2)

fig.tight_layout()
plt.show()

#
# ---- plot cdf with the VaR arrow -----
#

fig = plt.figure(figure_count)
figure_count = figure_count+1

ax = fig.add_subplot(111)
line1 = ax.plot(ret_uniq, cdf_calculated, linewidth=2, color='r')
line2 = ax.axhline(y=ConfidenceLevel, linewidth=1, color='k',linestyle='--')
line3 = ax.axvline(x=VaR, linewidth=1, color='k',linestyle='--')
#plt.text(VaR,3.0*ConfidenceLevel,'cdf = 5%', color='r',fontweight='bold')
ax.annotate('cdf = 5% \nVaR(5%)=-0.04', fontweight = 'bold',xy=(VaR, ConfidenceLevel), xytext=(1.2*VaR, 7.0*ConfidenceLevel),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

ax.set_ylim(0,1)


plt.xlabel('Return', fontweight = 'bold')
plt.ylabel('Cumulative Probability (cdf)', fontweight = 'bold')


#  --- random number normal distribution exaple

# generate random number
np.random.seed(1234)

Randomdata = np.random.randn(10000, 1)  # 100x1 matrix of N(0, 1) random draws
Randomdata_sorted = sorted(Randomdata)
CutoffPointID = int(ConfidenceLevel*len(Randomdata_sorted))

VaR1 = Randomdata_sorted[CutoffPointID]

# cdf
#cdf_Rand = mlab.normcdf(Randomdata_sorted, loc=0, scale=1)

# best fit of data
(mu, sigma) = norm.fit(Randomdata)

plt.figure(figure_count)
figure_count = figure_count+1
counts, bins, patches = plt.hist(Randomdata, density=True, bins=50, histtype='stepfilled', alpha=0.5)
plt.axvline(x=VaR1, linewidth=2, color='k')
#
# add a 'best fit' line
#y = mlab.normpdf( bins, mu, sigma)
y = norm.pdf(bins, mu, sigma) # new mlab.normpdf doesn't work anymore
l = plt.plot(bins, y, 'r--', linewidth=2)

