# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:26:49 2018

@author: Steve Xia
"""
import numpy as np
#%%

# logical operations
a=1 
b=4
c1= (a==b)
print(c1)
c2 = (a>=b)
print(c2)
c3= (a!=b)
print(c3)
d = 5
print(2<d<9)
# logical and, or and not
print(c1 and c2)
print(c1 or True)
print(not c3)

# boolean indexing
X = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9],[10, 11, 12]])
BIdx1 = X[:,0]==1
BIdx2 = X[:,1]<=5

BIdx3 = BIdx1 & BIdx2 #Note here we use & to perform boolean and of two logical arrays
BIdx4 = ~BIdx1
BIdx5 = BIdx1 | BIdx2
# taking only the rows where the first elements are 1
X1 = X[np.repeat(BIdx1,X.shape[1],axis=1)]
# Membership operator
print('good' in 'this is a great show')
print(2 not in np.linspace(1, 4, 2))
print(2 in np.linspace(1, 4, 4))
#%%
for x in range(0, 8):
    print(x) 
#%%
count = 0
while count < 5:
    print(count)
    count += 1
    print(100)

