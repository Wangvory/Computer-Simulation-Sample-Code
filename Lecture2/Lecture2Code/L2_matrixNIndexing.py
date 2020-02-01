# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:49:56 2018

@author: Steve Xia 
"""
import numpy as np
# create a 3x3 matrix
X = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# referencing the element of X at the first row and second column
X[0,1]
# reference the first column of X
X[:,0]
# reference the 2 to 3 row and 1 and 2nd column of the matrix. 
# Note 1:3 here mean only 2 and 3 row, 0:2 means only first and second column
X[1:2,0:2]

# use built in functions to initiate matrices
# create a matrix of zeros with three rows and two columns
Y = np.zeros((3,2))
# create a matrix of ones with three rows and two columns
Y1 = np.ones((3,2))
# create a identity matrix - with only the diagonal elements nonzero
np.identity(2)
# create an array from 2 to 4, with 5 elements
z = np.linspace(2, 4, 5)
# create an array from 0 to n-1. Note the largest number is n-1, not n
t1 = np.arange(35)
# create a matrix of inf with three rows and two columns
Y2 = np.full((3, 2), np.inf)
# create a matrix of nan with one rows and three columns
Y3 = np.full((1,3), np.nan)

# test stacking matrices together
Z = np.ones((1,3))
Z1 = np.vstack((X,Z))
Z2 = np.hstack((X,np.transpose(Z)))
Z2a=np.array(Z2) # convert from a mtrix into an array
Z3=Z2a[Z2a[:,0]>=4] # Boolean indexing - only take the rows that the first element is greater or equal 4
#
# Matrix addition, multiplication etc.
#
a = np.array([1, 2, 3, 4]) 
b = np.array([5, 6, 7, 8]) 
print(a + b)
A = np.ones((2, 2)) 
B = np.ones((2, 2))
print(A @ B)
print(Z1 @ Z2) #Multiply a 4x3 matrix with a 3x4 matrix gives you a 4x4 matrix
print(np.dot(A, B)) # The same as A@B
# take out first rows
W = np.delete(Z1, (0), axis=0)
W1 = np.delete(Z1, (1), axis=1)
#
# Matrix referencing using boolean indexing
#
# picking only elements of Z2 that are greater than 3
Z2[Z2 > 3]
ID_Gt1 = Z2[:,0]>1.0
ID_Lt5 = Z2[:,0]<5.0
ID_Gt1Lt5 = ID_Gt1 & ID_Lt5
# reshape a 35x1 column into a 5x7 matrix, with the first 7 element forming the first row
y = np.arange(35).reshape(5,7)
# select only the rows of y where the first element of each row is greater than 0
y1 = y[y[:,0]>0]


from scipy.stats import norm
from scipy.integrate import quad

fai = norm()
value, error = quad(fai.pdf, -2, 2)  # Integrate using Gaussian quadrature

