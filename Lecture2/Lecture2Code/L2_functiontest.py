# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 11:29:49 2018

@author: SteveX & Shan
"""
import numpy as np

def addfunc(x,y):
    return x+y

def addsubfunc(x,y):
    return x+y, x-y

# defining a class called Point, with two methods, assign and print
class Point:
    def assign(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def print(self):
        print(self.x, self.y, self.z)

a=addfunc(1,2)
c,d = addsubfunc(1,2)

# define a point object
P1 = Point()
P1.assign(1,2,3)
P1.print()

# using a method came with Python
X = np.array([1, 2, 3])
print(X.mean)

       
# calling the functions
a=addfunc(1,2)
c,d = addsubfunc(1,2)
