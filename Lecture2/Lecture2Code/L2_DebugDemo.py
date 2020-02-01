# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 23:38:00 2018

@author: Steve Xia
"""
import numpy as np
def demo(x):
    for i in range(5):
        print("i={}, x={}".format(i, x))
        x = x + 1
    return x
a = np.ones((3,1))
a1=3
b=demo(0)
c=2