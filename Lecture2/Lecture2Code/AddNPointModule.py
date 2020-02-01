# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:42:24 2017

@author: Steve Xia
"""

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