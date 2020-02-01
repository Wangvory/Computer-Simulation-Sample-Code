# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:26:33 2018

@author: flyin
"""
# where 0: indciate formatting for the first var, 1: for the second var
# 8.4f means print a floater with four digits after the . 
# and eight total digits - space will be printed in the front if there are less than egiht numbers
"a= {0:8.4f}, b={1:6,d}".format(12345.5678,1234567) 
"a= {0:8.4f}, b={1:6,d}".format(5.56,1234567)
# a: indicate it is to format the variable named a
"Art: {a:5d},  Price: {p:8.2f}".format(a=453, p=59.058)
#
print("%10.3e"% (356.08977))
