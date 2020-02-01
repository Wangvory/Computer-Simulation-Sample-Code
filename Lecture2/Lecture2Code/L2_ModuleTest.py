# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 11:29:49 2018

@author: Steve Xia
"""
import AddNPointModule as AP

a = AP.addfunc(1,2)
c,d = AP.addsubfunc(1,2)

# define a point object
P1 = AP.Point()
P1.assign(1,2,3)
P1.print()

  