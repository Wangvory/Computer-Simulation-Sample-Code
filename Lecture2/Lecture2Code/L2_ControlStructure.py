# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:02:21 2018

@author: SteveX
"""

# --- for loops ---
# Prints out the numbers 0,1,2,3,4
for x in range(5):
    print(x)

# Prints out 3,4,5
for x in range(3, 6):
    print(x)

# Prints out 4, 6, 8
for x in range(3, 8, 2):
    y=x+1
    print(y)
    
# while loops
# Prints out 0,1,2,3,4
count = 0
while count < 5:
    print(count)
    count += 1  # This is the same as count = count + 1    
    
# if else 
var1 = 100
if var1>100:
   print("Var Value {0:8.4f} is greater than 100".format(var1))
else:
   print("Var Value {0:8.4f} is equal or less than 100".format(var1))

# elif 
var = 100
if var >= 200:
   print("Var Value is {0:8.4f} and high".format(var1))
elif var >= 150:
   print("Var Value is {0:8.4f} and moderate".format(var1))
elif var >= 100:
   print("Var Value is {0:8.4f} and low".format(var1))
else:
  print("Var Value is {0:8.4f} and very low".format(var1))

    