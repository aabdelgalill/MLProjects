import tensorflow as tf
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt #needed for plotting

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
a = np.mean(speed)
b = np.median(speed)
c = stats.mode(speed) # returns the most common value and how many times it appears
d = np.std(speed) #standard dev, how far most values are from mean
e = np.var(speed)
f = np.percentile(speed, 75) # what is the 75th percentile
print(a,b,c,d,e,f)
# numpy gets us mean, median, standard dev (square root of variance)
# scipy gets us mode

## DATA DISTRUBUTION

#x = np.random.uniform(0.0, 5.0, 250) # creats an array containng 250 randmo floats betwee 0 and 5




# REGRESSION = ATTEMPT TO FIND RELATIONSHIP BETWEEN VARIABLES
## LINEAR REGRESSION ##
# DRAWING A TREND LINE THROUGH DATA POINTS TO PREDICT FUTURE VALUES
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
# x is how old a car is in years
# y is the cars speed
x = [5,7,8,7,2,17,2,9,4,11,12,9,6] # x values on the graph
y = [99,86,87,88,111,86,103,87,94,78,77,85,86] # y values
slope, intercept, r, p, std_err = stats.linregress(x,y)
def myfuc(x):
    return slope * x + intercept # actual prediction happens here
mymodel = list(map(myfuc, x))
print (r) #relationship coff/ -1 to 1 , 0 means no relationship, 1/-1 means 100%
speed = myfuc(10) #this predicts the speed of a car that is 10 years old
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()


import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

x= [1,2,3,4,5,6,7]
y= [2,4,6,8,10,12,14]
slope, intercept, r, p, std_err = stats.linregress(x,y)
def myfuc(x):
    return slope * x + intercept
speed = myfuc(8)
print(r) # r = 1 b/c there is an obvious relationship here
print(speed) # predicts 16