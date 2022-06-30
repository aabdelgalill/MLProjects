import tensorflow as tf
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt #needed for plotting
from sklearn.metrics import r2_score # gives relationship measure for poly reg

### POLYNOMIAL REGRESSION ###
# IF IT IS CLEAR LINEAR WON'T WORK USE POLY

# X = HOUR OF DAY
# Y = SPEED OF CAR
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = np.poly1d(np.polyfit(x, y, 3)) # polynomial model
myline = np.linspace(1, 22, 100) # start at x = 1 and end at 22
plt.scatter(x, y)
plt.plot(myline, mymodel(myline)) # polynomial reg line
plt.show()
print(r2_score(y,mymodel(x))) # prints the relationship between x and y
## relationship measure = 0 to 1, 1 = 100%, 0 = 0%
speed = mymodel(17) # predicts the speed at hour 17

import tensorflow as tf
print(tf.__version__)