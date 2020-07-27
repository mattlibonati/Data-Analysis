# Overview
Something that every student can relate to is how our attention deteriorates over time when working on homework or a project. This is also known as decreasing marginal productivity because our attention span (productivity proxy) reaches a point when it will begin to decrease as a function of time. To demonstrate the effect, productivity is set to the arbitrary equation: SQRT(x/2) where x = time in minutes.

The SymPy library is used to perform calculus functions such as evaluating derivatives. NumPy is used to create arrays of data for the functions to perform on. Matplotlib is used to visualize the SymPy calculations.


```python
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
```

Before any SymPy functions can be performed, the symbol must be initialized.


```python
# initialize x for SymPy functions
x = Symbol('x')
# set range for input (minutes)
xrange = np.arange(0,100,1)
# used for dividing by 0 error in xrange within chart
np.seterr(divide='ignore', invalid='ignore')
```



Derivatives are obtained with the SymPy library by using the .diff(x) function. Antiderivatives are found using the integrate(function). Each function is lambdified so it can be called on the entire array of x-values specified in the prior step.


```python
# productivitve function 
productivity = (sqrt(x/2))
# productivitve function - lambdified so it can be performed on entire array
productivity_fn = lambdify(x,productivity,'numpy')
# marginal productivity function
marginal_productivity = productivity.diff(x)
# marginal productivity function - - lambdified so it can be performed on entire array
marginal_productivity_fn = lambdify(x,productivity.diff(x),'numpy')
```

One of the really cool apsects of SymPy is how it formats the output of an equation. Notice how the called marginal_productivity function's output is formatted in an appealing way to the eyes. 


```python
marginal_productivity
```

![alt text](https://github.com/mattlibonati/Data-Analysis/blob/master/Quantitative%20Methods/Images/Statistics_OLS_Regression_fit.png)



And just in case you wanted to see integrate in action...


```python
integrate(marginal_productivity)
```

![alt text](https://github.com/mattlibonati/Data-Analysis/blob/master/Quantitative%20Methods/Images/sympy_integrate_function.PNG)


Now that all of the math is out of the way, we need to plot the data. To do so, a twin plot (two y-axis) will be used to visualize total productivity and marginal productivity. The plot will be broken down into five twenty minute sections shaded with decreasing alphas to signify the area under the curve, or productivity during each specified interval. 


```python
x  = xrange 
y0 = np.arange(0,10,0.1)
y  = productivity_fn(xrange)
y1 = marginal_productivity_fn(xrange)

# set up 20 minute interval bounds
b0   = 0+0.0*y0
b20  = 20+0.0*y0
b40  = 40+0.0*y0
b60  = 60+0.0*y0
b80  = 80+0.0*y0
b100 = 100+0.0*y0

fig, ax1 = plt.subplots()

# total productivity 
color = 'tab:red'
ax1.set_xlabel('Minutes Worked')
ax1.set_ylabel('Output', color=color)
ax1.plot(x, y, color=color)

# plot bounds 
ax1.plot(b0,   y0, color='k', alpha=0.2)
ax1.plot(b20,  y0, color='k', alpha=0.2)
ax1.plot(b40,  y0, color='k', alpha=0.2)
ax1.plot(b60,  y0, color='k', alpha=0.2)
ax1.plot(b80,  y0, color='k', alpha=0.2)
ax1.plot(b100, y0, color='k', alpha=0.2)

ax1.tick_params(axis='y', labelcolor=color)

# marginal productivity
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Marginal Productivity', color=color)  
ax2.plot(x, y1, color=color)
ax2.tick_params(axis='y', labelcolor=color)


# create shaded regions
ax2.fill_between(x, y1, b0, where = x <= 20,
                 facecolor='tab:blue',alpha=0.7, interpolate=True)
x2, y2 = [20, 20, 40, 40], [0, marginal_productivity_fn(20), marginal_productivity_fn(40), 0]
fill(x2,y2, color='tab:blue', alpha=0.5)
x3, y3 = [40, 40, 60, 60], [0, marginal_productivity_fn(40), marginal_productivity_fn(60), 0]
fill(x3,y3, color='tab:blue', alpha=0.4)
x4, y4 = [60, 60, 80, 80], [0, marginal_productivity_fn(60), marginal_productivity_fn(80), 0]
fill(x4,y4, color='tab:blue', alpha=0.3)
x5, y5 = [80, 80, 100, 100], [0, marginal_productivity_fn(80), marginal_productivity_fn(100), 0]
fill(x5,y5, color='tab:blue', alpha=0.15)


# print chart
title ('Total Output') 
plt.tight_layout()

# print results - the antiderivative of marginal productivity is the total productivity function
for i in range(20,101,20):
    iteration = i
    print(f'Minutes {i-20} - {i}: {productivity_fn(i) - productivity_fn(i-20)}')
```

    Minutes 0 - 20: 3.1622776601683795
    Minutes 20 - 40: 1.3098582948312
    Minutes 40 - 60: 1.0050896200520825
    Minutes 60 - 80: 0.847329745285097
    Minutes 80 - 100: 0.7465124915287165
    

![alt text](https://github.com/mattlibonati/Data-Analysis/blob/master/Quantitative%20Methods/Images/integral_calc_matplotlib_output.PNG)



The graph tells us with each sequential 20 minute interval, our productivity decreases at an increasing rate. During the first 20 minutes, our productivity peaked at 3.16u^2 compared to .74u^2 during the 80-100 minute interval, a 2.42u^2 or 67% decrease.
<br><br> It should be noted the values of the total productivity function were used within the example. However, this can also be done using the integral of the derivitive at each point x with SymPy's integrate() function. 
