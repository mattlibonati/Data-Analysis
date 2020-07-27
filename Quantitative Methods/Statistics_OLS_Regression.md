### Overview

Here is a quick overview of some statistical analysis within Python including:
1. Plotting x and y coordinates
2. Calculating pearson coefficient 
3. Running an OLS regression
4. Plotting the line of best fit from regression results 
5. Predicting y-values based on OLS model parameters


### Modules

As usual, numpy and matplotlib are must haves. Numpy is used within the process to calculate the pearson correlation coefficient, as well as allow us to apply the OLS model parameters to the entire list of x-values. Statsmodels is used to perform the OLS regression and provide a plethora of information around the model. 


```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
```

#### Scatter plot

The model is initiated by specifying the x(independent) and y(dependent) variables and creating a simple scatter plot to observe the baseline data points.


```python
x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y = (0.2,0.8,1.2,2.2,2.7,3.2,3.2,4.5,4.7,5.2)

plt.xlabel('Dependent(x)') 
plt.ylabel('Independent(y)')

plt.plot(x, y, 'o', color='black');
```


![alt text](https://github.com/mattlibonati/Quantitative_Methods/blob/master/Images/Statistics_OLS_Regression_scatter.png?raw=true)


#### Correlation Coefficient

The pearson correlation coefficient can easily by found thanks to Numpy's corrcoef(function). Pearson correlation, measured in the range of -1 to 1, is used to determent the strength and direction of the correlation between two variables.


```python
np.corrcoef(x, y)[0, 1]
```




    0.9913407550176283



#### Linear Regression (OLS)
Statsmodels is a great library to use when performing analysis such as an OLS regression. It should be noted x is identified as X within the model becuase adding a constant changes the shape of the original list. If the sample size is below 20, a warning will occur but the model still executes.   

```python
X = sm.add_constant(x)                    # specify constant 
model = sm.OLS(y, X.astype(float)).fit()  # fit model(output, input)
predictions = model.predict(X)            # predict based on fit model
model.summary()                           # print model results
```
    




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.983</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.981</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   455.9</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 09 Jun 2020</td> <th>  Prob (F-statistic):</th> <td>2.43e-08</td>
</tr>
<tr>
  <th>Time:</th>                 <td>08:47:00</td>     <th>  Log-Likelihood:    </th> <td>  1.2832</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    10</td>      <th>  AIC:               </th> <td>   1.434</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>     8</td>      <th>  BIC:               </th> <td>   2.039</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>   -0.2867</td> <td>    0.163</td> <td>   -1.764</td> <td> 0.116</td> <td>   -0.662</td> <td>    0.088</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.5594</td> <td>    0.026</td> <td>   21.353</td> <td> 0.000</td> <td>    0.499</td> <td>    0.620</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.411</td> <th>  Durbin-Watson:     </th> <td>   2.699</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.814</td> <th>  Jarque-Bera (JB):  </th> <td>   0.322</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.343</td> <th>  Prob(JB):          </th> <td>   0.851</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.450</td> <th>  Cond. No.          </th> <td>    13.7</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### Plot Line of Best Fit
model.params is used to extract the constant and coefficient from the statsmodes output. The coefficient is converted to a numpy array so it can be multiplied by the x-coordinates for the line of best fit.


```python
constant = model.params[0]      # declare constant from statsmodels output 
coefficient = model.params[1]   # declare coefficient from statsmodels output

plt.scatter(x, y, color = 'black', marker = "o") 
  
y_fit = constant + np.asarray(coefficient)*x  # convert coefficient to array
  
plt.plot(x, y_fit, color = 'r') 
  
plt.xlabel('x') 
plt.ylabel('y') 
  
plt.show() 
```


![alt text](https://github.com/mattlibonati/Quantitative_Methods/blob/master/Images/Statistics_OLS_Regression_fit.png)


#### Predict y values


```python
print(f'Predicted Value: {coefficient*int(input()) + constant}')
```

    11
    Predicted Value: 5.866666666666668
    
