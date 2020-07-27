## FFIEC Call Report Data Analysis: Processing Charge-Off Data

This script processes charge-off data collected from the call report files and runs a linear regression against unemployment data collected from the BLS API. This is an early version of this script - optimization is to come. Please note, this script requires running the following three scripts for the source data: 
<br><br> 1. Obtaining and processing Call Report Bulk data:
<br>https://github.com/mattlibonati/Call-Report-Data-Analysis/blob/master/ffiec_call_report_analysis_process_zip_files.md
<br> 2. Creating bank dictionary table: 
<br>https://github.com/mattlibonati/Call-Report-Data-Analysis/blob/master/ffiec_call_report_analysis_creating_bank_list.md
<br> 3. Obtaining BLS unemployment rates:
<br>https://github.com/mattlibonati/macro_economic_data_processing/blob/master/bls_api_unemployment_rates.md

### Modules


```python
import pandas as pd
import numpy as np
import glob
from datetime import datetime

import statsmodels.api as sm # for regression
```

### Process call report text files for charge-offs


```python
dataframe  = []
for infile in glob.glob("call_report_unzipped/*.txt"):
    
    if infile[-18:-13] == 'RIBII':
        data = pd.read_csv(infile,sep='\t')
        data['date'] = infile[-12:-4]
        dataframe.append(data)
    else:
        pass
    
dataframe= pd.concat(dataframe, sort = True)
dataframe = dataframe[dataframe.IDRSSD.notnull()]
dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
```


```python
# filter the data down to just total charge-offs by bank by date
charge_offs = dataframe[['IDRSSD','RIADC079','date']]
```


```python
# validate how many rows have Null Charge-Offs by date
null_co = charge_offs.RIADC079.isnull().groupby([charge_offs['date']]).sum().astype(int).reset_index(name='count')
null_co.loc[null_co['count'] > 0]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>03312001</td>
      <td>8857</td>
    </tr>
    <tr>
      <td>40</td>
      <td>09302004</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remove null charge-offs from dataframe
charge_offs = charge_offs.loc[charge_offs['RIADC079'].notnull()]
# the date needs to be turned into a date format
charge_offs['date'] = pd.to_datetime(charge_offs['date'], format='%m%d%Y')
charge_offs.dtypes
```




    IDRSSD             float64
    RIADC079            object
    date        datetime64[ns]
    dtype: object



### Import bank data


```python
bank_list = pd.read_csv('processed/bank_list.csv')
bank_list['date'] = pd.to_datetime(bank_list['date'], format='%m%d%Y')
bank_list.dtypes
```




    Unnamed: 0                                    int64
    IDRSSD                                        int64
    Financial Institution Address                object
    Financial Institution City                   object
    Financial Institution Filing Type             int64
    Financial Institution Name                   object
    Financial Institution State                  object
    Financial Institution Zip Code                int64
    date                                 datetime64[ns]
    dtype: object



### Import Unemployment Rates
Please see: https://github.com/mattlibonati/macro_economic_data_processing/blob/master/bls_api_unemployment_rates.md
for obtaining unemployment rates.


```python
# obtain unemployment rates from BLS API processed file
unemployment = pd.read_csv(r'C:\Users\Matt\Desktop\GitHub\Github Sites\macro_economic_indicators\Scripting and Storage\data\unemployment\unemploymentprocessed.csv')
# seriesid's have whitespace that needs removed 
unemployment = unemployment.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
# reformat date from object to datetime
unemployment['date'] = pd.to_datetime(unemployment['date'], format='%Y-%m-%d')
unemployment
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>seriesid</th>
      <th>year</th>
      <th>value</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2020-04-30</td>
      <td>LNS14000000</td>
      <td>2020.0</td>
      <td>14.7</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2020-03-31</td>
      <td>LASST360000000000003</td>
      <td>2020.0</td>
      <td>4.5</td>
      <td>3</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2020-03-31</td>
      <td>LASST490000000000003</td>
      <td>2020.0</td>
      <td>3.6</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2020-03-31</td>
      <td>LASST080000000000003</td>
      <td>2020.0</td>
      <td>4.5</td>
      <td>3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2020-03-31</td>
      <td>LASST550000000000003</td>
      <td>2020.0</td>
      <td>3.4</td>
      <td>3</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>12239</td>
      <td>2001-01-31</td>
      <td>LASST380000000000003</td>
      <td>2001.0</td>
      <td>2.8</td>
      <td>1</td>
    </tr>
    <tr>
      <td>12240</td>
      <td>2001-01-31</td>
      <td>LASST200000000000003</td>
      <td>2001.0</td>
      <td>3.9</td>
      <td>1</td>
    </tr>
    <tr>
      <td>12241</td>
      <td>2001-01-31</td>
      <td>LASST090000000000003</td>
      <td>2001.0</td>
      <td>2.6</td>
      <td>1</td>
    </tr>
    <tr>
      <td>12242</td>
      <td>2001-01-31</td>
      <td>LASST500000000000003</td>
      <td>2001.0</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>12243</td>
      <td>2001-01-31</td>
      <td>LASST290000000000003</td>
      <td>2001.0</td>
      <td>4.4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>12244 rows × 5 columns</p>
</div>



### Import state code translations


```python
states = pd.read_csv(r'C:\Users\Matt\Desktop\GitHub\Github Sites\macro_economic_indicators\Scripting and Storage\bls_state_codes.txt',dtype={'state_code':object})
states.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state_code</th>
      <th>state_name</th>
      <th>state_short</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>01</td>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <td>1</td>
      <td>02</td>
      <td>Alaska</td>
      <td>AK</td>
    </tr>
    <tr>
      <td>2</td>
      <td>04</td>
      <td>Arizona</td>
      <td>AS</td>
    </tr>
    <tr>
      <td>3</td>
      <td>05</td>
      <td>Arkansas</td>
      <td>AZ</td>
    </tr>
    <tr>
      <td>4</td>
      <td>06</td>
      <td>California</td>
      <td>AR</td>
    </tr>
  </tbody>
</table>
</div>



### Merge files together


```python
unemployment = pd.merge(unemployment,
                        states[['state_code','state_short']], 
                        left_on  = unemployment['seriesid'].str[5:7],
                        right_on = ['state_code'],
                        how ='left')

# national rates will result in null, therefore replace with US for national indicator
unemployment.loc[unemployment['state_short'].isnull(), ['state_short']] = 'US'
unemployment
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>seriesid</th>
      <th>year</th>
      <th>value</th>
      <th>month</th>
      <th>state_code</th>
      <th>state_short</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2020-04-30</td>
      <td>LNS14000000</td>
      <td>2020.0</td>
      <td>14.7</td>
      <td>4</td>
      <td>00</td>
      <td>US</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2020-03-31</td>
      <td>LASST360000000000003</td>
      <td>2020.0</td>
      <td>4.5</td>
      <td>3</td>
      <td>36</td>
      <td>NY</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2020-03-31</td>
      <td>LASST490000000000003</td>
      <td>2020.0</td>
      <td>3.6</td>
      <td>3</td>
      <td>49</td>
      <td>UT</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2020-03-31</td>
      <td>LASST080000000000003</td>
      <td>2020.0</td>
      <td>4.5</td>
      <td>3</td>
      <td>08</td>
      <td>CO</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2020-03-31</td>
      <td>LASST550000000000003</td>
      <td>2020.0</td>
      <td>3.4</td>
      <td>3</td>
      <td>55</td>
      <td>WI</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>12239</td>
      <td>2001-01-31</td>
      <td>LASST380000000000003</td>
      <td>2001.0</td>
      <td>2.8</td>
      <td>1</td>
      <td>38</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>12240</td>
      <td>2001-01-31</td>
      <td>LASST200000000000003</td>
      <td>2001.0</td>
      <td>3.9</td>
      <td>1</td>
      <td>20</td>
      <td>KS</td>
    </tr>
    <tr>
      <td>12241</td>
      <td>2001-01-31</td>
      <td>LASST090000000000003</td>
      <td>2001.0</td>
      <td>2.6</td>
      <td>1</td>
      <td>09</td>
      <td>CT</td>
    </tr>
    <tr>
      <td>12242</td>
      <td>2001-01-31</td>
      <td>LASST500000000000003</td>
      <td>2001.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>50</td>
      <td>VT</td>
    </tr>
    <tr>
      <td>12243</td>
      <td>2001-01-31</td>
      <td>LASST290000000000003</td>
      <td>2001.0</td>
      <td>4.4</td>
      <td>1</td>
      <td>29</td>
      <td>MO</td>
    </tr>
  </tbody>
</table>
<p>12244 rows × 7 columns</p>
</div>




```python
# merge charge_off and bank_list dataframes
co_dataframe = pd.merge(charge_offs,
                        bank_list[['IDRSSD','date','Financial Institution State']], 
                        left_on  = ['IDRSSD','date'],
                        right_on = ['IDRSSD','date'],
                        how ='left')

# merge unemployment rate data by state
co_dataframe = pd.merge(co_dataframe,
                        unemployment[['date','year','month','value','state_short']], 
                        left_on  = ['date','Financial Institution State'],
                        right_on = ['date','state_short'],
                        how ='left')

# merge unemployment rate data by state
us = unemployment.loc[unemployment['state_short'] == 'US'].rename(columns={'value': 'us_ump'})
co_dataframe = pd.merge(co_dataframe,
                        us[['date','us_ump']], 
                        left_on  = ['date'],
                        right_on = ['date'],
                        how ='left')

co_processed = co_dataframe.rename(columns={'RIADC079': 'charge_offs',
                                            'value': 'state_ump'}).drop(columns=['Financial Institution State'])

# reformat objects to numeric
convert_dict = {'charge_offs'                        : float, 
                'state_ump'                         : float,              
                'us_ump'                             : float}

co_processed = co_processed.astype(convert_dict)

# further process the data file to fit regression:
#co_processed = co_processed.loc[co_processed['charge_offs'] > 0]
co_processed = co_processed.loc[co_processed['state_ump'].notnull()]
co_processed = co_processed.loc[co_processed['IDRSSD'] == 474919.0]
co_processed
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IDRSSD</th>
      <th>charge_offs</th>
      <th>date</th>
      <th>year</th>
      <th>month</th>
      <th>state_ump</th>
      <th>state_short</th>
      <th>us_ump</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3235</td>
      <td>474919.0</td>
      <td>1594.0</td>
      <td>2002-03-31</td>
      <td>2002.0</td>
      <td>3.0</td>
      <td>5.6</td>
      <td>PA</td>
      <td>5.7</td>
    </tr>
    <tr>
      <td>11752</td>
      <td>474919.0</td>
      <td>2567.0</td>
      <td>2003-03-31</td>
      <td>2003.0</td>
      <td>3.0</td>
      <td>5.8</td>
      <td>PA</td>
      <td>5.9</td>
    </tr>
    <tr>
      <td>20121</td>
      <td>474919.0</td>
      <td>1588.0</td>
      <td>2004-03-31</td>
      <td>2004.0</td>
      <td>3.0</td>
      <td>5.5</td>
      <td>PA</td>
      <td>5.8</td>
    </tr>
    <tr>
      <td>28316</td>
      <td>474919.0</td>
      <td>1085.0</td>
      <td>2005-03-31</td>
      <td>2005.0</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>PA</td>
      <td>5.2</td>
    </tr>
    <tr>
      <td>36248</td>
      <td>474919.0</td>
      <td>505.0</td>
      <td>2007-03-31</td>
      <td>2007.0</td>
      <td>3.0</td>
      <td>4.2</td>
      <td>PA</td>
      <td>4.4</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>504741</td>
      <td>474919.0</td>
      <td>19320.0</td>
      <td>2015-12-31</td>
      <td>2015.0</td>
      <td>12.0</td>
      <td>5.2</td>
      <td>PA</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>510897</td>
      <td>474919.0</td>
      <td>17561.0</td>
      <td>2016-12-31</td>
      <td>2016.0</td>
      <td>12.0</td>
      <td>5.2</td>
      <td>PA</td>
      <td>4.7</td>
    </tr>
    <tr>
      <td>516794</td>
      <td>474919.0</td>
      <td>22494.0</td>
      <td>2017-12-31</td>
      <td>2017.0</td>
      <td>12.0</td>
      <td>4.6</td>
      <td>PA</td>
      <td>4.1</td>
    </tr>
    <tr>
      <td>522428</td>
      <td>474919.0</td>
      <td>55071.0</td>
      <td>2018-12-31</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>4.1</td>
      <td>PA</td>
      <td>3.9</td>
    </tr>
    <tr>
      <td>527805</td>
      <td>474919.0</td>
      <td>53187.0</td>
      <td>2019-12-31</td>
      <td>2019.0</td>
      <td>12.0</td>
      <td>4.6</td>
      <td>PA</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 8 columns</p>
</div>



### Run Linear Regression

Using statsmodels OLS(function), we are able to run regressions by specifying the independent and dependent variables. The following example uses state unemployment rates with the previously specified bank -- 474919.


```python
# model input
X = co_processed['state_ump']     # independent variables 
y = co_processed[['charge_offs']] # dependent variable
X = sm.add_constant(X)            # this is where you specify the constant 

# fit OLS 
model = sm.OLS(y, X.astype(float)).fit()     # sm.OLS(output, input) 

# predict based on fit model
predictions = model.predict(X)

# print out the statistics
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>charge_offs</td>   <th>  R-squared:         </th> <td>   0.266</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.256</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   26.13</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 22 May 2020</td> <th>  Prob (F-statistic):</th> <td>2.53e-06</td>
</tr>
<tr>
  <th>Time:</th>                 <td>22:28:40</td>     <th>  Log-Likelihood:    </th> <td> -823.09</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    74</td>      <th>  AIC:               </th> <td>   1650.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    72</td>      <th>  BIC:               </th> <td>   1655.</td>
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
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>     <td>-2.351e+04</td> <td> 8457.430</td> <td>   -2.780</td> <td> 0.007</td> <td>-4.04e+04</td> <td>-6655.408</td>
</tr>
<tr>
  <th>state_ump</th> <td> 7202.1087</td> <td> 1408.850</td> <td>    5.112</td> <td> 0.000</td> <td> 4393.618</td> <td>    1e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>25.818</td> <th>  Durbin-Watson:     </th> <td>   0.530</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  37.100</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.509</td> <th>  Prob(JB):          </th> <td>8.79e-09</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.711</td> <th>  Cond. No.          </th> <td>    27.0</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



An adjusted R^2 of 0.256 is not statistically signicant. Therefore, adjustments will need to be made to the model in order to strengthen its predictability.
