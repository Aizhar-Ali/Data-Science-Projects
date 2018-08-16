
# Logistic Regression with Python
we will be working with the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic).
We'll be trying to predict a classification- survival or deceased.
Let's begin our understanding of implementing Logistic Regression in Python for classification.
## Import Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
train = pd.read_csv('titanic_train.csv')
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



# Exploratory Data Analysis

Let's begin some exploratory data analysis! We'll start by checking out missing data!

## Missing Data

We can use seaborn to create a simple heatmap to see where we are missing data!


```python
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22164c6f278>




![png](output_4_1.png)


Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"

Let's continue on by visualizing some more of the data! Check out the video for full explanations over these plots, this code is just to serve as reference.


```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22164ccae80>




![png](output_6_1.png)



```python
sns.countplot('Survived',data=train,hue='Sex',palette='RdBu_r')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22164dffb38>




![png](output_7_1.png)



```python
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22164ec0518>




![png](output_8_1.png)



```python
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
```

    C:\Users\Lenovo\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    




    <matplotlib.axes._subplots.AxesSubplot at 0x22165148828>




![png](output_9_2.png)



```python
sns.countplot(x='SibSp',data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22165138a58>




![png](output_10_1.png)



```python
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22166324b70>




![png](output_11_1.png)


### Cufflinks for plots


```python
import cufflinks as cf
cf.go_offline()
```


<script type='text/javascript'>if(!window.Plotly){define('plotly', function(require, exports, module) {/**
* plotly.js v1.38.0
* Copyright 2012-2018, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
});require(['plotly'], function(Plotly) {window.Plotly = Plotly;});}</script>



```python
train['Fare'].iplot(kind='hist',bins=30,color='green')
```


<div id="150939a9-70fd-489c-b6d6-2a1408a60303" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("150939a9-70fd-489c-b6d6-2a1408a60303", [{"type": "histogram", "x": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708, 16.7, 26.55, 8.05, 31.275, 7.8542, 16.0, 29.125, 13.0, 18.0, 7.225, 26.0, 13.0, 8.0292, 35.5, 21.075, 31.3875, 7.225, 263.0, 7.8792, 7.8958, 27.7208, 146.5208, 7.75, 10.5, 82.1708, 52.0, 7.2292, 8.05, 18.0, 11.2417, 9.475, 21.0, 7.8958, 41.5792, 7.8792, 8.05, 15.5, 7.75, 21.6792, 17.8, 39.6875, 7.8, 76.7292, 26.0, 61.9792, 35.5, 10.5, 7.2292, 27.75, 46.9, 7.2292, 80.0, 83.475, 27.9, 27.7208, 15.2458, 10.5, 8.1583, 7.925, 8.6625, 10.5, 46.9, 73.5, 14.4542, 56.4958, 7.65, 7.8958, 8.05, 29.0, 12.475, 9.0, 9.5, 7.7875, 47.1, 10.5, 15.85, 34.375, 8.05, 263.0, 8.05, 8.05, 7.8542, 61.175, 20.575, 7.25, 8.05, 34.6542, 63.3583, 23.0, 26.0, 7.8958, 7.8958, 77.2875, 8.6542, 7.925, 7.8958, 7.65, 7.775, 7.8958, 24.15, 52.0, 14.4542, 8.05, 9.825, 14.4583, 7.925, 7.75, 21.0, 247.5208, 31.275, 73.5, 8.05, 30.0708, 13.0, 77.2875, 11.2417, 7.75, 7.1417, 22.3583, 6.975, 7.8958, 7.05, 14.5, 26.0, 13.0, 15.0458, 26.2833, 53.1, 9.2167, 79.2, 15.2458, 7.75, 15.85, 6.75, 11.5, 36.75, 7.7958, 34.375, 26.0, 13.0, 12.525, 66.6, 8.05, 14.5, 7.3125, 61.3792, 7.7333, 8.05, 8.6625, 69.55, 16.1, 15.75, 7.775, 8.6625, 39.6875, 20.525, 55.0, 27.9, 25.925, 56.4958, 33.5, 29.125, 11.1333, 7.925, 30.6958, 7.8542, 25.4667, 28.7125, 13.0, 0.0, 69.55, 15.05, 31.3875, 39.0, 22.025, 50.0, 15.5, 26.55, 15.5, 7.8958, 13.0, 13.0, 7.8542, 26.0, 27.7208, 146.5208, 7.75, 8.4042, 7.75, 13.0, 9.5, 69.55, 6.4958, 7.225, 8.05, 10.4625, 15.85, 18.7875, 7.75, 31.0, 7.05, 21.0, 7.25, 13.0, 7.75, 113.275, 7.925, 27.0, 76.2917, 10.5, 8.05, 13.0, 8.05, 7.8958, 90.0, 9.35, 10.5, 7.25, 13.0, 25.4667, 83.475, 7.775, 13.5, 31.3875, 10.5, 7.55, 26.0, 26.25, 10.5, 12.275, 14.4542, 15.5, 10.5, 7.125, 7.225, 90.0, 7.775, 14.5, 52.5542, 26.0, 7.25, 10.4625, 26.55, 16.1, 20.2125, 15.2458, 79.2, 86.5, 512.3292, 26.0, 7.75, 31.3875, 79.65, 0.0, 7.75, 10.5, 39.6875, 7.775, 153.4625, 135.6333, 31.0, 0.0, 19.5, 29.7, 7.75, 77.9583, 7.75, 0.0, 29.125, 20.25, 7.75, 7.8542, 9.5, 8.05, 26.0, 8.6625, 9.5, 7.8958, 13.0, 7.75, 78.85, 91.0792, 12.875, 8.85, 7.8958, 27.7208, 7.2292, 151.55, 30.5, 247.5208, 7.75, 23.25, 0.0, 12.35, 8.05, 151.55, 110.8833, 108.9, 24.0, 56.9292, 83.1583, 262.375, 26.0, 7.8958, 26.25, 7.8542, 26.0, 14.0, 164.8667, 134.5, 7.25, 7.8958, 12.35, 29.0, 69.55, 135.6333, 6.2375, 13.0, 20.525, 57.9792, 23.25, 28.5, 153.4625, 18.0, 133.65, 7.8958, 66.6, 134.5, 8.05, 35.5, 26.0, 263.0, 13.0, 13.0, 13.0, 13.0, 13.0, 16.1, 15.9, 8.6625, 9.225, 35.0, 7.2292, 17.8, 7.225, 9.5, 55.0, 13.0, 7.8792, 7.8792, 27.9, 27.7208, 14.4542, 7.05, 15.5, 7.25, 75.25, 7.2292, 7.75, 69.3, 55.4417, 6.4958, 8.05, 135.6333, 21.075, 82.1708, 7.25, 211.5, 4.0125, 7.775, 227.525, 15.7417, 7.925, 52.0, 7.8958, 73.5, 46.9, 13.0, 7.7292, 12.0, 120.0, 7.7958, 7.925, 113.275, 16.7, 7.7958, 7.8542, 26.0, 10.5, 12.65, 7.925, 8.05, 9.825, 15.85, 8.6625, 21.0, 7.75, 18.75, 7.775, 25.4667, 7.8958, 6.8583, 90.0, 0.0, 7.925, 8.05, 32.5, 13.0, 13.0, 24.15, 7.8958, 7.7333, 7.875, 14.4, 20.2125, 7.25, 26.0, 26.0, 7.75, 8.05, 26.55, 16.1, 26.0, 7.125, 55.9, 120.0, 34.375, 18.75, 263.0, 10.5, 26.25, 9.5, 7.775, 13.0, 8.1125, 81.8583, 19.5, 26.55, 19.2583, 30.5, 27.75, 19.9667, 27.75, 89.1042, 8.05, 7.8958, 26.55, 51.8625, 10.5, 7.75, 26.55, 8.05, 38.5, 13.0, 8.05, 7.05, 0.0, 26.55, 7.725, 19.2583, 7.25, 8.6625, 27.75, 13.7917, 9.8375, 52.0, 21.0, 7.0458, 7.5208, 12.2875, 46.9, 0.0, 8.05, 9.5875, 91.0792, 25.4667, 90.0, 29.7, 8.05, 15.9, 19.9667, 7.25, 30.5, 49.5042, 8.05, 14.4583, 78.2667, 15.1, 151.55, 7.7958, 8.6625, 7.75, 7.6292, 9.5875, 86.5, 108.9, 26.0, 26.55, 22.525, 56.4958, 7.75, 8.05, 26.2875, 59.4, 7.4958, 34.0208, 10.5, 24.15, 26.0, 7.8958, 93.5, 7.8958, 7.225, 57.9792, 7.2292, 7.75, 10.5, 221.7792, 7.925, 11.5, 26.0, 7.2292, 7.2292, 22.3583, 8.6625, 26.25, 26.55, 106.425, 14.5, 49.5, 71.0, 31.275, 31.275, 26.0, 106.425, 26.0, 26.0, 13.8625, 20.525, 36.75, 110.8833, 26.0, 7.8292, 7.225, 7.775, 26.55, 39.6, 227.525, 79.65, 17.4, 7.75, 7.8958, 13.5, 8.05, 8.05, 24.15, 7.8958, 21.075, 7.2292, 7.8542, 10.5, 51.4792, 26.3875, 7.75, 8.05, 14.5, 13.0, 55.9, 14.4583, 7.925, 30.0, 110.8833, 26.0, 40.125, 8.7125, 79.65, 15.0, 79.2, 8.05, 8.05, 7.125, 78.2667, 7.25, 7.75, 26.0, 24.15, 33.0, 0.0, 7.225, 56.9292, 27.0, 7.8958, 42.4, 8.05, 26.55, 15.55, 7.8958, 30.5, 41.5792, 153.4625, 31.275, 7.05, 15.5, 7.75, 8.05, 65.0, 14.4, 16.1, 39.0, 10.5, 14.4542, 52.5542, 15.7417, 7.8542, 16.1, 32.3208, 12.35, 77.9583, 7.8958, 7.7333, 30.0, 7.0542, 30.5, 0.0, 27.9, 13.0, 7.925, 26.25, 39.6875, 16.1, 7.8542, 69.3, 27.9, 56.4958, 19.2583, 76.7292, 7.8958, 35.5, 7.55, 7.55, 7.8958, 23.0, 8.4333, 7.8292, 6.75, 73.5, 7.8958, 15.5, 13.0, 113.275, 133.65, 7.225, 25.5875, 7.4958, 7.925, 73.5, 13.0, 7.775, 8.05, 52.0, 39.0, 52.0, 10.5, 13.0, 0.0, 7.775, 8.05, 9.8417, 46.9, 512.3292, 8.1375, 76.7292, 9.225, 46.9, 39.0, 41.5792, 39.6875, 10.1708, 7.7958, 211.3375, 57.0, 13.4167, 56.4958, 7.225, 26.55, 13.5, 8.05, 7.7333, 110.8833, 7.65, 227.525, 26.2875, 14.4542, 7.7417, 7.8542, 26.0, 13.5, 26.2875, 151.55, 15.2458, 49.5042, 26.55, 52.0, 9.4833, 13.0, 7.65, 227.525, 10.5, 15.5, 7.775, 33.0, 7.0542, 13.0, 13.0, 53.1, 8.6625, 21.0, 7.7375, 26.0, 7.925, 211.3375, 18.7875, 0.0, 13.0, 13.0, 16.1, 34.375, 512.3292, 7.8958, 7.8958, 30.0, 78.85, 262.375, 16.1, 7.925, 71.0, 20.25, 13.0, 53.1, 7.75, 23.0, 12.475, 9.5, 7.8958, 65.0, 14.5, 7.7958, 11.5, 8.05, 86.5, 14.5, 7.125, 7.2292, 120.0, 7.775, 77.9583, 39.6, 7.75, 24.15, 8.3625, 9.5, 7.8542, 10.5, 7.225, 23.0, 7.75, 7.75, 12.475, 7.7375, 211.3375, 7.2292, 57.0, 30.0, 23.45, 7.05, 7.25, 7.4958, 29.125, 20.575, 79.2, 7.75, 26.0, 69.55, 30.6958, 7.8958, 13.0, 25.9292, 8.6833, 7.2292, 24.15, 13.0, 26.25, 120.0, 8.5167, 6.975, 7.775, 0.0, 7.775, 13.0, 53.1, 7.8875, 24.15, 10.5, 31.275, 8.05, 0.0, 7.925, 37.0042, 6.45, 27.9, 93.5, 8.6625, 0.0, 12.475, 39.6875, 6.95, 56.4958, 37.0042, 7.75, 80.0, 14.4542, 18.75, 7.2292, 7.8542, 8.3, 83.1583, 8.6625, 8.05, 56.4958, 29.7, 7.925, 10.5, 31.0, 6.4375, 8.6625, 7.55, 69.55, 7.8958, 33.0, 89.1042, 31.275, 7.775, 15.2458, 39.4, 26.0, 9.35, 164.8667, 26.55, 19.2583, 7.2292, 14.1083, 11.5, 25.9292, 69.55, 13.0, 13.0, 13.8583, 50.4958, 9.5, 11.1333, 7.8958, 52.5542, 5.0, 9.0, 24.0, 7.225, 9.8458, 7.8958, 7.8958, 83.1583, 26.0, 7.8958, 10.5167, 10.5, 7.05, 29.125, 13.0, 30.0, 23.45, 30.0, 7.75], "name": "Fare", "marker": {"color": "rgba(0, 128, 0, 1.0)", "line": {"width": 1.3, "color": "#4D5663"}}, "orientation": "v", "opacity": 0.8, "histfunc": "count", "histnorm": "", "nbinsx": 30}], {"legend": {"bgcolor": "#F5F6F9", "font": {"color": "#4D5663"}}, "paper_bgcolor": "#F5F6F9", "plot_bgcolor": "#F5F6F9", "yaxis1": {"tickfont": {"color": "#4D5663"}, "gridcolor": "#E1E5ED", "titlefont": {"color": "#4D5663"}, "zerolinecolor": "#E1E5ED", "showgrid": true, "title": ""}, "xaxis1": {"tickfont": {"color": "#4D5663"}, "gridcolor": "#E1E5ED", "titlefont": {"color": "#4D5663"}, "zerolinecolor": "#E1E5ED", "showgrid": true, "title": ""}, "titlefont": {"color": "#4D5663"}, "barmode": "overlay"}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


## Data Cleaning
We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
However we can be smarter about this and check the average age by passenger class. For example:


```python
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2215fcc7208>




![png](output_16_1.png)


We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.


```python
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
```


```python
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
```


```python
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x221673ea198>




![png](output_20_1.png)



```python
train.drop('Cabin',axis=1,inplace=True)
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>24</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.dropna(inplace=True)
```

## Converting Categorical Features 

We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.


```python
#Multticolinearality issue (because one column telling the infor of other column)
#if female=0 then male=1
pd.get_dummies(train['Sex']).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>female</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Resolving MultiColinearality issue
sex = pd.get_dummies(train['Sex'],drop_first=True)
sex.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()
#there is no issue of MultiColinearality because 
#we already dropped 'C' coulmn of dataset so now no column predict the value of other
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train = pd.concat([train,sex,embark],axis=1)
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>male</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>24</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.drop(['PassengerId','Name','Sex','Ticket','Embarked'],axis=1,inplace=True)
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>male</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>24</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Great! Our data is ready for our model!

# Building a Logistic Regression model

Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)
```

## Training and Predicting


```python
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
predictions = logmodel.predict(X_test)
```

## Evaluation
We can check precision,recall,f1-score using classification report!


```python
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
```

                 precision    recall  f1-score   support
    
              0       0.81      0.92      0.86       163
              1       0.84      0.65      0.74       104
    
    avg / total       0.82      0.82      0.81       267
    
    


```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
```

    [[150  13]
     [ 36  68]]
    