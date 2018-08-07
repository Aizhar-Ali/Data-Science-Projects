
# 911 Calls Data Analyzation using numpy, pandas, matplotlib, seaborn

911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). 
The data contains the following fields:

* lat : String variable, Latitude
* lng: String variable, Longitude
* desc: String variable, Description of the Emergency Call
* zip: String variable, Zipcode
* title: String variable, Title
* timeStamp: String variable, YYYY-MM-DD HH:MM:SS
* twp: String variable, Township
* addr: String variable, Address
* e: String variable, Dummy variable (always 1)

## Data and Setup


```python
import numpy as np
import pandas as pd
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df = pd.read_csv('911.csv')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 99492 entries, 0 to 99491
    Data columns (total 9 columns):
    lat          99492 non-null float64
    lng          99492 non-null float64
    desc         99492 non-null object
    zip          86637 non-null float64
    title        99492 non-null object
    timeStamp    99492 non-null object
    twp          99449 non-null object
    addr         98973 non-null object
    e            99492 non-null int64
    dtypes: float64(3), int64(1), object(5)
    memory usage: 6.8+ MB
    


```python
df.head()
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
      <th>lat</th>
      <th>lng</th>
      <th>desc</th>
      <th>zip</th>
      <th>title</th>
      <th>timeStamp</th>
      <th>twp</th>
      <th>addr</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40.297876</td>
      <td>-75.581294</td>
      <td>REINDEER CT &amp; DEAD END;  NEW HANOVER; Station ...</td>
      <td>19525.0</td>
      <td>EMS: BACK PAINS/INJURY</td>
      <td>2015-12-10 17:40:00</td>
      <td>NEW HANOVER</td>
      <td>REINDEER CT &amp; DEAD END</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40.258061</td>
      <td>-75.264680</td>
      <td>BRIAR PATH &amp; WHITEMARSH LN;  HATFIELD TOWNSHIP...</td>
      <td>19446.0</td>
      <td>EMS: DIABETIC EMERGENCY</td>
      <td>2015-12-10 17:40:00</td>
      <td>HATFIELD TOWNSHIP</td>
      <td>BRIAR PATH &amp; WHITEMARSH LN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40.121182</td>
      <td>-75.351975</td>
      <td>HAWS AVE; NORRISTOWN; 2015-12-10 @ 14:39:21-St...</td>
      <td>19401.0</td>
      <td>Fire: GAS-ODOR/LEAK</td>
      <td>2015-12-10 17:40:00</td>
      <td>NORRISTOWN</td>
      <td>HAWS AVE</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40.116153</td>
      <td>-75.343513</td>
      <td>AIRY ST &amp; SWEDE ST;  NORRISTOWN; Station 308A;...</td>
      <td>19401.0</td>
      <td>EMS: CARDIAC EMERGENCY</td>
      <td>2015-12-10 17:40:01</td>
      <td>NORRISTOWN</td>
      <td>AIRY ST &amp; SWEDE ST</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40.251492</td>
      <td>-75.603350</td>
      <td>CHERRYWOOD CT &amp; DEAD END;  LOWER POTTSGROVE; S...</td>
      <td>NaN</td>
      <td>EMS: DIZZINESS</td>
      <td>2015-12-10 17:40:01</td>
      <td>LOWER POTTSGROVE</td>
      <td>CHERRYWOOD CT &amp; DEAD END</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Basic Questions

** What are the top 5 zipcodes for 911 calls? **


```python
df['zip'].value_counts().head()
```




    19401.0    6979
    19464.0    6643
    19403.0    4854
    19446.0    4748
    19406.0    3174
    Name: zip, dtype: int64



** What are the top 5 townships (twp) for 911 calls? **


```python
df['twp'].value_counts().head()
```




    LOWER MERION    8443
    ABINGTON        5977
    NORRISTOWN      5890
    UPPER MERION    5227
    CHELTENHAM      4575
    Name: twp, dtype: int64



** Take a look at the 'title' column, how many unique title codes are there? **


```python
df['title'].nunique()
```




    110



## Creating new features

** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 

**For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **


```python
df['Reason'] = df['title'].apply(lambda x:x.split(':')[0])
```

** What is the most common Reason for a 911 call based off of this new column? **


```python
df['Reason'].value_counts()
```




    EMS        48877
    Traffic    35695
    Fire       14920
    Name: Reason, dtype: int64



** Now use seaborn to create a countplot of 911 calls by Reason. **


```python
sns.countplot(df['Reason'],palette='rainbow')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x207c388bcf8>




![png](output_21_1.png)


___
** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **


```python
type(df['timeStamp'][0])
```




    str



** You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **


```python
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['timeStamp'].head(2)
```




    0   2015-12-10 17:40:00
    1   2015-12-10 17:40:00
    Name: timeStamp, dtype: datetime64[ns]



** You can now grab specific attributes from a Datetime object by calling them. For example:**

    time = df['timeStamp'].iloc[0]
    time.hour

**You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.**


```python
df['Hour'] = df['timeStamp'].apply(lambda x:x.hour)
df['Month'] = df['timeStamp'].apply(lambda x:x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x:x.dayofweek)
```

** Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: **

    dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


```python
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
```


```python
df['Day of Week'] = df['Day of Week'].map(dmap)
df['Day of Week'].head()
```




    0    Thu
    1    Thu
    2    Thu
    3    Thu
    4    Thu
    Name: Day of Week, dtype: object



** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **


```python
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='rainbow')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
```




    <matplotlib.legend.Legend at 0x207c5bf6ac8>




![png](output_32_1.png)


**Now do the same for Month:**


```python
sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
```




    <matplotlib.legend.Legend at 0x207c3e4f128>




![png](output_34_1.png)


**Did you notice something strange about the Plot?**

_____

** You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas... **

** Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame. **


```python
byMonth = df.groupby('Month').count()
byMonth.head()
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
      <th>lat</th>
      <th>lng</th>
      <th>desc</th>
      <th>zip</th>
      <th>title</th>
      <th>timeStamp</th>
      <th>twp</th>
      <th>addr</th>
      <th>e</th>
      <th>Reason</th>
      <th>Hour</th>
      <th>Day of Week</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>13205</td>
      <td>13205</td>
      <td>13205</td>
      <td>11527</td>
      <td>13205</td>
      <td>13205</td>
      <td>13203</td>
      <td>13096</td>
      <td>13205</td>
      <td>13205</td>
      <td>13205</td>
      <td>13205</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11467</td>
      <td>11467</td>
      <td>11467</td>
      <td>9930</td>
      <td>11467</td>
      <td>11467</td>
      <td>11465</td>
      <td>11396</td>
      <td>11467</td>
      <td>11467</td>
      <td>11467</td>
      <td>11467</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11101</td>
      <td>11101</td>
      <td>11101</td>
      <td>9755</td>
      <td>11101</td>
      <td>11101</td>
      <td>11092</td>
      <td>11059</td>
      <td>11101</td>
      <td>11101</td>
      <td>11101</td>
      <td>11101</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11326</td>
      <td>11326</td>
      <td>11326</td>
      <td>9895</td>
      <td>11326</td>
      <td>11326</td>
      <td>11323</td>
      <td>11283</td>
      <td>11326</td>
      <td>11326</td>
      <td>11326</td>
      <td>11326</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11423</td>
      <td>11423</td>
      <td>11423</td>
      <td>9946</td>
      <td>11423</td>
      <td>11423</td>
      <td>11420</td>
      <td>11378</td>
      <td>11423</td>
      <td>11423</td>
      <td>11423</td>
      <td>11423</td>
    </tr>
  </tbody>
</table>
</div>



** Now create a simple plot off of the dataframe indicating the count of calls per month. **


```python
sns.set_style('whitegrid')
byMonth['twp'].plot(figsize=(10,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x207c3ae00b8>




![png](output_39_1.png)


** Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **


```python
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
```




    <seaborn.axisgrid.FacetGrid at 0x207c4706e80>




![png](output_41_1.png)



```python
byMonth.reset_index().head(1)
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
      <th>Month</th>
      <th>lat</th>
      <th>lng</th>
      <th>desc</th>
      <th>zip</th>
      <th>title</th>
      <th>timeStamp</th>
      <th>twp</th>
      <th>addr</th>
      <th>e</th>
      <th>Reason</th>
      <th>Hour</th>
      <th>Day of Week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>13205</td>
      <td>13205</td>
      <td>13205</td>
      <td>11527</td>
      <td>13205</td>
      <td>13205</td>
      <td>13203</td>
      <td>13096</td>
      <td>13205</td>
      <td>13205</td>
      <td>13205</td>
      <td>13205</td>
    </tr>
  </tbody>
</table>
</div>




```python
byMonth.head(1)
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
      <th>lat</th>
      <th>lng</th>
      <th>desc</th>
      <th>zip</th>
      <th>title</th>
      <th>timeStamp</th>
      <th>twp</th>
      <th>addr</th>
      <th>e</th>
      <th>Reason</th>
      <th>Hour</th>
      <th>Day of Week</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>13205</td>
      <td>13205</td>
      <td>13205</td>
      <td>11527</td>
      <td>13205</td>
      <td>13205</td>
      <td>13203</td>
      <td>13096</td>
      <td>13205</td>
      <td>13205</td>
      <td>13205</td>
      <td>13205</td>
    </tr>
  </tbody>
</table>
</div>



**Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 


```python
df['Date'] = df['timeStamp'].apply(lambda x:x.date())
```

** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**


```python
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()
```


![png](output_47_0.png)


** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**


```python
df[df['Reason'] == 'Traffic'].groupby('Date').count()['twp'].plot()
plt.tight_layout()
plt.title('Traffic')
```




    Text(0.5,1,'Traffic')




![png](output_49_1.png)



```python
df[df['Reason'] == 'Fire'].groupby('Date').count()['twp'].plot()
plt.tight_layout()
plt.title('Fire')
```




    Text(0.5,1,'Fire')




![png](output_50_1.png)



```python
df[df['Reason'] == 'EMS'].groupby('Date').count()['twp'].plot()
plt.tight_layout()
plt.title('EMS')
```




    Text(0.5,1,'EMS')




![png](output_51_1.png)


____
** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**


```python
df.groupby(['Hour','Day of Week']).count()['Reason'].unstack(level=1)
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
      <th>Day of Week</th>
      <th>Fri</th>
      <th>Mon</th>
      <th>Sat</th>
      <th>Sun</th>
      <th>Thu</th>
      <th>Tue</th>
      <th>Wed</th>
    </tr>
    <tr>
      <th>Hour</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>275</td>
      <td>282</td>
      <td>375</td>
      <td>383</td>
      <td>278</td>
      <td>269</td>
      <td>250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>235</td>
      <td>221</td>
      <td>301</td>
      <td>306</td>
      <td>202</td>
      <td>240</td>
      <td>216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>191</td>
      <td>201</td>
      <td>263</td>
      <td>286</td>
      <td>233</td>
      <td>186</td>
      <td>189</td>
    </tr>
    <tr>
      <th>3</th>
      <td>175</td>
      <td>194</td>
      <td>260</td>
      <td>268</td>
      <td>159</td>
      <td>170</td>
      <td>209</td>
    </tr>
    <tr>
      <th>4</th>
      <td>201</td>
      <td>204</td>
      <td>224</td>
      <td>242</td>
      <td>182</td>
      <td>209</td>
      <td>156</td>
    </tr>
    <tr>
      <th>5</th>
      <td>194</td>
      <td>267</td>
      <td>231</td>
      <td>240</td>
      <td>203</td>
      <td>239</td>
      <td>255</td>
    </tr>
    <tr>
      <th>6</th>
      <td>372</td>
      <td>397</td>
      <td>257</td>
      <td>300</td>
      <td>362</td>
      <td>415</td>
      <td>410</td>
    </tr>
    <tr>
      <th>7</th>
      <td>598</td>
      <td>653</td>
      <td>391</td>
      <td>402</td>
      <td>570</td>
      <td>655</td>
      <td>701</td>
    </tr>
    <tr>
      <th>8</th>
      <td>742</td>
      <td>819</td>
      <td>459</td>
      <td>483</td>
      <td>777</td>
      <td>889</td>
      <td>875</td>
    </tr>
    <tr>
      <th>9</th>
      <td>752</td>
      <td>786</td>
      <td>640</td>
      <td>620</td>
      <td>828</td>
      <td>880</td>
      <td>808</td>
    </tr>
    <tr>
      <th>10</th>
      <td>803</td>
      <td>793</td>
      <td>697</td>
      <td>643</td>
      <td>837</td>
      <td>840</td>
      <td>800</td>
    </tr>
    <tr>
      <th>11</th>
      <td>859</td>
      <td>822</td>
      <td>769</td>
      <td>693</td>
      <td>773</td>
      <td>838</td>
      <td>789</td>
    </tr>
    <tr>
      <th>12</th>
      <td>885</td>
      <td>893</td>
      <td>801</td>
      <td>771</td>
      <td>889</td>
      <td>887</td>
      <td>903</td>
    </tr>
    <tr>
      <th>13</th>
      <td>890</td>
      <td>842</td>
      <td>831</td>
      <td>679</td>
      <td>936</td>
      <td>917</td>
      <td>872</td>
    </tr>
    <tr>
      <th>14</th>
      <td>932</td>
      <td>869</td>
      <td>789</td>
      <td>684</td>
      <td>876</td>
      <td>943</td>
      <td>904</td>
    </tr>
    <tr>
      <th>15</th>
      <td>980</td>
      <td>913</td>
      <td>796</td>
      <td>691</td>
      <td>969</td>
      <td>938</td>
      <td>867</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1039</td>
      <td>989</td>
      <td>848</td>
      <td>663</td>
      <td>935</td>
      <td>1026</td>
      <td>990</td>
    </tr>
    <tr>
      <th>17</th>
      <td>980</td>
      <td>997</td>
      <td>757</td>
      <td>714</td>
      <td>1013</td>
      <td>1019</td>
      <td>1037</td>
    </tr>
    <tr>
      <th>18</th>
      <td>820</td>
      <td>885</td>
      <td>778</td>
      <td>670</td>
      <td>810</td>
      <td>905</td>
      <td>894</td>
    </tr>
    <tr>
      <th>19</th>
      <td>696</td>
      <td>746</td>
      <td>696</td>
      <td>655</td>
      <td>698</td>
      <td>731</td>
      <td>686</td>
    </tr>
    <tr>
      <th>20</th>
      <td>667</td>
      <td>613</td>
      <td>628</td>
      <td>537</td>
      <td>617</td>
      <td>647</td>
      <td>668</td>
    </tr>
    <tr>
      <th>21</th>
      <td>559</td>
      <td>497</td>
      <td>572</td>
      <td>461</td>
      <td>553</td>
      <td>571</td>
      <td>575</td>
    </tr>
    <tr>
      <th>22</th>
      <td>514</td>
      <td>472</td>
      <td>506</td>
      <td>415</td>
      <td>424</td>
      <td>462</td>
      <td>490</td>
    </tr>
    <tr>
      <th>23</th>
      <td>474</td>
      <td>325</td>
      <td>467</td>
      <td>330</td>
      <td>354</td>
      <td>274</td>
      <td>335</td>
    </tr>
  </tbody>
</table>
</div>




```python
dayHour = df.groupby(['Hour','Day of Week']).count()['Reason'].unstack(level=0)
```

** Now create a HeatMap using this new DataFrame. **


```python
plt.figure(figsize=(12,8))
sns.heatmap(dayHour,cmap='viridis')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x207c8c6a780>




![png](output_56_1.png)


** Now create a clustermap using this DataFrame. **


```python
sns.clustermap(dayHour)
```




    <seaborn.matrix.ClusterGrid at 0x207c8d28dd8>




![png](output_58_1.png)


** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **


```python
dayMonth = df.groupby(['Month','Day of Week']).count()['Reason'].unstack(level=0)
```


```python
plt.figure(figsize=(10,8))
sns.heatmap(dayMonth)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x207c9aa7a20>




![png](output_61_1.png)



```python
sns.clustermap(dayMonth)
```




    <seaborn.matrix.ClusterGrid at 0x207cb9c8eb8>




![png](output_62_1.png)

