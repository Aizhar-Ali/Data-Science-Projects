

# Support Vector Machines with Python
## Import Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Get the Data

We'll use the built in breast cancer dataset from Scikit Learn. We can get with the load function:


```python
from sklearn.datasets import load_breast_cancer
```


```python
cancer = load_breast_cancer()
```

The data set is presented in a dictionary form:


```python
cancer.keys()
```




    dict_keys(['DESCR', 'target', 'data', 'target_names', 'feature_names'])



We can grab information and arrays out of this dictionary to set up our data frame and understanding of the features:


```python
print(cancer['DESCR'])
```

    Breast Cancer Wisconsin (Diagnostic) Database
    
    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry 
            - fractal dimension ("coastline approximation" - 1)
            
            The mean, standard error, and "worst" or largest (mean of the three
            largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 3 is Mean Radius, field
            13 is Radius SE, field 23 is Worst Radius.
            
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ======= ========
                                               Min     Max
        ===================================== ======= ========
        radius (mean):                         6.981   28.11
        texture (mean):                        9.71    39.28
        perimeter (mean):                      43.79   188.5
        area (mean):                           143.5   2501.0
        smoothness (mean):                     0.053   0.163
        compactness (mean):                    0.019   0.345
        concavity (mean):                      0.0     0.427
        concave points (mean):                 0.0     0.201
        symmetry (mean):                       0.106   0.304
        fractal dimension (mean):              0.05    0.097
        radius (standard error):               0.112   2.873
        texture (standard error):              0.36    4.885
        perimeter (standard error):            0.757   21.98
        area (standard error):                 6.802   542.2
        smoothness (standard error):           0.002   0.031
        compactness (standard error):          0.002   0.135
        concavity (standard error):            0.0     0.396
        concave points (standard error):       0.0     0.053
        symmetry (standard error):             0.008   0.079
        fractal dimension (standard error):    0.001   0.03
        radius (worst):                        7.93    36.04
        texture (worst):                       12.02   49.54
        perimeter (worst):                     50.41   251.2
        area (worst):                          185.2   4254.0
        smoothness (worst):                    0.071   0.223
        compactness (worst):                   0.027   1.058
        concavity (worst):                     0.0     1.252
        concave points (worst):                0.0     0.291
        symmetry (worst):                      0.156   0.664
        fractal dimension (worst):             0.055   0.208
        ===================================== ======= ========
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    A few of the images can be found at
    http://www.cs.wisc.edu/~street/images/
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    References
    ----------
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870, 
         San Jose, CA, 1993. 
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
         prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
         163-171.
    
    


```python
cancer['feature_names']
```




    array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
           'mean smoothness', 'mean compactness', 'mean concavity',
           'mean concave points', 'mean symmetry', 'mean fractal dimension',
           'radius error', 'texture error', 'perimeter error', 'area error',
           'smoothness error', 'compactness error', 'concavity error',
           'concave points error', 'symmetry error', 'fractal dimension error',
           'worst radius', 'worst texture', 'worst perimeter', 'worst area',
           'worst smoothness', 'worst compactness', 'worst concavity',
           'worst concave points', 'worst symmetry', 'worst fractal dimension'], 
          dtype='<U23')



## Set up DataFrame


```python
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 30 columns):
    mean radius                569 non-null float64
    mean texture               569 non-null float64
    mean perimeter             569 non-null float64
    mean area                  569 non-null float64
    mean smoothness            569 non-null float64
    mean compactness           569 non-null float64
    mean concavity             569 non-null float64
    mean concave points        569 non-null float64
    mean symmetry              569 non-null float64
    mean fractal dimension     569 non-null float64
    radius error               569 non-null float64
    texture error              569 non-null float64
    perimeter error            569 non-null float64
    area error                 569 non-null float64
    smoothness error           569 non-null float64
    compactness error          569 non-null float64
    concavity error            569 non-null float64
    concave points error       569 non-null float64
    symmetry error             569 non-null float64
    fractal dimension error    569 non-null float64
    worst radius               569 non-null float64
    worst texture              569 non-null float64
    worst perimeter            569 non-null float64
    worst area                 569 non-null float64
    worst smoothness           569 non-null float64
    worst compactness          569 non-null float64
    worst concavity            569 non-null float64
    worst concave points       569 non-null float64
    worst symmetry             569 non-null float64
    worst fractal dimension    569 non-null float64
    dtypes: float64(30)
    memory usage: 133.4 KB
    


```python
cancer['target']
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
           1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0,
           1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1,
           0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
           0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
           0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
           0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
           0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
           1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
           1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
           1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
           0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
           1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
           0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1,
           1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
           1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
           1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])




```python
df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
```

Now let's actually check out the dataframe!


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



## Train Test Split


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)
```

# Train the Support Vector Classifier


```python
from sklearn.svm import SVC
```


```python
model = SVC()
```


```python
model.fit(X_train,y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



## Predictions and Evaluations

Now let's predict using the trained model.


```python
predictions = model.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(confusion_matrix(y_test,predictions))
```

    [[  0  66]
     [  0 105]]
    


```python
print(classification_report(y_test,predictions))
```

                 precision    recall  f1-score   support
    
              0       0.00      0.00      0.00        66
              1       0.61      1.00      0.76       105
    
    avg / total       0.38      0.61      0.47       171
    
    

    /Users/marci/anaconda/lib/python3.5/site-packages/sklearn/metrics/classification.py:1074: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    

Woah! Notice that we are classifying everything into a single class! This means our model needs to have it parameters adjusted (it may also help to normalize the data).

# Gridsearch

Finding the right parameters (like what C or gamma values to use) is a tricky task! But luckily, we can be a little lazy and just try a bunch of combinations and see what works best! This idea of creating a 'grid' of parameters and just trying out all the possible combinations is called a Gridsearch, this method is common enough that Scikit-learn has this functionality built in with GridSearchCV! The CV stands for cross-validation which is the

GridSearchCV takes a dictionary that describes the parameters that should be tried and a model to train. The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested. 


```python
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
```


```python
from sklearn.model_selection import GridSearchCV
```

One of the great things about GridSearchCV is that it is a meta-estimator. It takes an estimator like SVC, and creates a new estimator, that behaves exactly the same - in this case, like a classifier. You should add refit=True and choose verbose to whatever number you want, higher the number, the more verbose (verbose just means the text output describing the process).


```python
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
```

What fit does is a bit more involved then usual. First, it runs the same loop with cross-validation, to find the best parameter combination. Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to built a single new model using the best parameter setting.


```python
# May take awhile!
grid.fit(X_train,y_train)
```

    Fitting 3 folds for each of 25 candidates, totalling 75 fits
    [CV] gamma=1, C=0.1, kernel=rbf ......................................
    [CV] ............. gamma=1, C=0.1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=1, C=0.1, kernel=rbf ......................................
    [CV] ............. gamma=1, C=0.1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=1, C=0.1, kernel=rbf ......................................
    [CV] ............. gamma=1, C=0.1, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.1, C=0.1, kernel=rbf ....................................
    [CV] ........... gamma=0.1, C=0.1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.1, C=0.1, kernel=rbf ....................................
    [CV] ........... gamma=0.1, C=0.1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.1, C=0.1, kernel=rbf ....................................
    [CV] ........... gamma=0.1, C=0.1, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.01, C=0.1, kernel=rbf ...................................
    [CV] .......... gamma=0.01, C=0.1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.01, C=0.1, kernel=rbf ...................................
    [CV] .......... gamma=0.01, C=0.1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.01, C=0.1, kernel=rbf ...................................
    [CV] .......... gamma=0.01, C=0.1, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.001, C=0.1, kernel=rbf ..................................
    [CV] ......... gamma=0.001, C=0.1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.001, C=0.1, kernel=rbf ..................................
    [CV] ......... gamma=0.001, C=0.1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.001, C=0.1, kernel=rbf ..................................
    [CV] ......... gamma=0.001, C=0.1, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.0001, C=0.1, kernel=rbf .................................
    [CV] ........ gamma=0.0001, C=0.1, kernel=rbf, score=0.902256 -   0.0s
    [CV] gamma=0.0001, C=0.1, kernel=rbf .................................
    [CV] ........ gamma=0.0001, C=0.1, kernel=rbf, score=0.962406 -   0.0s
    [CV] gamma=0.0001, C=0.1, kernel=rbf .................................
    [CV] ........ gamma=0.0001, C=0.1, kernel=rbf, score=0.916667 -   0.0s
    [CV] gamma=1, C=1, kernel=rbf ........................................
    [CV] ............... gamma=1, C=1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=1, C=1, kernel=rbf ........................................
    [CV] ............... gamma=1, C=1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=1, C=1, kernel=rbf ........................................
    [CV] ............... gamma=1, C=1, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.1, C=1, kernel=rbf ......................................
    [CV] ............. gamma=0.1, C=1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.1, C=1, kernel=rbf ......................................
    [CV] ............. gamma=0.1, C=1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.1, C=1, kernel=rbf ......................................
    [CV] ............. gamma=0.1, C=1, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.01, C=1, kernel=rbf .....................................
    [CV] ............ gamma=0.01, C=1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.01, C=1, kernel=rbf .....................................
    [CV] ............ gamma=0.01, C=1, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.01, C=1, kernel=rbf .....................................
    [CV] ............ gamma=0.01, C=1, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.001, C=1, kernel=rbf ....................................
    [CV] ........... gamma=0.001, C=1, kernel=rbf, score=0.902256 -   0.0s
    [CV] gamma=0.001, C=1, kernel=rbf ....................................
    [CV] ........... gamma=0.001, C=1, kernel=rbf, score=0.939850 -   0.0s
    [CV] gamma=0.001, C=1, kernel=rbf ....................................
    [CV] ........... gamma=0.001, C=1, kernel=rbf, score=0.954545 -   0.0s
    [CV] gamma=0.0001, C=1, kernel=rbf ...................................
    [CV] .......... gamma=0.0001, C=1, kernel=rbf, score=0.939850 -   0.0s
    [CV] gamma=0.0001, C=1, kernel=rbf ...................................
    [CV] .......... gamma=0.0001, C=1, kernel=rbf, score=0.969925 -   0.0s
    [CV] gamma=0.0001, C=1, kernel=rbf ...................................
    [CV] .......... gamma=0.0001, C=1, kernel=rbf, score=0.946970 -   0.0s
    [CV] gamma=1, C=10, kernel=rbf .......................................
    [CV] .............. gamma=1, C=10, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=1, C=10, kernel=rbf .......................................
    [CV] .............. gamma=1, C=10, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=1, C=10, kernel=rbf .......................................
    [CV] .............. gamma=1, C=10, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.1, C=10, kernel=rbf .....................................
    [CV] ............ gamma=0.1, C=10, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.1, C=10, kernel=rbf .....................................
    [CV] ............ gamma=0.1, C=10, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.1, C=10, kernel=rbf .....................................
    [CV] ............ gamma=0.1, C=10, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.01, C=10, kernel=rbf ....................................
    [CV] ........... gamma=0.01, C=10, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.01, C=10, kernel=rbf ....................................
    [CV] ........... gamma=0.01, C=10, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.01, C=10, kernel=rbf ....................................
    [CV] ........... gamma=0.01, C=10, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.001, C=10, kernel=rbf ...................................
    [CV] .......... gamma=0.001, C=10, kernel=rbf, score=0.894737 -   0.0s
    [CV] gamma=0.001, C=10, kernel=rbf ...................................
    [CV] .......... gamma=0.001, C=10, kernel=rbf, score=0.932331 -   0.0s
    [CV] gamma=0.001, C=10, kernel=rbf ...................................
    [CV] .......... gamma=0.001, C=10, kernel=rbf, score=0.916667 -   0.0s
    [CV] gamma=0.0001, C=10, kernel=rbf ..................................
    [CV] ......... gamma=0.0001, C=10, kernel=rbf, score=0.932331 -   0.0s
    [CV] gamma=0.0001, C=10, kernel=rbf ..................................
    [CV] ......... gamma=0.0001, C=10, kernel=rbf, score=0.969925 -   0.0s
    [CV] gamma=0.0001, C=10, kernel=rbf ..................................
    [CV] ......... gamma=0.0001, C=10, kernel=rbf, score=0.962121 -   0.0s
    [CV] gamma=1, C=100, kernel=rbf ......................................
    [CV] ............. gamma=1, C=100, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=1, C=100, kernel=rbf ......................................
    [CV] ............. gamma=1, C=100, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=1, C=100, kernel=rbf ......................................
    [CV] ............. gamma=1, C=100, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.1, C=100, kernel=rbf ....................................
    [CV] ........... gamma=0.1, C=100, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.1, C=100, kernel=rbf ....................................
    [CV] ........... gamma=0.1, C=100, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.1, C=100, kernel=rbf ....................................
    [CV] ........... gamma=0.1, C=100, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.01, C=100, kernel=rbf ...................................
    [CV] .......... gamma=0.01, C=100, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.01, C=100, kernel=rbf ...................................
    [CV] .......... gamma=0.01, C=100, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.01, C=100, kernel=rbf ...................................
    [CV] .......... gamma=0.01, C=100, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.001, C=100, kernel=rbf ..................................
    [CV] ......... gamma=0.001, C=100, kernel=rbf, score=0.894737 -   0.0s
    [CV] gamma=0.001, C=100, kernel=rbf ..................................
    [CV] ......... gamma=0.001, C=100, kernel=rbf, score=0.932331 -   0.0s
    [CV] gamma=0.001, C=100, kernel=rbf ..................................
    [CV] ......... gamma=0.001, C=100, kernel=rbf, score=0.916667 -   0.0s
    [CV] gamma=0.0001, C=100, kernel=rbf .................................
    [CV] ........ gamma=0.0001, C=100, kernel=rbf, score=0.917293 -   0.0s
    [CV] gamma=0.0001, C=100, kernel=rbf .................................
    [CV] ........ gamma=0.0001, C=100, kernel=rbf, score=0.977444 -   0.0s
    [CV] gamma=0.0001, C=100, kernel=rbf .................................
    [CV] ........ gamma=0.0001, C=100, kernel=rbf, score=0.939394 -   0.0s
    [CV] gamma=1, C=1000, kernel=rbf .....................................
    [CV] ............ gamma=1, C=1000, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=1, C=1000, kernel=rbf .....................................
    [CV] ............ gamma=1, C=1000, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=1, C=1000, kernel=rbf .....................................
    [CV] ............ gamma=1, C=1000, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.1, C=1000, kernel=rbf ...................................
    [CV] .......... gamma=0.1, C=1000, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.1, C=1000, kernel=rbf ...................................
    [CV] .......... gamma=0.1, C=1000, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.1, C=1000, kernel=rbf ...................................
    [CV] .......... gamma=0.1, C=1000, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.01, C=1000, kernel=rbf ..................................
    [CV] ......... gamma=0.01, C=1000, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.01, C=1000, kernel=rbf ..................................
    [CV] ......... gamma=0.01, C=1000, kernel=rbf, score=0.631579 -   0.0s
    [CV] gamma=0.01, C=1000, kernel=rbf ..................................
    [CV] ......... gamma=0.01, C=1000, kernel=rbf, score=0.636364 -   0.0s
    [CV] gamma=0.001, C=1000, kernel=rbf .................................
    [CV] ........ gamma=0.001, C=1000, kernel=rbf, score=0.894737 -   0.0s
    [CV] gamma=0.001, C=1000, kernel=rbf .................................
    [CV] ........ gamma=0.001, C=1000, kernel=rbf, score=0.932331 -   0.0s
    [CV] gamma=0.001, C=1000, kernel=rbf .................................
    [CV] ........ gamma=0.001, C=1000, kernel=rbf, score=0.916667 -   0.0s

    [Parallel(n_jobs=1)]: Done  31 tasks       | elapsed:    0.3s
    [Parallel(n_jobs=1)]: Done  75 out of  75 | elapsed:    0.8s finished
    

    
    [CV] gamma=0.0001, C=1000, kernel=rbf ................................
    [CV] ....... gamma=0.0001, C=1000, kernel=rbf, score=0.909774 -   0.0s
    [CV] gamma=0.0001, C=1000, kernel=rbf ................................
    [CV] ....... gamma=0.0001, C=1000, kernel=rbf, score=0.969925 -   0.0s
    [CV] gamma=0.0001, C=1000, kernel=rbf ................................
    [CV] ....... gamma=0.0001, C=1000, kernel=rbf, score=0.931818 -   0.0s
    




    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'C': [0.1, 1, 10, 100, 1000], 'kernel': ['rbf']},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=3)



You can inspect the best parameters found by GridSearchCV in the best_params_ attribute, and the best estimator in the best\_estimator_ attribute:


```python
grid.best_params_
```




    {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}




```python
grid.best_estimator_
```

Then you can re-run predictions on this grid object just like you would with a normal model.


```python
grid_predictions = grid.predict(X_test)
```


```python
print(confusion_matrix(y_test,grid_predictions))
```

    [[ 60   6]
     [  3 102]]
    


```python
print(classification_report(y_test,grid_predictions))
```

                 precision    recall  f1-score   support
    
              0       0.95      0.91      0.93        66
              1       0.94      0.97      0.96       105
    
    avg / total       0.95      0.95      0.95       171
    
    
