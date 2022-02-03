# OOP with Scikit-Learn - Lab

## Introduction

Now that you have learned some of the basics of object-oriented programming with scikit-learn, let's practice applying it!

## Objectives:

In this lesson you will practice:

* Recall the distinction between mutable and immutable types
* Define the four main inherited object types in scikit-learn
* Instantiate scikit-learn transformers and models
* Invoke scikit-learn methods
* Access scikit-learn attributes

## Mutable and Immutable Types

For each example below, think to yourself whether it is a mutable or immutable type. Then expand the details tag to reveal the answer.

<ol>
    <li>
        <details>
            <summary style="cursor: pointer">Python dictionary (click to reveal)</summary>
            <p><strong>Mutable.</strong> For example, the `update` method can be used to modify values within a dictionary.</p>
            <p></p>
        </details>
    </li>
    <li>
        <details>
            <summary style="cursor: pointer">Python tuple (click to reveal)</summary>
            <p><strong>Immutable.</strong> If you want to create a modified version of a tuple, you need to use <code>=</code> to reassign it.</p>
            <p></p>
        </details>
    </li>
    <li>
        <details>
            <summary style="cursor: pointer">pandas <code>DataFrame</code> (click to reveal)</summary>
            <p><strong>Mutable.</strong> Using the <code>inplace=True</code> argument with various different methods allows you to modify a dataframe in place.</p>
            <p></p>
        </details>
    </li>
    <li>
        <details>
            <summary style="cursor: pointer">scikit-learn <code>OneHotEncoder</code> (click to reveal)</summary>
            <p><strong>Mutable.</strong> Calling the <code>fit</code> method causes the transformer to store information about the data that is passed in, modifying its internal attributes.</p>
            <p></p>
        </details>
    </li>
</ol>

## The Data

For this lab we'll use data from the built-in iris dataset:


```python
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True, as_frame=True)
```


```python
X
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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 4 columns</p>
</div>




```python
y
```




    0      0
    1      0
    2      0
    3      0
    4      0
          ..
    145    2
    146    2
    147    2
    148    2
    149    2
    Name: target, Length: 150, dtype: int64



## Scikit-Learn Classes

For the following exercises, follow the documentation link to understand the class you are working with, but **do not** worry about understanding the underlying algorithm. The goal is just to get used to creating and using these types of objects.

### Estimators

For all estimators, the steps are:

1. Import the class from the `sklearn` library
2. Instantiate an object from the class
3. Pass in the appropriate data to the `fit` method

#### `MinMaxScaler` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html))

Import this scaler, instantiate an object called `scaler` with default parameters, and `fit` the scaler on `X`.


```python
# Import
from sklearn.preprocessing import MinMaxScaler
# Instantiate
scaler = MinMaxScaler()
# Fit
scaler.fit(X)
```




    MinMaxScaler()



#### `DecisionTreeClassifier` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html))

Import the classifier, instantiate an object called `clf` (short for "classifier") with default parameters, and `fit` the classifier on `X` and `y`.


```python
# Import
from sklearn.tree import DecisionTreeClassifier
# Instantiate
clf = DecisionTreeClassifier()
# Fit
clf.fit(X, y)
```




    DecisionTreeClassifier()



### Transformers

One of the two objects instantiated above (`scaler` or `clf`) is a transformer. Which one is it? Consult the documentation.

---

<details>
    <summary style="cursor: pointer">Hint (click to reveal)</summary>
    <p>The class with a <code>transform</code> method is a transformer.</p>
</details>

---

#### Using the transformer, print out two of the fitted attributes along with descriptions from the documentation.

---

<details>
    <summary style="cursor: pointer">Hint (click to reveal)</summary>
    <p>Attributes ending with <code>_</code> are fitted attributes.</p>
</details>


```python
# (Answers will vary)
print("Minimum feature seen in the data:", scaler.data_min_)
print("Maximum feature seen in the data:", scaler.data_max_)
```

    Minimum feature seen in the data: [4.3 2.  1.  0.1]
    Maximum feature seen in the data: [7.9 4.4 6.9 2.5]


#### Now, call the `transform` method on the transformer and pass in `X`. Assign the result to `X_scaled`


```python
X_scaled = scaler.transform(X)
```

### Predictors and Models

The other of the two scikit-learn objects instantiated above (`scaler` or `clf`) is a predictor and a model. Which one is it? Consult the documentation.

---

<details>
    <summary style="cursor: pointer">Hint (click to reveal)</summary>
    <p>The class with a <code>predict</code> method and a <code>score</code> method is a predictor and a model.</p>
</details>

---

#### Using the predictor, print out two of the fitted attributes along with descriptions from the documentation.


```python
# (Answers will vary)
print("Number of classes:", clf.n_classes_)
print("Number of features seen:", clf.n_features_in_)
```

    Number of classes: 3
    Number of features seen: 4


#### Now, call the `predict` method on the predictor, passing in `X`. Assign the result to `y_pred`


```python
y_pred = clf.predict(X)
```

#### Now, call the `score` method on the predictor, passing in `X` and `y`


```python
clf.score(X, y)
```




    1.0



#### What does that score represent? Write your answer below


```python
"""
According to the documentation, this score represents the mean accuracy
"""
```

## Summary

In this lab, you practiced identifying mutable and immutable types as well as identifying transformers, predictors, and models using scikit-learn. You also instantiated scikit-learn objects, invoked the most common scikit-learn methods, and accessed some scikit-learn attributes.
