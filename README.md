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


```python
y
```

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

# Instantiate

# Fit

```

#### `DecisionTreeClassifier` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html))

Import the classifier, instantiate an object called `clf` (short for "classifier") with default parameters, and `fit` the classifier on `X` and `y`.


```python
# Import

# Instantiate

# Fit

```

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
# Your code here
```

#### Now, call the `transform` method on the transformer and pass in `X`. Assign the result to `X_scaled`


```python
# Your code here
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
# Your code here
```

#### Now, call the `predict` method on the predictor, passing in `X`. Assign the result to `y_pred`


```python
# Your code here
```

#### Now, call the `score` method on the predictor, passing in `X` and `y`


```python
# Your code here
```

#### What does that score represent? Write your answer below


```python
"""
Your answer here
"""
```

## Summary

In this lab, you practiced identifying mutable and immutable types as well as identifying transformers, predictors, and models using scikit-learn. You also instantiated scikit-learn objects, invoked the most common scikit-learn methods, and accessed some scikit-learn attributes.
