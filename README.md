
# K-Nearest Neighbors

![wilson](img/wilson.jpg)

# Agenda
1. FSM and metric review
2. KNN Under the Hood: Voting for K
3. Different types of distance
4. Importance of Scaling
5. Let's unpack: KNN is a supervised, non-parametric, descriminative, lazy-learning algorithm
6. Tuning K and the BV Tradeoff

KNearest Neighbors is our second classification algorithm in our toolbelt added to our logistic regression classifier.

If we remember, logistic regression is a supervised, parametric, discriminative model.

KNN is a **supervised, non-parametric, discriminative, lazy-learning algorithm.**



```python
# This is always a good idea
%load_ext autoreload
%autoreload 2

from src.student_caller import one_random_student
from src.student_list import student_first_names

import os
import sys
module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
```

## Let's load in the Titanic dataset

![titanic](https://media.giphy.com/media/uhB0n3Eac8ybe/giphy.gif)


```python
titanic = pd.read_csv('data/cleaned_titanic.csv')
titanic = titanic.iloc[:,:-2]
titanic.head()
```

#### For visualization purposes, we will use only two features for our first model


```python
X = titanic[['Age', 'Fare']]
y = titanic['Survived']
y.value_counts()
```

Titanic is a binary classification problem, with our target being the Survived feature


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = .25)
```

#### Then perform another tts, and put aside the test set from above until the end

We will hold of from KFold or crossval for now, so that our notebook is more comprehensible.


```python
X_t, X_val, y_t, y_val = train_test_split(X_train,y_train, random_state=42, test_size = .25)
```


```python
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
from src.confusion import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

ss = StandardScaler()
X_t = ss.fit_transform(X_t)
X_val = ss.transform(X_val)

knn.fit(X_t, y_t)
print(f"training accuracy: {knn.score(X_t, y_t)}")
print(f"Val accuracy: {knn.score(X_val, y_val)}")

y_hat = knn.predict(X_val)


```


```python
plot_confusion_matrix(confusion_matrix(y_val, y_hat), classes=['Perished', 'Survived'])
```

# Quick review of confusion matrix and our metrics: 
  


```python
question = 'How many true positives?'
one_random_student(student_first_names)

```


```python
question = 'How many true negatives?'
one_random_student(student_first_names)

```


```python
question = 'How many false positives?'
one_random_student(student_first_names)

```


```python
question = 'How many  how many false negatives?'
one_random_student(student_first_names)

```


```python
question = 'Which will be higher: precision or recall'
one_random_student(student_first_names)

```

# 2. KNN Under the Hood: Voting for K

For visualization purposes, let's pull out a small subset of our training data, and create a model using only two dimensions: Age and Fare.



```python
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = .25)
X_t, X_val, y_t, y_val = train_test_split(X_train,y_train, random_state=42, test_size = .25)
```


```python
import seaborn as sns

X_for_viz = X_t.sample(15, random_state=40)
y_for_viz = y_t[X_for_viz.index]

fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(X_for_viz['Age'], X_for_viz['Fare'], 
                hue=y_for_viz, palette={0:'red', 1:'green'}, 
                s=200, ax=ax)

ax.set_xlim(0,80)
ax.set_ylim(0,80)
plt.legend()
plt.title('Subsample of Training Data')
```

The KNN algorithm works by simply storing the training set in memory, then measuring the distance from the training points to a a new point.

Let's drop a point from our validation set into the plot above.


```python
X_for_viz = X_t.sample(15, random_state=40)
y_for_viz = y_t[X_for_viz.index]

fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(X_for_viz['Age'], X_for_viz['Fare'], hue=y_for_viz, palette={0:'red', 1:'green'}, s=200, ax=ax)

plt.legend()

#################^^^Old code^^^##############
####################New code#################

# Let's take one sample from our validation set and plot it
new_x = pd.DataFrame(X_val.loc[484]).T
new_y = y_val[new_x.index]

sns.scatterplot(new_x['Age'], new_x['Fare'], color='blue', s=200, ax=ax, label='New', marker='P')

ax.set_xlim(0,100)
ax.set_ylim(0,100)
```


```python
new_x.head()
```

Then, KNN finds the K nearest points. K corresponds to the n_neighbors parameter defined when we instantiate the classifier object.


```python
knn = KNeighborsClassifier(n_neighbors=1)
```

Let's fit our training data, then predict what our validation point will be based on the closest 1 neighbor.

# Chat poll: What will our 1 neighbor KNN classifier predict our new point to be?




```python
knn.fit(X_for_viz, y_for_viz)
knn.predict(new_x)
```

When we raise the value of K, KNN acts democratically.  It finds the K closest points, and takes a vote based on the labels.

Let's raise K to 3.


```python
knn = KNeighborsClassifier(n_neighbors=3)
```

# Chat poll: What will our 3 neighbor KNN classifier predict our new point to be?



```python
knn.fit(X_for_viz, y_for_viz)
knn.predict(new_x)
```

It is a bit harder to tell what which points are closest by eye.

Let's update our plot to add indexes.


```python
X_for_viz = X_t.sample(15, random_state=40)
y_for_viz = y_t[X_for_viz.index]

fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(X_for_viz['Age'], X_for_viz['Fare'], hue=y_for_viz, 
                palette={0:'red', 1:'green'}, s=200, ax=ax)


# Now let's take another sample

# new_x = X_val.sample(1, random_state=33)
new_x = pd.DataFrame(X_val.loc[484]).T
new_x.columns = ['Age','Fare']
new_y = y_val[new_x.index]

print(new_x)
sns.scatterplot(new_x['Age'], new_x['Fare'], color='blue', s=200, ax=ax, label='New', marker='P')
ax.set_xlim(0,100)
ax.set_ylim(0,100)
plt.legend()

#################^^^Old code^^^##############
####################New code#################

# add annotations one by one with a loop
for index in X_for_viz.index:
    ax.text(X_for_viz.Age[index]+0.7, X_for_viz.Fare[index], s=index, horizontalalignment='left', size='medium', color='black', weight='semibold')
 


```

We can the sklearn NearestNeighors object to see the exact calculations.


```python
from sklearn.neighbors import NearestNeighbors

df_for_viz = pd.merge(X_for_viz, y_for_viz, left_index=True, right_index=True)
neighbor = NearestNeighbors(3)
neighbor.fit(X_for_viz)
nearest = neighbor.kneighbors(new_x)

nearest
```


```python
df_for_viz.iloc[nearest[1][0]]
```


```python
new_x
```

# Chat poll: What will our 5 neighbor KNN classifier predict our new point to be?


```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_for_viz, y_for_viz)
knn.predict(new_x)
```

Let's iterate through K, 1 through 10, and see the predictions.


```python
for k in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_for_viz, y_for_viz)
    print(knn.predict(new_x))

```

What K was correct?


```python
new_y
```

# 3. Different types of distance

How did the algo calculate those distances? 


```python
nearest
```

### Euclidean Distance

**Euclidean distance** refers to the distance between two points. These points can be in different dimensional space and are represented by different forms of coordinates. In one-dimensional space, the points are just on a straight number line.


### Measuring distance in a 2-d Space

In two-dimensional space, the coordinates are given as points on the x- and y-axes

![alt text](img/euclidean_2d.png)
### Measuring distance in a 3-d Space

In three-dimensional space, x-, y- and z-axes are used. 

$$\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 +  (z_1-z_2)^2}$$
![alt text](img/vectorgraph.jpg)


```python
# Let's reproduce those numbers:
nearest
```


```python
df_for_viz.iloc[11]

```


```python
new_x
```


```python
def euclid(train_X, val_X):
    """
    :param train_X: one record from the training set
                    (type series or dataframe including target (survived))
    :param val_X: one record from the validation set
                    series or dataframe include target (survived)
    :return: The euclidean distance between train_X and val_X
    """
    diff = train_X - val_X

    # Remove survived column
    diff = diff.iloc[:, :-1]

    dist = np.sqrt((diff ** 2).sum(axis=1))

    return dist

    
```


```python
euclid(df_for_viz.iloc[11], new_x)
```


```python
euclid(df_for_viz.iloc[5], new_x)
```


```python
euclid(df_for_viz.iloc[0], new_x)
```

# Manhattan distance

Manhattan distance is the distance measured if you walked along a city block instead of a straight line. 

> if 𝑥=(𝑎,𝑏) and 𝑦=(𝑐,𝑑),  
> Manhattan distance = |𝑎−𝑐|+|𝑏−𝑑|

![](img/manhattan.png)

# Pairs: 

Write an function that calculates Manhattan distance between two points

Calculate the distance between new_X and the 15 training points.

Based on 5 K, determine what decision a KNN algorithm would make if it used Manhattan distance.





```python
# your code here

```


```python
#__SOLUTION__
def manhattan(train_x, val_x):
    
    """
    :param train_X: one record from the training set

    :param val_X: one record from the validation set
                
    :return: the Manhattan distance between train_X and val_X
    """
    
    
    manhattan_dist = np.abs(train_x['Age'] - val_x['Age'])\
                     + np.abs(train_x['Fare'] - val_x["Fare"])
        
    return manhattan_dist.values[0]

```


```python
manh_diffs = []
for index in df_for_viz.index:
    manh_diffs.append(manhattan(df_for_viz.loc[index], new_x))
    
sorted(manh_diffs)
```


```python
from sklearn.neighbors import NearestNeighbors

neighbor = NearestNeighbors(10, p=1)
neighbor.fit(X_for_viz)
nearest = neighbor.kneighbors(new_x)

nearest
```


```python
df_for_viz.iloc[nearest[1][0]]
```


```python
from src.plot_train import plot_train
plot_train(X_t, y_t, X_val, y_val)
```

If we change the distance metric, our prediction should change for K = 5.


```python
from sklearn.neighbors import KNeighborsClassifier

knn_euc = KNeighborsClassifier(5, p=2)
knn_euc.fit(X_for_viz, y_for_viz)
knn_euc.predict(new_x)
```


```python
knn_man = KNeighborsClassifier(5, p=1)
knn_man.fit(X_for_viz, y_for_viz)
knn_man.predict(new_x)
```


```python
# Which got it right? 
new_y
```

# 4. Importance of Scaling

You may have suspected that we were leaving something out. For any distance based algorithms, scaling is very important.  Look at how the shape of array changes before and after scaling.

![non-normal](img/nonnormal.png)

![normal](img/normalized.png)

Let's look at our data for viz dataset


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = .25)
X_t, X_val, y_t, y_val = train_test_split(X_train,y_train, random_state=42, test_size = .25)

knn = KNeighborsClassifier()

ss = StandardScaler()
X_ind = X_t.index
X_col = X_t.columns

X_t_s = pd.DataFrame(ss.fit_transform(X_t))
X_t_s.index = X_ind
X_t_s.columns = X_col

X_v_ind = X_val.index
X_val_s = pd.DataFrame(ss.transform(X_val))
X_val_s.index = X_v_ind
X_val_s.columns = X_col

knn.fit(X_t_s, y_t)
print(f"training accuracy: {knn.score(X_t_s, y_t)}")
print(f"Val accuracy: {knn.score(X_val_s, y_val)}")

y_hat = knn.predict(X_val_s)


```


```python
plot_train(X_t, y_t, X_val, y_val)
plot_train(X_t_s, y_t, X_val_s, y_val, -2,2, text_pos=.1 )
```

Look at how much that changes things.

Look at 166 and 150.  
Look at 827 to 786

Now let's run our classifier on scaled data and compare to unscaled.


```python
from src.k_classify import predict_one

titanic = pd.read_csv('data/cleaned_titanic.csv')
X = titanic[['Age', 'Fare']]
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = .25)
X_t, X_val, y_t, y_val = train_test_split(X_train,y_train, random_state=42, test_size = .25)

predict_one(X_t, X_val, y_t, y_val)
```


```python
ss = StandardScaler()

X_t_s = pd.DataFrame(ss.fit_transform(X_t))
X_t_s.index = X_t.index
X_t_s.columns = X_t.columns

X_val_s = pd.DataFrame(ss.transform(X_val))
X_val_s.index = X_val.index
X_val_s.columns = X_val.columns


predict_one(X_t_s, X_val_s, y_t, y_val)
```

# 5. Let's unpack: KNN is a supervised, non-parametric, descriminative, lazy-learning algorithm

## Supervised
You should be very comfortable with the idea of supervised learning by now.  Supervised learning involves labels.  KNN needs labels for the voting process.



# Non-parametric

Let's look at the fit KNN classifier.


```python
knn = KNeighborsClassifier()
knn.__dict__
```


```python
knn.fit(X_t_s, y_t)
knn.__dict__
```

What do you notice? No coefficients! In linear and logistic regression, fitting the model involves calculation of parameters associated with a best fit hyperplane.

KNN does not use such a process.  It simply calculates the distance from each point, and votes.

# Descriminative

### Example training data

This example uses a multi-class problem and each color represents a different class. 


### KNN classification map (K=1)

![1NN classification map](img/04_1nn_map.png)

### KNN classification map (K=5)

![5NN classification map](img/04_5nn_map.png)

## What are those white spaces?

Those are spaces where ties occur.  

How can we deal with ties?  
  - for binary classes  
      - choose an odd number for k
        
  - for multiclass  
      - Reduce the K by 1 to see who wins.  
      - Weight the votes based on the distance of the neighbors  

# Lazy-Learning
![lazy](https://media.giphy.com/media/QSzIZKD16bNeM/giphy.gif)

Lazy-learning has also to do with KNN's training, or better yet, lack of a training step.  Whereas models like linear and logistic fit onto training data, doing the hard work of calculating paramaters when .fit is called, the training phase of KNN is simply storing the training data in memory.  The training step of KNN takes no time at all. All the work is done in the prediction phase, where the distances are calculated. Prediction is therefore memory intensive, and can take a long time.    KNN is lazy because it puts off the work until a later time than most algos.


# Pair 

Use the timeit function to compare the time of fitting and predicting in Logistic vs KNN

Time it example


```python
%%timeit
import nltk 
emma = nltk.corpus.gutenberg.words('austen-emma.txt')

newlist = []
for word in emma:
    newlist.append(word.upper())

```


```python
import nltk 
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
%timeit newlist = [s.upper() for s in emma]
```


```python
%timeit newlist = map(str.upper, emma)
```


```python
# Your code here
```


```python
#__SOLUTION__
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
%timeit lr.fit(X,y)

```


```python
#__SOLUTION__

knn = KNeighborsClassifier()
%timeit knn.fit(X,y)
```


```python
#__SOLUTION__

%timeit knn.predict(X)
```

# 6. Tuning K and the BV Tradeoff


```python
titanic = pd.read_csv('data/cleaned_titanic.csv')
titanic = titanic.iloc[:,:-2]
titanic.head()

X = titanic.drop('Survived', axis=1)
X.head()
X['youngin'] = X.youngin.astype(int)
X.head()
```


```python
from sklearn.model_selection import train_test_split, KFold

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)
# Set test set aside until we are confident in our model
```


```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score

kf = KFold(n_splits=5)

k_scores_train = {}
k_scores_val = {}


for k in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=k, p=2)
    accuracy_score_t = []
    accuracy_score_v = []
    for train_ind, val_ind in kf.split(X_train, y_train):
        
        X_t, y_t = X_train.iloc[train_ind], y_train.iloc[train_ind] 
        X_v, y_v = X_train.iloc[val_ind], y_train.iloc[val_ind]
        mm = MinMaxScaler()
        
        X_t_ind = X_t.index
        X_v_ind = X_v.index
        
        X_t = pd.DataFrame(mm.fit_transform(X_t))
        X_t.index = X_t_ind
        X_v = pd.DataFrame(mm.transform(X_v))
        X_v.index = X_v_ind
        
        knn.fit(X_t, y_t)
        
        y_pred_t = knn.predict(X_t)
        y_pred_v = knn.predict(X_v)
        
        accuracy_score_t.append(accuracy_score(y_t, y_pred_t))
        accuracy_score_v.append(accuracy_score(y_v, y_pred_v))
        
        
    k_scores_train[k] = np.mean(accuracy_score_t)
    k_scores_val[k] = np.mean(accuracy_score_v)
```


```python
fig, ax = plt.subplots(figsize=(15,15))

ax.plot(list(k_scores_train.keys()), list(k_scores_train.values()),color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10, label='Train')
ax.plot(list(k_scores_val.keys()), list(k_scores_val.values()), color='green', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10, label='Val')
ax.set_xlabel('k')
ax.set_ylabel('Accuracy')
plt.legend()
```

### What value of K performs best on our Test data?

### How do you think K size relates to our concepts of bias and variance?

![alt text](img/K-NN_Neighborhood_Size_print.png)


```python
mm = MinMaxScaler()

X_train_ind = X_train.index
X_train = pd.DataFrame(mm.fit_transform(X_train))
X_train.index = X_train_ind

X_test_ind = X_test.index
X_test =  pd.DataFrame(mm.transform(X_test))
X_test.index = X_test_ind


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)



print(f"training accuracy: {knn.score(X_train, y_train)}")
print(f"Test accuracy: {knn.score(X_test, y_test)}")

y_hat = knn.predict(X_test)

plot_confusion_matrix(confusion_matrix(y_test, y_hat), classes=['Perished', 'Survived'])
```


```python
recall_score(y_test, y_hat)
```


```python
precision_score(y_test, y_hat)
```
