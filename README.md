
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


## Let's load in the Titanic dataset

![titanic](https://media.giphy.com/media/uhB0n3Eac8ybe/giphy.gif)

#### For visualization purposes, we will use only two features for our first model

Titanic is a binary classification problem, with our target being the Survived feature

#### Then perform another tts, and put aside the test set from above until the end

We will hold of from KFold or crossval for now, so that our notebook is more comprehensible.

# Quick review of confusion matrix and our metrics: 
  

# 2. KNN Under the Hood: Voting for K

For visualization purposes, let's pull out a small subset of our training data, and create a model using only two dimensions: Age and Fare.


The KNN algorithm works by simply storing the training set in memory, then measuring the distance from the training points to a a new point.

Let's drop a point from our validation set into the plot above.

Then, KNN finds the K nearest points. K corresponds to the n_neighbors parameter defined when we instantiate the classifier object.

Let's fit our training data, then predict what our validation point will be based on the closest 1 neighbor.

# Chat poll: What will our 1 neighbor KNN classifier predict our new point to be?



When we raise the value of K, KNN acts democratically.  It finds the K closest points, and takes a vote based on the labels.

Let's raise K to 3.

# Chat poll: What will our 3 neighbor KNN classifier predict our new point to be?


It is a bit harder to tell what which points are closest by eye.

Let's update our plot to add indexes.

We can the sklearn NearestNeighors object to see the exact calculations.

# Chat poll: What will our 5 neighbor KNN classifier predict our new point to be?

Let's iterate through K, 1 through 10, and see the predictions.

What K was correct?

# 3. Different types of distance

How did the algo calculate those distances? 

### Euclidean Distance

**Euclidean distance** refers to the distance between two points. These points can be in different dimensional space and are represented by different forms of coordinates. In one-dimensional space, the points are just on a straight number line.


### Measuring distance in a 2-d Space

In two-dimensional space, the coordinates are given as points on the x- and y-axes

![alt text](img/euclidean_2d.png)
### Measuring distance in a 3-d Space

In three-dimensional space, x-, y- and z-axes are used. 

$$\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 +  (z_1-z_2)^2}$$
![alt text](img/vectorgraph.jpg)

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

If we change the distance metric, our prediction should change for K = 5.

# 4. Importance of Scaling

You may have suspected that we were leaving something out. For any distance based algorithms, scaling is very important.  Look at how the shape of array changes before and after scaling.

![non-normal](img/nonnormal.png)

![normal](img/normalized.png)

Let's look at our data for viz dataset

Look at how much that changes things.

Look at 166 and 150.  
Look at 827 to 786

Now let's run our classifier on scaled data and compare to unscaled.

# 5. Let's unpack: KNN is a supervised, non-parametric, descriminative, lazy-learning algorithm

## Supervised
You should be very comfortable with the idea of supervised learning by now.  Supervised learning involves labels.  KNN needs labels for the voting process.



# Non-parametric

Let's look at the fit KNN classifier.

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
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
%timeit lr.fit(X,y)

```


```python

knn = KNeighborsClassifier()
%timeit knn.fit(X,y)
```


```python

%timeit knn.predict(X)
```

# 6. Tuning K and the BV Tradeoff

### What value of K performs best on our val data?

### How do you think K size relates to our concepts of bias and variance?

![alt text](img/K-NN_Neighborhood_Size_print.png)
