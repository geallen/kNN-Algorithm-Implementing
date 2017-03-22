# kNN-Algorithm-Implementing

This code applies k-NN classification method on already labeled data. 
Each stage of the k-NN algorithm will be implemented step by step, it will run on the given dataset and results will be displayed.

Part 1: Load and Read the Data

The data provided is loaded and read. It is the Iris flower data set (also known as Fisher's Iris data set) which is a multivariate data
set introduced by Sir Ronald Fisher in 1936. The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris 
virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals,
in centimeters. Data matrix has five columns; the first four columns contain the samples measurements, whereas the last column contains
the labels. The code needed for this part is already in the script.

Part 2: Plotting Data in 3D

This part of the code plots the data features in 3D according to their classes for making pre analysis. Only 1st, 3rd and 4th
(more informative) feature columns are displayed and used for the rest of this exercise. The code needed for this part is already 
in the script.

Part 3: Implementing k-NN sub-functions 

1. Random separation: Split data into training and test data randomly. 
2. Similarity: Calculate the distance between any two points by using the Euclidean distance metric.
3. Neighbors: Find the k nearest neighbors of a test sample based on these pairwise distances. 
4. Decision making: Assign the class label of the test sample based on k nearest neighbors’ majority (among the training dataset).

Part 4: Implementing the k-NN function

Use a k-NN classifier to classify the iris data. In order to see the effect of the value of k on the performance of the k-NN classifier, 
try k=1,2,3,4,5,6,7,8,9,10,11,12,13 values each for 50 times on randomly separated dataset. Randomly choose two third of the data
(from each class) for training dataset and the remaining one third for testing the classifier (don’t use python library defined functions
like “cross_validation” for randomized separation). Repeat this data-splitting fifteen times. (That is doing this experiment 50 times
by randomly choosing training and testing data for each k values). Collect the error rates (average) of the test set for these values of k.

Part 5: Finding the optimal k

Now that we have all the required functions, we find the optimal k value. Plot the average error rate as a function of k. Then choose
the optimal k value/values and shortly explain your reason and how k value characteristic should be according to class number?
(Write your decision into the area denoted with “Your Respond”)
