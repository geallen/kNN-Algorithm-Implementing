import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as mt
from operator import itemgetter


# Part 1: Load the data
data = np.loadtxt('iris.data')

# Part 2: Plot 1st, 3rd and 4th features in 3D
fig_size = (30, 20)
marker_size = 60
fig = plt.figure(figsize=fig_size)
data0 = data[data[:, 4] == 0]
data1 = data[data[:, 4] == 1]
data2 = data[data[:, 4] == 2]
ax = fig.add_subplot(111, projection='3d')
s = ax.scatter(data0[:, 0], data0[:, 2], data0[:, 3], marker='o', c='r', s=marker_size)
s.set_edgecolors = s.set_facecolors = lambda *args: None
s = ax.scatter(data1[:, 0], data1[:, 2], data1[:, 3], marker='o', c='g', s=marker_size)
s.set_edgecolors = s.set_facecolors = lambda *args: None
s = ax.scatter(data2[:, 0], data2[:, 2], data2[:, 3], marker='o', c='b', s=marker_size)
s.set_edgecolors = s.set_facecolors = lambda *args: None


# Part 3: k-NN sub-functions implementation

# Part 3-1: Random seperation for training and test data.

# , We compute the necessary number of
# training & test sample dinamicly for choosing two third of the data (from each class) for train data
# and the remaining one third for test data


def rand_Train_Test(data):
    RandTrainData = []
    RandTestData = []

    # YOUR CODE HERE

    random_seperation = np.random.rand(len(data)) < (2.0 / 3.0)
    RandTrainData = data[random_seperation]
    RandTestData = data[~random_seperation]

    return RandTrainData, RandTestData


# Part 3-2: Calculate the distance between any two points by using the Euclidean distance metric.
def find_Dist(sample1, sample2):
    euclid_Dist = 0

    # YOUR CODE HERE
    diffs_squared_distance = pow((sample1[0] - sample2[0]),2) + pow((sample1[2] - sample2[2]), 2) + pow((sample1[3] - sample2[3]), 2)
    euclid_Dist = mt.sqrt(diffs_squared_distance)
    return euclid_Dist


# Part 3-3: Find the k nearest(most similar) neighbors of a test sample based on these pairwise distances(among the training dataset).
# Each test sample should be compared with all training data samples.


def find_Neighbours(training_data, test_data, k):
    all_Test_neighbours = []

    # YOUR CODE HERE
    distances = []
    for x in range(len(training_data)):
        distance = find_Dist(test_data, training_data[x])
        distances.append((training_data[x], distance))

    distances.sort(key=itemgetter(1))
    for i in range(k):
        all_Test_neighbours.append(distances[i])

    return all_Test_neighbours

# Part 3-4: Assign the class label of the test sample based on k nearest neighbors' majority.
#  (which class comes up the most often among the nearest neighbours).
#  ***** If the labels of k nearest neighbours are equally distributed for a test sample, you reject this test sample and use remaining test samples for computing the average error***.

#
def assign_Class(all_Test_neighbours):
    Test_Class_ids = []

    # YOUR CODE HERE
    classes = []
    class0 = []
    class1 = []
    class2 = []
    for x in range(len(all_Test_neighbours)):
        classes.append(all_Test_neighbours[x][0][4])

    for x in range(len(classes)):
        if classes[x] == 0.0:
            class0.append((0, 0.0))
        elif classes[x] == 1.0:
            class1.append((1, 1.0))
        else:
            class2.append((2, 2.0))

    if len(class0) == len(class1) == len(class2):
        Test_Class_ids.append('x')
    else:
        maximum = max(class1, class2, class0, key=len)
        Test_Class_ids.append(maximum)
    return Test_Class_ids


# Part 4: Implement the function that runs k-NN with given different k values (1,2,3,4,5,6,7,8,9,10,11,12,13). For each k value, apply your k-NN on randomly seperated train-test data with 50 times.
# For all trials(times), calculate and record error rate then find the average error rate for each k values.

k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
iterNum = 50
average_err_for_ks = []

# YOUR CODE HERE

def kNN(k, average_err_for_ks, iterNum, data):

    equal_labels = 0

    for nK in range(len(k)):
        iN = 0
        err = 0
        while iN < 50:
            training_data, test_data = rand_Train_Test(data)
            for item in range(len(test_data)):
                test_neighbours = find_Neighbours(training_data, test_data[item], nK+1)

                class_ids = assign_Class(test_neighbours)

                if class_ids[0] == 'x':
                    equal_labels += 1

                elif class_ids[0][0][0] != int(test_data[item][4]):
                    err += 1

            iN +=1
        error_rate = err / ((float(len(test_data)) * 50) - equal_labels)
        average_err_for_ks.append(error_rate)
    return average_err_for_ks

kNN(k,average_err_for_ks,iterNum,data)

# Print Average error rate for corresponding k value on command window like below.
for i in range(len(k)):
    print 'Average error rate for k=' + repr(k[i]) + '--> ' + repr(average_err_for_ks[i])

# Part 5: Plot the average error rate as a function of k. Then choose your optimal k value/values.
# Shortly explain your reason and how k value characteristic should be according to class number? (write your respond as a comment)

# YOUR CODE HERE


plt.figure()
plt.plot(k,average_err_for_ks, 'ro')
plt.show()
'''
#################  Your Respond  ############################################

In this Knn algorithm, this code splits data into traning and test data randomly. For randomly selected test data,
knn algorithm works. So each time error differs for each k value. Optimal k value should be with minimal error rate.
That is why for every trial it is different. For example for this trial

Average error rate for k=1--> 0.05021276595744681
Average error rate for k=2--> 0.05
Average error rate for k=3--> 0.03890909090909091
Average error rate for k=4--> 0.042352941176470586
Average error rate for k=5--> 0.044651162790697675
Average error rate for k=6--> 0.04666666666666667
Average error rate for k=7--> 0.044583333333333336
Average error rate for k=8--> 0.041403508771929824
Average error rate for k=9--> 0.03309090909090909
Average error rate for k=10--> 0.04
Average error rate for k=11--> 0.03538461538461538
Average error rate for k=12--> 0.05024390243902439
Average error rate for k=13--> 0.035

optimal k can be 9 with minimum error rate. For general speaking as I observed error rate is less for odd number of k's
than even number of k's.


As class number increases , time used increases. Because in assigning classes, there exists more classes to look for.
Comparision should be done to find which class is more exists.
##############################################################################
'''
