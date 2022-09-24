from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
import time

def eucliden_distance(data1, data2):
    distance = 0
    for i in range(len(data1) - 1):
        distance += (data1[i] - data2[i]) ** 2
    return sqrt(distance)


def dist(a,b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def KNN(obj, X_train, Y_train, k):

    mark = [0, 0]
    
    #test_dist = [[dist(obj, X_train[i]), i] for i in range(len(X_train
    test_dist = [[np.sqrt( ( (obj[0] - X_train[i][0])**2 + (obj[1] - X_train[i][1])**2 ) ), i] for i in range(len(X_train))]
    test_dist = sorted(test_dist, key=lambda x: x[0])
    
    for i in range(k):
        if Y_train[test_dist[i][1]] == 1:
            mark[0] += 1 * (1 - (test_dist[i][0] / test_dist[k][0]))
        else:
            mark[1] += 1 * (1 - (test_dist[i][0] / test_dist[k][0]))
    if mark[0] > mark[1]:
        return 1
    else:
        return 0

def LOO_KNN(obj, X, Y, k):
    res_mark = [0, 0] 
    for i in range(k):
        if Y[i] == 1:
            res_mark[0] += 1
        else:
            res_mark[1] += 1
    if res_mark[0] > res_mark[1]:
        return 1
    else:
        return 0

def LOO(k, X_test, Y_test):
    miss = 0
    for i in range(len(X_test)):
        if Y_test[i] != LOO_KNN(X_test[i], np.delete(X_test, i, 0), Y_test, k):
            miss += 1
    return miss

#====================================main=================================
data = np.loadtxt('data4.csv', delimiter=',', skiprows=1)
   
X_data = np.stack((data[:, 0], data[:, 1]), axis=1)
Y_data = data[:, 2]

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, stratify=Y_data, test_size=0.33, random_state=12345)


k_neighbors = [[LOO(i, X_train, Y_train), i] for i in range(3, 24, 2)]
print("=SEARCH BEST K\n")
for i in range(len(k_neighbors)):
    print(f"{k_neighbors[i][1]} : {k_neighbors[i][0] / len(X_train)}")

minK = min(k_neighbors, key=lambda k_neighbors: k_neighbors[0])[1]
print(f"\nBest K: {minK}")

miss = 0

for i in range(len(Y_test)):
    predict = KNN(X_test[i], X_train, Y_train, minK)
    if predict != Y_test[i]:
        miss += 1

print(f"Accuracy: {100 - (miss / len(Y_test) * 100)}")
